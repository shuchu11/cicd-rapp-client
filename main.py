import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import time

# Configuration
RAPP_URL = "http://192.168.8.69:32500"
#INSTANCE_ID = "c4c1202c-9e61-4c7f-8595-5eed1249bdaf" # Joule
INSTANCE_ID = "0fedeadc-c46b-4675-9f5e-38f3e5470bd6" # Lavoisier Baremetal
#BITRATES = [10, 50, 100, 200, 300, 400, 500, 600, 700]  # Mbps
#BITRATES = [10, 50, 100, 200, 300, 400]  # Mbps
BITRATES = [600, 700, 800]
#BITRATES = [50, 200, 500, 800]  # Mbps
#BITRATES = [1]  # Mbps
IPERF_DURATION = 10  # seconds
PERF_DURATION = 10  # seconds (longer to capture full iperf test)
PTP_INTERFACE = "eth0"  # PTP interface name


def trigger_iperf(bitrate):
    """Trigger iperf test on UE"""
    print(f"Starting iperf test at {bitrate} Mbps...")
    url = f"{RAPP_URL}/ue/iperf"
    payload = {
        "duration": IPERF_DURATION,
        "bitrate": bitrate
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            print(f"  iperf started at {bitrate} Mbps")
            return response.json()
        else:
            print(f"  iperf failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  iperf error: {e}")
        return None


def fetch_thread_cpu():
    """Fetch thread CPU data during test"""
    print("  Fetching thread CPU data...")
    url = f"http://192.168.206.82:32501/process/threads"
    payload = {
        
        "duration": PERF_DURATION,
        "pgrep": "gnb",
        "include_timeseries": True,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=PERF_DURATION + 30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  thread_cpu failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  thread_cpu error: {e}")
        return None


def fetch_ptp_status():
    """Fetch PTP synchronization status"""
    print("  Fetching PTP status...")
    url = f"{RAPP_URL}/sideload/measure/ptp"
    payload = {
        
        "duration": 5,
        "interface": PTP_INTERFACE,
        "include_timeseries": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=15)
        if response.status_code == 200:
            data = response.json()
            # Extract offset from ptp monitor response
            offset = data.get("offset_ns") or data.get("mean_offset_ns") or 0
            return abs(float(offset))
        else:
            print(f"  ptp_status failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ptp_status error: {e}")
        return None


def generate_plots(data, bitrate, timestamp):
    """Generate heatmaps for given bitrate"""

    # CPU usage heatmap
    df = pd.DataFrame(data['threads'])
    df['tid_numeric'] = pd.to_numeric(df['tid'])
    df = df.sort_values('tid_numeric')
    df['label'] = df['name'] + " (TID: " + df['tid'] + ")"
    df.set_index('label', inplace=True)
    df["min_cpu"] = df["cpu_usage"].apply(lambda x: x["min"])
    df["avg_cpu"] = df["cpu_usage"].apply(lambda x: x["avg"])
    df["max_cpu"] = df["cpu_usage"].apply(lambda x: x["max"])
    heatmap_data = df[["min_cpu", "avg_cpu", "max_cpu"]]

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title(f'Thread CPU Usage - {bitrate} Mbps\n{timestamp}')
    plt.xlabel('CPU Metrics')
    plt.ylabel('Thread Name & ID')
    plt.tight_layout()
    cpu_file = f'cpu_heatmap_{bitrate}mbps_{timestamp}.png'
    plt.savefig(cpu_file, dpi=150)
    print(f"  Saved {cpu_file}")
    plt.close()

    # Core affinity heatmap
    threads_with_cores = [t for t in data['threads'] if 'core_distribution' in t and t['core_distribution']]

    if threads_with_cores:
        system_max_cpu = data.get('system_max_cpu', 31)
        offline_cpus = set(data.get('offline_cpus', []))

        max_core_in_data = max(max(int(k) for k in t['core_distribution'].keys()) for t in threads_with_cores)
        max_core = max(max_core_in_data, system_max_cpu)

        matrix = []
        labels = []

        for thread in sorted(threads_with_cores, key=lambda x: float(x['tid'])):
            row = []
            core_dist = {int(k): v for k, v in thread['core_distribution'].items()}
            total_samples = sum(core_dist.values())

            for core in range(max_core + 1):
                count = core_dist.get(core, 0)
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                row.append(percentage)

            matrix.append(row)
            labels.append(f"{thread['name']} (TID: {thread['tid']})")

        df_cores = pd.DataFrame(matrix, index=labels, columns=[f"CPU{i}" for i in range(max_core + 1)])

        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(df_cores, annot=True, cmap='YlOrRd', fmt='.0f',
                    cbar_kws={'label': '% of samples'}, ax=ax)

        for cpu_id in offline_cpus:
            if cpu_id <= max_core:
                ax.add_patch(plt.Rectangle((cpu_id, 0), 1, len(df_cores),
                                           fill=False, edgecolor='blue',
                                           hatch='///', linewidth=2, zorder=10))

        if offline_cpus:
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='none', edgecolor='blue',
                                    hatch='///', label='Offline CPU')]
            ax.legend(handles=legend_elements, loc='upper right')

        plt.title(f'Thread-to-Core Affinity - {bitrate} Mbps\n{timestamp}\nOffline CPUs: {sorted(offline_cpus) if offline_cpus else "None"}')
        plt.xlabel('CPU Core')
        plt.ylabel('Thread Name & ID')
        plt.tight_layout()
        affinity_heatmap = f'affinity_heatmap_{bitrate}mbps_{timestamp}.png'
        plt.savefig(affinity_heatmap, dpi=150)
        print(f"  Saved {affinity_heatmap}")
        plt.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    print(f"Starting bitrate sweep test - {len(BITRATES)} steps")
    print(f"Timestamp: {timestamp}\n")

    for bitrate in BITRATES:
        print(f"\n{'='*60}")
        print(f"Testing at {bitrate} Mbps")
        print(f"{'='*60}")

        # Run iperf and CPU monitoring in parallel
        import threading
        _iperf = [None]
        _cpu = [None]
        def _run_iperf(): _iperf[0] = trigger_iperf(bitrate)
        def _run_cpu():
            time.sleep(1)
            _cpu[0] = fetch_thread_cpu()
        t1 = threading.Thread(target=_run_iperf)
        t2 = threading.Thread(target=_run_cpu)
        t1.start(); t2.start()
        t1.join(); t2.join()
        iperf_result = _iperf[0]
        cpu_data = _cpu[0]

        # Fetch PTP status
        ptp_offset = fetch_ptp_status()
        if ptp_offset is not None:
            print(f"  PTP offset: {ptp_offset:.1f} ns")

        if cpu_data:
            # Save raw data
            json_file = f'threads_{bitrate}mbps_{timestamp}.json'
            with open(json_file, 'w') as f:
                json.dump(cpu_data, f, indent=2)
            print(f"  Saved {json_file}")

            # Generate plots
            generate_plots(cpu_data, bitrate, timestamp)

            # Store summary
            results.append({
                'bitrate': bitrate,
                'total_threads': cpu_data.get('total_threads', 0),
                'active_threads': cpu_data.get('total_threads', 0),
                'ptp_offset_ns': ptp_offset,
                'iperf_result': iperf_result
            })

        # Wait before next test
        print(f"  Waiting 5s before next test...")
        time.sleep(5)

    # Save summary
    summary_file = f'test_summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All tests complete!")
    print(f"Summary saved to {summary_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
