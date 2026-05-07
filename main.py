import json
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import requests
from datetime import datetime
import time
import threading

# Configuration
RAPP_URL = "http://192.168.8.69:32500"
SIDELOADER_URL = "http://192.168.206.82:32501"
INSTANCE_ID = "0fedeadc-c46b-4675-9f5e-38f3e5470bd6"
BITRATES = [600, 700, 800]
IPERF_DURATION = 10  # seconds
PERF_DURATION = 10   # seconds
PTP_INTERFACE = "eth0"


def trigger_iperf(bitrate):
    """Trigger iperf test on UE"""
    print(f"Starting iperf test at {bitrate} Mbps...")
    try:
        response = requests.post(
            f"{RAPP_URL}/ue/iperf",
            json={"duration": IPERF_DURATION, "bitrate": bitrate},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
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
    """Fetch thread CPU data directly from sideloader"""
    print("  Fetching thread CPU data...")
    try:
        response = requests.post(
            f"{SIDELOADER_URL}/process/threads",
            json={"duration": PERF_DURATION, "pgrep": "gnb", "include_timeseries": True},
            headers={"Content-Type": "application/json"},
            timeout=PERF_DURATION + 30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  thread_cpu failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  thread_cpu error: {e}")
        return None


def fetch_cpu_monitor():
    """Fetch per-core CPU usage timeseries directly from sideloader"""
    print("  Fetching per-core CPU data...")
    try:
        response = requests.post(
            f"{SIDELOADER_URL}/cpu/monitor",
            json={"duration": PERF_DURATION, "include_timeseries": True},
            timeout=PERF_DURATION + 30,
        )
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  cpu_monitor failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  cpu_monitor error: {e}")
        return None


def fetch_ptp_status():
    """Fetch PTP synchronization status"""
    print("  Fetching PTP status...")
    try:
        response = requests.post(
            f"{RAPP_URL}/sideload/measure/ptp",
            json={"instance_id": INSTANCE_ID, "duration": 5,
                  "interface": PTP_INTERFACE, "include_timeseries": False},
            timeout=15,
        )
        if response.status_code == 200:
            data = response.json()
            offset = data.get("offset_ns") or data.get("mean_offset_ns") or 0
            return abs(float(offset))
        else:
            print(f"  ptp_status failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"  ptp_status error: {e}")
        return None


def _sort_cpu_keys(keys):
    """Sort cpu keys numerically: cpu0, cpu1, ..."""
    def _k(k):
        m = re.match(r"^cpu(\d+)$", k)
        return (0, int(m.group(1))) if m else (1, k)
    return sorted(keys, key=_k)


def generate_plots(data, bitrate, timestamp):
    """Generate thread heatmap and affinity heatmap for given bitrate"""

    # Thread CPU usage heatmap
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

        matrix, labels = [], []
        for thread in sorted(threads_with_cores, key=lambda x: float(x['tid'])):
            core_dist = {int(k): v for k, v in thread['core_distribution'].items()}
            total_samples = sum(core_dist.values())
            row = [(core_dist.get(c, 0) / total_samples * 100) if total_samples > 0 else 0
                   for c in range(max_core + 1)]
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
            ax.legend(handles=[Patch(facecolor='none', edgecolor='blue',
                                     hatch='///', label='Offline CPU')], loc='upper right')

        plt.title(f'Thread-to-Core Affinity - {bitrate} Mbps\n{timestamp}')
        plt.xlabel('CPU Core')
        plt.ylabel('Thread Name & ID')
        plt.tight_layout()
        affinity_file = f'affinity_heatmap_{bitrate}mbps_{timestamp}.png'
        plt.savefig(affinity_file, dpi=150)
        print(f"  Saved {affinity_file}")
        plt.close()


def generate_combined_cpu_plots(all_core_data, bitrates, timestamp):
    """
    Generate combined CPU plots across all bitrates.
    all_core_data: list of cpu_monitor results, one per bitrate (same order as bitrates)
    """
    if not all_core_data or all(d is None for d in all_core_data):
        print("  No per-core CPU data to plot")
        return

    # Collect all cpu keys across all segments
    all_cpu_keys = set()
    for data in all_core_data:
        if data and "cpus" in data:
            for k in data["cpus"]:
                if re.match(r"^cpu\d+$", k):
                    all_cpu_keys.add(k)
    cpu_keys = _sort_cpu_keys(list(all_cpu_keys))

    # Build concatenated timeseries per core
    # elapsed_time resets at start of each bitrate segment, we offset them
    combined_timestamps = []   # global elapsed seconds
    combined_percent = {k: [] for k in cpu_keys}
    bitrate_boundaries = []    # x positions of bitrate switches (in seconds)

    current_offset = 0.0

    for seg_idx, (data, bitrate) in enumerate(zip(all_core_data, bitrates)):
        if data is None or "cpus" not in data:
            # Fill with zeros for missing segment
            segment_len = PERF_DURATION
            seg_times = [current_offset + i for i in range(segment_len)]
            combined_timestamps.extend(seg_times)
            for k in cpu_keys:
                combined_percent[k].extend([0.0] * segment_len)
            current_offset += segment_len
            bitrate_boundaries.append(current_offset)
            continue

        # Get timestamps from any available core
        seg_times_raw = None
        for k in cpu_keys:
            cpu_entry = data["cpus"].get(k)
            if cpu_entry:
                ts_data = cpu_entry["usage"].get("timeseries")
                if ts_data and "timestamps" in ts_data and ts_data["timestamps"]:
                    t0 = ts_data["timestamps"][0]
                    seg_times_raw = [t - t0 for t in ts_data["timestamps"]]
                    break

        if seg_times_raw is None:
            seg_times_raw = list(range(PERF_DURATION))

        seg_times_global = [current_offset + t for t in seg_times_raw]
        combined_timestamps.extend(seg_times_global)

        for k in cpu_keys:
            cpu_entry = data["cpus"].get(k)
            if cpu_entry:
                ts_data = cpu_entry["usage"].get("timeseries")
                if ts_data and "percent" in ts_data:
                    pct = ts_data["percent"]
                    # Pad or trim to match seg_times_raw length
                    n = len(seg_times_raw)
                    if len(pct) < n:
                        pct = pct + [0.0] * (n - len(pct))
                    combined_percent[k].extend(pct[:n])
                else:
                    combined_percent[k].extend([0.0] * len(seg_times_raw))
            else:
                combined_percent[k].extend([0.0] * len(seg_times_raw))

        current_offset = seg_times_global[-1] if seg_times_global else current_offset + PERF_DURATION
        if seg_idx < len(bitrates) - 1:
            bitrate_boundaries.append(current_offset)

    # ------------------------------------------------------------------ #
    # Plot 1: CPU Core Heatmap (time x core, color = usage %)
    # ------------------------------------------------------------------ #
    # Build 2D matrix: rows=cores, cols=time samples
    matrix = np.array([combined_percent[k] for k in cpu_keys])  # shape: (n_cores, n_time)

    fig, ax = plt.subplots(figsize=(max(14, len(combined_timestamps) * 0.15), max(8, len(cpu_keys) * 0.4)))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=100,
                   extent=[combined_timestamps[0], combined_timestamps[-1],
                            len(cpu_keys) - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label='Usage (%)')

    # Y axis: cpu labels
    ax.set_yticks(range(len(cpu_keys)))
    ax.set_yticklabels(cpu_keys, fontsize=8)

    # Bitrate boundary lines and labels
    for boundary in bitrate_boundaries:
        ax.axvline(x=boundary, color='white', linewidth=1.5, linestyle='--', alpha=0.8)

    # Bitrate labels in middle of each segment
    segment_starts = [0.0] + bitrate_boundaries
    segment_ends = bitrate_boundaries + [combined_timestamps[-1]]
    for i, bitrate in enumerate(bitrates):
        mid = (segment_starts[i] + segment_ends[i]) / 2
        ax.text(mid, len(cpu_keys) * 0.05, f'{bitrate}M',
                ha='center', va='top', fontsize=10, fontweight='bold',
                color='white',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

    ax.set_title(f'CPU Heatmap - {" / ".join(str(b) + "M" for b in bitrates)} Mbps\n{timestamp}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('CPU Core')
    plt.tight_layout()
    heatmap_file = f'cpu_core_heatmap_combined_{timestamp}.png'
    plt.savefig(heatmap_file, dpi=150)
    print(f"  Saved {heatmap_file}")
    plt.close()

    # ------------------------------------------------------------------ #
    # Plot 2: CPU Core Timeseries (active cores only)
    # ------------------------------------------------------------------ #
    # Only plot cores with avg > 1% across all segments
    active_cores = [k for k in cpu_keys
                    if np.mean([v for v in combined_percent[k] if v is not None]) > 1.0]
    if not active_cores:
        active_cores = cpu_keys

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap("tab20")
    for i, cpu_key in enumerate(active_cores):
        ax.plot(combined_timestamps, combined_percent[cpu_key],
                label=cpu_key, color=cmap(i % 20), linewidth=1.5)

    # Bitrate boundary lines
    for boundary in bitrate_boundaries:
        ax.axvline(x=boundary, color='gray', linewidth=1, linestyle='--', alpha=0.7)

    # Bitrate labels
    for i, bitrate in enumerate(bitrates):
        mid = (segment_starts[i] + segment_ends[i]) / 2
        ax.text(mid, 102, f'{bitrate}M', ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    ax.set_title(f'CPU Core Usage - {" / ".join(str(b) + "M" for b in bitrates)} Mbps\n{timestamp}\nActive Cores (>1%)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('CPU Usage (%)')
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=8, ncol=4,
              title='Active Cores (>1%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    timeseries_file = f'cpu_core_timeseries_combined_{timestamp}.png'
    plt.savefig(timeseries_file, dpi=150)
    print(f"  Saved {timeseries_file}")
    plt.close()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []
    all_core_data = []   # collect per-core CPU data across all bitrates

    print(f"Starting bitrate sweep test - {len(BITRATES)} steps")
    print(f"Timestamp: {timestamp}\n")

    for bitrate in BITRATES:
        print(f"\n{'='*60}")
        print(f"Testing at {bitrate} Mbps")
        print(f"{'='*60}")

        # Run iperf, thread CPU, per-core CPU in parallel
        _iperf = [None]
        _cpu = [None]
        _core_cpu = [None]

        def _run_iperf(): _iperf[0] = trigger_iperf(bitrate)
        def _run_cpu():
            time.sleep(1)
            _cpu[0] = fetch_thread_cpu()
        def _run_core_cpu():
            time.sleep(1)
            _core_cpu[0] = fetch_cpu_monitor()

        t1 = threading.Thread(target=_run_iperf)
        t2 = threading.Thread(target=_run_cpu)
        t3 = threading.Thread(target=_run_core_cpu)
        t1.start(); t2.start(); t3.start()
        t1.join(); t2.join(); t3.join()

        iperf_result = _iperf[0]
        cpu_data = _cpu[0]
        core_cpu_data = _core_cpu[0]

        # Store per-core data for combined plot
        all_core_data.append(core_cpu_data)

        # Fetch PTP status
        ptp_offset = fetch_ptp_status()
        if ptp_offset is not None:
            print(f"  PTP offset: {ptp_offset:.1f} ns")

        if cpu_data:
            json_file = f'threads_{bitrate}mbps_{timestamp}.json'
            with open(json_file, 'w') as f:
                json.dump(cpu_data, f, indent=2)
            print(f"  Saved {json_file}")
            generate_plots(cpu_data, bitrate, timestamp)

        if core_cpu_data:
            core_json_file = f'cpu_cores_{bitrate}mbps_{timestamp}.json'
            with open(core_json_file, 'w') as f:
                json.dump(core_cpu_data, f, indent=2)
            print(f"  Saved {core_json_file}")

        results.append({
            'bitrate': bitrate,
            'total_threads': cpu_data.get('total_threads', 0) if cpu_data else 0,
            'ptp_offset_ns': ptp_offset,
            'iperf_result': iperf_result
        })

        print(f"  Waiting 5s before next test...")
        time.sleep(5)

    # Generate combined CPU plots across all bitrates
    print(f"\nGenerating combined CPU plots...")
    generate_combined_cpu_plots(all_core_data, BITRATES, timestamp)

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
