import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import time
import numpy as np

# Configuration
RAPP_URL = "http://192.168.8.35:5000"
NFO_URL = "http://192.168.8.35:8080"
INSTANCE_ID = "be0cc18b-1b8e-4b58-a2bc-9f4681ac6142"
IPERF_BITRATE = 200  # Mbps
IPERF_DURATION = 10
PERF_DURATION = 15
PTP_INTERFACE = "eth0"  # PTP interface name

# --- TEST MATRIX CONFIGURATION ---
RUNS_PER_CASE = 1
CPU_VARIANTS = [8, 10, 12, 14]

# Define the vendor specifics
VENDORS = [
    {"name": "pegatron", "branch": "starlingx/pegatron", "id_prefix": "PEG"},
    {"name": "liteon",   "branch": "starlingx/liteon",   "id_prefix": "LIT"}
]

# Generate the full matrix
TEST_SCENARIOS = []
for vendor in VENDORS:
    for cpu_count in CPU_VARIANTS:
        test_id = f"{vendor['id_prefix']}-CPU-{cpu_count}"
        TEST_SCENARIOS.append({
            "test_id": test_id,
            "oru": vendor['name'],
            "branch": vendor['branch'],
            "runs": RUNS_PER_CASE,
            "custom_values": {
                "resources": {
                    "define": True,
                    "limits": {
                        "nf": {
                            "wr_isolcpus": cpu_count
                        }
                    },
                    "requests": {
                        "nf": {
                            "wr_isolcpus": cpu_count
                        }
                    }
                }
            }
        })


def build_gnb_config(scenario):
    safe_name = f"oai-gnb-{scenario['test_id'].lower().replace('_', '-')}"
    return {
        "name": safe_name,
        "description": f"{scenario['test_id']}: {scenario['oru'].upper()} - {scenario['custom_values']['resources']['limits']['nf']['wr_isolcpus']} Cores",
        "profile_type": "kubernetes",
        "artifact_repo_url": "https://github.com/motangpuar/ocloud-helm-templates.git",
        "artifact_name": "oai-gnb-fhi-72",
        "artifact_repo_branch": scenario['branch'],
        "target_cluster": "cc1397ba-b1c4-4a3e-bc8d-6af58ef53818",
        "values": scenario['custom_values']
    }


def deploy_gnb(config):
    print(f"  Deploying {config['name']}...")
    try:
        resp = requests.post(f"{RAPP_URL}/nfo/deploy", json=config, timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            deployment_id = data.get('instance_id')
            print(f"    ✓ Deployed: {deployment_id}")
            return deployment_id
        else:
            print(f"    ✗ Deploy failed: {resp.status_code}")
            return None
    except Exception as e:
        print(f"    ✗ Deploy error: {e}")
        return None


def terminate_gnb(deployment_id):
    print(f"  Terminating gNB {deployment_id}...")
    try:
        resp = requests.post(f"{NFO_URL}/api/o2dms/v2/deployments/{deployment_id}/terminate/", timeout=60)
        if resp.status_code == 200:
            print("    ✓ Terminated")
            return True
        return False
    except Exception as e:
        print(f"    ✗ Terminate error: {e}")
        return False


def check_ue_status():
    try:
        resp = requests.get(f"{RAPP_URL}/ue/status", timeout=10)
        if resp.status_code == 200:
            status = resp.json()
            attached = status.get('attached', False)
            has_data_ip = status.get('data_ip') is not None
            return attached and has_data_ip, status
        return False, None
    except:
        return False, None


def get_ptp_status():
    """Fetch PTP offset via rAPP proxy"""
    try:
        resp = requests.post(
            f"{RAPP_URL}/sideload/measure/ptp",
            json={
                "instance_id": INSTANCE_ID,
                "duration": 5,
                "interface": PTP_INTERFACE,
                "include_timeseries": False,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            offset = data.get("offset_ns") or data.get("mean_offset_ns") or 0
            return abs(float(offset))
        return None
    except:
        return None


def toggle_airplane_mode():
    try:
        requests.post(f"{RAPP_URL}/ue/airplane/on", json={"enable": True}, timeout=10)
        time.sleep(2)
        requests.post(f"{RAPP_URL}/ue/airplane/off", json={"enable": False}, timeout=10)
        time.sleep(5)
    except:
        pass


def run_iperf_test():
    try:
        resp = requests.post(
            f"{RAPP_URL}/ue/iperf",
            json={"duration": IPERF_DURATION, "bitrate": IPERF_BITRATE},
            timeout=IPERF_DURATION + 20,
        )
        if resp.status_code == 200:
            data = resp.json()
            stream = data.get('end', {}).get('streams', [{}])[0].get('udp', {})
            return {
                'throughput_mbps': stream.get('bits_per_second', 0) / 1_000_000,
                'jitter_ms': stream.get('jitter_ms', 0),
                'lost_percent': stream.get('lost_percent', 0)
            }
        return None
    except:
        return None


def fetch_thread_cpu():
    """Fetch thread CPU data via rAPP proxy"""
    print("  Fetching thread CPU data...")
    try:
        resp = requests.post(
            f"{RAPP_URL}/sideload/measure/thread_cpu",
            json={
                "instance_id": INSTANCE_ID,
                "duration": PERF_DURATION,
                "pgrep": "softmodem",
                "include_timeseries": True,
            },
            timeout=PERF_DURATION + 10,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None


def generate_plots(data, test_label, timestamp, iperf_result, ue_status):
    try:
        df = pd.DataFrame(data['threads'])
        df['tid_numeric'] = pd.to_numeric(df['tid'])
        df = df.sort_values('tid_numeric')
        df['label'] = df['name'] + " (TID: " + df['tid'] + ")"
        df.set_index('label', inplace=True)
        heatmap_data = df[['min_cpu', 'avg_cpu', 'max_cpu']]

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.1f')
        plt.title(f'Thread CPU Usage - {test_label}\n{timestamp}')
        plt.tight_layout()
        plt.savefig(f'cpu_heatmap_{test_label}_{timestamp}.png', dpi=150)
        plt.close()
    except Exception as e:
        print(f"  ! Plot error: {e}")


def run_single_test(scenario, run_number, timestamp):
    print(f"\n  Run #{run_number + 1}/{scenario['runs']}")

    toggle_airplane_mode()
    attached, ue_status = check_ue_status()

    # --- PLMN CHECK ---
    signal = None
    if ue_status:
        cell_info = ue_status.get('cell', {})
        mcc = str(cell_info.get('mcc', ''))
        mnc = str(cell_info.get('mnc', ''))

        if mcc == '101' and mnc == '01':
            signal = ue_status.get('signal')
        elif attached:
            print(f"    ! WRONG PLMN DETECTED: {mcc}{mnc} (Expected 10101)")
            signal = None

    if attached and signal is None:
        print("    ! Signal missing or invalid PLMN. Retrying...")
        time.sleep(5)
        _, ue_status = check_ue_status()

        if ue_status:
            cell_info = ue_status.get('cell', {})
            mcc = str(cell_info.get('mcc', ''))
            mnc = str(cell_info.get('mnc', ''))

            if mcc == '101' and mnc == '01':
                signal = ue_status.get('signal')
            else:
                print(f"    ! WRONG PLMN ON RETRY: {mcc}{mnc}")
                signal = None

    if not attached or signal is None:
        print("    ! NO VALID SIGNAL (10101). Skipping traffic test.")
        return {
            'run': run_number + 1,
            'status': 'NO_SIGNAL',
            'throughput_mbps': 0.0,
            'jitter_ms': 0.0,
            'lost_percent': 0.0,
            'ptp_rms_ns': get_ptp_status(),
            'ue_rsrp': None,
            'ue_rsrq': None,
            'ue_sinr': None,
            'cpu_data': None
        }

    rsrp = signal.get('rsrp') if signal else None
    rsrq = signal.get('rsrq') if signal else None
    sinr = signal.get('sinr') if signal else None

    print(f"    UE (10101): RSRP={rsrp} dBm, RSRQ={rsrq} dB, SINR={sinr} dB")

    iperf_result = run_iperf_test()
    time.sleep(1)
    cpu_data = fetch_thread_cpu()

    if cpu_data:
        with open(f"{scenario['test_id']}_run{run_number + 1}_{timestamp}.json", 'w') as f:
            json.dump({'cpu': cpu_data, 'iperf': iperf_result, 'ue': ue_status}, f)
        generate_plots(cpu_data, f"{scenario['test_id']}_R{run_number + 1}", timestamp, iperf_result, ue_status)

    return {
        'run': run_number + 1,
        'status': 'SUCCESS',
        'throughput_mbps': iperf_result['throughput_mbps'] if iperf_result else 0.0,
        'jitter_ms': iperf_result['jitter_ms'] if iperf_result else 0.0,
        'lost_percent': iperf_result['lost_percent'] if iperf_result else 0.0,
        'ptp_rms_ns': get_ptp_status(),
        'ue_rsrp': rsrp,
        'ue_rsrq': rsrq,
        'ue_sinr': sinr,
        'cpu_data': cpu_data
    }


def run_test_scenario(scenario, timestamp):
    print(f"\n{'='*70}")
    print(f"Test {scenario['test_id']}: {scenario['oru'].upper()}")
    print(f"Resources: {scenario['custom_values']['resources']}")
    print(f"{'='*70}")

    gnb_config = build_gnb_config(scenario)
    deployment_id = deploy_gnb(gnb_config)

    if not deployment_id:
        return {'test_id': scenario['test_id'], 'successful_runs': 0}

    print("  Waiting 30s for stabilization...")
    time.sleep(30)

    run_results = []
    try:
        for run_num in range(scenario['runs']):
            run_results.append(run_single_test(scenario, run_num, timestamp))
            if run_num < scenario['runs'] - 1:
                time.sleep(3)
    finally:
        terminate_gnb(deployment_id)

    valid_runs = [r for r in run_results if r['status'] in ['SUCCESS', 'NO_SIGNAL']]

    if valid_runs:
        def safe_mean(key):
            vals = [r[key] for r in valid_runs if r.get(key) is not None]
            return np.mean(vals) if vals else None

        stats = {
            'test_id': scenario['test_id'],
            'oru': scenario['oru'].upper(),
            'isolcpus': scenario['custom_values']['resources']['limits']['nf']['wr_isolcpus'],
            'total_runs': scenario['runs'],
            'successful_runs': len(valid_runs),
            'avg_throughput_mbps': np.mean([r['throughput_mbps'] for r in valid_runs]),
            'avg_jitter_ms': np.mean([r['jitter_ms'] for r in valid_runs]),
            'avg_loss_percent': np.mean([r['lost_percent'] for r in valid_runs]),
            'avg_ptp_rms_ns': safe_mean('ptp_rms_ns'),
            'avg_rsrp_dbm': safe_mean('ue_rsrp'),
            'avg_rsrq_db': safe_mean('ue_rsrq'),
            'avg_sinr_db': safe_mean('ue_sinr'),
            'raw_results': run_results
        }
    else:
        stats = {'test_id': scenario['test_id'], 'successful_runs': 0}

    return stats


def generate_summary_report(all_results, timestamp):
    valid_results = [r for r in all_results if r and 'test_id' in r]
    if not valid_results:
        return

    # --- Prepare CPU Stacked Data ---
    cpu_breakdown = {}
    for r in valid_results:
        test_id = r['test_id']
        cpu_breakdown[test_id] = {}
        if r.get('raw_results'):
            thread_totals = {}
            run_count = 0
            for run in r['raw_results']:
                if run.get('cpu_data') and run['cpu_data'].get('threads'):
                    run_count += 1
                    for t in run['cpu_data']['threads']:
                        thread_totals[t['name']] = thread_totals.get(t['name'], 0) + t['avg_cpu']
            if run_count > 0:
                for name, total in thread_totals.items():
                    cpu_breakdown[test_id][name] = total / run_count

    df_cpu = pd.DataFrame.from_dict(cpu_breakdown, orient='index').fillna(0)
    if not df_cpu.empty:
        df_cpu.sort_index(inplace=True)
        top_threads = df_cpu.sum().nlargest(8).index
        df_cpu['Others'] = df_cpu.loc[:, ~df_cpu.columns.isin(top_threads)].sum(axis=1)
        df_cpu = df_cpu[list(top_threads) + ['Others']]

    # --- Plotting (3x3 Grid) ---
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.3)

    test_ids = [r['test_id'] for r in valid_results]
    colors = []
    for tid in test_ids:
        if "PEG" in tid:
            colors.append('#ff7f0e')
        elif "LIT" in tid:
            colors.append('#1f77b4')
        else:
            colors.append('gray')

    def simple_bar(ax, data, title, ylabel):
        ax.bar(test_ids, data, color=colors, alpha=0.7)
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

    # ROW 1: Network Performance
    simple_bar(fig.add_subplot(gs[0, 0]), [r.get('avg_throughput_mbps', 0) for r in valid_results], 'DL Throughput', 'Mbps')
    simple_bar(fig.add_subplot(gs[0, 1]), [r.get('avg_jitter_ms', 0) or 0 for r in valid_results], 'Average Jitter', 'ms')
    simple_bar(fig.add_subplot(gs[0, 2]), [r.get('avg_loss_percent', 0) or 0 for r in valid_results], 'Packet Loss', '%')

    # ROW 2: Signal Quality
    simple_bar(fig.add_subplot(gs[1, 0]), [r.get('avg_rsrp_dbm', 0) or 0 for r in valid_results], 'RSRP', 'dBm')
    simple_bar(fig.add_subplot(gs[1, 1]), [r.get('avg_rsrq_db', 0) or 0 for r in valid_results], 'RSRQ', 'dB')
    simple_bar(fig.add_subplot(gs[1, 2]), [r.get('avg_sinr_db', 0) or 0 for r in valid_results], 'SINR', 'dB')

    # ROW 3: CPU Stack & Table
    ax7 = fig.add_subplot(gs[2, :2])
    if not df_cpu.empty:
        df_cpu.plot(kind='bar', stacked=True, ax=ax7, colormap='tab20', width=0.6)
        ax7.set_title('CPU Usage Breakdown by Thread', fontweight='bold')
        ax7.set_ylabel('CPU Usage (%)')
        ax7.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=5, fontsize=9)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=0, fontsize=9)
        ax7.grid(axis='y', alpha=0.3)

    # Parameter Table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    table_data = [[r['test_id'], r['oru'], r['isolcpus']] for r in valid_results]
    table = ax9.table(cellText=table_data, colLabels=['ID', 'Vendor', 'Cores'], loc='center')
    table.scale(1, 1.5)

    plt.suptitle(f'O-RAN Resource Scaling (PEG vs LIT) - {timestamp}', fontsize=16, fontweight='bold')
    plt.savefig(f'resource_test_full_{timestamp}.png', dpi=150, bbox_inches='tight')
    print("✓ Plots generated")


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Starting Multi-Vendor Resource Test (PLMN 10101 Guard)")
    print(f"Vendors: {[v['name'] for v in VENDORS]}")
    print(f"CPU Variants: {CPU_VARIANTS}")

    all_results = []
    for scenario in TEST_SCENARIOS:
        all_results.append(run_test_scenario(scenario, timestamp))
        time.sleep(10)

    generate_summary_report(all_results, timestamp)
    print(f"\nAll tests complete.")


if __name__ == "__main__":
    main()
