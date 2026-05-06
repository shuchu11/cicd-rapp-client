with open('/home/ubuntu/leo/cicd-rapp-client/main.py', 'r') as f:
    lines = f.readlines()

old_block = [
    '        # Start iperf test\n',
    '        iperf_result = trigger_iperf(bitrate)\n',
    '\n',
    '        # Wait for traffic to stabilize\n',
    '        time.sleep(2)\n',
    '\n',
    '        # Measure thread CPU during the traffic\n',
    '        cpu_data = fetch_thread_cpu()\n',
]

new_block = [
    '        # Run iperf and CPU monitoring in parallel\n',
    '        import threading\n',
    '        _iperf = [None]\n',
    '        _cpu = [None]\n',
    '        def _run_iperf(): _iperf[0] = trigger_iperf(bitrate)\n',
    '        def _run_cpu():\n',
    '            time.sleep(1)\n',
    '            _cpu[0] = fetch_thread_cpu()\n',
    '        t1 = threading.Thread(target=_run_iperf)\n',
    '        t2 = threading.Thread(target=_run_cpu)\n',
    '        t1.start(); t2.start()\n',
    '        t1.join(); t2.join()\n',
    '        iperf_result = _iperf[0]\n',
    '        cpu_data = _cpu[0]\n',
]

for i in range(len(lines) - len(old_block) + 1):
    if lines[i:i+len(old_block)] == old_block:
        lines[i:i+len(old_block)] = new_block
        print(f"Replaced at line {i+1}")
        break
else:
    print("Not found!")

with open('/home/ubuntu/leo/cicd-rapp-client/main.py', 'w') as f:
    f.writelines(lines)
