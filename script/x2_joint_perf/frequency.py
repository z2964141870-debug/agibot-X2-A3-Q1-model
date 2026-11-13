"""Read-only CPU/devfreq and official power-tool observations; never changes clocks."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from script.x2_fgp_accel.build import resources


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--privileged-show", action="store_true", help="Only sudo -n jetson_clocks --show; read-only")
    a = p.parse_args()
    if a.output.exists():
        p.error("Preserve prior observation")
    cpus, devices = [], []
    for directory in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
        row = dict(policy=directory.name)
        for name in ('affected_cpus', 'scaling_governor', 'scaling_min_freq', 'scaling_max_freq',
                     'scaling_cur_freq', 'cpuinfo_max_freq', 'scaling_available_frequencies'):
            path = directory/name
            row[name] = path.read_text().strip() if path.is_file() else None
        cpus.append(row)
    for directory in sorted(Path('/sys/class/devfreq').glob('*')):
        row = dict(device=directory.name)
        for name in ('name', 'governor', 'cur_freq', 'min_freq', 'max_freq', 'available_frequencies'):
            path = directory/name
            try:
                row[name] = path.read_text().strip()
            except (OSError, PermissionError):
                row[name] = None
        devices.append(row)
    commands = []
    show = ["sudo", "-n", "/usr/bin/jetson_clocks", "--show"] if a.privileged_show else ["jetson_clocks", "--show"]
    for command in (["nvpmodel", "-q", "--verbose"], show):
        result = subprocess.run(command, capture_output=True, text=True, timeout=5)
        commands.append(dict(command=command, exit_code=result.returncode,
                             stdout=result.stdout, stderr=result.stderr))
    receipt = dict(status="READ_ONLY_FREQUENCY_OBSERVATION", wall_time_ns=time.time_ns(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        uid=os.getuid(), user=os.environ.get('USER'),
        cpu_policies=cpus, devfreq=devices, queries=commands, resources=resources(),
        frequency_writes=0, power_mode_changes=0, sudo_commands=int(a.privileged_show),
        power_mode_verified=commands[0]['exit_code'] == 0,
        limitation="Point-in-time visible sysfs limits, not frequency telemetry during inference or proof of every hardware power mode")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
