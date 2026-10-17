"""Read-only A3 host inventory without importing model or tensor libraries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


def run(argv):
    try:
        value = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return dict(returncode=value.returncode, stdout=value.stdout.strip(),
                    stderr=value.stderr.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(returncode=None, error_type=type(exc).__name__)


def package_metadata(environment):
    result = {}
    for site in environment.glob('lib/python*/site-packages'):
        for directory in site.glob('*.dist-info'):
            if not directory.name.lower().startswith(
                    ('torch-', 'isaacsim-', 'isaaclab-', 'mujoco-', 'numpy-',
                     'accelerate-', 'transformers-')):
                continue
            metadata = directory / 'METADATA'
            if not metadata.is_file():
                continue
            fields = {}
            for line in metadata.read_text(errors='replace').splitlines():
                if line.startswith(('Name: ', 'Version: ')):
                    key, value = line.split(': ', 1)
                    fields[key] = value
            if 'Name' in fields:
                result[fields['Name']] = fields.get('Version')
    return result


def inventory(root):
    expected = Path('/home/humanplus/YUANQI_A3')
    if root.resolve() != expected or Path.home() != Path('/home/humanplus'):
        raise ValueError('Only the user-authorized humanplus workspace is supported')
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        if name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            memory[name] = int(value.split()[0]) * 1024
    temperatures = []
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temperatures.append(dict(kind=(zone / 'type').read_text().strip(),
                                     celsius=int((zone / 'temp').read_text()) / 1000))
        except (OSError, ValueError):
            continue
    processes = []
    unreadable = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            fields = dict(line.split(':', 1) for line in
                          (directory / 'status').read_text().splitlines() if ':' in line)
            rss = int(fields.get('VmRSS', '0 kB').split()[0]) * 1024
            name = fields['Name'].strip()
            if rss > 1024**3 or name.lower().startswith(('python', 'kit', 'isaac')):
                row = dict(pid=int(directory.name), comm=name, rss_bytes=rss)
                try:
                    row['cwd'] = os.readlink(directory / 'cwd')
                except OSError:
                    row['cwd'] = 'UNREADABLE'
                processes.append(row)
        except (OSError, ValueError, KeyError):
            if directory.exists():
                unreadable.append(int(directory.name))
    plan = json.loads((root / 'data/manifests/a3_finite_retention_plan_20261017.json').read_text())
    inputs = []
    for item in plan['inputs']:
        path = root / item['path']
        row = dict(path=item['path'], expected_sha256=item['sha256'], exists=path.is_file())
        # This inventory does not read large artifacts while other work is active.
        if path.is_file() and path.stat().st_size <= 2 * 1024**2:
            row['actual_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs.append(row)
    stat = os.statvfs(root)
    env = Path('/home/humanplus/anaconda3/envs/x2-sonic-isaaclab')
    return dict(at_utc=datetime.now(timezone.utc).isoformat(), host=os.uname().nodename,
        root=str(root), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        kernel=os.uname().release, memory_bytes=memory,
        cgroup=Path('/proc/self/cgroup').read_text().strip(),
        temperatures=temperatures, disk_available_bytes=stat.f_bavail * stat.f_frsize,
        gpu=run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits']),
        gpu_apps=run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                      '--format=csv,noheader,nounits']),
        kernel_readability=run(['dmesg', '--level=emerg,alert,crit,err', '--color=never']),
        visible_compute=sorted(processes, key=lambda row: row['rss_bytes'], reverse=True),
        unreadable_process_pids=unreadable, candidate_environment=str(env),
        candidate_packages=package_metadata(env),
        git_head=run(['git', '-C', str(root), 'rev-parse', 'HEAD']),
        input_inventory=inputs, missing_source=[name for name in plan['source_sha256']
                                              if not (root / name).is_file()],
        source_host_kernel_evidence_reusable_as_new_baseline=False,
        tensors_imported=False, runtime_validated=False, training_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root.resolve())))
