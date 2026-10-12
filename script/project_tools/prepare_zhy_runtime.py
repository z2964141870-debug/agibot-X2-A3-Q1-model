#!/usr/bin/env python3
"""Relocate only the copied A3 overlay and editable Isaac Lab registrations."""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    base = root / 'data/env'
    overlay = root / 'data/environments/a3-sonic'
    isaac = root / 'script/vendor/IsaacLab'
    if not (base / 'bin/python').is_file() or not (isaac / 'source/isaaclab').is_dir():
        raise RuntimeError('The verified runtime and Isaac Lab copy must exist first')
    changed = []
    mappings = {
        '/home/yu/projects/IsaacLab': str(isaac),
        '/home/yu/IsaacLab': str(isaac),
        '/media/yu/FAFF-E9771/YUANQI/script/vendor/sonic_for_a3': str(root / 'script/vendor/sonic_for_a3'),
        '/media/yu/FAFF-E9771/YUANQI/data/environments/a3-sonic': str(overlay),
    }
    files = list((base / 'lib/python3.11/site-packages').glob('__editable*finder.py'))
    files += list((overlay / 'lib/python3.11/site-packages').glob('*.pth'))
    files += [p for p in (overlay / 'bin').iterdir() if p.is_file() and not p.is_symlink()]
    for path in files:
        try:
            original = path.read_text()
        except UnicodeDecodeError:
            continue
        updated = original
        for source, destination in mappings.items():
            updated = updated.replace(source, destination)
        if updated != original:
            path.write_text(updated)
        if any(destination in updated for destination in mappings.values()):
            changed.append({'path': str(path.relative_to(root)), 'sha256': sha(path)})
    config = configparser.ConfigParser()
    config.read_string('[venv]\n' + (overlay / 'pyvenv.cfg').read_text())
    config['venv'].update(home=str(base / 'bin'), executable=str(base / 'bin/python3.11'),
                          command='Migrated A3 overlay; see README_ZHY_RUNTIME_20261012.md',
                          **{'include-system-site-packages': 'true'})
    (overlay / 'pyvenv.cfg').write_text(''.join(k + ' = ' + v + '\n' for k, v in config['venv'].items()))
    changed.append({'path': str((overlay / 'pyvenv.cfg').relative_to(root)),
                    'sha256': sha(overlay / 'pyvenv.cfg')})
    python = overlay / 'bin/python'
    if not python.is_symlink():
        raise RuntimeError('Expected the copied overlay Python to be a symlink')
    python.unlink()
    python.symlink_to(os.path.relpath(base / 'bin/python', python.parent))
    check = subprocess.run([str(python), '-c',
        'import sys,importlib.util,json; print(json.dumps({"version":sys.version,"prefix":sys.prefix,'
        '"base_prefix":sys.base_prefix,"modules":{k:bool(importlib.util.find_spec(k)) '
        'for k in ["torch","isaacsim","isaaclab","accelerate","gear_sonic","smpl_sim"]}}))'],
        text=True, capture_output=True, check=True)
    receipt = {'status': 'PATHS_RELOCATED', 'changed': changed, 'python': json.loads(check.stdout),
               'source_environment_modified': False, 'gpu_compute_performed': False}
    if receipt['python']['base_prefix'] != str(base):
        raise RuntimeError('Overlay still resolves to a foreign base interpreter')
    if not all(receipt['python']['modules'][name] for name in
               ('torch', 'isaacsim', 'isaaclab', 'accelerate', 'gear_sonic')):
        raise RuntimeError('Missing required module specifications: ' + check.stdout)
    receipt['optional_smpl_metrics_available'] = receipt['python']['modules']['smpl_sim']
    output = root / 'logs/zhy_runtime_20261012/relocation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    prepare(p.parse_args().root.resolve())
