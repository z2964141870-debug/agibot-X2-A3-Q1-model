#!/usr/bin/env python3
"""Hash supplemental Isaac Lab and overlay files before path relocation."""

import json
import os
from pathlib import Path

from script.project_tools import migrate_zhy as migration


def main():
    migration.SSH[2] = '/tmp/zhy-runtime-20261012.sock'
    roots = [(Path('/home/yu/projects/IsaacLab'), 'script/vendor/IsaacLab/'),
             (Path('/media/yu/FAFF-E9771/YUANQI/data/environments/a3-sonic'),
              'data/environments/a3-sonic/')]
    rows = []
    for root, prefix in roots:
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in ('.git', '__pycache__'))
            for name in sorted(files):
                path = Path(directory) / name
                if path.is_symlink() or name.endswith(('.pyc', '.pyo')):
                    continue
                rows.append(dict(path=prefix + path.relative_to(root).as_posix(), kind='file',
                                 size=path.stat().st_size, sha256=migration.digest(path)))
    log = migration.SOURCE / 'logs/zhy_runtime_20261012/supplemental'
    log.mkdir(parents=True, exist_ok=True)
    with (log / 'source_manifest.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    result = migration.verify(rows, log)
    result['manifest_sha256'] = migration.digest(log / 'source_manifest.jsonl')
    (log / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
