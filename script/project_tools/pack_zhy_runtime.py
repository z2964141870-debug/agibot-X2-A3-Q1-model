#!/usr/bin/env python3
"""Use conda-pack on the target with prefix modes from the original install."""

import argparse
import hashlib
import json
from pathlib import Path

ORIGINAL = '/home/yu/miniconda3/envs/x2-sonic-isaaclab'
TARGET = Path('/opt/data/private/hechengzhu/ZHY')


def modes(output):
    from conda_pack import CondaEnv
    import conda_pack
    env = CondaEnv.from_prefix(ORIGINAL, ignore_editable_packages=True)
    result = dict(source_prefix=ORIGINAL, tool_version=conda_pack.__version__,
                  modes={f.target:f.file_mode for f in env.files if f.file_mode in ('text', 'binary')})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(mode_records=len(result['modes']), path=str(output))))


def pack(mode_path):
    from conda_pack import CondaEnv
    from conda_pack.core import File
    from conda_pack.formats import NoArchive
    if not TARGET.is_dir() or Path.cwd().resolve() != TARGET:
        raise RuntimeError('Packing is restricted to the target workspace')
    receipt = json.loads((TARGET / 'logs/zhy_runtime_20261012/raw_verification.json').read_text())
    if receipt['failures']:
        raise RuntimeError('Raw runtime checksum verification failed')
    original = json.loads(mode_path.read_text())
    if original['source_prefix'] != ORIGINAL or original['tool_version'] != '0.8.1':
        raise RuntimeError('Prefix-mode provenance mismatch')
    raw = TARGET / 'data/environments/isaac-runtime-raw'
    destination = TARGET / 'data/env'
    if destination.exists():
        raise FileExistsError('Do not overwrite an existing prepared runtime')
    env = CondaEnv.from_prefix(str(raw), ignore_editable_packages=True)
    files = []
    for f in env.files:
        mode = original['modes'].get(f.target, f.file_mode)
        # Copied installed files already contain ORIGINAL, not build placeholders.
        files.append(File(f.source, f.target, is_conda=f.is_conda,
                          prefix_placeholder=ORIGINAL if mode in ('text', 'binary') else None,
                          file_mode=mode))
    portable = CondaEnv(ORIGINAL, files)
    # Version 0.8.1's directory writer suppresses exceptions by returning itself.
    NoArchive.__exit__ = lambda self, exc_type, exc_value, traceback: False
    portable.pack(output=str(destination), format='no-archive', dest_prefix=str(destination))
    missing = [f.target for f in files if not (destination / f.target).exists()
               and not (destination / f.target).is_symlink()]
    if missing:
        raise RuntimeError('Incomplete packed runtime: ' + repr(missing[:10]))
    public = TARGET / 'data/environments/isaac-runtime'
    public.symlink_to('../env', target_is_directory=True)
    result = dict(status='PACKED_ON_TARGET', prefix=str(destination),
                  mode_manifest_sha256=hashlib.sha256(mode_path.read_bytes()).hexdigest(),
                  input_verified_files=receipt['verified_files'], input_verified_bytes=receipt['verified_bytes'],
                  preserved_original_prefix=ORIGINAL, packed_entries=len(files), backup_status='LOCAL_ONLY')
    (TARGET / 'logs/zhy_runtime_20261012/packing.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('modes', 'pack'))
    p.add_argument('--modes', type=Path, required=True)
    args = p.parse_args()
    dict(modes=modes, pack=pack)[args.action](args.modes)
