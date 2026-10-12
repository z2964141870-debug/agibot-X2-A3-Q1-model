"""Copy selected original CLOTHO labels without importing tensor libraries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import zipfile

SELECTION = (
    ('G_0113_zj_train/183926_zj_walk', 'train'),
    ('G_0113_zj_train/192333_zj_warmUp', 'train'),
    ('G_20251226_fjw/fjw_140008_walk', 'train'),
    ('G_20251226_zls/zls_155013_walk', 'validation'),
    ('G_20251229_yy/yy_123744_squat', 'test'),
    ('G_20251229_yy/yy_125106_walk', 'test'),
    ('G_20251229_zzx/zzx_111505_walk', 'test'),
)


def pack(archive, inventory_path, destination):
    archive, inventory_path, destination = map(Path, (archive, inventory_path, destination))
    inventory = json.loads(inventory_path.read_text())
    if destination.exists():
        raise FileExistsError('Keep an existing selection; do not reset or overwrite it')
    before = archive.stat()
    if before.st_size != inventory['archive']['bytes']:
        raise ValueError('Source archive size changed')
    destination.mkdir(parents=True)
    labels = destination / 'selected_labels.zip'
    rows = []
    with zipfile.ZipFile(archive) as source, zipfile.ZipFile(labels, 'x', compression=zipfile.ZIP_STORED) as output:
        for sequence, partition in SELECTION:
            entries = {}
            for name in ('pose.pt', 'joint.pt'):
                expected = inventory['sequences'][sequence][name]
                entry = source.getinfo(expected['entry'])
                if (entry.file_size != expected['bytes']
                        or f'{entry.CRC:08x}' != expected['zip_crc32']):
                    raise ValueError('ZIP entry identity changed: ' + entry.filename)
                raw = source.read(entry)
                output.writestr(entry.filename, raw)
                entries[name] = dict(entry=entry.filename, bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), zip_crc32=expected['zip_crc32'])
            rows.append(dict(sequence=sequence, capture_folder=sequence.split('/')[0],
                             partition=partition, labels=entries))
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Source archive changed during the copy')
    report = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_provenance=dict(kind='SELF_COLLECTED_CLOTHING_MODEL_TRAINING_DATA',
            evidence='User confirmation in this conversation on 2026-10-12',
            training_use_authorized=True, external_dataset_fallback_authorized=True),
        source_archive=inventory['archive'],
        source_archive_sha256_recomputed=False,
        inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        packer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        label_archive=dict(name=labels.name, bytes=labels.stat().st_size,
            sha256=hashlib.sha256(labels.read_bytes()).hexdigest()),
        sequences=rows, selection_rule='FIXED_NAMES_BEFORE_TENSOR_OR_RETARGET_INSPECTION',
        partition_unit='CAPTURE_FOLDER_NOT_VERIFIED_INDEPENDENT_SUBJECT',
        source_fps=30, source_fps_evidence='Existing CLOTHO project playback contract; no per-frame timestamps in selected labels',
        live_timing_verified=False, training_ready=False,
        protected_test_only_artifacts_unchanged=True, backup_status='LOCAL_ONLY')
    with (destination / 'selection.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(sequences=len(rows), label_files=2 * len(rows),
        label_archive=report['label_archive'], training_ready=False)))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pack(args.archive, args.inventory, args.output)
