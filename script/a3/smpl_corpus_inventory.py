"""Inventory existing SMPL labels from ZIP metadata without extracting tensors."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile


def inventory(archive, prior_inspection):
    archive = Path(archive)
    previous = json.loads(Path(prior_inspection).read_text())['archive']
    if archive.stat().st_size != previous['bytes']:
        raise ValueError('Archive size differs from its prior inspection')
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
    labels, notices = {}, []
    directory_digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda row: row.filename):
        identity = [entry.filename, entry.CRC, entry.file_size, entry.compress_size]
        directory_digest.update(json.dumps(identity, ensure_ascii=True).encode('ascii'))
        path = PurePosixPath(entry.filename)
        if any(word in path.name.lower() for word in ('license', 'readme', 'notice', 'citation')):
            notices.append(entry.filename)
        if len(path.parts) == 4 and path.parts[0] == 'train_released' and path.name in ('pose.pt', 'joint.pt'):
            sequence = '/'.join(path.parts[1:-1])
            if path.name in labels.setdefault(sequence, {}):
                raise ValueError('Duplicate tensor entry: ' + entry.filename)
            labels[sequence][path.name] = dict(entry=entry.filename, bytes=entry.file_size,
                compressed_bytes=entry.compress_size, zip_crc32=f'{entry.CRC:08x}')
    complete = {name: value for name, value in labels.items() if set(value) == {'pose.pt', 'joint.pt'}}
    folders = {}
    for name in complete:
        folder = name.split('/')[0]
        folders[folder] = folders.get(folder, 0) + 1
    return dict(schema_version=1, inspected_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='ZIP_DIRECTORY_ONLY_NO_TENSOR_LOADING_EXTRACTION_OR_TRAINING',
        archive=dict(path=str(archive.resolve()), bytes=archive.stat().st_size,
            previous_file_sha256=previous['sha256'],
            previous_file_sha256_evidence=str(Path(prior_inspection).resolve()),
            file_sha256_recomputed_this_run=False,
            central_directory_sha256=directory_digest.hexdigest()),
        complete_sequences=len(complete), capture_folder_count=len(folders),
        capture_folders=folders, folders_are_not_verified_independent_people=True,
        label_entries=sum(len(value) for value in complete.values()),
        label_bytes=sum(row['bytes'] for value in complete.values() for row in value.values()),
        label_compressed_bytes=sum(row['compressed_bytes'] for value in complete.values() for row in value.values()),
        incomplete_sequences=sorted(set(labels) - set(complete)),
        notice_entries=notices, tensor_shapes_and_timestamps_validated=False,
        source_training_use_verified=False, paired_a3_references_created=False,
        retarget_quality_verified=False, train_validation_test_sessions_fixed=False,
        training_ready=False, backup_status='LOCAL_ONLY', sequences=complete)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--prior-inspection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep an existing corpus inventory; use a separate output for a new check')
    result = inventory(args.archive, args.prior_inspection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: result[key] for key in ('complete_sequences', 'capture_folder_count',
        'label_entries', 'label_bytes', 'label_compressed_bytes', 'incomplete_sequences',
        'notice_entries', 'training_ready')}, indent=2))
