"""Inspect bounded metadata prefixes inside the supplied EgoLocate archive.

Does not extract videos, load pose tensors, or treat inferred poses as truth.
Only selected non-secret metadata is returned. Full recording integrity and
cross-stream alignment remain unverified by this prefix inventory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def inventory(path):
    output = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        manifests = sorted(n for n in names if '/recordings/' in n and n.endswith('/manifest.json'))
        if len(manifests) > 64:
            raise ValueError('recording count exceeds bounded inventory')
        for name in manifests:
            if archive.getinfo(name).file_size > 65536:
                raise ValueError('oversized manifest')
            raw = archive.read(name)
            manifest = json.loads(raw)
            directory = name.rsplit('/', 1)[0]
            sizes = {k: archive.getinfo(directory + '/' + k).file_size
                     if directory + '/' + k in names else None
                     for k in ('frames.jsonl', 'camera_frames.jsonl', 'camera.mp4')}
            config = manifest.get('args', {})
            record = {'session': directory.rsplit('/', 1)[1],
                      'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                      'declared_state': manifest.get('state'),
                      'declared_frame_count': manifest.get('frame_count'),
                      'member_sizes': sizes, 'camera_source': config.get('camera_source'),
                      'paired_nonempty_files': all(bool(sizes[x]) for x in sizes),
                      'prefix_checks': {}}
            for filename in ('frames.jsonl', 'camera_frames.jsonl'):
                if not sizes[filename]:
                    continue
                checked, raw_prefix = [], bytearray()
                with archive.open(directory + '/' + filename) as stream:
                    for _ in range(8):
                        line = stream.readline(262145)
                        if not line:
                            break
                        if len(line) > 262144:
                            raise ValueError('JSONL line exceeds read budget')
                        raw_prefix.extend(line)
                        item = json.loads(line)
                        checked.append({
                            'frame_index': item.get('frame_index'),
                            'camera_video_frame_index': item.get('camera_video_frame_index'),
                            'camera_timestamp_s': item.get('camera_timestamp_s'),
                            'shape': item.get('shape'),
                            'has_exposure_field': any('exposure' in k for k in item),
                            'fields_present': sorted(item),
                        })
                record['prefix_checks'][filename] = {'rows': len(checked),
                    'prefix_sha256': hashlib.sha256(raw_prefix).hexdigest(), 'items': checked}
            output.append(record)
    return {'archive_name': Path(path).name, 'archive_bytes': Path(path).stat().st_size,
            'scope': 'manifest_and_first_eight_rows_only', 'full_integrity_verified': False,
            'training_ready': False, 'sessions': output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    print(json.dumps(inventory(p.parse_args().archive), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
