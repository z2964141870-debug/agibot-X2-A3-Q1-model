"""Bounded metadata audit of existing recordings; never loads a model/video.

Reads complete JSONL members in a ZIP, validates their CRC via EOF, and reports
host-timestamp statistics. None of these timestamps is relabeled as exposure
time. Results are diagnostic evidence, not ground truth or training approval.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import zipfile

SESSIONS = ('ble_transport_smoke_20260824', 'manual_full_test_001',
            'worn_acceptance_retry_20260824')
MAX_BYTES = 256 * 1024 * 1024
MAX_LINE = 262144
MAX_ROWS = 30000


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def stats(values):
    v = sorted(x for x in values if finite(x))
    def q(p):
        if not v: return None
        k = (len(v) - 1) * p; i = int(k)
        return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (k - i)
    return {'n': len(v), 'min': v[0] if v else None, 'p50': q(.5),
            'p95': q(.95), 'max': v[-1] if v else None}


def scan_jsonl(archive, member, consume, deadline):
    info = archive.getinfo(member)
    if info.file_size > MAX_BYTES: raise ValueError('member exceeds read budget')
    digest = hashlib.sha256(); rows = bad = 0; examples = []
    with archive.open(member) as stream:
        while True:
            if time.monotonic() > deadline: raise TimeoutError('audit budget expired')
            line = stream.readline(MAX_LINE + 1)
            if not line: break
            if len(line) > MAX_LINE: raise ValueError('line exceeds budget')
            rows += 1
            if rows > MAX_ROWS: raise ValueError('row count exceeds budget')
            digest.update(line)
            try:
                record = json.loads(line)
                if not isinstance(record, dict): raise ValueError('not an object')
            except (ValueError, UnicodeError):
                bad += 1
                if len(examples) < 8: examples.append(rows)
                continue
            consume(record)
    return {'bytes': info.file_size, 'sha256': digest.hexdigest(), 'lines': rows,
            'malformed_lines': bad, 'malformed_line_numbers': examples,
            'zip_crc_verified_at_eof': True}


def analyze(archive, session, seconds=60):
    start = time.monotonic(); deadline = start + seconds
    prefix = 'EgoLocate/recordings/' + session + '/'
    manifest_raw = archive.read(prefix + 'manifest.json')
    manifest = json.loads(manifest_raw)
    camera_rows, frames, fields, errors = [], [], Counter(), Counter()
    ages, imu_ages, host_offsets, gaps, writer_lags = [], [], [], [], []
    pose_sources, fusion_states, shapes = Counter(), Counter(), Counter()
    last_imu = None; sensor_repeats = 0; valid_imu_rows = 0
    positions = []; last_frame = None

    def camera(record):
        idx, video_idx, ts = (record.get(k) for k in
                             ('frame_index', 'camera_video_frame_index', 'camera_timestamp_s'))
        if not isinstance(idx, int) or not isinstance(video_idx, int) or not finite(ts):
            errors['invalid_camera_index_or_timestamp'] += 1
        camera_rows.append((idx, video_idx, ts, record.get('written_at_unix_s')))
        shapes[str(record.get('shape'))] += 1

    def frame(record):
        nonlocal last_imu, sensor_repeats, valid_imu_rows, last_frame
        fields.update(record.keys())
        idx = record.get('frame_index'); mono = record.get('monotonic_s')
        ts = record.get('camera_timestamp_s'); wall = record.get('wall_time_s')
        video_idx = record.get('camera_video_frame_index')
        frames.append((idx, video_idx, ts, wall))
        if not isinstance(idx, int) or not finite(mono): errors['invalid_frame_id_or_time'] += 1
        if finite(mono) and finite(ts): ages.append(mono - ts)
        if finite(mono) and finite(wall): host_offsets.append(wall - mono)
        if finite(mono) and last_frame is not None: gaps.append(mono - last_frame)
        if finite(mono): last_frame = mono
        imu = record.get('raw_imu') or {}
        if finite(wall) and finite(imu.get('timestamp_s')): imu_ages.append(wall - imu['timestamp_s'])
        ordered = [imu.get('upper_ordered_6x7'), imu.get('lower_ordered_5x7')]
        if all(x is not None for x in ordered):
            digest = hashlib.sha256(json.dumps(ordered, separators=(',', ':')).encode()).digest()
            sensor_repeats += int(digest == last_imu); last_imu = digest; valid_imu_rows += 1
        tran = record.get('translation_m')
        if isinstance(tran, list) and len(tran) == 3 and all(finite(x) for x in tran):
            positions.append(tran)
        else: errors['invalid_translation'] += 1
        if record.get('cam_pos') is not None: pose_sources['cam_pos_nonnull'] += 1
        if record.get('cam_ori') is not None: pose_sources['cam_ori_nonnull'] += 1
        f = record.get('fusion') or {}
        fusion_states[str(f.get('state', f.get('reason')))] += 1

    camera_integrity = scan_jsonl(archive, prefix + 'camera_frames.jsonl', camera, deadline)
    frame_integrity = scan_jsonl(archive, prefix + 'frames.jsonl', frame, deadline)
    camera_map = {}
    for idx, vid, ts, written in camera_rows:
        if vid in camera_map: errors['duplicate_camera_video_index'] += 1
        camera_map[vid] = (idx, ts, written)
    used = set(); matched = 0
    for idx, vid, ts, wall in frames:
        if vid is None: continue
        used.add(vid)
        match = camera_map.get(vid)
        if match is None: errors['frame_without_camera_index_record'] += 1; continue
        if match[0] != idx or match[1] != ts:
            errors['cross_file_index_or_timestamp_mismatch'] += 1
        else: matched += 1
        if finite(wall) and finite(match[2]): writer_lags.append(match[2] - wall)
    errors['orphan_camera_index_records'] = len(set(camera_map) - used)
    ids = [x[0] for x in frames]; vids = [x[1] for x in camera_rows]
    camera_ts = [x[2] for x in camera_rows if finite(x[2])]
    camera_deltas = [b - a for a, b in zip(camera_ts, camera_ts[1:])]
    count = frame_integrity['lines'] - frame_integrity['malformed_lines']
    if manifest.get('frame_count') is not None and count != manifest['frame_count']:
        errors['declared_frame_count_mismatch'] += 1
    camera_count = camera_integrity['lines'] - camera_integrity['malformed_lines']
    if manifest.get('camera_video_frame_count') is not None and camera_count != manifest['camera_video_frame_count']:
        errors['declared_camera_count_mismatch'] += 1
    valid_time_rows = len(gaps) + int(last_frame is not None)
    duration = sum(gaps) if gaps else 0
    extents = [max(p[i] for p in positions) - min(p[i] for p in positions)
               for i in range(3)] if positions else None
    return {'session': session, 'declared_state': manifest.get('state'),
            'manifest_sha256': hashlib.sha256(manifest_raw).hexdigest(),
            'declared_frame_count': manifest.get('frame_count'),
            'declared_video_frame_count': manifest.get('camera_video_frame_count'),
            'integrity': {'frames': frame_integrity, 'camera_index': camera_integrity},
            'counts': {'frames': count, 'camera_records': camera_count, 'matched': matched,
                       'unique_camera_host_timestamps': len(set(camera_ts)),
                       'consecutive_reused_camera_timestamps': sum(x == 0 for x in camera_deltas),
                       'backward_camera_timestamps': sum(x < 0 for x in camera_deltas),
                       'nonpositive_body_time_gaps': sum(x <= 0 for x in gaps),
                       'negative_camera_ages': sum(x < 0 for x in ages),
                       'valid_imu_rows': valid_imu_rows, 'identical_consecutive_imu_rows': sensor_repeats,
                       'body_indices_start_zero_contiguous': ids == list(range(len(ids))),
                       'video_indices_start_zero_contiguous': vids == list(range(len(vids)))},
            'errors': dict(errors), 'fields_present_counts': dict(fields),
            'camera_shapes': dict(shapes), 'fusion_states': dict(fusion_states),
            'timing_seconds': {'recording_span': duration,
                'effective_body_logging_hz': (valid_time_rows - 1) / duration if duration > 0 else None,
                'body_gap': stats(gaps), 'camera_host_timestamp_gap': stats(camera_deltas),
                'camera_age_at_body_record': stats(ages), 'imu_snapshot_to_body_record': stats(imu_ages),
                'wall_minus_monotonic': stats(host_offsets), 'async_video_write_after_body': stats(writer_lags)},
            'translation_logged_only': {'count': len(positions), 'axis_extent_m': extents,
                'all_zero': all(all(x == 0 for x in p) for p in positions) if positions else None},
            'elapsed_seconds': time.monotonic() - start,
            'video_bitstream_decoded': False, 'video_sample_count_verified': False,
            'exposure_synchronization_verified': False, 'independent_camera_world_pose_verified': False,
            'training_ready': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--session', choices=SESSIONS, required=True)
    args = p.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        print(json.dumps(analyze(archive, args.session), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__': main()
