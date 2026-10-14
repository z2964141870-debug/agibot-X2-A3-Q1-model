"""Bounded ISO-BMFF sample-table inventory, without decoding or extracting video.

Only conventional non-fragmented MP4 sample tables are interpreted. This cannot
prove pictures are decodable, distinct, correctly timestamped, or synchronized.
"""
import argparse
import hashlib
import json
import struct
import time
import zipfile


def boxes(data):
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8: raise ValueError('truncated box header')
        size, kind = struct.unpack_from('>I4s', data, offset); header = 8
        if size == 1:
            if len(data) - offset < 16: raise ValueError('truncated extended size')
            size = struct.unpack_from('>Q', data, offset + 8)[0]; header = 16
        elif size == 0: size = len(data) - offset
        if size < header or offset + size > len(data): raise ValueError('invalid box extent')
        yield kind, data[offset + header:offset + size]
        offset += size


def track_tables(data):
    tracks = []
    for kind, track in boxes(data):
        if kind != b'trak': continue
        info = {}
        def visit(buffer):
            for name, payload in boxes(buffer):
                if name in (b'mdia', b'minf', b'stbl'): visit(payload)
                elif name == b'hdlr': info['handler'] = payload[8:12].decode('ascii')
                elif name == b'mdhd':
                    if payload[0] == 0:
                        scale, duration = struct.unpack_from('>II', payload, 12)
                    elif payload[0] == 1:
                        scale, duration = struct.unpack_from('>IQ', payload, 20)
                    else: raise ValueError('unknown mdhd version')
                    info.update(timescale=scale, media_duration_ticks=duration,
                                media_duration_s=duration / scale if scale else None)
                elif name == b'stsz':
                    sample_size, count = struct.unpack_from('>II', payload, 4)
                    if len(payload) != 12 + (0 if sample_size else count * 4):
                        raise ValueError('inconsistent stsz sample table')
                    info['sample_count'] = count
                elif name == b'stts':
                    n = struct.unpack_from('>I', payload, 4)[0]
                    if len(payload) != 8 + n * 8: raise ValueError('inconsistent stts')
                    entries = [struct.unpack_from('>II', payload, 8 + i * 8) for i in range(n)]
                    info['stts_count'] = sum(x for x, _ in entries)
                    info['stts_duration_ticks'] = sum(x * y for x, y in entries)
        visit(track); tracks.append(info)
    return tracks


def inspect(archive, member):
    size = archive.getinfo(member).file_size
    if size > 128 * 1024 * 1024: raise ValueError('video inventory budget exceeded')
    deadline = time.monotonic() + 60; digest = hashlib.sha256(); remaining = size
    records, movie = [], None
    with archive.open(member) as stream:
        def read(n):
            data = stream.read(n)
            if len(data) != n: raise ValueError('truncated MP4')
            digest.update(data)
            if time.monotonic() > deadline: raise TimeoutError('MP4 inventory expired')
            return data
        while remaining:
            if remaining < 8: raise ValueError('trailing incomplete box')
            length, kind = struct.unpack('>I4s', read(8)); header = 8
            if length == 1: length = struct.unpack('>Q', read(8))[0]; header = 16
            elif length == 0: length = remaining
            if not header <= length <= remaining: raise ValueError('invalid top-level box')
            records.append({'type': kind.decode('ascii'), 'bytes': length})
            if kind == b'moov':
                if movie is not None or length > 2 * 1024 * 1024: raise ValueError('moov budget/duplication')
                movie = read(length - header)
            else:
                count = length - header
                while count:
                    chunk = min(count, 65536); read(chunk); count -= chunk
            remaining -= length
        if stream.read(1): raise ValueError('unexpected trailing bytes')
    if any(x['type'] == 'moof' for x in records): raise ValueError('fragmented MP4 unsupported')
    if movie is None: raise ValueError('missing moov')
    tracks = track_tables(movie)
    for track in tracks:
        if track.get('sample_count') != track.get('stts_count'): raise ValueError('sample count mismatch')
    return {'member': member, 'bytes': size, 'sha256': digest.hexdigest(),
            'zip_crc_verified_at_eof': True, 'boxes': records, 'tracks': tracks,
            'video_decoded': False, 'synchronization_verified': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True); parser.add_argument('--member', required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as z:
        print(json.dumps(inspect(z, args.member), ensure_ascii=False, indent=2))
