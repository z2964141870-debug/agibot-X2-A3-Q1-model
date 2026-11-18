"""Add optional watchdog source IDs without repacking the original pose values."""
from __future__ import annotations

import json
import math
import struct

FIELD = "wd_pose_provenance"
SOURCE_CODES = {"upstream": 1, "hold": 2, "blend": 3, "cold_idle": 4, "idle_clip": 5}
DTYPE_BYTES = {"f32": 4, "f64": 8, "i32": 4, "i64": 8, "u8": 1, "bool": 1}


def tag_pose(msg: bytes, topic: str, header_size: int, sequence: int, source: str) -> bytes:
    """Return the original bytes if this packet cannot carry diagnostic data."""
    if not 0 <= sequence < 1 << 60:
        raise ValueError("watchdog sequence is outside the signed wire range")
    stamp = (sequence << 3) | SOURCE_CODES[source]
    prefix = topic.encode("utf-8")
    end = len(prefix) + header_size
    if not msg.startswith(prefix) or len(msg) < end:
        return msg
    try:
        header = json.loads(msg[len(prefix):end].rstrip(b"\0"))
    except (ValueError, UnicodeError):
        return msg
    if not isinstance(header, dict) or header.get("endian") != "le" or header.get("count") != 1:
        return msg
    fields = header.get("fields")
    if (not isinstance(fields, list) or not all(isinstance(field, dict) for field in fields)
            or any(field.get("name") == FIELD for field in fields)):
        return msg
    expected = 0
    for field in fields:
        dtype, shape = field.get("dtype"), field.get("shape")
        if (not isinstance(dtype, str) or dtype not in DTYPE_BYTES
                or not isinstance(shape, list) or any(type(dim) is not int or dim <= 0 for dim in shape)):
            return msg
        expected += DTYPE_BYTES[dtype] * math.prod(shape)
    if expected != len(msg) - end:
        return msg
    header["fields"] = fields + [dict(name=FIELD, dtype="i64", shape=[1])]
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    if len(encoded) > header_size:
        return msg
    return prefix + encoded.ljust(header_size, b"\0") + msg[end:] + struct.pack("<q", stamp)


def unpack_origin(stamp: int) -> dict | None:
    code = stamp & 7
    if stamp <= 0 or code not in SOURCE_CODES.values():
        return None
    return dict(source=next(name for name, value in SOURCE_CODES.items() if value == code),
                watchdog_sequence=stamp >> 3)
