"""Bounded, standard-library inspection of Nori sample BMP metadata.

Independent implementation from supplied protocol documentation. Never opens a
camera, executes vendor code, writes images, or supplies robot control signals.
No checksum is documented: successful decoding is not proof of frame integrity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

MAX_BMP_BYTES = 32 * 1024 * 1024
MOD32 = 1 << 32


def forward_us(previous, current, max_gap_us):
    """Only for a known continuous clock/stream; resets require a new epoch."""
    if not all(isinstance(x, int) and 0 <= x < MOD32 for x in (previous, current)):
        raise ValueError("timestamps must be unsigned 32-bit integers")
    if not 0 < max_gap_us < MOD32 // 2:
        raise ValueError("a bounded gap below half a clock period is required")
    delta = (current - previous) % MOD32
    if not 0 < delta <= max_gap_us:
        raise ValueError("duplicate, backward/reset, or excessive timestamp gap")
    return delta


def parse_header(data):
    if len(data) != 16:
        raise ValueError("header must contain exactly 16 bytes")
    if data[:8] == data[8:]:
        version, slots = 0, [(1, 11)]
        start, end = struct.unpack(">II", data[:8])
    else:
        bits = int.from_bytes(data[:8], "big")
        version = bits >> 60
        if version != 1:
            raise ValueError("unsupported protocol version")
        slots = [((bits >> (48 - i * 12 + 4)) & 255,
                  (bits >> (48 - i * 12)) & 15) for i in range(5)]
        start, end = struct.unpack(">II", data[8:])
    if sum(count for _, count in slots) > 11:
        raise ValueError("declared sensor groups exceed documented limit")
    if any(count and device != 1 for device, count in slots):
        raise ValueError("unsupported populated sensor type")
    return {"protocol_version": version, "slots": slots,
            "exposure_start_us32": start, "exposure_end_us32": end}


def decode_payload(data, accel_range_g=None, gyro_range_dps=None):
    if len(data) < 16:
        raise ValueError("truncated header")
    header = parse_header(data[:16])
    count = sum(n for _, n in header["slots"])
    if len(data) != (count + 1) * 16:
        raise ValueError("payload length differs from header")
    if accel_range_g is not None and accel_range_g not in (2, 4, 8, 16):
        raise ValueError("unsupported accelerometer full scale")
    if gyro_range_dps is not None and gyro_range_dps not in (125, 250, 500, 1000, 2000):
        raise ValueError("unsupported gyroscope full scale for this validator")
    samples = []
    for offset in range(16, len(data), 16):
        timestamp, *axes = struct.unpack(">I6h", data[offset:offset + 16])
        valid = axes[:3] != [-1, -1, -1] and axes[3] != -32768
        sample = {"timestamp_us32": timestamp, "accel_raw": axes[:3],
                  "gyro_raw": axes[3:], "valid": valid}
        # Keep invalid sentinels visible; never turn them into valid zero motion.
        if valid and accel_range_g is not None:
            sample["accel_m_s2"] = [v * accel_range_g * 9.80665 / 32768 for v in axes[:3]]
        if valid and gyro_range_dps is not None:
            sample["gyro_rad_s"] = [v * gyro_range_dps * 0.017453292519943295 / 32768 for v in axes[3:]]
        samples.append(sample)
    return {**header, "payload_hex": data.hex(), "sensor_group_count": count,
            "imu_samples": samples,
            "scale_source": "caller_supplied_not_read_from_firmware",
            "accel_range_g": accel_range_g, "gyro_range_dps": gyro_range_dps,
            "crc_verified": False, "actual_user_device_verified": False}


def decode_bmp(data, **kwargs):
    if not 54 <= len(data) <= MAX_BMP_BYTES or data[:2] != b"BM":
        raise ValueError("invalid or oversized BMP")
    declared, offset = struct.unpack_from("<I", data, 2)[0], struct.unpack_from("<I", data, 10)[0]
    dib = struct.unpack_from("<I", data, 14)[0]
    width, signed_height, planes, bpp, compression = struct.unpack_from("<iiHHI", data, 18)
    height = abs(signed_height)
    if dib < 40 or not 0 < width <= 8192 or not 0 < height <= 8192:
        raise ValueError("unsupported BMP geometry")
    if planes != 1 or bpp != 24 or compression != 0:
        raise ValueError("only uncompressed 24-bit BMP is supported")
    stride = ((width * bpp + 31) // 32) * 4
    if offset < 14 + dib or offset + stride * height > len(data) or declared != len(data):
        raise ValueError("truncated or inconsistent BMP header")
    payload, expected = bytearray(), None
    line_count = 0
    # Vendor sample: top visual row 2, then every eight rows. Honor BMP offset,
    # orientation and row padding rather than assuming packed positive-height BMP.
    for top_y in range(2, min(height, 2 + 192 * 8), 8):
        storage_y = height - 1 - top_y if signed_height > 0 else top_y
        row = memoryview(data)[offset + storage_y * stride:offset + storage_y * stride + width * 3]
        if len(row) < 6 or any(v <= 220 for v in row[:4]):
            raise ValueError("missing white metadata prefix")
        first = next((x for x in range(width) if row[x * 3 + 1] < 220), None)
        if first is None:
            raise ValueError("missing encoded line")
        bits, terminated = [], False
        for x in range(first + 4, width - 1, 8):
            value = row[x * 3 + 1] + row[(x + 1) * 3 + 1]
            if value >= 440:
                terminated = True
                break
            bits.append(int(value >= 100))
        if not terminated or not bits or len(bits) % 8:
            raise ValueError("incomplete or unterminated encoded byte")
        payload.extend(sum(bits[i + k] << k for k in range(8)) for i in range(0, len(bits), 8))
        line_count += 1
        if expected is None and len(payload) >= 16:
            header = parse_header(payload[:16])
            expected = 16 * (1 + sum(n for _, n in header["slots"]))
        if expected is not None and len(payload) >= expected:
            if len(payload) != expected:
                raise ValueError("unexpected trailing encoded bytes")
            return {"bmp": {"width": width, "height": height, "stride": stride,
                             "offset": offset, "metadata_lines": line_count},
                    **decode_payload(bytes(payload), **kwargs)}
    raise ValueError("metadata is truncated")


def inspect_archive(path):
    with zipfile.ZipFile(path) as archive:
        matches = [x for x in archive.infolist() if x.filename.lower().endswith('.bmp')]
        if len(matches) != 1 or not 54 <= matches[0].file_size <= MAX_BMP_BYTES:
            raise ValueError("expected one bounded vendor sample BMP")
        with archive.open(matches[0]) as stream:
            data = stream.read(MAX_BMP_BYTES + 1)
        decoded = decode_bmp(data)  # No hardware scale assumptions.
        return {"member": matches[0].filename, "bytes": len(data),
                "bmp_sha256": hashlib.sha256(data).hexdigest(), **decoded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_archive(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
