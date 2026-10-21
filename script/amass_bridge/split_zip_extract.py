#!/usr/bin/env python3
"""split_zip_extract.py -- deterministic multi-volume (split) zip extractor.

Info-ZIP on this host cannot faithfully unpack AMASS_smplxg.z01..zNN+.zip:
  * 'zip -s 0 --out' merge corrupts entries that span volumes
    (unzip: "invalid compressed data to inflate"; python: overlapped entries)
  * 'zip -FF' repair is interactive and loops on piped input
  * naive 'cat volumes | unzip' fails: local header offsets are DISK-relative,
    central-directory offset is FINAL-VOLUME-relative (verified byte-level
    2026-10-20, see unpack.sh validate history)

This tool reads the split set directly (no merge, no temp copy):
  1. scan the FINAL volume (.zip) tail for EOCD / zip64-EOCD by signature
  2. parse the central directory from the final volume
  3. per entry: absolute local-header position = volume_start[start_disk] +
     stored offset; data is read through a virtual concatenated reader
     (entries may span volumes), inflated (method 0/8), CRC32-verified
Usage:  split_zip_extract.py <final_volume.zip> <out_dir> [--every N] [--list]
Resume: existing files with matching size are skipped (use --overwrite to redo).
"""
from __future__ import annotations

import bisect
import binascii
import os
import re as _re
import struct
import sys
import time
import zlib

SIG_EOCD = b"PK\x05\x06"
SIG_EOCD64 = b"PK\x06\x06"
SIG_CDH = b"PK\x01\x02"
SIG_LH = b"PK\x03\x04"


class SplitReader:
    """Virtual concatenated view over z01..zNN + final .zip (sorted by number)."""

    def __init__(self, final_path, vol_paths=None):
        d, base = os.path.split(final_path)
        name = os.path.splitext(base)[0]
        if vol_paths is None:
            vols = []
            pat = _re.compile(_re.escape(name) + r"\.z(\d+)$")
            for fn in os.listdir(d or "."):
                m = pat.match(fn)
                if m:
                    vols.append((int(m.group(1)), os.path.join(d, fn)))
            vols.sort()
            vol_paths = [p for _, p in vols]
            if not vol_paths:
                raise SystemExit("no %s.zNN volumes found next to %s" % (name, final_path))
        self.paths = list(vol_paths) + [final_path]
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.starts = []
        off = 0
        for s in self.sizes:
            self.starts.append(off)
            off += s
        self.total = off
        self.final_path = final_path
        self.handles = [open(p, "rb") for p in self.paths]

    def read_at(self, pos, n):
        if pos < 0 or pos + n > self.total:
            raise IOError("read [%d,%d) outside archive (total %d)" % (pos, pos + n, self.total))
        parts = []
        while n > 0:
            idx = bisect.bisect_right(self.starts, pos) - 1
            local = pos - self.starts[idx]
            take = min(n, self.sizes[idx] - local)
            h = self.handles[idx]
            h.seek(local)
            buf = h.read(take)
            if len(buf) != take:
                raise IOError("short read in %s at %d" % (self.paths[idx], local))
            parts.append(buf)
            pos += take
            n -= take
        return b"".join(parts)

    def read_final(self, offset, n):
        return self.read_at(self.starts[-1] + offset, n)


def _cd_looks_valid(final_path, cd_off, cd_size):
    if cd_off < 0 or cd_size <= 0:
        return False
    with open(final_path, "rb") as f:
        f.seek(cd_off)
        if f.read(4) != SIG_CDH:
            return False
        return cd_off + cd_size <= f.seek(0, 2)


def find_central_directory(final_path):
    """Scan the final volume's tail (direct file read) for EOCD/zip64-EOCD.

    -> (cd_offset_in_final, cd_size, n_entries, final_disk_index)
    Accepts the LAST candidate whose central directory actually starts with the
    central-header signature (guards against signatures inside entry payloads)."""
    fsize = os.path.getsize(final_path)
    tail_len = min(16 << 20, fsize)
    with open(final_path, "rb") as f:
        f.seek(fsize - tail_len)
        tail = f.read()
    sizes = [fsize]
    # zip64 EOCD candidates, last first
    pos = len(tail)
    while True:
        k64 = tail.rfind(SIG_EOCD64, 0, pos)
        if k64 < 0:
            break
        pos = k64
        try:
            (_sig, _sz, _v1, _v2, _disk, _cddisk, _ndisk, n_total,
             cd_size, cd_off) = struct.unpack("<4sQHHIIQQQQ", tail[k64:k64 + 56])
        except struct.error:
            continue
        if _cd_looks_valid(final_path, cd_off, cd_size):
            return cd_off, cd_size, n_total, _disk
    pos = len(tail)
    while True:
        k = tail.rfind(SIG_EOCD, 0, pos)
        if k < 0:
            break
        pos = k
        try:
            (_sig, _disk, _cddisk, _ndisk, _ntot, cd_size, cd_off,
             _clen) = struct.unpack("<IHHHHIIH", tail[k:k + 22])
        except struct.error:
            continue
        if _cd_looks_valid(final_path, cd_off, cd_size):
            return cd_off, cd_size, _ntot, _disk
    raise SystemExit("no EOCD with a valid central directory found in final volume")


def parse_central_directory(sr, cd_off, cd_size):
    data = sr.read_final(cd_off, cd_size)
    entries = []
    pos = 0
    while pos + 46 <= len(data) and data[pos:pos + 4] == SIG_CDH:
        (ver_made, ver_need, flag, method, mtime, mdate, crc, comp, uncomp,
         name_len, extra_len, comment_len, disk_start, int_attr, ext_attr, lho) = \
            struct.unpack("<HHHHHHIIIHHHHHII", data[pos + 4:pos + 46])
        name = data[pos + 46:pos + 46 + name_len].decode("utf-8", "surrogateescape")
        extra = data[pos + 46 + name_len:pos + 46 + name_len + extra_len]
        e = 46 + name_len + extra_len + comment_len
        # zip64 extra
        if comp == 0xFFFFFFFF or uncomp == 0xFFFFFFFF or lho == 0xFFFFFFFF or disk_start == 0xFFFF:
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                body = extra[q + 4:q + 4 + hsz]
                if hid == 0x0001:
                    b = 0
                    if uncomp == 0xFFFFFFFF and b + 8 <= len(body):
                        uncomp = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                    if comp == 0xFFFFFFFF and b + 8 <= len(body):
                        comp = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                    if lho == 0xFFFFFFFF and b + 8 <= len(body):
                        lho = struct.unpack("<Q", body[b:b + 8])[0]; b += 8
                    if disk_start == 0xFFFF and b + 4 <= len(body):
                        disk_start = struct.unpack("<I", body[b:b + 4])[0]
                q += 4 + hsz
        entries.append(dict(name=name, flag=flag, method=method, crc=crc, comp=comp,
                            uncomp=uncomp, disk=disk_start, lho=lho))
        pos += e
    return entries


def safe_dest(out_dir, name):
    parts = [p for p in name.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if not parts:
        return None
    return os.path.join(out_dir, *parts)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    final_path, out_dir = sys.argv[1], sys.argv[2]
    every = 500
    if "--every" in sys.argv:
        every = int(sys.argv[sys.argv.index("--every") + 1])
    list_only = "--list" in sys.argv
    overwrite = "--overwrite" in sys.argv
    # authoritative volume set from the final volume's EOCD disk numbering:
    # disks 0..k-1 are the z-volumes, disk k IS the final .zip. Any extra
    # .zNN files beyond that (e.g. a ghost z29 written by the downloader's
    # manifest) are ignored with a warning.
    d, base = os.path.split(final_path)
    name = os.path.splitext(base)[0]
    pat = _re.compile(_re.escape(name) + r"\.z(\d+)$")
    vols = sorted((int(m.group(1)), os.path.join(d, fn))
                  for fn in os.listdir(d or ".")
                  for m in [pat.match(fn)] if m)
    cd_off, cd_size, n_total_eocd, final_disk = find_central_directory(final_path)
    need_z = final_disk
    if len(vols) < need_z:
        raise SystemExit("archive declares %d split volumes, found only %d" % (need_z, len(vols)))
    if len(vols) > need_z:
        print("WARNING: ignoring %d extra volume file(s) beyond declared set: %s"
              % (len(vols) - need_z, ", ".join(p for _, p in vols[need_z:])), flush=True)
    vol_paths = [p for _, p in vols[:need_z]]
    sr = SplitReader(final_path, vol_paths)
    print("volumes=%d (declared %d) total_bytes=%d" % (len(sr.paths), need_z + 1, sr.total), flush=True)
    entries = parse_central_directory(sr, cd_off, cd_size)
    print("central_dir: offset_in_final=%d size=%d entries=%d (eocd n_total=%d)"
          % (cd_off, cd_size, len(entries), n_total_eocd), flush=True)
    if list_only:
        for e in entries[:50]:
            print("  %-60s comp=%d uncomp=%d disk=%d lho=%d" % (e["name"], e["comp"], e["uncomp"], e["disk"], e["lho"]))
        return 0
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    n_ok = n_skip = n_dir = 0
    for i, e in enumerate(entries):
        dest = safe_dest(out_dir, e["name"])
        if dest is None:
            continue
        if e["name"].endswith("/"):
            os.makedirs(dest, exist_ok=True)
            n_dir += 1
        else:
            if not overwrite and os.path.exists(dest) and os.path.getsize(dest) == e["uncomp"]:
                n_skip += 1
            else:
                abs_pos = sr.starts[e["disk"]] + e["lho"]
                lh = sr.read_at(abs_pos, 30)
                if lh[:4] != SIG_LH:
                    raise SystemExit("entry %s: no local header at %d (disk %d, lho %d)"
                                     % (e["name"], abs_pos, e["disk"], e["lho"]))
                name_len, extra_len = struct.unpack("<HH", lh[26:30])
                data = sr.read_at(abs_pos + 30 + name_len + extra_len, e["comp"])
                if e["method"] == 0:
                    out = data
                elif e["method"] == 8:
                    dco = zlib.decompressobj(-15)
                    out = dco.decompress(data, e["uncomp"]) + dco.flush()
                    if len(out) != e["uncomp"]:
                        raise SystemExit("entry %s: inflated %d != %d" % (e["name"], len(out), e["uncomp"]))
                else:
                    raise SystemExit("entry %s: unsupported method %d" % (e["name"], e["method"]))
                got = binascii.crc32(out) & 0xFFFFFFFF
                if got != e["crc"]:
                    raise SystemExit("entry %s: CRC mismatch stored=%08x got=%08x" % (e["name"], e["crc"], got))
                os.makedirs(os.path.dirname(dest) or out_dir, exist_ok=True)
                with open(dest, "wb") as f:
                    f.write(out)
                n_ok += 1
        if (i + 1) % every == 0:
            print("[%s] %d/%d entries (ok=%d skip=%d dir=%d, %.0fs, CRC-verified)"
                  % (time.strftime("%T"), i + 1, len(entries), n_ok, n_skip, n_dir, time.time() - t0), flush=True)
    print("SPLIT_EXTRACT_DONE entries=%d ok=%d skipped=%d dirs=%d seconds=%.0f"
          % (len(entries), n_ok, n_skip, n_dir, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
