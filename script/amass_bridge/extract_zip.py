#!/usr/bin/env python3
"""extract_zip.py -- zip64-safe streaming extractor with CRC checking and resume.

Usage: extract_zip.py <archive.zip> <out_dir> [--every N]
Skips members that already exist with the same size; aborts non-zero on any
CRC/zip error so the caller can keep the merged archive for retry.
"""
import os
import sys
import time
import zipfile


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    arc, out = sys.argv[1], sys.argv[2]
    every = 500
    if "--every" in sys.argv:
        every = int(sys.argv[sys.argv.index("--every") + 1])
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    n_ok = n_skip = 0
    with zipfile.ZipFile(arc) as z:
        infos = z.infolist()
        total = len(infos)
        for i, info in enumerate(infos):
            dest = os.path.join(out, info.filename)
            if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
                n_skip += 1
            else:
                os.makedirs(os.path.dirname(dest) or out, exist_ok=True)
                with z.open(info) as src, open(dest, "wb") as dst:
                    while True:
                        chunk = src.read(1 << 20)
                        if not chunk:
                            break
                        dst.write(chunk)
                n_ok += 1
            if (i + 1) % every == 0:
                print("[%s] %d/%d members (extracted=%d skipped=%d, %.0fs)"
                      % (time.strftime("%T"), i + 1, total, n_ok, n_skip, time.time() - t0), flush=True)
    print("EXTRACT_DONE members=%d extracted=%d skipped=%d seconds=%.0f"
          % (total, n_ok, n_skip, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
