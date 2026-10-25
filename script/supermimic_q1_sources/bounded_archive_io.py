"""Sequential archive reads with file-local cache advice and bounded buffers."""

import hashlib
import os
from pathlib import Path


WINDOW = 16 * 1024 ** 2


class ReleasingReader:
    def __init__(self, stream):
        self.stream, self.released = stream, 0
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_SEQUENTIAL)

    def read(self, count=-1):
        if count < 0:
            raise ValueError("Archive reads require an explicit bounded length")
        value = self.stream.read(count)
        self.release()
        return value

    def release(self, final=False):
        end = self.stream.tell() // 4096 * 4096
        if hasattr(os, "posix_fadvise") and end > self.released and (final or end - self.released >= WINDOW):
            os.posix_fadvise(self.stream.fileno(), self.released, end - self.released, os.POSIX_FADV_DONTNEED)
            self.released = end

    def close(self):
        if not self.stream.closed:
            self.release(final=True)
            self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getattr__(self, name):
        return getattr(self.stream, name)


class ArchivePath(type(Path())):
    def open(self, mode="r", *args, **kwargs):
        stream = super().open(mode, *args, **kwargs)
        return ReleasingReader(stream) if mode == "rb" else stream


def bounded_identity(path, progress=None):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest, count = hashlib.sha256(), 0
    with ArchivePath(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            count += len(block)
            if progress is not None and (count % (64 * 1024 ** 2) == 0 or count == before.st_size):
                progress(count, before.st_size)
    after = path.stat()
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or count != after.st_size):
        raise ValueError("Source changed during the bounded archive read")
    return {"path": str(path), "bytes": after.st_size, "sha256": digest.hexdigest()}
