"""Check streamed hashing and gzip/tar extraction through the cache-aware reader."""

import hashlib
import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from bounded_archive_io import ArchivePath, bounded_identity
from intake_bones_archive import stream_extract


class ArchiveIOTests(unittest.TestCase):
    def test_hash_and_explicit_lengths(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.bin"
            payload = b"abcdef" * 10000
            path.write_bytes(payload)
            self.assertEqual(bounded_identity(path)["sha256"], hashlib.sha256(payload).hexdigest())
            with ArchivePath(path).absolute().open("rb") as stream:
                self.assertEqual(stream.read(10), payload[:10])
                stream.seek(0)
                self.assertEqual(stream.read(11), payload[:11])
                with self.assertRaises(ValueError):
                    stream.read()

    def test_stream_selected_file_and_crc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "input.tar.gz"
            with tarfile.open(path, "w:gz") as package:
                info = tarfile.TarInfo("g1/csv/example.csv")
                payload = b"Frame,value\n0,1\n"
                info.size = len(payload)
                package.addfile(info, io.BytesIO(payload))
            output = root / "output"
            output.mkdir()
            report = stream_extract(ArchivePath(path), {info.name}, output)
            self.assertTrue(report["gzip_read_to_eof_and_crc_checked"])
            self.assertEqual((output / "files" / info.name).read_bytes(), payload)
            damaged = bytearray(path.read_bytes())
            damaged[-8] ^= 1
            path.write_bytes(damaged)
            output2 = root / "output2"
            output2.mkdir()
            with self.assertRaises((OSError, EOFError)):
                stream_extract(ArchivePath(path), {info.name}, output2)


if __name__ == "__main__":
    unittest.main()
