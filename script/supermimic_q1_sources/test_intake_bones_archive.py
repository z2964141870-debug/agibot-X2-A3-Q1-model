import argparse
import contextlib
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from intake_bones_archive import canonical_member, run, stream_extract
from inventory_amass import identity


def fixture(path, members):
    with tarfile.open(path, "w:gz") as package:
        for name, payload, kind in members:
            member = tarfile.TarInfo(name)
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(payload)
                package.addfile(member, io.BytesIO(payload))
            else:
                member.linkname = "outside.csv"
                package.addfile(member)


class BonesArchiveTests(unittest.TestCase):
    def intake_fixture(self, root):
        source = root / "source"
        source.mkdir()
        archive = source / "g1.tar.gz"
        fixture(archive, [("g1/csv/a.csv", b"a\n", tarfile.REGTYPE)])
        selected = root / "selected.jsonl"
        selected.write_text(json.dumps({"source_group": "synthetic_fixture_group", "split": "train",
                                        "is_mirror": False, "quarantine": "", "move_g1_path": "g1/csv/a.csv"}) + "\n")
        all_splits = root / "all.csv"
        with all_splits.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["move_g1_path"])
            writer.writerow(["g1/csv/a.csv"])
        metadata = root / "metadata.json"
        metadata.write_text(json.dumps({"status": "METADATA_AUDIT_AND_CANDIDATE_SPLITS_COMPLETED",
                                        "synthetic_test_fixture": True,
                                        "selected_candidates": identity(selected), "all_splits": identity(all_splits),
                                        "selected_candidate_groups": {"train": 1}}))
        goal = root / "goal.json"
        goal.write_text(json.dumps({"storage": {"minimum_free_bytes": 0,
                                               "maximum_experiment_unique_allocated_bytes": 1024 ** 2,
                                               "maximum_new_source_unique_allocated_bytes": 1024 ** 2}}))
        return argparse.Namespace(archive=archive, output_dir=root / "output", source_root=source,
                                  expected_bytes=archive.stat().st_size,
                                  expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                                  goal_config=goal, expected_oom_kill=None, metadata_summary=metadata,
                                  member_prefix="")

    def test_full_intake_receipt_is_transport_only_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.intake_fixture(Path(folder))
            before = args.archive.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                run(args)
            receipt = json.loads((args.output_dir / "receipt.json").read_text())
            self.assertEqual(receipt["status"], "ARCHIVE_ACCEPTED_FROZEN_SELECTION_EXTRACTED")
            self.assertEqual(receipt["new_q1_references"], 0)
            self.assertEqual(receipt["new_optimizer_updates"], 0)
            self.assertEqual(args.archive.read_bytes(), before)
            self.assertEqual(receipt["selected_split_counts"], {"train": 1})

    def test_wrong_complete_archive_hash_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            args = self.intake_fixture(Path(folder))
            args.expected_sha256 = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                run(args)
            self.assertFalse((args.output_dir / "files").exists())
            self.assertEqual(json.loads((args.output_dir / "receipt.json").read_text())["status"], "FAILED")

    def test_changed_frozen_selection_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.intake_fixture(root)
            with (root / "selected.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaisesRegex(ValueError, "Frozen metadata artifact changed"):
                run(args)
            self.assertFalse((args.output_dir / "files").exists())
            self.assertEqual(json.loads((args.output_dir / "receipt.json").read_text())["status"], "FAILED")

    def test_only_frozen_selection_is_extracted_and_hashes_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "g1.tar.gz"
            fixture(archive, [("g1/csv/a.csv", b"a\n", tarfile.REGTYPE),
                              ("g1/csv/b.csv", b"b\n", tarfile.REGTYPE)])
            output = root / "output"
            output.mkdir()
            result = stream_extract(archive, {"g1/csv/a.csv"}, output,
                                    catalog={"g1/csv/a.csv", "g1/csv/b.csv", "g1/csv/c.csv"})
            self.assertEqual((output / "files/g1/csv/a.csv").read_bytes(), b"a\n")
            self.assertFalse((output / "files/g1/csv/b.csv").exists())
            self.assertEqual(result["selected"]["g1/csv/a.csv"]["sha256"], hashlib.sha256(b"a\n").hexdigest())
            self.assertEqual(result["catalog_paths_missing"], 1)
            self.assertTrue(result["gzip_read_to_eof_and_crc_checked"])

    def test_explicit_prefix_handles_archive_root_without_guessing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "g1.tar.gz", [("./csv/a.csv", b"a", tarfile.REGTYPE)])
            result = stream_extract(root / "g1.tar.gz", {"g1/csv/a.csv"}, root, prefix="g1")
            self.assertEqual(result["selected_files"], 1)
            self.assertTrue((root / "files/g1/csv/a.csv").exists())

    def test_parent_absolute_windows_and_prefix_paths_are_rejected(self):
        for name, prefix in (("../escape.csv", ""), ("/absolute.csv", ""),
                             ("C:/escape.csv", ""), ("a\\b.csv", ""),
                             ("a.csv", "../outside"), ("a.csv", "/outside")):
            with self.assertRaises(ValueError):
                canonical_member(name, prefix)

    def test_links_and_repeated_canonical_names_are_rejected(self):
        for members in ([("g1/csv/a.csv", b"", tarfile.SYMTYPE)],
                        [("g1/csv/a.csv", b"", tarfile.LNKTYPE)],
                        [("g1/csv/a.csv", b"a", tarfile.REGTYPE),
                         ("./g1/csv/a.csv", b"b", tarfile.REGTYPE)]):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture(root / "g1.tar.gz", members)
                with self.assertRaises(ValueError):
                    stream_extract(root / "g1.tar.gz", {"g1/csv/a.csv"}, root)

    def test_unselected_link_is_also_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture(root / "g1.tar.gz", [("g1/csv/a.csv", b"a", tarfile.REGTYPE),
                                        ("unselected", b"", tarfile.SYMTYPE)])
            with self.assertRaisesRegex(ValueError, "regular files"):
                stream_extract(root / "g1.tar.gz", {"g1/csv/a.csv"}, root)

    def test_missing_selection_and_storage_bounds_refuse_acceptance(self):
        for kwargs, wanted in (({}, {"g1/csv/missing.csv"}),
                               ({"maximum_member_bytes": 1}, {"g1/csv/a.csv"}),
                               ({"maximum_selected_bytes": 1}, {"g1/csv/a.csv"}),
                               ({"maximum_expanded_bytes": 1}, {"g1/csv/a.csv"}),
                               ({"maximum_members": 0}, {"g1/csv/a.csv"})):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture(root / "g1.tar.gz", [("g1/csv/a.csv", b"a\n", tarfile.REGTYPE)])
                with self.assertRaises(ValueError):
                    stream_extract(root / "g1.tar.gz", wanted, root, **kwargs)

    def test_complete_tar_without_gzip_footer_is_not_accepted(self):
        for corrupt_crc in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                archive = root / "g1.tar.gz"
                fixture(archive, [("g1/csv/a.csv", b"a\n", tarfile.REGTYPE)])
                original = archive.read_bytes()
                modified = original[:-8] + bytes([original[-8] ^ 1]) + original[-7:] if corrupt_crc else original[:-8]
                archive.write_bytes(modified)
                with self.assertRaises((EOFError, gzip.BadGzipFile)):
                    stream_extract(archive, {"g1/csv/a.csv"}, root)

    def test_existing_extraction_directory_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "g1.tar.gz"
            fixture(archive, [("g1/csv/a.csv", b"a\n", tarfile.REGTYPE)])
            stream_extract(archive, {"g1/csv/a.csv"}, root)
            with self.assertRaises(FileExistsError):
                stream_extract(archive, {"g1/csv/a.csv"}, root)


if __name__ == "__main__":
    unittest.main()
