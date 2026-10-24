import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

import test_bones_retarget as fixtures
import bones_verify_reference as verification
from bones_retarget import HERE, convert
from bones_verify_reference import check, read_source, verify
from intake_bones_archive import run as intake_archive
from supermimic.q1_assets import identity


def rewrite(payload, mutate):
    reader = csv.DictReader(io.StringIO(payload.decode()))
    rows = list(reader)
    mutate(rows)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode()


class BonesIndependentReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.BonesRetargetTests.setUpClass()
        cls.fixture = fixtures.BonesRetargetTests()

    @contextlib.contextmanager
    def converted_fixture(self):
        fixture = self.fixture
        payload = fixture.payload(angle=False, moving=False)
        with tempfile.TemporaryDirectory(prefix="bones-independent-control-",
                                          dir=fixture.root / "data/supermimic_for_q1") as directory:
            work = Path(directory)
            source = work / "source"
            source.mkdir()
            member = "g1/csv/000000/synthetic.csv"
            archive = source / "fixture.tar.gz"
            with tarfile.open(archive, "w:gz") as package:
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                package.addfile(info, io.BytesIO(payload))
            row = {"source_group": hashlib.sha256(b"independent-synthetic-control-only").hexdigest(),
                   "move_g1_path": member, "move_duration_frames": 13, "split": "train",
                   "is_mirror": False, "quarantine": "", "candidate_kind": "synthetic_control"}
            selected, all_splits = work / "selection.jsonl", work / "all.csv"
            selected.write_text(json.dumps(row) + "\n")
            all_splits.write_text("move_g1_path\n" + member + "\n")
            summary = work / "summary.json"
            summary.write_text(json.dumps({"status": "METADATA_AUDIT_AND_CANDIDATE_SPLITS_COMPLETED",
                                           "selected_candidates": identity(selected), "all_splits": identity(all_splits),
                                           "selected_candidate_groups": {"train": 1}}))
            intake = work / "intake"
            converted = work / "converted"
            with contextlib.redirect_stdout(io.StringIO()):
                intake_archive(SimpleNamespace(archive=archive, output_dir=intake, source_root=source,
                    expected_bytes=archive.stat().st_size, expected_sha256=identity(archive)["sha256"],
                    metadata_summary=summary, goal_config=HERE / "bones_goal_20261024.json",
                    member_prefix="", expected_oom_kill=14))
                convert(SimpleNamespace(root=fixture.root, intake=intake / "receipt.json",
                    model_comparison=fixture.comparison, output=converted, offset=0, count=1, expected_oom_kill=14))
            manifest = json.loads((converted / "manifest.json").read_text())
            yield converted, manifest

    def test_independent_reader_keeps_explicit_native_frames_and_units(self):
        times, q, _ = read_source(self.fixture.payload(), self.fixture.model, 13)
        np.testing.assert_allclose(times, np.arange(13) / 120.)
        np.testing.assert_allclose(q[:, 0], 1.23 + .6 * times)
        self.assertEqual(q.shape, (13, 36))
        self.assertAlmostEqual(q[0, 7 + 3], np.deg2rad(10.))
        spaced = self.fixture.payload().replace(b",", b", ")
        np.testing.assert_allclose(read_source(spaced, self.fixture.model, 13)[1], q)

    def test_numeric_identity_ignores_global_origin_heading_and_frame_offset(self):
        original = self.fixture.payload()

        def change(rows):
            rotation = Rotation.from_euler("z", 70., degrees=True).as_matrix()
            for row in rows:
                p = np.asarray([float(row["root_translate" + axis]) for axis in "XYZ"])
                p = rotation @ p + [1000., -200., 50.]
                for axis, value in zip("XYZ", p):
                    row["root_translate" + axis] = value
                row["root_rotateZ"] = float(row["root_rotateZ"]) + 70.
                row["Frame"] = int(row["Frame"]) + 200

        shifted = rewrite(original, change)
        a = read_source(original, self.fixture.model, 13)[2]
        b = read_source(shifted, self.fixture.model, 13)[2]
        self.assertEqual(a, b)
        changed = rewrite(original, lambda rows: rows[5].update(left_knee_joint_dof=30.))
        self.assertNotEqual(a, read_source(changed, self.fixture.model, 13)[2])

    def test_bad_alias_header_and_gapped_frames_are_rejected(self):
        payload = self.fixture.payload()
        invalid = payload.replace(b"root_translate", b"Hips_translate")
        with self.assertRaisesRegex(ValueError, "header"):
            read_source(invalid, self.fixture.model, 13)
        invalid = rewrite(payload, lambda rows: rows[4].update(Frame=999))
        with self.assertRaisesRegex(ValueError, "Frame sequence"):
            read_source(invalid, self.fixture.model, 13)

    def test_full_independent_recompute_does_not_admit_a_training_corpus(self):
        with self.converted_fixture() as (directory, manifest):
            output = directory / "independent.json"
            with contextlib.redirect_stdout(io.StringIO()):
                verify(SimpleNamespace(root=self.fixture.root, motions=directory, output=output, expected_oom_kill=14))
            result = json.loads(output.read_text())
            self.assertFalse(result["corpus_ready_for_training"])
            self.assertEqual(result["physics_steps"], 0)
            self.assertEqual(result["optimizer_updates"], 0)
            row = result["clips"][0]
            for gate in ("source_fk", "independent_target_mapping", "native_120_to_50hz",
                         "root_displacement_preserved", "root_orientation_preserved", "qpos_export",
                         "qvel_export", "joint_export", "body_fk_export"):
                self.assertTrue(row["gates"][gate], msg=gate)

    def test_rehashed_converter_target_error_is_caught_by_reconstruction(self):
        with self.converted_fixture() as (_, manifest):
            row = manifest["motions"][0]
            path = Path(row["kinematic_trace"]["path"])
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays["target_position_m_world"][:, :, 0] += .025
            arrays["source_position_m_world"][:, :, 1] += .01
            np.savez_compressed(path, **arrays)
            row["kinematic_trace"] = identity(path)
            result = check(self.fixture.assets, self.fixture.model, row, manifest["scale"])
            self.assertFalse(result["gates"]["independent_target_mapping"])
            self.assertFalse(result["gates"]["source_fk"])
            self.assertFalse(result["qualified_for_training_pilot"])

    def test_rehashed_time_stretch_is_refused(self):
        with self.converted_fixture() as (_, manifest):
            row = manifest["motions"][0]
            path = Path(row["output"]["path"])
            with np.load(path, allow_pickle=False) as saved:
                arrays = {name: saved[name].copy() for name in saved.files}
            arrays["time_s"] *= 2
            np.savez_compressed(path, **arrays)
            row["output"] = identity(path)
            with self.assertRaisesRegex(ValueError, "native time"):
                check(self.fixture.assets, self.fixture.model, row, manifest["scale"])

    def test_changed_displacement_scale_is_refused_before_qualification(self):
        with self.converted_fixture() as (directory, manifest):
            manifest["scale"] *= .5
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "standing leg lengths"):
                verify(SimpleNamespace(root=self.fixture.root, motions=directory,
                                       output=directory / "independent.json", expected_oom_kill=14))

    def test_verifier_source_change_is_refused_without_editing_shared_files(self):
        with self.converted_fixture() as (directory, _):
            original, calls = verification.identity, 0

            def changing_identity(path):
                nonlocal calls
                record = original(path)
                if Path(path).resolve() == Path(verification.__file__).resolve():
                    calls += 1
                    if calls > 1:
                        record["sha256"] = "0" * 64
                return record

            with patch.object(verification, "identity", side_effect=changing_identity):
                with self.assertRaisesRegex(ValueError, "Frozen converter"):
                    verify(SimpleNamespace(root=self.fixture.root, motions=directory,
                                           output=directory / "independent.json", expected_oom_kill=14))


if __name__ == "__main__":
    unittest.main()
