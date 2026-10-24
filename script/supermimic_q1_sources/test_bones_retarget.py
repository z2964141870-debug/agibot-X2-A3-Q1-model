import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from bones_g1_csv import JOINT_NAMES_29, decode_bones_g1
from bones_retarget import (BODY_MAP, HERE, ContactCorrectedAssets, convert, displacement_scale,
                            grounded_solver, initialize_first_pose, source_reference)
from intake_bones_archive import run as intake_archive
from inventory_amass import identity


class BonesRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ.get("Q1_BONES_ROOT", "/opt/data/private/YuanQi_Q1"))
        cls.comparison = cls.root / "data/supermimic_for_q1/BONES_P02_model_20261024/official_comparison.json"
        record = json.loads(cls.comparison.read_text())
        cls.model = mujoco.MjModel.from_xml_path(record["candidate"]["path"])
        cls.assets = ContactCorrectedAssets(cls.root)
        cls.names = list(BODY_MAP)
        current = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, current)
        cls.standing_height = float(current.xpos[cls.model.body("pelvis").id, 2])

    def payload(self, *, angle=True, moving=True):
        header = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
                  "root_rotateX", "root_rotateY", "root_rotateZ",
                  *(name + "_dof" for name in JOINT_NAMES_29)]
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(header)
        for frame in range(13):
            values = {name: 0. for name in header}
            values.update(Frame=100 + frame, root_translateX=123. + (60. * frame / 120 if moving else 0),
                          root_translateY=-25., root_translateZ=self.standing_height * 100)
            if angle:
                values.update(root_rotateX=10., root_rotateY=20., root_rotateZ=40.,
                              left_knee_joint_dof=10. + 6. * frame / 120.)
            writer.writerow([values[name] for name in header])
        return stream.getvalue().encode()

    def test_native_clock_and_fixed_horizontal_origin(self):
        motion = decode_bones_g1(self.payload(), source_hz=120., expected_frames=13)
        reference, details = source_reference(self.model, motion, self.names)
        np.testing.assert_allclose(reference.times, np.arange(6) / 50, atol=1e-15)
        pelvis = self.names.index("pelvis")
        np.testing.assert_allclose(reference.positions[:, pelvis, 0], reference.times * .6, atol=1e-14)
        np.testing.assert_allclose(reference.positions[:, pelvis, 1], 0., atol=1e-15)
        self.assertEqual(details["source_frame_start"], 100)
        self.assertEqual(details["source_frame_end"], 112)
        self.assertEqual(details["time_scale"], 1.)
        self.assertAlmostEqual(details["source_duration_s"], .1)
        self.assertAlmostEqual(details["target_duration_s"], .1)

    def test_body_fk_matches_independently_constructed_states(self):
        motion = decode_bones_g1(self.payload(), source_hz=120.)
        reference, _ = source_reference(self.model, motion, self.names)
        data = mujoco.MjData(self.model)
        ids = [self.model.body(BODY_MAP[name]).id for name in self.names]
        for frame in (0, 3, 5):
            at = frame / 50.
            q = self.model.qpos0.copy()
            q[:3] = [.6 * at, 0., self.standing_height]
            q[3:7] = Rotation.from_euler("xyz", [10., 20., 40.], degrees=True).as_quat()[[3, 0, 1, 2]]
            q[self.model.jnt_qposadr[self.model.joint("left_knee_joint").id]] = np.deg2rad(10. + 6. * at)
            data.qpos[:] = q
            mujoco.mj_forward(self.model, data)
            np.testing.assert_allclose(reference.positions[frame], data.xpos[ids], atol=1e-14)
            np.testing.assert_allclose(reference.orientations[frame], data.xquat[ids], atol=1e-14)

    def test_wrong_rate_or_named_joint_layout_is_refused(self):
        motion = decode_bones_g1(self.payload(), source_hz=30.)
        with self.assertRaisesRegex(ValueError, "native rate"):
            source_reference(self.model, motion, self.names)
        motion["source_hz"] = 120.
        motion["joint_names"] = tuple(reversed(motion["joint_names"]))
        with self.assertRaisesRegex(ValueError, "named 29-joint"):
            source_reference(self.model, motion, self.names)

    def test_q1_initialization_and_ik_preserve_root_motion_at_native_speed(self):
        motion = decode_bones_g1(self.payload(angle=False), source_hz=120.)
        solver, _ = grounded_solver(self.assets)
        reference, _ = source_reference(self.model, motion, solver.names)
        scale = displacement_scale(self.model, self.assets)
        initialize_first_pose(solver, reference, scale)
        result = solver.retarget(reference, scale)
        np.testing.assert_allclose(result[1][:, :2], reference.positions[:, 0, :2] * scale, atol=1e-12)
        self.assertEqual(result[1].shape, (6, 29))
        self.assertTrue(np.isfinite(result[1]).all())
        self.assertGreater(scale, 0.)

    def test_full_synthetic_intake_to_q1_is_never_training_qualified(self):
        payload = self.payload(angle=False, moving=False)
        with tempfile.TemporaryDirectory(prefix="bones-retarget-control-",
                                          dir=self.root / "data/supermimic_for_q1") as directory:
            work = Path(directory)
            source = work / "source"
            source.mkdir()
            member = "g1/csv/000000/synthetic.csv"
            archive = source / "fixture.tar.gz"
            with tarfile.open(archive, "w:gz") as package:
                info = tarfile.TarInfo(member)
                info.size = len(payload)
                package.addfile(info, io.BytesIO(payload))
            row = {"source_group": hashlib.sha256(b"synthetic-control-only").hexdigest(),
                   "move_g1_path": member, "move_duration_frames": 13, "split": "train",
                   "is_mirror": False, "quarantine": "", "candidate_kind": "synthetic_control"}
            selected, all_splits = work / "selection.jsonl", work / "all.csv"
            selected.write_text(json.dumps(row) + "\n")
            all_splits.write_text("move_g1_path\n" + member + "\n")
            summary = work / "summary.json"
            summary.write_text(json.dumps({"status": "METADATA_AUDIT_AND_CANDIDATE_SPLITS_COMPLETED",
                                           "selected_candidates": identity(selected), "all_splits": identity(all_splits),
                                           "selected_candidate_groups": {"train": 1}}))
            output = work / "intake"
            with contextlib.redirect_stdout(io.StringIO()):
                intake_archive(SimpleNamespace(archive=archive, output_dir=output, source_root=source,
                    expected_bytes=archive.stat().st_size, expected_sha256=identity(archive)["sha256"],
                    metadata_summary=summary, goal_config=HERE / "bones_goal_20261024.json",
                    member_prefix="", expected_oom_kill=14))
                converted = work / "converted"
                convert(SimpleNamespace(root=self.root, intake=output / "receipt.json",
                    model_comparison=self.comparison, output=converted, offset=0, count=1, expected_oom_kill=14))
            manifest = json.loads((converted / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "BONES_RETARGETING_COMPLETED_INDEPENDENT_CHECK_PENDING")
            self.assertFalse(manifest["qualified_for_training"])
            self.assertEqual(manifest["optimizer_updates"], 0)
            clip = manifest["motions"][0]
            self.assertNotIn("error", clip)
            self.assertFalse(clip["qualified_for_training_pilot"])
            with np.load(clip["output"]["path"], allow_pickle=False) as saved:
                self.assertEqual(saved["qpos"].shape, (6, 29))
                np.testing.assert_allclose(saved["time_s"], np.arange(6) / 50.)


if __name__ == "__main__":
    unittest.main()
