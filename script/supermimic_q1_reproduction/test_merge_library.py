"""Verify batch coverage and reference reuse without geometry admission gates."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from merge_library import merge_libraries
from reference_admission import POLICY, TRACKED, assemble, identity


def batch(root, name):
    motions = root / (name + "_converted")
    motions.mkdir()
    path = motions / (name + ".npz")
    frames = 3
    qpos = np.zeros((frames, 29))
    qpos[:, 2:4] = 1.
    np.savez(path, time_s=np.arange(frames) / 50., fps=50., body_names=TRACKED,
             qpos=qpos, qvel=np.zeros((frames, 28)), joint_pos=qpos[:, 7:],
             joint_vel=np.full((frames, 22), 8.), body_pos_w=np.ones((frames, len(TRACKED), 3)),
             body_quat_w=np.tile([1., 0., 0., 0.], (frames, len(TRACKED), 1)),
             body_lin_vel_w=np.zeros((frames, len(TRACKED), 3)),
             body_ang_vel_w=np.zeros((frames, len(TRACKED), 3)))
    manifest = {"status": "BONES_RETARGETING_COMPLETED_INDEPENDENT_CHECK_PENDING",
                "reference_protocol": "TEST_NATIVE_TIME", "q1_assets": {"robot": "Q1"},
                "motions": [{"clip": name, "split": "train", "output": identity(path),
                             "qualified_for_training_pilot": False}]}
    (motions / "manifest.json").write_text(json.dumps(manifest))
    library = root / (name + "_library")
    assemble(motions, library)
    return library, path


class MergeTests(unittest.TestCase):
    def test_merge_preserves_diagnostics_and_uses_existing_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, path = batch(root, "train_first")
            second, _ = batch(root, "train_second")
            before = identity(path)
            result = merge_libraries([first, second], root / "merged")
            self.assertEqual(result["admitted_clips"], ["train_first", "train_second"])
            self.assertEqual(result["admission_policy"], POLICY)
            self.assertAlmostEqual(result["admitted_duration_s"], .08)
            link = root / "merged/references/train_first.npz"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.resolve(), path)
            self.assertEqual(identity(path), before)
            manifest = json.loads((root / "merged/references/manifest.json").read_text())
            self.assertFalse(manifest["motions"][0]["qualified_for_training_pilot"])

    def test_duplicate_batch_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library, _ = batch(root, "train_first")
            with self.assertRaisesRegex(ValueError, "Duplicate clip"):
                merge_libraries([library, library], root / "merged")
            self.assertFalse((root / "merged").exists())


if __name__ == "__main__":
    unittest.main()
