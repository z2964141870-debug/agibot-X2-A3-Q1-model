import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from build_support_projection import SETTINGS, shifted_states, support_shift
from audit_support_projection import file_id, run as audit_projection


class SupportProjectionTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy required for sparse projection")
    def test_constant_and_affine_floor_gaps_preserve_zero_curvature(self):
        for height in (np.full(21, .02), np.linspace(.01, .025, 21)):
            shift, result = support_shift(height)
            np.testing.assert_allclose(height + shift, SETTINGS["target_lower_foot_z_m"], atol=1e-10)
            self.assertLess(result["second_difference_rms_m"], 1e-10)

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy required for sparse projection")
    def test_curved_correction_stays_inside_band_and_reduces_objective(self):
        height = .01 + .005 * np.sin(np.arange(100) * .3)
        shift, _ = support_shift(height)
        lower = height + shift
        self.assertGreaterEqual(lower.min(), SETTINGS["minimum_lower_foot_z_m"] - 1e-10)
        self.assertLessEqual(lower.max(), SETTINGS["maximum_lower_foot_z_m"] + 1e-10)
        exact = SETTINGS["target_lower_foot_z_m"] - height
        def objective(value):
            return np.sum((value - exact) ** 2) + SETTINGS["second_difference_weight"] ** 2 * np.sum(np.diff(value, n=2) ** 2)
        self.assertLess(objective(shift), objective(exact))

    @unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy required for sparse projection")
    def test_invalid_or_overlarge_corrections_are_rejected(self):
        for height in ([0.] * 4, np.full(10, np.nan), np.full(10, .2)):
            with self.assertRaises(ValueError):
                support_shift(height)

    def test_only_root_vertical_pose_and_velocity_change(self):
        rng = np.random.default_rng(51)
        poses, speeds = rng.normal(size=(13, 29)), rng.normal(size=(13, 28))
        original_poses, original_speeds = poses.copy(), speeds.copy()
        shift = .001 * np.sin(np.arange(13))
        new_poses, new_speeds = shifted_states(poses, speeds, shift, .02)
        keep = np.arange(29) != 2
        np.testing.assert_array_equal(new_poses[:, keep], poses[:, keep])
        np.testing.assert_array_equal(new_speeds[:, np.arange(28) != 2], speeds[:, np.arange(28) != 2])
        np.testing.assert_allclose(new_poses[:, 2], poses[:, 2] + shift)
        np.testing.assert_allclose(new_speeds[:-1, 2], np.diff(new_poses[:, 2]) / .02)
        self.assertEqual(new_speeds[-1, 2], new_speeds[-2, 2])
        np.testing.assert_array_equal(poses, original_poses)
        np.testing.assert_array_equal(speeds, original_speeds)

    def test_state_layout_and_time_fail_closed(self):
        for count, dt in ((4, .02), (8, 0)):
            with self.assertRaises(ValueError):
                shifted_states(np.zeros((count, 29)), np.zeros((count, 28)), np.zeros(count), dt)


class ProjectionArtifactTests(unittest.TestCase):
    def fixture(self, root, corrupt=None):
        folder = root / "data"; folder.mkdir()
        n = 8
        poses = np.zeros((n, 29)); poses[:, 2] = .5; poses[:, 3] = 1
        body_positions = np.zeros((n, 2, 3)); body_positions[:, :, 2] = .5
        body_quats = np.zeros((n, 2, 4)); body_quats[:, :, 0] = 1
        parent = {"qpos": poses, "qvel": np.zeros((n, 28)), "body_pos_w": body_positions,
                  "body_quat_w": body_quats, "body_lin_vel_w": np.zeros((n, 2, 3)),
                  "body_ang_vel_w": np.zeros((n, 2, 3)), "time_s": np.arange(n) * .02,
                  "fps": np.array(50.), "body_names": np.array(["root", "foot"]),
                  "joint_pos": np.zeros((n, 22)), "joint_vel": np.zeros((n, 22))}
        candidate = {key: value.copy() for key, value in parent.items()}
        shift = np.full(n, -.01025)
        candidate["qpos"][:, 2] += shift
        candidate["body_pos_w"][:, :, 2] += shift[:, None]
        if corrupt == "joint":
            candidate["qpos"][3, 10] = .01
        elif corrupt == "velocity":
            candidate["qvel"][3, 2] = .01
        elif corrupt == "body":
            candidate["body_pos_w"][3, 1, 0] = .01
        def save(name, arrays):
            path = folder / name
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            return file_id(path)
        parent_id, candidate_id = save("parent.npz", parent), save("candidate.npz", candidate)
        parent_gap = np.full((n, 2), .01)
        projection_id = save("projection.npz", {
            "root_height_shift_m": shift, "parent_foot_gap_m": parent_gap,
            "candidate_foot_gap_m": parent_gap + shift[:, None],
            "parent_self_contact_count": np.zeros(n, dtype=int),
            "candidate_self_contact_count": np.zeros(n, dtype=int)})
        indices = np.arange(2, n - 2)
        raw_id = save("dynamics.npz", {"frame_index": indices,
                                       "foot_gap_m": (parent_gap + shift[:, None])[indices]})
        dynamic_path = folder / "dynamics.json"
        dynamic_path.write_text(json.dumps({"clips": [
            {"clip": "fixture", "reference": candidate_id, "raw_arrays": raw_id}]}))
        receipt = folder / "receipt.json"
        receipt.write_text(json.dumps({"status": "SUPPORT_HEIGHT_CANDIDATE_BUILT_NOT_ADMITTED",
            "inputs": [], "candidate_dynamics": file_id(dynamic_path), "settings": SETTINGS,
            "clips": [{"clip": "fixture", "parent": parent_id, "candidate": candidate_id,
                       "projection_arrays": projection_id, "qualified_for_training_pilot": False}]}))
        return receipt, folder / "audit.json"

    def test_valid_export_passes_independent_artifact_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            receipt, output = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                audit_projection(root, receipt, output)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "SUPPORT_PROJECTION_EXPORT_AUDIT_PASSED")
            self.assertTrue(result["clips"][0]["only_vertical_root_motion_changed"])

    def test_wrong_joint_velocity_or_body_export_is_rejected(self):
        for corruption in ("joint", "velocity", "body"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                receipt, output = self.fixture(root, corruption)
                with self.assertRaises(ValueError):
                    audit_projection(root, receipt, output)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
