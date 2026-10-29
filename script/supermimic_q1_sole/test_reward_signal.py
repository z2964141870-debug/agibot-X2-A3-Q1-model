"""Guard source alignment and reward decomposition in saved-state analysis."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from reward_signal import position_z_replacement, saved_clip
from target_cost_attribution import cost_parts, run


class RewardSignalTest(unittest.TestCase):
    def test_foot_replacement_preserves_other_components(self):
        errors = np.arange(1, 85, dtype=float).reshape(2, 14, 3) / 10000
        original = np.square(errors).sum(-1).mean(-1)
        ankle = errors[:, [3, 6], 2]
        sole = np.array([[.012, -.025], [.03, .005]])
        changed = errors.copy()
        changed[:, [3, 6], 2] = sole
        np.testing.assert_allclose(position_z_replacement(original, ankle, sole, 14),
                                   np.square(changed).sum(-1).mean(-1), rtol=1e-14)

    def test_rejects_incompatible_or_invalid_errors(self):
        for error, ankle, sole, count in (([-1.], [[0., 0.]], [[0., 0.]], 14),
                ([0.], [[1., 0.]], [[0., 0.]], 14),
                ([0.], [[0., 0.]], [[np.nan, 0.]], 14),
                ([0.], [[0.]], [[0.]], 14),
                ([0.], [[0., 0.]], [[0., 0.]], True)):
            with self.subTest(error=error, ankle=ankle, count=count), self.assertRaises(ValueError):
                position_z_replacement(error, ankle, sole, count)

    def trace(self):
        result = {name:np.zeros((5, 1) + width) for name, width in
                  {"states":(57,), "targets":(22,), "force":(2,), "slip":(2,)}.items()}
        result.update(frames=np.arange(1, 6, dtype=np.int64)[:, None], steps=np.array([3]),
                      reference_steps=np.array([5]), alive=np.ones((5, 1), dtype=bool))
        return result

    def test_excludes_reset_suffix_and_copies_arrays(self):
        trace = self.trace()
        trace["states"][3:] = np.nan
        selected = saved_clip(trace, 0, 3)
        self.assertEqual(selected["states"].shape, (3, 57))
        selected["states"][:] = 9
        self.assertEqual(trace["states"][0, 0, 0], 0)

    def test_rejects_duration_frame_and_alive_mismatch(self):
        for field in ("duration", "frame", "alive", "state_shape", "nonfinite"):
            trace = self.trace()
            if field == "duration":
                trace["steps"][0] = 4
            elif field == "frame":
                trace["frames"][1] = 99
            elif field == "alive":
                trace["alive"][1] = False
            elif field == "state_shape":
                trace["states"] = np.zeros((5, 1, 56))
            else:
                trace["force"][0] = np.nan
            with self.subTest(field=field), self.assertRaises(ValueError):
                saved_clip(trace, 0, 3)

    def test_cost_partition_retains_total_clamp_and_different_action_scales(self):
        targets = np.zeros((3, 22))
        targets[1:, :14] = .01
        targets[2, 14:] = 10.
        scale = np.r_[np.full(14, .5), np.full(8, 2.)]
        values = cost_parts(targets, scale, -.1, .02)
        np.testing.assert_allclose(values["base_controlled14"], [.002 * 14 * .02**2, 0.])
        np.testing.assert_allclose(values["base_upper8"], [0., .002 * 8 * 5**2])
        np.testing.assert_allclose(values["ppo_controlled14"] + values["ppo_upper8"],
                                   [.002 * 14 * .1**2 / 22, .05])

    def test_attribution_accepts_json_paths_and_detects_changed_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "evaluation_library.json"
            library.write_text(json.dumps({"splits": {"train": [{"name": "walk_fixture"}]}}))
            trace = root / "initial_static075_repeat0.npz"
            np.savez(trace, targets=np.zeros((3, 1, 22)))
            def identity(path):
                content = path.read_bytes()
                return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
            (root / "summary.json").write_text(json.dumps(dict(sources=[identity(library)],
                trace_sources=[identity(trace)], action_scales=[.5]*22,
                action_rate_weight=-.1, control_dt=.02)))
            (root / "fixed_reward_signal.csv").write_text(
                "label,repeat,clip,steps,base_saved_action_rate_cost_mean,ppo_saved_smooth_cost_mean\n"
                "initial_static075,0,walk_fixture,3,0,0\n")
            with patch("target_cost_attribution.plot"), contextlib.redirect_stdout(io.StringIO()):
                run(SimpleNamespace(review=root))
                destination = root / "target_cost_attribution.json"
                self.assertTrue(json.loads(destination.read_text())["reconstruction_of_prior_total_costs_passed"])
                destination.unlink()
                np.savez(trace, targets=np.ones((3, 1, 22)))
                with self.assertRaisesRegex(ValueError, "Saved trace changed"):
                    run(SimpleNamespace(review=root))


if __name__ == "__main__":
    unittest.main()
