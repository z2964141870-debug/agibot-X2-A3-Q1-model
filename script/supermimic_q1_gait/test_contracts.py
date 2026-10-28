"""Budget and physical-ranking edge cases, independent of optimization."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_ppo")]
from clock import TrainingClock
from metrics import gait_clip, checkpoint_rank
from evaluation import FIXED_DEMOS


class Contracts(unittest.TestCase):
    def test_pauses_do_not_spend_inherited_budget(self):
        with tempfile.TemporaryDirectory() as root:
            clock = TrainingClock(Path(root) / "clock.json", 3500., 14400.)
            with patch("clock.time.monotonic", side_effect=[10., 13., 100., 105.]):
                clock.begin()
                clock.end()
                clock.begin()
                clock.end()
            self.assertEqual(clock.total, 3508.)
            self.assertEqual(clock.inherited, 3500.)

    def test_missing_slip_cannot_win_checkpoint_selection(self):
        clip = {"failed": False, "completion": 1., "forward_displacement_m": 1.,
                "loaded_slip_mps": None, "foot_lift_tracking_rmse_m": 0.,
                "torso_tilt_rms_rad": 0., "target_delta_rms_rad": 0.}
        summary = {"summary": {"validation": {"failures": 0}},
                   "gait": {name: dict(clip) for name in FIXED_DEMOS}}
        self.assertEqual(checkpoint_rank([summary])[0], 999999)

    def test_physical_metrics_joint_axis_and_truncated_prefix(self):
        count = 8
        states = np.zeros((count, 1, 57), dtype=np.float32)
        states[..., 3] = 1.
        states[:, 0, 0] = np.linspace(.01, .08, count)
        reference = np.zeros((count, 1, 3), dtype=np.float32)
        reference[:, 0, 0] = np.linspace(.01, .08, count)
        feet = np.zeros((count, 1, 2, 3), dtype=np.float32)
        feet[..., 0] = states[..., :1]
        trace = dict(states=states, feet=feet, reference_feet=feet.copy(), reference_p=reference,
                     reference_quat=states[..., 3:7].copy(), reference_q=np.zeros((count, 1, 22)),
                     force=np.full((count, 1, 2), 10.), slip=np.full((count, 1, 2), .1),
                     targets=np.zeros((count, 1, 22)), initial=states[0].copy(),
                     steps=np.array([count]), reference_steps=np.array([count]), failed=np.array([False]))
        result = gait_clip(trace, 0, 4)
        self.assertAlmostEqual(result["knee_reference_rmse_rad"], 0.)
        self.assertAlmostEqual(result["loaded_slip_mps"], .1)
        self.assertEqual(result["common_prefix_steps"], 4)
        self.assertGreater(result["forward_displacement_m"], 0.)


if __name__ == "__main__":
    unittest.main()
