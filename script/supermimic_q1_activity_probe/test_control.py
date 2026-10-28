"""Activity scheduling bounds, state reset, and task-independent counterexamples."""

import unittest
import torch

from activity_control import CANDIDATES, UpperActivityBlend


class TestActivity(unittest.TestCase):
    def test_static_candidates_exact(self):
        for name in ("baseline", "static025", "static075"):
            gate = UpperActivityBlend(3, name, "cpu")
            for _ in range(4):
                alpha = gate.update(torch.tensor([0., .5, 2.]), torch.tensor([0., 1., 3.]))
                torch.testing.assert_close(alpha, torch.full((3,), CANDIDATES[name][0]), rtol=0, atol=0)

    def test_activity_reduces_blend_and_limits_transition(self):
        gate = UpperActivityBlend(2, "activity025075", "cpu")
        torch.testing.assert_close(gate.update(torch.zeros(2), torch.zeros(2)), torch.full((2,), .75))
        previous = gate.alpha.clone()
        for _ in range(100):
            current = gate.update(torch.tensor([.6, 0.]), torch.tensor([0., 2.]))
            self.assertTrue(bool(((current >= .25) & (current <= .75)).all()))
            self.assertLessEqual(float((current-previous).abs().max()), .5*gate.factor+1e-6)
            previous = current
        torch.testing.assert_close(current, torch.full((2,), .25), atol=1e-4, rtol=0)

    def test_reset_is_per_environment(self):
        gate = UpperActivityBlend(2, "activity025075", "cpu")
        gate.update(torch.zeros(2), torch.zeros(2))
        gate.reset(torch.tensor([True, False]))
        alpha = gate.update(torch.ones(2), torch.zeros(2))
        self.assertEqual(float(alpha[0]), .25)
        self.assertGreater(float(alpha[1]), .25)

    def test_invalid_activity_rejected(self):
        gate = UpperActivityBlend(1, "activity025075", "cpu")
        for speed, yaw in (([-1.], [0.]), ([float("nan")], [0.]), ([0.], [float("inf")])):
            with self.assertRaises(ValueError):
                gate.update(torch.tensor(speed), torch.tensor(yaw))
        with self.assertRaises(ValueError):
            gate.update(torch.zeros(2), torch.zeros(1))


if __name__ == "__main__":
    unittest.main()
