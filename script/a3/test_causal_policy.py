"""Check raw-arrival causality including the velocity lookahead."""

from types import SimpleNamespace
import unittest

import numpy as np

from script.a3.causal_policy import CausalPolicyReplay
from script.a3.reference_contract import causal_window


class ReplayTests(unittest.TestCase):
    def make_replay(self):
        source = np.arange(100) / 30
        reference = SimpleNamespace(path="test.csv", num_frames=165,
                                    values=np.arange(165, dtype=float))
        class Runner:
            def _record_metrics_step(self, *args):
                pass
        sim = SimpleNamespace(LoopSimRunner=Runner,
            load_a3_flat_csv=lambda *a, **k: (source, None, None, len(source)),
            build_encoder_input=lambda ref, base, *args: ref.values[base:base + 11].copy())
        return CausalPolicyReplay(sim, 30, 1, 180), reference, source

    def test_velocity_requires_extra_20ms(self):
        times = np.arange(100) / 30
        window = causal_window(times, .22, nominal_delay=.20, lookahead_s=.20)
        self.assertLessEqual(window["base"] + .20, times[6] + 1e-9)
        self.assertLess(window["base"], causal_window(times, .22)["base"])

    def test_mutating_future_cannot_change_policy_window(self):
        replay, reference, times = self.make_replay()
        for frame in range(140):
            before = replay.build(reference, frame, None)
            row = replay.rows[-1]
            self.assertLessEqual(row[4], row[1] + 1e-9)
            old = reference.values.copy()
            unavailable = np.arange(reference.num_frames) / 50 > times[int(row[5])] + 1e-9
            reference.values[unavailable] = 1e10
            np.testing.assert_array_equal(before, replay.build(reference, frame, None))
            reference.values[:] = old
        self.assertEqual(replay.prefill_s, .20)

    def test_metrics_use_commanded_target(self):
        replay, reference, _ = self.make_replay()
        seen = []
        replay.original_record = lambda runner, step, frame, result: seen.append(frame)
        replay.install()
        replay.build(reference, 2, None)
        replay.sim.LoopSimRunner()._record_metrics_step(2, 2, None)
        self.assertEqual(seen, [replay.last_target_frame])


if __name__ == "__main__":
    unittest.main()
