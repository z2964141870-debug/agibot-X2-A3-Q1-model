"""Known-motion checks for saved velocity scope, decomposition and load labels."""

import unittest

import numpy as np

from saved_velocity_review import matched_support, summarize, velocity_series


class SavedVelocityTests(unittest.TestCase):
    def packet(self, n=6):
        return np.zeros((n, 27, 3)), np.zeros((n, 27, 3)), np.zeros((n, 2))

    def test_known_uniform_translation_is_root_not_relative_motion(self):
        a,b,load = self.packet()
        a[:, 1:, 0] = np.arange(6)[:, None]*.02
        result = summarize(velocity_series(a,b,load), 5)
        self.assertAlmostEqual(result["world_body_velocity_rmse_m_s"], 1.)
        self.assertAlmostEqual(result["root_velocity_rmse_m_s"], 1.)
        self.assertAlmostEqual(result["relative_body_velocity_rmse_m_s"], 0.)

    def test_signed_cross_term_cannot_be_dropped(self):
        a,b,load = self.packet()
        a[:, 1, 0] = np.arange(6)*.02
        result = summarize(velocity_series(a,b,load), 5)
        self.assertLess(result["signed_cross_term"], 0.)
        self.assertAlmostEqual(result["world_mean_squared_vector_error"], 1./26.)
        self.assertAlmostEqual(result["root_mean_squared_vector_error"]
            +result["relative_mean_squared_vector_error"]+result["signed_cross_term"], 1./26.)

    def test_adding_equal_time_varying_translation_preserves_world_error(self):
        a,b,load = self.packet()
        a[:, 2:, 1] = np.arange(6)[:, None]**2*.01
        before = velocity_series(a,b,load)
        common = np.arange(6)[:, None, None]**2*np.array([.02,.03,.04])[None,None]
        a[:, 1:] += common; b[:, 1:] += common
        after = velocity_series(a,b,load)
        np.testing.assert_allclose(before["world_energy"], after["world_energy"], atol=1e-12)

    def test_future_perturbation_never_changes_prior_velocity_rows(self):
        a,b,load = self.packet()
        before = velocity_series(a,b,load)
        a[4:, 1:] = 3.
        after = velocity_series(a,b,load)
        np.testing.assert_array_equal(before["world_energy"][:3], after["world_energy"][:3])
        self.assertGreater(after["world_energy"][3], 0.)

    def test_current_load_threshold_and_missing_modes(self):
        a,b,load = self.packet()
        load[1:] = [[0,0],[5,0],[0,5],[5,5],[4.999,5]]
        result = velocity_series(a,b,load)
        np.testing.assert_array_equal(result["mode"], [0,1,2,3,2])
        summary = summarize(result, 1)
        self.assertIsNone(summary["load_modes"][1]["world_body_velocity_rmse_m_s"])

    def test_invalid_or_excess_support_is_rejected(self):
        a,b,load = self.packet()
        for steps in (0, 6, 1.5, True):
            with self.assertRaises(ValueError): summarize(velocity_series(a,b,load), steps)
        with self.assertRaises(ValueError): velocity_series(a,b,load,dt=0.)
        a[0,0,0] = 1.
        with self.assertRaises(ValueError): velocity_series(a,b,load)
        a[0,0,0] = np.nan
        with self.assertRaises(ValueError): velocity_series(a,b,load)

    def test_terminal_transition_is_not_matched_support(self):
        rows = [dict(arm=arm, attempted_steps=3, failure_frame=13 if i == 0 else None,
            valid_prefix_steps=2 if i == 0 else 3, anchor=10, clip="known", split="train")
            for i,arm in enumerate(("student_f0", "privileged_teacher_h10"))]
        arrays = [dict(actual=np.zeros((4,79)),reference=np.zeros((4,79)),canonical_action=np.zeros((3,29)),
            body_actual=np.zeros((4,27,3)),body_reference=np.zeros((4,27,3)),load=np.zeros((4,2))) for _ in rows]
        self.assertEqual(matched_support(rows, arrays, 2), 2)
        with self.assertRaises(ValueError): matched_support(rows, arrays, 3)
        rows[0]["valid_prefix_steps"] = 3
        with self.assertRaises(ValueError): matched_support(rows, arrays, 3)


if __name__ == "__main__":
    unittest.main()
