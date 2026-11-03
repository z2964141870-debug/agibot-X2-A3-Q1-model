"""Failed clips, absent load and repeated seeds cannot improve diagnostic means."""

import copy
import unittest

from diagnostic_summary import summarize_complete


def diagnosis():
    return dict(torso_tilt_rms_rad=.1, tilt_error_rms_rad=.05, upper_joint_rmse_rad=.2,
        sole_p95_m=[.001, -.001], reference_sole_p95_m=[.02, .025],
        loaded_samples=[10, 10], loaded_slip_mean_m_s=[.1, .2],
        lift_events_above_10mm=0, alternating_lift_transitions=0,
        upper_joint_peak_to_peak_rad=[.8, .5], upper_reference_peak_to_peak_rad=[.5, .6])


def case(clip="a", seed=0, complete=True):
    return dict(clip=clip, seed=seed, complete=complete,
        failure_frame=None if complete else 3, valid_steps=10 if complete else 2,
        attempted_steps=10 if complete else 3, reference_steps=10, diagnostics=diagnosis())


class DiagnosticSummaryTests(unittest.TestCase):
    def test_initial_failures_are_excluded_and_empty_means_are_not_zero(self):
        failed = case(complete=False)
        failed["diagnostics"] = {}
        result = summarize_complete([failed])
        self.assertEqual(result["complete_cases"], 0)
        self.assertEqual(result["excluded_failed_cases"], 1)
        self.assertEqual(result["statistics"], {})
        self.assertEqual(result["counts"], {})

    def test_negative_clearance_and_missing_load_are_preserved(self):
        row = case()
        row["diagnostics"].update(loaded_samples=[0, 10], loaded_slip_mean_m_s=[None, .2])
        result = summarize_complete([row])
        self.assertEqual(result["statistics"]["sole_p95_right_m"]["mean"], -.001)
        missing = result["statistics"]["loaded_slip_left_m_s"]
        self.assertEqual(missing["included_clips"], 0)
        self.assertEqual(missing["missing_clips"], 1)
        self.assertIsNone(missing["mean"])
        self.assertEqual(result["counts"]["clips_with_reference_both_p95_above_10mm_but_actual_both_at_most_5mm"], 1)

    def test_repeated_seeds_have_equal_clip_weights(self):
        rows = [case("a", seed) for seed in range(3)] + [case("b")]
        for row in rows:
            row["diagnostics"]["upper_joint_rmse_rad"] = .2 if row["clip"] == "a" else .8
        result = summarize_complete(rows)
        self.assertAlmostEqual(result["statistics"]["upper_joint_rmse_rad"]["mean"], .5)
        self.assertEqual(result["statistics"]["upper_joint_rmse_rad"]["included_clips"], 2)

    def test_no_load_in_one_repeat_excludes_clip_slip_and_reports_missing(self):
        rows = [case("a", 0), case("a", 1), case("b")]
        rows[1]["diagnostics"].update(loaded_samples=[0, 10], loaded_slip_mean_m_s=[None, .2])
        summary = summarize_complete(rows)["statistics"]["loaded_slip_left_m_s"]
        self.assertEqual((summary["included_clips"], summary["missing_clips"]), (1, 1))
        self.assertEqual(summary["mean"], .1)

    def test_incomplete_or_invalid_evidence_is_rejected(self):
        for edit in (lambda row: row.update(attempted_steps=9),
                     lambda row: row["diagnostics"].update(loaded_samples=[0, 10]),
                     lambda row: row["diagnostics"].update(alternating_lift_transitions=1),
                     lambda row: row["diagnostics"].update(upper_joint_rmse_rad=float("nan")),
                     lambda row: row["diagnostics"].pop("reference_sole_p95_m")):
            row = copy.deepcopy(case())
            edit(row)
            with self.assertRaises(RuntimeError):
                summarize_complete([row])
        with self.assertRaises(RuntimeError):
            summarize_complete([case(), case()])


if __name__ == "__main__":
    unittest.main()
