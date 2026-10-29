"""Check saved-event boundaries and interpretations on small synthetic traces."""

import csv
from pathlib import Path
import tempfile
import unittest

from analyze_q1_saved_sample_runs import analyze_case, alternating_labels, masks, read_curves, runs


def feet(left_height, right_height, left_force, right_force, reference=None):
    count = len(left_height)
    reference = [True] * count if reference is None else reference
    return {side: [dict(seconds=(i + 1) * .02, training_seconds=0., global_reference_frame=100 + i,
        sole_z_m=z[i], reference_sole_z_m=.02 if reference[i] else 0., native_force_N=force[i],
        intent_swing=reference[i]) for i in range(count)] for side, z, force in
        (("left", left_height, left_force), ("right", right_height, right_force))}


class SavedSampleTests(unittest.TestCase):
    def test_runs_keep_one_sample_gaps_and_censored_boundaries(self):
        self.assertEqual(runs([True, True, False, True, False, True]), [(0, 2), (3, 4), (5, 6)])
        self.assertEqual(runs([]), [])
        with self.assertRaises(ValueError):
            runs([1, 0])

    def test_height_is_strict_and_force_uses_the_original5N_boundary(self):
        trace = feet([.01, .011, .011, .011], [0.] * 4, [0., 5., 5.01, 0.], [10., 10., 10., 5.])
        result = masks(trace, "left", .01)
        self.assertEqual(result["sole_high"], [False, True, True, True])
        self.assertEqual(result["low_force_high_other_loaded"], [False, True, False, False])

    def test_missing_other_support_cannot_be_a_qualified_run(self):
        trace = feet([.02] * 4, [.02] * 4, [0.] * 4, [0.] * 4)
        foot, _, episodes, case = analyze_case(("base", "initial", 0, "synthetic"), trace, .01, 3)
        self.assertEqual([row["qualified_runs"] for row in foot], [0, 0])
        self.assertEqual(case["run_label_sequence"], "")
        self.assertFalse(any(row["kind"] == "low_force_high_other_loaded" for row in episodes))

    def test_three_samples_span40ms_and_two_samples_fail_minimum(self):
        trace = feet([.02, .02, .02, 0., .02, .02], [0.] * 6, [0.] * 6, [10.] * 6)
        foot, _, episodes, _ = analyze_case(("base", "initial", 0, "synthetic"), trace, .01, 3)
        selected = [row for row in episodes if row["side"] == "left" and row["kind"] == "low_force_high_other_loaded"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["samples"], 3)
        self.assertAlmostEqual(selected[0]["endpoint_span_s"], .04)
        self.assertTrue(selected[0]["left_censored"])
        self.assertFalse(selected[0]["right_censored"])
        self.assertEqual(foot[0]["qualified_samples"], 3)

    def test_reference_overlap_requires_three_consecutive_inside_samples(self):
        trace = feet([.02] * 5, [0.] * 5, [0.] * 5, [10.] * 5, [False, True, True, False, False])
        foot, windows, _, _ = analyze_case(("base", "initial", 0, "synthetic"), trace, .01, 3)
        self.assertEqual(foot[0]["reference_windows"], 0)
        self.assertIsNone(foot[0]["matched_reference_window_fraction"])
        trace["left"][3].update(intent_swing=True, reference_sole_z_m=.02)
        foot, windows, _, _ = analyze_case(("base", "initial", 0, "synthetic"), trace, .01, 3)
        self.assertEqual(foot[0]["matched_reference_windows"], 1)
        self.assertTrue(windows[0]["matched_by_consecutive_saved_samples"])
        trace["left"][3]["sole_z_m"] = 0.
        foot, windows, _, _ = analyze_case(("base", "initial", 0, "synthetic"), trace, .01, 3)
        self.assertEqual(foot[0]["qualified_runs"], 1)
        self.assertEqual(foot[0]["matched_reference_windows"], 0)
        self.assertEqual(windows[0]["longest_qualified_samples_inside"], 2)

    def test_repeated_same_side_runs_do_not_form_an_alternating_chain(self):
        result = alternating_labels([(0, 3, "left"), (4, 7, "left"), (9, 12, "right"), (13, 16, "left")])
        self.assertEqual(result["run_label_sequence"], "LLRL")
        self.assertEqual(result["longest_alternating_label_chain"], 3)
        self.assertEqual(result["same_side_adjacent_runs"], 1)

    def test_reader_rejects_missing_foot_duplicate_time_and_nonfinite_input(self):
        trace = feet([.02] * 3, [0.] * 3, [0.] * 3, [10.] * 3)
        fields = ["line", "label", "repeat", "clip", "side", *trace["left"][0]]
        rows = [dict(line="base", label="initial", repeat=0, clip="synthetic", side=side, **sample)
                for side, samples in trace.items() for sample in samples]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "curves.csv"
            def write(values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(values)
            write(rows)
            self.assertEqual(len(read_curves(path)), 1)
            for values in (rows[:3], [*rows[:1], rows[0], *rows[2:]],
                           [dict(row, sole_z_m="nan") if i == 0 else row for i, row in enumerate(rows)]):
                write(values)
                with self.assertRaises(ValueError):
                    read_curves(path)


if __name__ == "__main__":
    unittest.main()
