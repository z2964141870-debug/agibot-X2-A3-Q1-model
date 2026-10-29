import csv
from pathlib import Path
import tempfile
import unittest

from analyze_q1_lift_windows import analyze, read_series, shared_source


def fixture():
    actual = {(label, "0", "walk", side): {
        tick: {"sole_height_m": height, "native_force_N": 80.}
        for tick, height in enumerate((.001, .002, .001), 1)}
        for label in ("initial", "trained") for side in ("left", "right")}
    reference = {("walk", side): {tick: {"sole_height_m": height}
        for tick, height in enumerate((.03, .04, .001), 1)} for side in ("left", "right")}
    support = {("walk", side): {tick: {"declared_stable_contact": int(tick == 0 or tick == 3)}
        for tick in range(4)} for side in ("left", "right")}
    return actual, reference, support


class LiftWindows(unittest.TestCase):
    def test_post_step_alignment_and_loaded_low_sole(self):
        rows, events, supports, changes = analyze(*fixture(), "initial")
        row = next(r for r in rows if r["label"] == "initial" and r["side"] == "right"
                   and r["phase"] == "reference_sole_above20mm")
        self.assertEqual(row["frames"], 2)
        self.assertEqual(row["native_loaded_fraction"], 1.)
        stable = next(r for r in rows if r["label"] == "initial" and r["side"] == "right"
                      and r["phase"] == "declared_stable")
        self.assertEqual(stable["frames"], 1)
        self.assertAlmostEqual(events[0]["start_seconds"], .02)
        self.assertAlmostEqual(events[0]["duration_seconds"], .04)
        self.assertAlmostEqual(events[0]["peak_shortfall_m"], .038)
        self.assertEqual(supports[0]["final_substep_both_loaded_fraction"], 1.)
        self.assertTrue(all(r["height_rmse_m_delta"] == 0 for r in changes))

    def test_shortest_prefix_excludes_unmatched_suffix(self):
        actual, reference, support = fixture()
        for side in ("left", "right"):
            del actual["trained", "0", "walk", side][3]
            actual["initial", "0", "walk", side][3]["sole_height_m"] = 100.
        rows, _, _, _ = analyze(actual, reference, support, "initial")
        self.assertTrue(all(r["common_prefix_frames"] == 2 for r in rows))
        self.assertTrue(all(r["actual_height_p95_m"] < .01 for r in rows if r["frames"]))
        stable = [r for r in rows if r["phase"] == "declared_stable"]
        self.assertTrue(all(r["frames"] == 0 and r["native_loaded_fraction"] is None for r in stable))

    def test_missing_baseline_or_foot_rejected(self):
        actual, reference, support = fixture()
        del actual["initial", "0", "walk", "right"]
        with self.assertRaisesRegex(ValueError, "both feet"):
            analyze(actual, reference, support, "initial")
        with self.assertRaisesRegex(ValueError, "baseline"):
            analyze(*fixture(), "different")

    def test_missing_entire_scenario_rejected(self):
        actual, reference, support = fixture()
        for side in ("left", "right"):
            actual["initial", "1", "walk", side] = actual["initial", "0", "walk", side]
        with self.assertRaisesRegex(ValueError, "same baseline"):
            analyze(actual, reference, support, "initial")

    def test_reference_phase_offset_rejected(self):
        actual, reference, support = fixture()
        for series in support.values():
            del series[0]
        with self.assertRaisesRegex(ValueError, "alignment"):
            analyze(actual, reference, support, "initial")

    def test_empty_lift_window_is_not_zero_error(self):
        actual, reference, support = fixture()
        for series in reference.values():
            for row in series.values():
                row["sole_height_m"] = .001
        rows, events, _, _ = analyze(actual, reference, support, "initial")
        self.assertEqual(events, [])
        self.assertTrue(all(r["height_rmse_m"] is None for r in rows
                            if r["phase"] == "reference_sole_above20mm"))

    def test_saved_csv_gap_duplicate_and_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "series.csv"
            for times, heights in (([.02, .06], [0., 0.]), ([.02, .02], [0., 0.]),
                                   ([.02], [float("nan")])):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, ("clip", "side", "seconds", "sole_height_m"))
                    writer.writeheader()
                    writer.writerows(dict(clip="walk", side="left", seconds=t, sole_height_m=h)
                                     for t, h in zip(times, heights))
                with self.assertRaises(ValueError):
                    read_series(path, ("clip", "side"), ("sole_height_m",), first_tick=1)

    def test_scene_source_must_match(self):
        source = dict(path="/server/scene.xml", bytes=20, sha256="a")
        self.assertEqual(shared_source([dict(sources=[source])]*3, "scene.xml"),
                         dict(bytes=20, sha256="a", recorded_paths=[source["path"]]))
        with self.assertRaisesRegex(ValueError, "differs"):
            shared_source([dict(sources=[source]),
                           dict(sources=[dict(source, sha256="b")])], "scene.xml")

    def test_same_content_at_different_paths_accepted(self):
        sources = [dict(sources=[dict(path=path, bytes=20, sha256="a")])
                   for path in ("/trial/scene.xml", "/reference/scene.xml")]
        self.assertEqual(shared_source(sources, "scene.xml")["recorded_paths"],
                         ["/reference/scene.xml", "/trial/scene.xml"])


if __name__ == "__main__":
    unittest.main()
