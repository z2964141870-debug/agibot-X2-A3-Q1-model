"""Missing episodes, short prefixes and early exposure cannot become good results."""

import copy
import unittest

from comparison import case_index, exposure_comparison, summary
from common import TARGET_TRANSITIONS
from test_diagnostic_summary import diagnosis


def case(clip, seed=0, complete=True):
    row = dict(clip=clip, seed=seed, valid_steps=10 if complete else 2,
        attempted_steps=10 if complete else 3, reference_steps=10,
        complete=complete, failure_frame=None if complete else 3,
        evaluation_randomization=False, reference_sha256="a" * 64, trace="/server/trace.npz", interface="motion")
    if complete:
        row["full_clip_metrics"] = dict(joint_rmse_rad=.5, body_relative_rmse_m=.1)
        row["diagnostics"] = diagnosis()
    return row


def pair(left, right):
    return dict(clip=left["clip"], seed=left["seed"],
        common_steps=min(left["valid_steps"], right["valid_steps"]),
        initial_joint_rmse_rad=1., final_joint_rmse_rad=.4,
        initial_body_relative_rmse_m=.2, final_body_relative_rmse_m=.05)


class ComparisonTests(unittest.TestCase):
    def test_missing_duplicate_failed_and_interface_mix_are_rejected(self):
        valid = dict(status="PASSED", cases=[case("a")])
        self.assertEqual(len(case_index(valid, ["a"], [0], "motion")), 1)
        for edit in (lambda v: v.update(status="RUNNING"), lambda v: v["cases"].clear(),
                     lambda v: v["cases"].append(case("a")),
                     lambda v: v["cases"][0].update(interface="human"),
                     lambda v: v["cases"][0].update(attempted_steps=9)):
            value = copy.deepcopy(valid)
            edit(value)
            with self.assertRaises(RuntimeError):
                case_index(value, ["a"], [0], "motion")
        failed = case("a", complete=False)
        self.assertEqual(len(case_index(dict(status="PASSED", cases=[failed]), ["a"], [0])), 1)
        failed["full_clip_metrics"] = dict(joint_rmse_rad=0, body_relative_rmse_m=0)
        with self.assertRaises(RuntimeError):
            case_index(dict(status="PASSED", cases=[failed]), ["a"], [0])

    def test_complete_motion_and_short_common_prefix_are_separate(self):
        a, b = case("a", complete=False), case("a")
        left, right = {("a", 0): a}, {("a", 0): b}
        result = summary(left, right, [pair(a, b)], ["a"])
        self.assertEqual(result["status"], "COMPARABLE")
        self.assertEqual(result["complete_motion"]["left"]["complete_cases"], 0)
        self.assertEqual(result["complete_motion"]["left"]["metrics"], {})
        self.assertEqual(result["complete_motion"]["right"]["metrics"]["joint_rmse_rad"], .5)
        self.assertEqual(result["common_prefix_metrics"]["joint_rmse_rad"]["right"], .4)
        self.assertFalse(result["natural_gait_qualified"])

    def test_missing_prefix_never_reports_mean_over_only_survivors(self):
        a, b = case("a", complete=False), case("a")
        a["valid_steps"] = 0
        result = summary({("a", 0): a}, {("a", 0): b}, [pair(a, b)], ["a"])
        self.assertEqual(result["status"], "MISSING_COMPARABLE_PREFIX")
        self.assertEqual(result["common_prefix_metrics"], {})
        self.assertEqual(len(result["missing_common_prefix"]), 1)

    def test_missing_full_motion_diagnostics_cannot_publish_a_complete_comparison(self):
        a, b = case("a", complete=False), case("a")
        b["diagnostics"].pop("sole_p95_m")
        with self.assertRaisesRegex(RuntimeError, "sole_p95_m"):
            summary({("a", 0): a}, {("a", 0): b}, [pair(a, b)], ["a"])

    def test_reference_changes_and_wrong_prefix_are_rejected(self):
        a, b = case("a", complete=False), case("a")
        p = pair(a, b)
        for field, value in (("reference_sha256", "b" * 64), ("reference_steps", 11)):
            changed = dict(b, **{field: value})
            with self.assertRaises(RuntimeError):
                summary({("a", 0): a}, {("a", 0): changed}, [p], ["a"])
        with self.assertRaises(RuntimeError):
            summary({("a", 0): a}, {("a", 0): b}, [dict(p, common_steps=3)], ["a"])

    def test_equal_clip_weight_and_no_exposure_fabrication(self):
        left, right, pairs = {}, {}, []
        for clip, count, score in (("a", 3, .2), ("b", 1, .8)):
            for seed in range(count):
                a, b = case(clip, seed), case(clip, seed)
                left[clip, seed], right[clip, seed] = a, b
                p = pair(a, b)
                p["final_joint_rmse_rad"] = score
                pairs.append(p)
        self.assertAlmostEqual(summary(left, right, pairs, ["a", "b"])["common_prefix_metrics"]["joint_rmse_rad"]["right"], .5)
        exposure = exposure_comparison([dict(global_env_steps=102563840)])
        self.assertEqual(exposure["status"], "UNAVAILABLE_EARLY_STOP_BEFORE_TARGET")
        self.assertFalse(exposure["additional_training_authorized"])
        self.assertEqual(exposure_comparison([dict(global_env_steps=TARGET_TRANSITIONS)])["status"], "MATCHED_AUDITED_CHECKPOINT")
        self.assertEqual(exposure_comparison([dict(global_env_steps=2 * TARGET_TRANSITIONS)])["status"], "UNAVAILABLE_NO_CHECKPOINT_WITHIN_ONE_PERCENT")


if __name__ == "__main__":
    unittest.main()
