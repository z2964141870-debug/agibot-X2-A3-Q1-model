"""HB06 small scalar controls with analytical answers, not public data results."""
import copy
import unittest

from script.headband.evaluation_core import evaluate


def sequence(times, frames, names=("pelvis", "joint1", "joint2")):
    return {"timestamps_ns": times, "joints_m": frames, "joint_names": list(names),
            "root_index": 0, "coordinate_frame": "synthetic_world_z_up",
            "clock_domain": "synthetic_capture", "position_unit": "m", "timestamp_unit": "ns"}


POSE = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]


class EvaluationControls(unittest.TestCase):
    def test_identity_and_input_immutability(self):
        gt = sequence([0, 100_000_000], [POSE, POSE])
        original = copy.deepcopy(gt)
        out = evaluate(gt, {"identity": gt})
        self.assertEqual(gt, original)
        m = out["methods"]["identity"]["all_available_metrics"]
        for key in ("world_mpjpe_m", "world_joint_mse_m2", "root_position_mse_m2",
                    "root_relative_joint_mse_m2", "root_local_cross_m2"):
            self.assertEqual(m[key], 0)
        self.assertEqual(out["common_gt_frame_coverage"], 1)

    def test_pure_root_translation_three_four_five(self):
        gt = sequence([0], [POSE])
        pred = sequence([0], [[[3, 4, 0], [4, 4, 0], [5, 4, 0]]])
        out = evaluate(gt, {"baseline": gt, "translated": pred})
        m = out["methods"]["translated"]["all_available_metrics"]
        self.assertEqual(m["world_mpjpe_m"], 5)
        self.assertEqual(m["root_position_mse_m2"], 25)
        self.assertEqual(m["root_relative_joint_mse_m2"], 0)
        self.assertEqual(m["root_local_cross_m2"], 0)
        self.assertTrue(out["pairwise_local_equivalence"][0]["root_only_attribution_supported"])

    def test_cross_term_and_gt_root_correction_is_not_an_error_bound(self):
        gt = sequence([0], [POSE])
        pred = sequence([0], [[[1, 0, 0], [1, 0, 0], [2, 0, 0]]])
        m = evaluate(gt, {"biased": pred})["methods"]["biased"]["all_available_metrics"]
        self.assertAlmostEqual(m["world_joint_mse_m2"], 1 / 3)
        self.assertAlmostEqual(m["root_position_mse_m2"], 1)
        self.assertAlmostEqual(m["root_relative_joint_mse_m2"], 2 / 3)
        self.assertAlmostEqual(m["root_local_cross_m2"], -4 / 3)
        self.assertAlmostEqual(m["decomposition_residual_m2"], 0)
        self.assertAlmostEqual(m["world_mpjpe_m"], 1 / 3)
        self.assertAlmostEqual(m["gt_root_replacement_counterfactual_mpjpe_m"], 2 / 3)
        self.assertFalse(m["gt_root_replacement_is_guaranteed_error_bound"])

    def test_irregular_time_tie_large_utc_and_missing_gap(self):
        epoch = 1_800_000_000_000_000_000
        gt = sequence([epoch + t for t in [5_000_000, 120_000_000, 200_000_000]], [POSE] * 3)
        pred = sequence([epoch + t for t in [0, 10_000_000, 200_000_000]], [POSE] * 3)
        out = evaluate(gt, {"irregular": pred})["methods"]["irregular"]
        self.assertEqual(out["matched_source_indices"], [0, 2])
        self.assertEqual(out["source_minus_gt_ns_for_matched_frames"], [-5_000_000, 0])
        self.assertEqual(out["missing_gt_indices"], [1])
        self.assertEqual(out["gt_frame_coverage"], 2 / 3)

    def test_exact_gate_boundary_and_repeated_assignment_are_visible(self):
        gt = sequence([0, 50_000_000, 50_000_001], [POSE] * 3)
        pred = sequence([0], [POSE])
        m = evaluate(gt, {"held": pred})["methods"]["held"]
        self.assertEqual(m["matched_gt_indices"], [0, 1])
        self.assertEqual(m["missing_gt_indices"], [2])
        self.assertEqual(m["reused_source_assignments"], 1)

    def test_unequal_coverage_and_common_timeline(self):
        times = [0, 100_000_000, 200_000_000, 300_000_000]
        gt = sequence(times, [[[0, 0, 0]]] * 4, names=("pelvis",))
        a = sequence(times[:3], [[[1, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]], names=("pelvis",))
        b = sequence(times[1:], [[[0, 0, 0]], [[0, 0, 0]], [[5, 0, 0]]], names=("pelvis",))
        out = evaluate(gt, {"a": a, "b": b})
        self.assertEqual(out["common_gt_indices"], [1, 2])
        self.assertEqual(out["common_gt_frame_coverage"], .5)
        for name in ("a", "b"):
            self.assertEqual(out["methods"][name]["gt_frame_coverage"], .75)
            self.assertEqual(out["methods"][name]["common_gt_metrics"]["world_mpjpe_m"], 0)
        self.assertAlmostEqual(out["methods"]["a"]["all_available_metrics"]["world_mpjpe_m"], 1 / 3)
        self.assertAlmostEqual(out["methods"]["b"]["all_available_metrics"]["world_mpjpe_m"], 5 / 3)

    def test_total_dropout_has_no_zero_error_fiction(self):
        gt = sequence([0], [POSE])
        out = evaluate(gt, {"empty": sequence([], []), "good": gt})
        self.assertEqual(out["common_gt_frame_coverage"], 0)
        self.assertIsNone(out["methods"]["empty"]["all_available_metrics"])
        self.assertIsNone(out["methods"]["good"]["common_gt_metrics"])
        self.assertEqual(out["methods"]["good"]["all_available_metrics"]["world_mpjpe_m"], 0)
        self.assertFalse(out["pairwise_local_equivalence"][0]["root_only_attribution_supported"])

    def test_local_pose_or_time_difference_blocks_root_only_attribution(self):
        gt = sequence([0], [POSE])
        changed = sequence([0], [[[0, 0, 0], [1.001, 0, 0], [2, 0, 0]]])
        shifted_time = sequence([1], [POSE])
        for alternative in (changed, shifted_time):
            out = evaluate(gt, {"original": gt, "alternative": alternative})
            self.assertFalse(out["pairwise_local_equivalence"][0]["root_only_attribution_supported"])

    def test_invalid_data_contracts_are_rejected(self):
        gt = sequence([0], [POSE])
        invalid = []
        for field, value in [("joints_m", []), ("root_index", -1), ("root_index", True),
                             ("coordinate_frame", "unknown"), ("clock_domain", "host_arrival"),
                             ("position_unit", "cm"), ("timestamp_unit", "us"),
                             ("timestamps_ns", [0.0]), ("timestamps_ns", [True]),
                             ("joint_names", ["root_guess", "joint1", "joint2"])]:
            value_case = copy.deepcopy(gt)
            value_case[field] = value
            invalid.append(value_case)
        for bad_number in (float("nan"), float("inf"), True, "1"):
            value_case = copy.deepcopy(gt)
            value_case["joints_m"][0][1][0] = bad_number
            invalid.append(value_case)
        invalid.extend([sequence([0, 0], [POSE, POSE]), sequence([1, 0], [POSE, POSE]),
                        sequence([0], [[[0, 0, 0], [1, 0, 0]]])])
        for i, bad in enumerate(invalid):
            with self.subTest(case=i), self.assertRaises(ValueError):
                evaluate(gt, {"bad": bad})
        mismatched = copy.deepcopy(gt)
        mismatched["joint_names"] = ["pelvis", "joint2", "joint1"]
        with self.assertRaises(ValueError):
            evaluate(gt, {"other_order": mismatched})
        with self.assertRaises(ValueError):
            evaluate(sequence([], []), {"good": gt})


if __name__ == "__main__":
    unittest.main()
