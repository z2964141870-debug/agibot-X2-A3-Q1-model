"""Reject partial residual reviews and fabricated full-clip improvements."""

import copy
import unittest

from student_proprio_summary import STAGE, completed_gate, join_comparisons, review_cases
from deliver_student_proprio_review import saved_gate
from student_summary import full_group


def rows(variant, horizon):
    return [dict(event=e, clip=c, variant=variant, horizon_frames=horizon, human_real_future_frames=0,
        complete=False, completion_fraction=.2, derived_reference_sha256=e+c) for e,c in review_cases()]


class ProprioSummaryTests(unittest.TestCase):
    def test_failed_old_and_new_cases_have_no_full_motion_difference(self):
        controls = [r for v in ("teacher", "student") for r in rows(v, 4)]
        previous, current = rows("original_f0", 0), rows("f0_student", 0)
        joined = join_comparisons(current, controls, previous)
        self.assertEqual(len(joined), 28)
        self.assertEqual(len([r for r in joined if r["variant"] == "proprio_f0"]), 7)
        self.assertTrue(all(r["change_from_original_f0"] is None for r in joined[21:]))
        self.assertTrue(all(r["variant"] == "original_f0" for r in previous))

    def test_old_targets_cannot_be_substituted_or_duplicated(self):
        controls = [r for v in ("teacher", "student") for r in rows(v, 4)]
        old = rows("original_f0", 0)
        for wrong in (old[:-1], old+[old[0]], [dict(old[0], derived_reference_sha256="wrong")]+old[1:]):
            with self.assertRaises(RuntimeError):
                join_comparisons(rows("f0_student", 0), controls, wrong)

    def test_completion_requires_all_guards_settled_cost_and_unchanged_scope(self):
        auth = dict(stage=STAGE, new_physical_cases=7, reused_baseline_cases=21, new_optimizer_updates=0,
            human_real_future_frames=0, cases=review_cases(), selected_update=3000,
            cpu_conversion_candidate_used=False, heldout_events_opened=False)
        phases = [dict(name=f"{e}_f0_student_h0_{c}_{p}") for e,c in review_cases() for p in ("replay", "media")]
        result = dict(auth, status="COMPLETED", source_integrity="PASSED", total_comparison_cases=28,
            cases=[dict(path=str(i)) for i in range(7)], phases=phases, controller=dict(pid=2147483647),
            cost_settlement=dict(state="SETTLED"))
        completed_gate(result, auth)
        for change in (dict(status="RUNNING"), dict(source_integrity="PENDING"), dict(phases=phases[:-1]),
                dict(cost_settlement=dict(state="RESERVED")), dict(new_optimizer_updates=1)):
            with self.assertRaises(RuntimeError):
                completed_gate(dict(result, **change), auth)
        for change in (dict(heldout_events_opened=True), dict(selected_update=0),
                dict(cpu_conversion_candidate_used=True)):
            with self.assertRaises(RuntimeError):
                completed_gate(result, dict(auth, **change))

    def test_saved_figures_preserve_failures_and_no_hardware_latency_claim(self):
        controls = [r for v in ("teacher", "student") for r in rows(v, 4)]
        combined = join_comparisons(rows("f0_student", 0), controls, rows("original_f0", 0))
        groups = [dict(variant=v, **full_group([r for r in combined if r["event"] == "nominal" and r["variant"] == v]))
            for v in ("teacher", "student", "original_f0", "proprio_f0")]
        value = dict(stage="F0_PROPRIO_MATCHED_SAVED_ANALYSIS", total_comparison_cases=28, rows=combined,
            new_saved_cases=7, reused_saved_controls=21, selected_update=3000, complete_new_cases=0,
            new_inference_calls=0, new_physical_transitions=0, new_training_updates=0,
            measured_hardware_end_to_end_latency_ms=None, natural_teleoperation_qualified=False,
            nominal_groups=groups)
        audit = dict(status="PASSED", source_inputs_unchanged=True)
        saved_gate(value, audit)
        self.assertTrue(all(g["full_fixed_set_means"] is None for g in groups))
        for change in (dict(selected_update=500), dict(complete_new_cases=7), dict(total_comparison_cases=21),
                dict(new_physical_transitions=1), dict(measured_hardware_end_to_end_latency_ms=0)):
            with self.assertRaises(RuntimeError):
                saved_gate(dict(value, **change), audit)
        wrong = copy.deepcopy(value)
        wrong["rows"][-1]["upper_rmse_rad"] = .01
        with self.assertRaises(RuntimeError):
            saved_gate(wrong, audit)


if __name__ == "__main__":
    unittest.main()
