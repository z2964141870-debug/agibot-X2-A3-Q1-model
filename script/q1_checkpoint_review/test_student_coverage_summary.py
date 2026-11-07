"""Keep original failed f0 controls and reject partial coverage campaigns."""

import copy
import unittest

from student_coverage_summary import STAGE,completed_gate,join_comparisons,review_cases
from deliver_student_coverage_review import saved_gate
from student_summary import full_group


def rows(variant,horizon):
    return [dict(event=e,clip=c,variant=variant,horizon_frames=horizon,human_real_future_frames=0,
        complete=False,completion_fraction=.2,derived_reference_sha256=e+c) for e,c in review_cases()]


class CoverageSummaryTests(unittest.TestCase):
    def test_old_failures_are_retained_and_cannot_gain_full_motion_differences(self):
        controls=[r for v in ("teacher","student") for r in rows(v,4)]
        old,new=rows("f0_student",0),rows("f0_student",0)
        joined=join_comparisons(new,controls,old)
        self.assertEqual(len(joined),28)
        self.assertEqual(len([r for r in joined if r["variant"] == "original_f0"]),7)
        self.assertTrue(all(r["change_from_original_f0"] is None for r in joined[21:]))
        self.assertTrue(all(r["variant"] == "f0_student" for r in old))

    def test_different_or_duplicated_old_targets_cannot_form_a_comparison(self):
        controls=[r for v in ("teacher","student") for r in rows(v,4)]
        old=rows("f0_student",0)
        for bad in (old[:-1],old+[old[0]], [dict(old[0],derived_reference_sha256="wrong")]+old[1:]):
            with self.assertRaises(RuntimeError):join_comparisons(rows("f0_student",0),controls,bad)

    def test_pending_missing_media_or_changed_stage_cannot_be_analyzed(self):
        auth=dict(stage=STAGE,new_physical_cases=7,reused_baseline_cases=14,new_optimizer_updates=0,
            human_real_future_frames=0,cases=review_cases())
        phases=[dict(name=f"{e}_f0_student_h0_{c}_{p}") for e,c in review_cases() for p in ("replay","media")]
        result=dict(auth,status="COMPLETED",source_integrity="PASSED",total_comparison_cases=21,
            cases=[dict(path=str(i)) for i in range(7)],phases=phases,controller=dict(pid=2147483647))
        completed_gate(result,auth)
        for change in (dict(status="RUNNING"),dict(stage="R2_F0_MATCHED_CLOSED_LOOP"),dict(phases=phases[:-1]),
                dict(source_integrity="PENDING"),dict(new_optimizer_updates=1)):
            with self.assertRaises(RuntimeError):completed_gate(dict(result,**change),auth)

    def test_figures_require_real_saved28_and_keep_failures_without_means(self):
        controls=[r for v in ("teacher","student") for r in rows(v,4)]
        combined=join_comparisons(rows("f0_student",0),controls,rows("f0_student",0))
        groups=[dict(variant=v,**full_group([r for r in combined if r["event"] == "nominal"
            and r["variant"] == v])) for v in ("teacher","student","original_f0","coverage_f0")]
        value=dict(stage="COVERAGE01_MATCHED_SAVED_ANALYSIS",total_comparison_cases=28,rows=combined,
            new_saved_cases=7,reused_saved_controls=21,selected_update=500,complete_new_cases=0,
            new_inference_calls=0,new_physical_transitions=0,new_training_updates=0,
            measured_hardware_end_to_end_latency_ms=None,natural_teleoperation_qualified=False,
            nominal_groups=groups)
        audit=dict(status="PASSED",source_inputs_unchanged=True)
        saved_gate(value,audit)
        self.assertTrue(all(g["full_fixed_set_means"] is None for g in groups))
        for change in (dict(selected_update=3000),dict(complete_new_cases=7),dict(total_comparison_cases=21),
                dict(new_physical_transitions=1),dict(measured_hardware_end_to_end_latency_ms=0)):
            with self.assertRaises(RuntimeError):saved_gate(dict(value,**change),audit)
        wrong=copy.deepcopy(value)
        wrong["rows"][-1]["upper_rmse_rad"]=.01
        with self.assertRaises(RuntimeError):saved_gate(wrong,audit)


if __name__ == "__main__":unittest.main()
