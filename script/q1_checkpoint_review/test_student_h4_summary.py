"""Saved comparisons cannot invent complete errors for failed H4 clips."""

import copy
import unittest

from student_h4_summary import STAGE, completed_gate, join_comparisons, review_cases
from student_summary import KEYS


def fixture():
    controls = [dict(event=e,clip=c,variant=v,horizon_frames=4,complete=True,completion_fraction=1.,
        derived_reference_sha256=e+":"+c,**{k:1. for k in KEYS}) for e,c in review_cases() for v in ("teacher","student")]
    controls += [dict(r,horizon_frames=10) for r in controls[:6:2]]
    rows = [dict(event=e,clip=c,variant="h4_specialist",horizon_frames=4,complete=True,completion_fraction=1.,
        derived_reference_sha256=e+":"+c,**{k:.9 for k in KEYS}) for e,c in review_cases()]
    return rows,controls


class H4SummaryTests(unittest.TestCase):
    def test_requires_all_thirteen_continuation_guards_and_settled_cost(self):
        cases = review_cases()
        auth = dict(stage=STAGE,physical_cases=7,new_physical_cases=6,reused_physical_cases=1,
            reused_baseline_cases=17,new_optimizer_updates=0,human_real_future_frames=4,
            selected_update=1500,cases=cases)
        result = dict(auth,status="COMPLETED",source_integrity="PASSED",total_comparison_cases=24,
            cost_settlement=dict(state="SETTLED"),controller=dict(pid=999999999),
            cases=[dict(path=str(i)) for i in range(7)],
            phases=[dict(name=f"{e}_student_h4_{c}_{p}") for i,(e,c) in enumerate(cases)
                for p in ("replay","media") if not (i == 0 and p == "replay")])
        completed_gate(result,auth)
        for bad in (dict(result,phases=result["phases"][:-1]),
                dict(result,cost_settlement=dict(state="RESERVED")),dict(result,status="FAILED")):
            with self.assertRaises(RuntimeError):
                completed_gate(bad,auth)

    def test_preserve_all_seventeen_controls_without_modification(self):
        rows,controls = fixture()
        previous = copy.deepcopy(controls)
        joined = join_comparisons(rows,controls)
        self.assertEqual(len(joined),24)
        self.assertEqual(controls,previous)
        self.assertEqual({r["variant"] for r in joined},{"teacher_h10","teacher_h4","r1_h4","h4_specialist"})

    def test_incomplete_case_does_not_receive_full_error_improvement(self):
        rows,controls = fixture()
        rows[0].update(complete=False,completion_fraction=.2)
        join_comparisons(rows,controls)
        self.assertIsNone(rows[0]["change_from_r1_h4"])
        self.assertIsNone(rows[0]["change_from_teacher_h10"])
        self.assertAlmostEqual(rows[0]["completion_fraction_change_from_r1_h4"],-.8)

    def test_changed_targets_and_missing_or_duplicated_cases_are_rejected(self):
        rows,controls = fixture()
        for candidates in (rows[:-1],rows+[rows[0]]):
            with self.assertRaises(RuntimeError):
                join_comparisons(candidates,controls)
        rows[-1]["derived_reference_sha256"] = "different physical target"
        with self.assertRaises(RuntimeError):
            join_comparisons(rows,controls)


if __name__ == "__main__":
    unittest.main()
