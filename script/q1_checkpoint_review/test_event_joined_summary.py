"""Coverage and failure accounting for the independent saved-artifact union."""

import copy
import unittest

from event_joined_summary import CLIPS,CONFIGS,EVENTS,MISSING,groups,key,merge


def fixture():
    rows = [dict(event=e,method=m,horizon_frames=h,clip=c,complete=True,
        upper_rmse_rad=.1,joint_rmse_rad=.1,slip_left_m_s=.01,slip_right_m_s=.01,
        sole_left_m=.01,sole_right_m=.01,alternations=2,tilt_rms_rad=.1)
        for e in EVENTS for m,h in CONFIGS for c in CLIPS]
    added = [r for r in rows if key(r) in MISSING]
    old = [r for r in rows if key(r) not in MISSING]
    return dict(source_campaign_status="FAILED",source_final_gate_passed=False,rows=old),added


class JoinedTests(unittest.TestCase):
    def test_exact_union_preserves_every_saved_case(self):
        partial,added = fixture()
        rows = merge(partial,added)
        self.assertEqual(len(rows),84)
        self.assertEqual(len({key(r) for r in rows}),84)
        self.assertEqual(len(groups(rows)),28)

    def test_duplicate_or_relabelled_parent_is_rejected(self):
        partial,added = fixture()
        added[1] = copy.deepcopy(added[0])
        with self.assertRaises(ValueError): merge(partial,added)
        partial,added = fixture()
        partial["source_campaign_status"] = "COMPLETED"
        with self.assertRaises(ValueError): merge(partial,added)

    def test_failed_added_clip_does_not_gain_a_group_mean(self):
        partial,added = fixture()
        added[0]["complete"] = False
        added[0]["upper_rmse_rad"] = None
        rows = merge(partial,added)
        group = next(g for g in groups(rows) if g["event"]==added[0]["event"]
                     and g["method"]==added[0]["method"] and g["horizon_frames"]==added[0]["horizon_frames"])
        self.assertEqual(group["complete_cases"],2)
        self.assertIsNone(group["full_fixed_set_means"])


if __name__ == "__main__":
    unittest.main()
