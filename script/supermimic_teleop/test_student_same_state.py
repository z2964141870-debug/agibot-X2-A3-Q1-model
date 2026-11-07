"""Reject padded traces, changed actions and missing same-state references."""

import unittest
import numpy as np
from student_same_state_math import CLIPS, VARIANTS, cleanup_confirmed, compare, parity, prefix


def case(steps=106):
    return dict(clip=CLIPS[1], event="nominal", complete=False,
                valid_steps=steps-1, attempted_steps=steps, failure_frame=steps)


class SameStateTests(unittest.TestCase):
    def test_unacknowledged_or_unreaped_phase_keeps_reservation(self):
        clean=dict(cleanup=dict(worker_reaped=True,worker_return_code=-15))
        self.assertTrue(cleanup_confirmed([clean],1))
        self.assertFalse(cleanup_confirmed([clean],2))
        self.assertFalse(cleanup_confirmed([dict(cleanup=dict(worker_reaped=False,worker_return_code=None))],1))

    def test_failed_call_retained_and_no_held_padding(self):
        value = case()
        a,b = prefix(np.zeros((107,79)), np.zeros((106,29)), value)
        self.assertEqual((a.shape,b.shape), ((106,79),(106,29)))
        with self.assertRaises(ValueError):prefix(np.zeros((152,79)),np.zeros((151,29)),value)
        a,b=prefix(np.zeros((145,79)),np.zeros((144,29)),case(144))
        self.assertEqual(len(b),128)

    def test_parity_rejects_real_change_or_nonfinite(self):
        self.assertEqual(parity(np.zeros(29),np.zeros(29)),0.)
        for value in (np.ones(29)*.001, np.ones(29)*np.nan, np.zeros(22)):
            with self.assertRaises(ValueError):parity(value,np.zeros(29))

    def test_present_joints_only_and_failed_prefix_metrics(self):
        present=np.arange(29)<22
        actions={v:np.zeros((106,29)) for v in VARIANTS}
        actions["coverage_f0"][:,0]=2
        actions["original_f0"][:,0]=1
        result=compare(actions,present,[14,15,16,17,18,19,20,21],case())
        self.assertIsNone(result["full_clip_metrics"])
        self.assertTrue(result["failure_call_included"])
        self.assertEqual(len(result["residuals"]),12)
        rows=result["residuals"]
        self.assertGreater(rows[0]["rmse_by_group"]["present"],rows[6]["rmse_by_group"]["present"])
        actions["coverage_f0"][:,28]=1
        with self.assertRaises(ValueError):compare(actions,present,list(range(14,22)),case())

    def test_misaligned_missing_reference_is_rejected(self):
        actions={v:np.zeros((106,29)) for v in VARIANTS}
        actions["teacher_h10"]=np.zeros((105,29))
        with self.assertRaises(ValueError):compare(actions,np.arange(29)<22,list(range(14,22)),case())


if __name__ == "__main__":
    unittest.main()
