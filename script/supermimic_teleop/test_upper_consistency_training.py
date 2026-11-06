"""Paired overlap and shared-history regularization contracts."""

import json
from pathlib import Path
import unittest

import numpy as np
import torch

from upper_consistency_training import collect_pairs, paired_loss, specification


class ConsistencyTests(unittest.TestCase):
    def test_preregistered_plan(self):
        path = Path(__file__).with_name("upper_consistency_training_plan_20261106.json")
        self.assertEqual(json.loads(path.read_text()), specification())

    def test_oracle_shared_targets_have_zero_revision_and_loss(self):
        signal = torch.arange(11,dtype=torch.float32).reshape(1,11,1).repeat(2,1,8)
        paired = torch.stack((signal[:,:-1],signal[:,1:]),dim=1)
        loss, tracking, revision = paired_loss(torch, paired, paired, .5)
        self.assertEqual([value.item() for value in (loss,tracking,revision)],[0.,0.,0.])

    def test_regularization_gradient_reaches_both_origins(self):
        values = torch.zeros(2,2,10,8)
        values[:,1] = 1.
        forecast = values.requires_grad_()
        target = values.detach().clone()
        loss,_,_ = paired_loss(torch,forecast,target,.5)
        loss.backward()
        self.assertTrue(torch.isfinite(forecast.grad).all())
        self.assertTrue((forecast.grad[:,0,1:]!=0.).all())
        self.assertTrue((forecast.grad[:,1,:-1]!=0.).all())
        self.assertTrue((forecast.grad[:,0,0]==0.).all())
        self.assertTrue((forecast.grad[:,1,-1]==0.).all())
        with self.assertRaises(ValueError):
            paired_loss(torch,forecast,target,.1)

    def test_pair_sampling_keeps_both_origins_within_source(self):
        signal=np.repeat(np.arange(70)[:,None],8,axis=1)
        class Corpus:
            def load(self,name,split):
                self_test.assertEqual(split,"train")
                return signal
        self_test=self
        plan=dict(history_frames=25,future_frames=10,clips=[dict(relative="a/clip.npz",split="train",frames=70)])
        histories,targets,items=collect_pairs(plan,Corpus(),10,lambda:None)
        self.assertEqual(len(items),10)
        self.assertEqual(histories.shape,(10,26,8))
        for index,(_,cutoff) in enumerate(items):
            np.testing.assert_array_equal(histories[index],signal[cutoff-24:cutoff+2])
            np.testing.assert_array_equal(targets[index],signal[cutoff+1:cutoff+12])
            self.assertLess(cutoff+11,len(signal))
        self.assertEqual(plan["future_frames"],10)


if __name__ == "__main__":
    unittest.main()
