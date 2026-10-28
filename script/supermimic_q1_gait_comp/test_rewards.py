"""Synthetic physical counterexamples for the four gait reward groups."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from rewards import FootReference


class Rewards(unittest.TestCase):
    def setUp(self):
        self.state = torch.zeros(2,57)
        self.state[:,3] = 1.
        self.state[:,2] = .5
        self.reference = torch.tensor([[[0.,0.,.05],[0.,0.,.10]]]*2)
        self.measured = self.reference.clone()
        self.foot = FootReference.__new__(FootReference)
        self.foot.motion = SimpleNamespace(root_p=torch.tensor([[0.,0.,.5]]*2),
                                           root_R=torch.eye(3).repeat(2,1,1))
        self.stance = torch.tensor([[True,False]]*2)
        self.swing = ~self.stance
        self.still = torch.tensor([True,False])
        self.foot.measured = lambda state: (self.measured,torch.zeros(2,2,3))
        self.foot.targets = lambda frames: (self.reference,torch.zeros(2,2,3),self.reference[...,2],
                                           self.stance,self.swing,self.still)
        self.force = torch.full((2,2),10.)
        self.slip = torch.zeros(2,2)
        self.target = torch.zeros(2,22)

    def measure(self,target=None):
        return self.foot.reward(torch.zeros(2),self.state,torch.arange(2),
                                 self.target if target is None else target,
                                 self.target,torch.zeros(2,14),self.force,self.slip)

    def test_loaded_stance_slip_is_penalized(self):
        baseline,parts = self.measure()
        self.slip[:,0] = .3
        candidate,changed = self.measure()
        self.assertTrue(bool((candidate < baseline).all()))
        self.assertTrue(bool((changed["stance_bonus"] < parts["stance_bonus"]).all()))

    def test_unloaded_stance_cannot_get_no_slip_bonus(self):
        _,parts = self.measure()
        self.force.zero_()
        _,candidate = self.measure()
        self.assertTrue(bool((parts["stance_bonus"] > candidate["stance_bonus"]).all()))
        self.assertEqual(float(candidate["stance_bonus"].sum()),0.)

    def test_swing_height_error_is_penalized(self):
        _,baseline = self.measure()
        self.measured[:,1,2] -= .06
        _,candidate = self.measure()
        self.assertTrue(bool((candidate["swing_bonus"] < baseline["swing_bonus"]).all()))

    def test_command_jump_cost_and_stationary_phase_mask(self):
        _,baseline = self.measure()
        _,candidate = self.measure(torch.full((2,22),.2))
        self.assertTrue(bool((candidate["smooth_cost"] > baseline["smooth_cost"]).all()))
        self.assertGreater(float(candidate["stationary_bonus"][0]),0.)
        self.assertEqual(float(candidate["stationary_bonus"][1]),0.)


if __name__ == "__main__":
    runtime = Path("/opt/data/private/ZHY/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027")
    sys.path[:0] = [str(runtime / "src"),str(runtime)]
    unittest.main()
