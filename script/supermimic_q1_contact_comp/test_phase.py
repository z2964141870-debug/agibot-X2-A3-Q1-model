"""Phase mapping and immutable command counterexamples."""

import unittest
from pathlib import Path
import sys

import torch
from contact_reference import phase_map
from contact_limits import physical_deadline_reached, OPTIMIZATION_DEADLINE


class PhaseTests(unittest.TestCase):
    def test_physical_deadline_preserves_nontraining_delivery(self):
        self.assertTrue(physical_deadline_reached(OPTIMIZATION_DEADLINE,None))
        self.assertFalse(physical_deadline_reached(OPTIMIZATION_DEADLINE,"delivery"))
        self.assertFalse(physical_deadline_reached(OPTIMIZATION_DEADLINE-60,None))

    def test_source_clock_and_neutral(self):
        c=torch.tensor([[1,0],[1,0],[0,1]],dtype=torch.bool)
        s,w=phase_map(c,4)
        self.assertEqual(s.tolist(),[[True,False],[True,False],[False,False],[False,False]])
        self.assertEqual(w.tolist(),[[False,True],[False,True],[False,False],[False,False]])

    def test_stationary_contact_remains_contact(self):
        s,w=phase_map(torch.ones(164,2,dtype=torch.bool),272)
        self.assertTrue(s.all())
        self.assertFalse(w.any())

    def test_last_frame_no_next_clip(self):
        c=torch.tensor([[0,0],[1,1]],dtype=torch.bool)
        s,w=phase_map(c,2)
        self.assertEqual(s[-1].tolist(),[False,False])
        self.assertFalse(bool((s&w).any()))

    def test_malformed_rejected(self):
        with self.assertRaises(ValueError):
            phase_map(torch.zeros(3,2),4)


if __name__=="__main__":
    unittest.main()

