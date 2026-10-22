import unittest
from script.teleop_research.source_clock import SourceClockDerivative

class Tests(unittest.TestCase):
    def model(self): return SourceClockDerivative(1, max_gap_s=0.2, max_abs_velocity=10, clock_id='recording-1')
    def run_rate(self, rate):
        m=self.model();out=[]
        for i in range(4):out.append(m.step([i/30],i/30,i/rate,'recording-1'))
        return out
    def test_publication_speed_invariance(self):
        for rate in (15,30,60):
            self.assertIsNone(self.run_rate(rate)[0]['velocity'])
            for row in self.run_rate(rate)[1:]:self.assertAlmostEqual(row['velocity'][0],1)
    def test_duplicate_and_out_of_order(self):
        m=self.model();m.step([0],0,0,'recording-1')
        self.assertIsNone(m.step([1],0,1,'recording-1')['velocity'])
        self.assertIsNone(m.step([1],-1,1,'recording-1')['velocity'])
        self.assertAlmostEqual(m.step([.1],.1,1,'recording-1')['velocity'][0],1)
    def test_gap_reprime(self):
        m=self.model();m.step([0],0,0,'recording-1')
        self.assertEqual(m.step([1],1,1,'recording-1')['status'],'GAP_REPRIME_NO_VELOCITY')
        self.assertAlmostEqual(m.step([1.1],1.1,1.1,'recording-1')['velocity'][0],1)
    def test_spike_not_clipped(self):
        m=self.model();m.step([0],0,0,'recording-1')
        out=m.step([9],.01,.01,'recording-1')
        self.assertIn('REJECT_IMPLAUSIBLE',out['status']);self.assertIsNone(out['velocity'])
    def test_clock_change(self):
        with self.assertRaises(ValueError):self.model().step([0],0,0,'other')
    def test_nan(self):
        with self.assertRaises(ValueError):self.model().step([float('nan')],0,0,'recording-1')
    def test_arrival_reversal(self):
        m=self.model();m.step([0],0,1,'recording-1')
        self.assertEqual(m.step([.1],.1,.5,'recording-1')['status'],'REJECT_ARRIVAL_CLOCK_REVERSAL')
    def test_reset(self):
        m=self.model();m.step([0],0,0,'recording-1');m.reset()
        self.assertEqual(m.step([5],1,1,'recording-1')['status'],'FIRST_FRAME_NO_VELOCITY')

if __name__=='__main__':unittest.main()
