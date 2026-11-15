import unittest
from .recover import recover
from .mc_handoff import HandoffError
from .test_mc_handoff import FakeClock, Controller, Mc


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.c=Controller(); self.c.row.update(stage='FAULT_DAMPING',mode='hardware',run_id='test')
        self.mc=Mc(self.c);self.mc.running=False;self.mc.action='PASSIVE_DEFAULT'
        original=self.mc.set_mode
        def set_mode(mode):
            original(mode)
            if mode=='JOINT_DEFAULT':self.c.row['stage']='RELEASED'
        self.mc.set_mode=set_mode
        self.clock=FakeClock()
    def run_recovery(self):
        return recover(self.mc,self.c.status,clock=self.clock.now,sleep=self.clock.sleep)
    def test_stopped_mc_returns_after_observed_release(self):
        self.assertEqual(self.run_recovery()['status'],'MC_RECOVERY_CONFIRMED')
        self.assertEqual(self.mc.calls,['start_app','JOINT_DEFAULT','STAND_DEFAULT'])
    def test_no_bus_takeover_never_stands(self):
        self.mc.set_mode=lambda mode:self.mc.calls.append(mode) or setattr(self.mc,'action',mode)
        with self.assertRaisesRegex(HandoffError,'bus takeover'):
            self.run_recovery()
        self.assertNotIn('STAND_DEFAULT',self.mc.calls)
    def test_live_mc_is_not_restarted(self):
        self.mc.running=True
        self.run_recovery()
        self.assertNotIn('start_app',self.mc.calls)
    def test_nonfaulted_writer_cannot_enter_recovery(self):
        self.c.row['stage']='CONTROL'
        with self.assertRaises(HandoffError):self.run_recovery()
        self.assertEqual(self.mc.calls,[])
    def test_unknown_mc_runtime_cannot_restart(self):
        self.mc.runtime=lambda:dict(status='Unknown')
        with self.assertRaises(HandoffError):self.run_recovery()
        self.assertEqual(self.mc.calls,[])
    def test_unavailable_mode_service_never_changes_mode(self):
        def unavailable():raise HandoffError('service not ready')
        self.mc.mode=unavailable
        with self.assertRaisesRegex(HandoffError,'services did not return'):self.run_recovery()
        self.assertEqual(self.mc.calls,['start_app'])
    def test_identity_change_cannot_confirm_release(self):
        original=self.mc.set_mode
        def switched(mode):original(mode);self.c.row['run_id']='other'
        self.mc.set_mode=switched
        with self.assertRaisesRegex(HandoffError,'identity changed'):self.run_recovery()
        self.assertNotIn('STAND_DEFAULT',self.mc.calls)
