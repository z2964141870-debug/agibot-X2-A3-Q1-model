import unittest
from .mc_handoff import Handoff, HandoffError, stopped


class FakeClock:
    def __init__(self): self.t = 1.
    def now(self): return self.t
    def sleep(self, seconds): self.t += seconds


class Controller:
    def __init__(self):
        self.row = dict(stage="STANDBY", ready_target=True, mc_capture_complete=True,
                        external_age_s=1., transport_failed=False)
        self.calls = []
    def status(self): return self.row.copy()
    def command(self, op, **fields):
        self.calls.append((op, fields))
        if op=="capture_return":
            self.row["mc_capture_frozen"]=True
            return
        self.row["stage"] = {"arm":"CONTROL", "return":"HOLD_FOR_MC"}[op]


class Mc:
    def __init__(self, controller):
        self.controller, self.action, self.running = controller, "STAND_DEFAULT", True
        self.calls = []
    def runtime(self): return dict(status="Running(12)" if self.running else "Stopped", pid=12)
    def mode(self): return self.action
    def set_mode(self, mode): self.calls.append(mode); self.action = mode
    def app(self, action):
        self.calls.append(action)
        self.running = action == "start_app"
        if self.running:
            self.action = "PASSIVE_DEFAULT"
            if self.controller.row["stage"] == "HOLD_FOR_MC":
                self.controller.row["stage"] = "RELEASED"


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.clock, self.controller = FakeClock(), Controller()
        self.mc = Mc(self.controller)
        self.h = Handoff(self.mc, self.controller, clock=self.clock.now, sleep=self.clock.sleep)
    def test_normal_transfer_and_return(self):
        self.h.acquire(); self.h.release()
        self.assertEqual(self.mc.calls, ["DAMPING_DEFAULT", "stop_app", "start_app", "JOINT_DEFAULT", "STAND_DEFAULT"])
        self.assertEqual(self.controller.row["stage"], "RELEASED")
    def test_stopped_pid_is_not_running(self):
        self.assertTrue(stopped(dict(status="Stopped",pid=12)))
        self.assertFalse(stopped(dict(status="Running(12)",pid=12)))
    def test_no_target_never_stops_mc(self):
        self.controller.row["ready_target"] = False
        with self.assertRaises(HandoffError): self.h.acquire()
        self.assertEqual(self.mc.calls, [])
    def test_missing_return_capture_never_stops_mc(self):
        self.controller.row["mc_capture_complete"] = False
        with self.assertRaises(HandoffError): self.h.acquire()
        self.assertEqual(self.mc.calls, [])
    def test_not_standing_never_stops_mc(self):
        self.mc.action = "PASSIVE_DEFAULT"
        with self.assertRaises(HandoffError): self.h.acquire()
        self.assertEqual(self.mc.calls, [])
    def test_stop_ack_without_runtime_stop_does_not_arm(self):
        self.mc.app = lambda action: self.mc.calls.append(action)
        with self.assertRaises(HandoffError): self.h.acquire()
        self.assertFalse(any(op=="arm" for op,_ in self.controller.calls))
    def test_busy_bus_does_not_arm(self):
        self.controller.row["external_age_s"] = .01
        with self.assertRaises(HandoffError): self.h.acquire()
        self.assertFalse(any(op=="arm" for op,_ in self.controller.calls))
    def test_rollback_after_unarmed_failure(self):
        self.controller.row["external_age_s"] = .01
        with self.assertRaises(HandoffError): self.h.acquire()
        self.h.rollback_unarmed()
        self.assertTrue(self.mc.running)
    def test_fault_never_auto_stands_or_clears_latch(self):
        self.h.acquire(); self.controller.row["stage"] = "FAULT_DAMPING"
        with self.assertRaises(HandoffError): self.h.release()
        self.assertNotIn("start_app", self.mc.calls)
        with self.assertRaises(HandoffError): self.h.rollback_unarmed()
    def test_service_ack_without_takeover_does_not_stand(self):
        self.h.acquire()
        def start(action): self.mc.calls.append(action); self.mc.running=True; self.mc.action="PASSIVE_DEFAULT"
        self.mc.app = start
        with self.assertRaises(HandoffError): self.h.release()
        self.assertNotEqual(self.mc.calls[-1], "STAND_DEFAULT")
    def test_passive_can_be_silent_until_joint_mode(self):
        self.h.acquire()
        def start(action): self.mc.calls.append(action); self.mc.running=True; self.mc.action="PASSIVE_DEFAULT"
        self.mc.app = start
        original_set = self.mc.set_mode
        def set_mode(mode):
            original_set(mode)
            if mode == "JOINT_DEFAULT": self.controller.row["stage"]="RELEASED"
        self.mc.set_mode = set_mode
        self.h.release()
        self.assertEqual(self.mc.action,"STAND_DEFAULT")
    def test_denied_damping_aborts_before_stop(self):
        def denied(mode): raise HandoffError("denied")
        self.mc.set_mode=denied
        with self.assertRaises(HandoffError): self.h.acquire()
        self.assertEqual(self.mc.calls, [])


if __name__ == "__main__": unittest.main()
