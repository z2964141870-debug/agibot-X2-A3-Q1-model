"""Predecessor and diagnostic configuration contracts, without GPU imports."""
import unittest
from script.diagnostics.gpu_operator_compare import CASES, allowed, predecessor_allowed


class CompareTests(unittest.TestCase):
    def test_failure_or_changed_boot_prevents_escalation(self):
        p=dict(state=dict(status="COMPLETED",boot_id="a"),trace=dict(errors=[],incomplete=[]))
        self.assertTrue(predecessor_allowed(p,"a"))
        self.assertFalse(predecessor_allowed(p,"b"))
        for status in ("RUNNING","READY","STOPPED","STOPPED_REBOOT"):
            self.assertFalse(predecessor_allowed(dict(state=dict(status=status,boot_id="a")),"a"))
        for evidence in (dict(errors=["fault"]),dict(incomplete=["call"])):
            self.assertFalse(predecessor_allowed(dict(state=p["state"],trace=evidence),"a"))

    def test_fixed_changes_and_no_relaunch(self):
        self.assertEqual([CASES[c]["duration_seconds"] for c in CASES],[0,30,30])
        self.assertEqual([CASES[c]["extra_syncs"] for c in CASES],[True,True,False])
        self.assertTrue(all(CASES[c]["cuda_launch_blocking"]=="0" for c in CASES))
        self.assertFalse(allowed(dict(status="OBSERVING"),dict(boot_id="a"),"a"))


if __name__ == "__main__":
    unittest.main()
