"""Ensure a stopped or interrupted launch can never silently restart."""

import unittest

from script.a3.reboot_trial import launch_allowed
from script.diagnostics.host_loadcheck import protection


class RebootTrialTests(unittest.TestCase):
    def test_registered_launch_requires_original_boot(self):
        self.assertTrue(launch_allowed({"status":"READY"}, {"initial_boot":"a"}, "a"))
        self.assertFalse(launch_allowed({"status":"READY"}, {"initial_boot":"a"}, "b"))

    def test_interrupted_and_completed_launches_cannot_restart(self):
        for status in ("RUNNING", "STOPPED", "TRAINING_COMPLETE"):
            self.assertFalse(launch_allowed({"status":status}, {"initial_boot":"a"}, "a"))

    def test_original_thermal_and_missing_telemetry_protection(self):
        health = dict(temperatures_c={"x86_pkg_temp":90}, gpus=[{"temperature.gpu":30}],
                      memory_bytes={"MemAvailable":8*1024**3})
        reason, count = protection(health, 0)
        self.assertIsNone(reason)
        self.assertEqual(protection(health, count)[0], "TEMPERATURE_PROTECTION")
        health["gpus"] = []
        self.assertEqual(protection(health, 0)[0], "TELEMETRY_UNREADABLE")


if __name__ == "__main__":
    unittest.main()
