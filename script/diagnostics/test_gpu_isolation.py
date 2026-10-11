"""Verify restart isolation and protective thresholds without CUDA."""
import unittest
from script.diagnostics.gpu_isolation import allowed
from script.diagnostics.host_loadcheck import protection


class GuardTests(unittest.TestCase):
    def health(self, cpu=49, gpu=30, memory=8*1024**3):
        return dict(temperatures_c=dict(x86_pkg_temp=cpu),
                    gpus=[{"temperature.gpu":gpu}], memory_bytes=dict(MemAvailable=memory))

    def test_launch_once_and_original_boot_only(self):
        self.assertTrue(allowed(dict(status="READY"), dict(boot_id="a"), "a"))
        self.assertFalse(allowed(dict(status="READY"), dict(boot_id="a"), "b"))
        for status in ("RUNNING", "STOPPED", "STOPPED_REBOOT", "COMPLETED"):
            self.assertFalse(allowed(dict(status=status), dict(boot_id="a"), "a"))

    def test_consecutive_hot_samples_and_reset(self):
        for cpu, gpu in ((90,30),(49,85)):
            reason, count = protection(self.health(cpu,gpu),0)
            self.assertIsNone(reason)
            self.assertEqual(count,1)
            self.assertEqual(protection(self.health(cpu,gpu),count)[0], "TEMPERATURE_PROTECTION")
            self.assertEqual(protection(self.health(),count), (None,0))

    def test_missing_temperature_and_memory(self):
        self.assertEqual(protection(self.health(cpu=None),0)[0], "TELEMETRY_UNREADABLE")
        self.assertEqual(protection(self.health(gpu=None),0)[0], "TELEMETRY_UNREADABLE")
        self.assertEqual(protection(self.health(memory=4*1024**3-1),0)[0], "MEMORY_RESERVE_PROTECTION")


if __name__ == "__main__":
    unittest.main()
