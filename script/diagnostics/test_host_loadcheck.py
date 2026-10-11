"""Verify safety boundaries without starting any load."""

import unittest

from script.diagnostics.host_loadcheck import protection, utilization


def healthy(cpu=40, gpu=30, available=30 * 1024**3):
    return dict(temperatures_c={"x86_pkg_temp": cpu}, gpus=[{"temperature.gpu": gpu}],
                memory_bytes={"MemAvailable": available})


class ProtectionTests(unittest.TestCase):
    def test_temperature_requires_two_consecutive_samples_and_resets(self):
        self.assertEqual(protection(healthy(cpu=90), 0), (None, 1))
        self.assertEqual(protection(healthy(cpu=90), 1), ("TEMPERATURE_PROTECTION", 2))
        self.assertEqual(protection(healthy(gpu=85), 1), ("TEMPERATURE_PROTECTION", 2))
        self.assertEqual(protection(healthy(), 1), (None, 0))

    def test_missing_sensors_fail_closed(self):
        health = healthy()
        health["gpus"] = []
        self.assertEqual(protection(health, 0)[0], "TELEMETRY_UNREADABLE")
        self.assertEqual(protection(healthy(cpu=None), 0)[0], "TELEMETRY_UNREADABLE")

    def test_memory_reserve(self):
        self.assertEqual(protection(healthy(available=3 * 1024**3), 0)[0],
                         "MEMORY_RESERVE_PROTECTION")

    def test_utilization_uses_deltas_not_boot_totals(self):
        self.assertEqual(utilization({"cpu": (1000, 900)}, {"cpu": (1200, 1000)})["cpu"], 50)


if __name__ == "__main__":
    unittest.main()
