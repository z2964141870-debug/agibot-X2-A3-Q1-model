"""Test resource accounting and refusal to reinterpret a device fault as OOM."""

import unittest

from common import capacity_failure
from formal import disk_budget


class Admission(unittest.TestCase):
    def test_full_replay_and_all_snapshots_are_budgeted(self):
        profile = dict(final=dict(checkpoint=dict(bytes=300_000_000), rank_shard=dict(bytes=400_000_000),
            initialized_replay_bytes=399_000_000, full_capacity_replay_bytes=6_500_000_000))
        result = disk_budget(profile)
        self.assertEqual(result["checkpoint_count_bound"], 55)
        self.assertGreater(result["maximum_experiment_bytes"], 55 * 300_000_000 + 2 * 6_500_000_000)
        self.assertEqual(result["minimum_free_reserve_bytes"], 100 * 1024**3)

    def test_oom_with_device_fault_does_not_authorize_fallback(self):
        self.assertFalse(capacity_failure("CUDA out of memory; CUDA error: unspecified launch failure"))


if __name__ == "__main__":
    unittest.main()
