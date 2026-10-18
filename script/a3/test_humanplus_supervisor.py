"""Synthetic supervision controls; no tensor, driver, or simulator imports."""

import copy
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from script.a3 import humanplus_supervisor as guard


def evidence():
    return dict(journal={'fatal': []}, new_oom_events=[],
                memory={'available_bytes': 10 * guard.health.GIB},
                gpu_uuid='test-gpu', gpu_celsius=40, cpu_celsius=60,
                gpu_utilization=0, gpu_pids=[], processes=[])


class SupervisorTests(unittest.TestCase):
    def check(self, row, **kwargs):
        return guard.issues(row, monitor_pid=100, **kwargs)

    def test_startup_blocks_gpu_utilization_without_apps(self):
        row = evidence(); row['gpu_utilization'] = 1
        self.assertIn('GPU_NOT_IDLE', self.check(row, startup=True))

    def test_other_compute_not_exempt_by_idle_gpu(self):
        row = evidence()
        row['processes'] = [dict(pid=200, ppid=1, pgid=200, name='python3', rss_bytes=1)]
        self.assertTrue(self.check(row, startup=True)[0].startswith('OTHER_COMPUTE:'))

    def test_own_group_only_and_external_gpu(self):
        row = evidence()
        row['processes'] = [dict(pid=201, ppid=200, pgid=200, name='python3', rss_bytes=1)]
        row['gpu_pids'] = [201]
        self.assertEqual(self.check(row, startup=False, worker_group=200), [])
        row['gpu_pids'].append(999)
        self.assertIn('OTHER_GPU_COMPUTE', self.check(row, startup=False, worker_group=200))

    def test_large_non_python_process_blocks(self):
        row = evidence()
        row['processes'] = [dict(pid=200, ppid=1, pgid=200, name='binary',
                                 rss_bytes=guard.health.GIB + 1)]
        self.assertTrue(self.check(row, startup=True))

    def test_two_consecutive_hot_samples_required(self):
        cold = evidence(); hot = copy.deepcopy(cold)
        hot.update(cpu_celsius=90, gpu_celsius=85)
        self.assertEqual(self.check(hot, startup=False, previous=cold), [])
        self.assertEqual(len(self.check(hot, startup=False, previous=hot)), 2)
        self.assertEqual(self.check(cold, startup=False, previous=hot), [])
        self.assertIn('STARTUP_TEMPERATURE_HIGH', self.check(hot, startup=True))

    def test_memory_thresholds_no_cache_allowance(self):
        row = evidence(); row['memory']['available_bytes'] = 7 * guard.health.GIB
        self.assertIn('STARTUP_MEMORY_BELOW_8GIB', self.check(row, startup=True))
        self.assertEqual(self.check(row, startup=False), [])
        row['memory']['available_bytes'] = 4 * guard.health.GIB - 1
        self.assertIn('MEMORY_RESERVE_BELOW_4GIB', self.check(row, startup=False))

    def test_oom_kernel_and_gpu_identity(self):
        row = evidence(); row['new_oom_events'] = [{'increase': 1}]
        row['journal']['fatal'] = [{'message': 'Xid'}]
        errors = self.check(row, startup=False, gpu_uuid='changed')
        self.assertEqual(set(errors), {'CGROUP_OOM_EVENT', 'CRITICAL_KERNEL_EVENT',
                                       'GPU_IDENTITY_CHANGED'})

    def test_stuck_telemetry_is_bounded(self):
        with patch.object(guard.subprocess, 'run', side_effect=subprocess.TimeoutExpired('sensor', 4)):
            with self.assertRaises(subprocess.TimeoutExpired):
                guard.sample(Path('baseline.json'))

    def test_stop_only_created_group(self):
        process = unittest.mock.Mock(pid=12345)
        with patch.object(guard.os, 'killpg') as kill, patch.object(guard.time, 'sleep'):
            guard.stop_group(process)
        self.assertEqual([call.args[0] for call in kill.call_args_list], [12345, 12345])
        process.wait.assert_called_once_with(timeout=5)

    def test_process_metadata_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory); (proc / '200').mkdir()
            with self.assertRaises(guard.health.TelemetryError):
                guard.process_rows(proc)

    def run_lifecycle(self, directory, *, samples, process=None, clock=None):
        with patch.object(guard, 'ROOT', Path.cwd()), \
             patch.object(guard.Path, 'home', return_value=Path.cwd().parent), \
             patch.object(guard, 'sample', side_effect=samples), \
             patch.object(guard.subprocess, 'Popen', return_value=process) as spawn, \
             patch.object(guard.time, 'sleep'), \
             patch.object(guard, 'stop_group') as stop:
            if clock is None:
                with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                    guard.run_stage(directory, ['worker'], Path('baseline'))
            else:
                with patch.object(guard.time, 'monotonic', side_effect=clock):
                    with self.assertRaisesRegex(RuntimeError, 'DEADLINE'):
                        guard.run_stage(directory, ['worker'], Path('baseline'))
        import json
        state = json.loads((directory / 'state.json').read_text())
        self.assertEqual(state['status'], 'STOPPED')
        return spawn, stop

    def test_preflight_does_not_launch_and_cannot_overwrite(self):
        row = evidence(); row['gpu_utilization'] = 1
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'stage'
            spawn, stop = self.run_lifecycle(target, samples=[row])
            spawn.assert_not_called()
            with patch.object(guard, 'ROOT', Path.cwd()), \
                 patch.object(guard.Path, 'home', return_value=Path.cwd().parent):
                with self.assertRaises(FileExistsError):
                    guard.run_stage(target, ['worker'], Path('baseline'))

    def test_runtime_sensor_failure_stops_owned_worker(self):
        process = unittest.mock.Mock(pid=12345)
        samples = [evidence() for _ in range(3)] + [subprocess.TimeoutExpired('sensor', 4)]
        with tempfile.TemporaryDirectory() as directory:
            spawn, stop = self.run_lifecycle(Path(directory) / 'stage', samples=samples, process=process)
            spawn.assert_called_once()
            stop.assert_called_once_with(process)

    def test_900_second_deadline_stops_before_extra_sampling(self):
        process = unittest.mock.Mock(pid=12345)
        # start; three startup scheduling reads; first runtime deadline; final elapsed.
        with tempfile.TemporaryDirectory() as directory:
            spawn, stop = self.run_lifecycle(Path(directory) / 'stage',
                samples=[evidence() for _ in range(3)], process=process,
                clock=[0, 0, 0, 5, 5, 10, 897, 897])
            stop.assert_called_once_with(process)


if __name__ == '__main__':
    unittest.main()
