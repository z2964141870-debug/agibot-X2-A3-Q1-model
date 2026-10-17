"""Standard-library controls for new-host telemetry; no model or GPU work."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from script.a3.humanplus_health import (
    GIB, TelemetryError, cgroup_location, journal_evidence,
    memory_changes, memory_evidence, reasons)


BOOT = 'afb6796b-3883-4df4-99ea-bc381c52ea13'


def journal(messages):
    return '\n'.join(json.dumps({'__CURSOR': str(i), '_BOOT_ID': BOOT.replace('-', ''),
        '__MONOTONIC_TIMESTAMP': str(i), 'MESSAGE': message})
        for i, message in enumerate(messages))


class HealthTests(unittest.TestCase):
    def memory_fixture(self, root):
        proc = root / 'proc'
        (proc / 'self').mkdir(parents=True)
        group = root / 'cgroup'
        (group / 'session').mkdir(parents=True)
        (proc / 'meminfo').write_text('MemTotal: 33554432 kB\nMemAvailable: 16777216 kB\n')
        (proc / 'self/cgroup').write_text('0::/session\n')
        (proc / 'self/mountinfo').write_text(f'1 0 0:1 / {group} rw - cgroup2 cgroup rw\n')
        for folder in (group, group / 'session'):
            (folder / 'memory.current').write_text(str(2 * GIB))
            for name in ('memory.max', 'memory.high'):
                (folder / name).write_text('max')
            for name in ('memory.events', 'memory.events.local'):
                (folder / name).write_text('low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n')
        return proc, group

    def test_unlimited_uses_physical_available(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, _ = self.memory_fixture(Path(directory))
            self.assertEqual(memory_evidence(proc)['available_bytes'], 16 * GIB)

    def test_tight_ancestor_and_high_are_both_accounted(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, group = self.memory_fixture(Path(directory))
            (group / 'memory.max').write_text(str(10 * GIB))
            (group / 'session/memory.high').write_text(str(7 * GIB))
            self.assertEqual(memory_evidence(proc)['available_bytes'], 5 * GIB)

    def test_over_limit_has_zero_available(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, group = self.memory_fixture(Path(directory))
            (group / 'memory.max').write_text(str(GIB))
            self.assertEqual(memory_evidence(proc)['available_bytes'], 0)

    def test_real_root_without_controller_files_keeps_child_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, group = self.memory_fixture(Path(directory))
            for filename in ('memory.current', 'memory.max', 'memory.high',
                             'memory.events', 'memory.events.local'):
                (group / filename).unlink()
            (group / 'session/memory.max').write_text(str(9 * GIB))
            value = memory_evidence(proc)
            self.assertEqual(value['available_bytes'], 7 * GIB)
            self.assertEqual(value['ancestors'][-1]['memory.max'], 'root_unconstrained')

    def test_hidden_ancestors_and_traversal_rejected(self):
        with self.assertRaises(TelemetryError):
            cgroup_location('0::/session', '1 0 0:1 /hidden /cgroup rw - cgroup2 cgroup rw')
        with self.assertRaises(TelemetryError):
            cgroup_location('0::/../session', '1 0 0:1 / /cgroup rw - cgroup2 cgroup rw')

    def test_missing_counter_file_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, group = self.memory_fixture(Path(directory))
            (group / 'memory.events.local').unlink()
            with self.assertRaises(OSError):
                memory_evidence(proc)

    def test_ancestor_oom_detected_without_resetting_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, _ = self.memory_fixture(Path(directory))
            before = memory_evidence(proc)
            after = copy.deepcopy(before)
            after['ancestors'][-1]['memory.events']['oom_kill'] += 1
            events = memory_changes(before, after)
            self.assertEqual(events[0]['event'], 'oom_kill')
            self.assertEqual(before['ancestors'][-1]['memory.events']['oom_kill'], 0)

    def test_counter_decrease_and_limit_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            proc, _ = self.memory_fixture(Path(directory))
            before = memory_evidence(proc)
            after = copy.deepcopy(before)
            after['ancestors'][0]['memory.events']['high'] = -1
            with self.assertRaises(TelemetryError):
                memory_changes(before, after)
            after = copy.deepcopy(before)
            after['ancestors'][0]['memory.high'] = 2 * GIB
            with self.assertRaises(TelemetryError):
                memory_changes(before, after)

    def test_journal_append_and_quiet_interval(self):
        before = journal_evidence(journal(['boot', 'ready']), BOOT)
        self.assertEqual(journal_evidence(journal(['boot', 'ready']), BOOT, before), before)
        after = journal_evidence(journal(['boot', 'ready', 'normal']), BOOT, before)
        self.assertEqual(after['records'], 3)

    def test_lost_or_changed_prefix_rejected(self):
        before = journal_evidence(journal(['boot', 'ready']), BOOT)
        for text in (journal(['ready']), journal(['changed', 'ready'])):
            with self.assertRaises(TelemetryError):
                journal_evidence(text, BOOT, before)

    def test_boot_and_duplicate_cursor_rejected(self):
        with self.assertRaises(TelemetryError):
            journal_evidence(journal(['boot']), '00000000-0000-0000-0000-000000000000')
        row = journal(['boot'])
        with self.assertRaises(TelemetryError):
            journal_evidence(row + '\n' + row, BOOT)

    def test_empty_or_malformed_journal_rejected(self):
        with self.assertRaises(TelemetryError):
            journal_evidence('', BOOT)
        with self.assertRaises((ValueError, KeyError, TelemetryError)):
            journal_evidence('{"MESSAGE":"missing evidence"}', BOOT)

    def test_external_oom_and_xid_not_exempted(self):
        messages = ['boot', 'other invoked oom-killer: external', 'NVRM: Xid 79']
        self.assertEqual(len(journal_evidence(journal(messages), BOOT)['fatal']), 2)

    def test_original_memory_thresholds_and_kernel_gate(self):
        evidence = {'journal': {'fatal': []}, 'new_oom_events': [],
                    'memory': {'available_bytes': 8 * GIB}}
        self.assertEqual(reasons(evidence, startup=True), [])
        evidence['memory']['available_bytes'] -= 1
        self.assertEqual(reasons(evidence, startup=True), ['STARTUP_MEMORY_BELOW_8GIB'])
        self.assertEqual(reasons(evidence, startup=False), [])
        evidence['memory']['available_bytes'] = 4 * GIB - 1
        evidence['new_oom_events'] = [{'event': 'oom'}]
        evidence['journal']['fatal'] = [{'message': 'fatal'}]
        self.assertEqual(reasons(evidence, startup=False),
            ['CRITICAL_KERNEL_EVENT', 'CGROUP_OOM_EVENT', 'MEMORY_RESERVE_BELOW_4GIB'])


if __name__ == '__main__':
    unittest.main()
