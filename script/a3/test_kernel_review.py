"""Synthetic standard-library tests; no tensor, model, or simulation runtime."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from script.a3.kernel_review import baseline_digest, check_kernel, digest, memory_cgroup


class KernelReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.own = '/kubepods/burstable/pod-own/container-own'
        self.other = '/kubepods/burstable/pod-other/container-other'
        self.cgroup = '3:memory:' + self.own + '\n'
        self.baseline = dict(boot_id='boot', kernel_baseline=['[1.0] ordinary'])
        self.fatal = '[2.002] Memory cgroup out of memory: Killed process 123 (python)'
        self.origin = ('[2.001] oom-kill:constraint=CONSTRAINT_MEMCG,oom_memcg=' + self.other +
                       ',task_memcg=' + self.other + ',task=python,pid=123,uid=0')
        self.evidence = dict(boot_id='boot', own_memory_cgroup=self.own,
                             contexts=[[self.origin, self.fatal]])
        self.review = dict(schema=1, policy='EXACT_HISTORICAL_EXTERNAL_MEMCG_OOM_ONLY',
                           boot_id='boot', baseline_kernel_sha256=baseline_digest(self.baseline['kernel_baseline']),
                           approved_fatal_line_sha256=[digest(self.fatal.encode())])
        self.kernel = dict(returncode=0, stdout=self.origin + '\n' + self.fatal)

    def reference(self):
        def save(name, value):
            raw = json.dumps(value).encode()
            (self.root / name).write_bytes(raw)
            return dict(path=name, sha256=digest(raw))
        self.review['evidence'] = save('evidence.json', self.evidence)
        return save('review.json', self.review)

    def check(self, reference=None, **kwargs):
        return check_kernel(self.root, self.kernel, self.baseline, reference,
                            boot_id=kwargs.get('boot_id', 'boot'), cgroup_text=kwargs.get('cgroup', self.cgroup))

    def test_exact_external_historical_record_passes(self):
        value = self.check(self.reference())
        self.assertEqual(value['approved_exact_lines_count'], 1)
        self.assertFalse(value['future_events_exempted'])

    def test_default_and_unknown_future_oom_fail(self):
        with self.assertRaises(RuntimeError):
            self.check()
        ref = self.reference()
        self.kernel['stdout'] += '\n[3.002] Memory cgroup out of memory: Killed process 456 (python)'
        with self.assertRaises(RuntimeError):
            self.check(ref)

    def test_own_ancestor_and_descendant_fail(self):
        for scope in (self.own, '/kubepods/burstable', self.own + '/child'):
            with self.subTest(scope=scope):
                self.evidence['contexts'] = [[self.origin.replace(self.other, scope), self.fatal]]
                with self.assertRaises(ValueError):
                    self.check(self.reference())

    def test_global_oom_and_hardware_errors_fail(self):
        ref = self.reference()
        for line in ('[3.0] Out of memory: Killed process 456 (python)',
                     '[3.0] NVRM: Xid 79', '[3.0] Hardware Error', '[3.0] Kernel panic'):
            with self.subTest(line=line):
                self.kernel['stdout'] = self.origin + '\n' + self.fatal + '\n' + line
                with self.assertRaises(RuntimeError):
                    self.check(ref)

    def test_wrong_pid_time_scope_and_missing_context_fail(self):
        for context in ([self.origin.replace('pid=123', 'pid=456'), self.fatal],
                        [self.origin.replace('[2.001]', '[0.001]'), self.fatal],
                        [self.origin.replace('CONSTRAINT_MEMCG', 'CONSTRAINT_NONE'), self.fatal],
                        [self.fatal]):
            with self.subTest(context=context):
                self.evidence['contexts'] = [context]
                with self.assertRaises(ValueError):
                    self.check(self.reference())

    def test_boot_container_and_old_baseline_mismatch_fail(self):
        ref = self.reference()
        for kwargs in (dict(boot_id='new-boot'), dict(cgroup='3:memory:/another/container\n')):
            with self.assertRaises(ValueError):
                self.check(ref, **kwargs)
        self.baseline['kernel_baseline'].append('modified baseline')
        with self.assertRaises(ValueError):
            self.check(ref)

    def test_unreadable_kernel_and_tampered_files_fail(self):
        ref = self.reference()
        self.kernel['returncode'] = 1
        with self.assertRaises(RuntimeError):
            self.check(ref)
        self.kernel['returncode'] = 0
        (self.root / 'evidence.json').write_text('{}')
        with self.assertRaises(ValueError):
            self.check(ref)

    def test_cgroup_and_review_paths_are_bounded(self):
        for value in ('3:memory:/\n', '0::/container\n', '3:memory:/a/../b\n'):
            with self.assertRaises(ValueError):
                memory_cgroup(value)
        with self.assertRaises(ValueError):
            self.check(dict(path='../outside', sha256='0' * 64))

    def test_no_new_fatal_does_not_need_exemption(self):
        self.kernel['stdout'] = self.baseline['kernel_baseline'][0]
        self.assertEqual(self.check()['new_fatal_count'], 0)

    def test_review_does_not_mutate_source_records(self):
        before = copy.deepcopy((self.kernel, self.baseline, self.evidence))
        self.check(self.reference())
        self.assertEqual(before, (self.kernel, self.baseline, self.evidence))


if __name__ == '__main__':
    unittest.main()
