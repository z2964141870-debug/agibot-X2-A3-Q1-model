"""Broader source planning preserves the old split and never recollects prefixes."""

import unittest

from student_coverage_plan import build


class CoveragePlanTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(relative=f'{group}/motion_{group}_{clip}.npz', split=split,
            group_id=group, frames=500, human=dict(bytes=1, sha256='a'*64))
            for group, split, clips in (('a','train',3), ('b','train',2), ('v','validation',2), ('t','test',2))
            for clip in range(clips)]
        old = [{k:rows[0][k] for k in ('relative','split','group_id','human')}]
        return dict(clips=rows), old

    def test_cover_groups_without_recollecting_original_or_reading_test(self):
        plan, old = self.fixture()
        result = build(plan, old)
        self.assertEqual(result['coverage']['train']['planned_groups'], 2)
        self.assertEqual(result['coverage']['train']['planned_clips'], 4)
        self.assertEqual(len(result['new_collection_jobs']), 5)
        self.assertNotIn(old[0]['relative'], [r['relative'] for r in result['new_collection_jobs']])
        self.assertTrue(all(r['split'] != 'test' for r in result['selected_sources']))
        self.assertFalse(result['training_dispatched'])
        self.assertEqual(result['original_jobs'], old)
        self.assertTrue(all(j['maximum_steps']+10 <= 500 for j in result['new_collection_jobs']))

    def test_selection_is_metadata_order_invariant_and_one_clip_groups_are_explicit(self):
        plan, old = self.fixture()
        first = build(plan, old)
        plan['clips'].reverse()
        self.assertEqual(first, build(plan, old))
        plan['clips'].append(dict(relative='short/only.npz', split='train', group_id='short', frames=30, human={}))
        result = build(plan, old)
        self.assertEqual(result['excluded_groups'][0]['group_id'], 'short')

    def test_reject_split_crossing_changed_prior_and_ambiguous_names(self):
        for defect in ('split', 'prior', 'name'):
            plan, old = self.fixture()
            if defect == 'split':
                plan['clips'][1]['split'] = 'test'
            elif defect == 'prior':
                old[0]['human'] = dict(bytes=2, sha256='b'*64)
            else:
                for row in plan['clips']:
                    if row['group_id'] == 'b':
                        row['relative'] = 'b/'+old[0]['relative'].split('/')[-1]
            with self.assertRaises(ValueError):
                build(plan, old)


if __name__ == '__main__':
    unittest.main()
