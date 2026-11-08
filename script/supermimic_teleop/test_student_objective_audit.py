"""Train-only visibility, exact sequence schedule and non-mutating gradients."""

import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from student_fit import module_sha
from student_objective_audit import audit, gradient_record, group_mean, load_train, train_data
from student_retained_sequence_data import sample_windows, sequence_data
from test_student_retained_sequence import fixtures


class ObjectiveAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_training_mapper_matches_original_without_decoding_validation(self):
        _,_,origins,candidate,pure,jobs,summary = fixtures()
        old,_,current,sources,windows = sequence_data(origins,pure,candidate,jobs,summary)
        a,b,actual,w = train_data([o['train'] for o in origins],pure['train'],candidate,jobs,summary)
        self.assertEqual(actual,sources)
        for key in a:
            np.testing.assert_array_equal(a[key],old['train'][key])
            np.testing.assert_array_equal(b[key],current['train'][key])
        for kind in ('old','pure'):
            np.testing.assert_array_equal(w[kind],windows[kind]['train'])
        one,two = np.random.default_rng(2026110806),np.random.default_rng(2026110806)
        for index in range(4):
            for kind,data in (('old',a),('pure',b)):
                np.testing.assert_array_equal(sample_windows(data,w[kind],sources,one,16,index*16),
                    sample_windows(data,windows[kind]['train'],sources,two,16,index*16))

    def test_reassigned_source_or_terminal_crossing_is_rejected(self):
        _,_,origins,candidate,pure,jobs,summary = fixtures()
        bad = copy.deepcopy(jobs);bad[0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'splits'):
            train_data([o['train'] for o in origins],pure['train'],candidate,bad,summary)
        summary['records'][0]['relative']=jobs[1]['relative']
        with self.assertRaisesRegex(ValueError,'source mapping'):
            train_data([o['train'] for o in origins],pure['train'],candidate,jobs,summary)
        _,_,origins,candidate,pure,jobs,summary = fixtures()
        pure['train']['terminal_after_action'][6]=True
        with self.assertRaisesRegex(ValueError,'terminal'):
            train_data([o['train'] for o in origins],pure['train'],candidate,jobs,summary)

    def test_loader_refuses_validation_and_does_not_decode_future_values(self):
        _,_,origins,_,_,_,_ = fixtures()
        with patch('student_objective_audit.np.load',side_effect=AssertionError('must reject before opening')):
            with self.assertRaisesRegex(ValueError,'training arrays'):
                load_train('validation_data.npz')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'train_data.npz'
            np.savez(path,**origins[0]['train'],next_state=np.asarray([object()],dtype=object))
            values=load_train(path)
            self.assertNotIn('next_state',values)
            self.assertNotIn('actual_state',values)

    def test_gram_sign_zero_norm_and_first_order_dot_have_known_answers(self):
        x=torch.tensor([1.,2.],requires_grad=True)
        losses=dict(old_point=x.sum(),pure_point=-x.sum(),retention=(x-x.detach()).square().mean(),
            increment=(2*x).sum(),absent=None)
        result=gradient_record(losses,[x],{'known':torch.tensor([3.,4.],dtype=torch.float64)},20.,.5)
        names=result['objectives']
        self.assertAlmostEqual(result['cosine'][names.index('old_point')][names.index('pure_point')],-1.)
        self.assertEqual(result['gradient_norms']['retention'],0.)
        self.assertIsNone(result['cosine'][names.index('retention')][names.index('old_point')])
        self.assertIsNone(result['gradient_norms']['absent'])
        self.assertEqual(result['dot_actual_final_displacement']['known']['old_point'],7.)
        self.assertIsNone(x.grad)

    def test_group_mean_has_equal_group_clip_origin_weighting(self):
        rows=[dict(group_id='a',source_clip_id=0,action_mse=0.),
            dict(group_id='a',source_clip_id=0,action_mse=2.),
            dict(group_id='a',source_clip_id=1,action_mse=3.),
            dict(group_id='b',source_clip_id=2,action_mse=6.)]
        self.assertEqual(group_mean(rows),4.)

    def test_actual_forward_and_autograd_do_not_update_any_policy_or_use_optimizer(self):
        native,payload,origins,candidate,pure,jobs,summary=fixtures()
        payloads={'initial':payload}
        for index,name in enumerate(('retained_point_last','retained_delta_last')):
            value=copy.deepcopy(payload)
            value['adapter']['2.bias']+=.01*(index+1)
            payloads[name]=value
        before=module_sha(native.policy)
        settings=dict(seed=2026110806,minibatches=2,windows_per_distribution=2,retention_weight=20.,
            increment_weight=.5,maximum_forward_batches=1000,maximum_forward_rows=30000,maximum_autograd_calls=84)
        with patch('torch.optim.Adam',side_effect=AssertionError('No optimizer in a read-only audit')):
            result=audit(native,payloads,[o['train'] for o in origins],pure['train'],candidate,jobs,summary,
                settings,time.monotonic_ns()+120_000_000_000)
        self.assertEqual(result['status'],'TRAIN_ONLY_OBJECTIVE_AUDIT_PASSED')
        self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual(result['validation_arrays_decoded'],0)
        self.assertEqual(result['training_records'],dict(old=24,pure=8))
        self.assertEqual(result['counts']['student_forward_rows'],192)
        for row in result['policies'][0]['gradient_probes']:
            self.assertEqual(row['gradient_norms']['retention'],0.)
        self.assertTrue(all(r['policy_weights_unchanged'] for r in result['policies']))
        self.assertEqual(module_sha(native.policy),before)
        settings['maximum_forward_rows']=1
        with self.assertRaisesRegex(RuntimeError,'allowance'):
            audit(native,payloads,[o['train'] for o in origins],pure['train'],candidate,jobs,summary,
                settings,time.monotonic_ns()+120_000_000_000)


if __name__=='__main__':
    unittest.main()
