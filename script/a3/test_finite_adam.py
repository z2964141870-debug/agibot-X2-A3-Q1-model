"""ZHY-only tiny FP32 transaction controls, before any robot model loads."""

import copy
import unittest
import torch
from script.a3.finite_adam import FiniteAdamW, CandidateRejected


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


class TransactionTests(unittest.TestCase):
    def make(self, fused, evaluate, record=None):
        p = torch.nn.Parameter(torch.tensor([.2, -.3], device='cuda' if fused else 'cpu'))
        p.grad = torch.tensor([.4, -.2], device=p.device)
        rows = []
        opt = FiniteAdamW([dict(params=[p], lr=.01, weight_decay=.01, fused=fused)],
                          evaluate=evaluate, record=record or rows.append)
        return p, opt, rows

    def test_full_step_native_and_hooks(self):
        for fused in (False, True):
            p, opt, rows = self.make(fused, lambda p, v: dict(objective=0., teacher_rmse=0., teacher_max_abs=0.))
            q = torch.nn.Parameter(p.detach().clone())
            native = torch.optim.AdamW([dict(opt.param_groups[0], params=[q])])
            hooks = []
            opt.register_step_post_hook(lambda *args: hooks.append(1))
            for i in range(2):
                p.grad = torch.full_like(p, .3 + i)
                q.grad = p.grad.clone()
                native.step()
                opt.step()
                self.assertTrue(torch.equal(p, q))
                self.assertTrue(equal(opt.state_dict(), native.state_dict()))
            self.assertEqual(hooks, [1, 1])

    def test_backtrack_moments_commit_once(self):
        count = [0]
        def evaluate(p, v):
            count[0] += 1
            return dict(objective=0., teacher_rmse=.06 if count[0] == 2 else .04, teacher_max_abs=.4)
        p, opt, rows = self.make(True, evaluate)
        start = p.detach().clone()
        q = torch.nn.Parameter(start.clone()); q.grad = p.grad.clone()
        native = torch.optim.AdamW([dict(opt.param_groups[0], params=[q])]); native.step()
        opt.step()
        self.assertTrue(torch.equal(p, torch.lerp(start, q.detach(), .5)))
        self.assertTrue(equal(opt.state_dict(), native.state_dict()))
        self.assertEqual(rows[-1]['candidates'][-1]['scale'], .5)
        self.assertEqual(int(opt.state[p]['step']), 1)

    def test_rejection_preserves_existing_moments_and_no_post_hook(self):
        p, opt, rows = self.make(True, lambda p, v: dict(objective=0., teacher_rmse=.01, teacher_max_abs=.1))
        opt.step()
        before, params = copy.deepcopy(opt.state_dict()), p.detach().clone()
        calls, hooks = [0], []
        def reject(p, v):
            calls[0] += 1
            return dict(objective=0., teacher_rmse=.01 if calls[0] == 1 else .06, teacher_max_abs=.1)
        opt.evaluate_candidate = reject
        opt.register_step_post_hook(lambda *a: hooks.append(1))
        with self.assertRaises(CandidateRejected):
            opt.step()
        self.assertTrue(equal(before, opt.state_dict()))
        self.assertTrue(torch.equal(params, p))
        self.assertEqual(calls[0], 9)
        self.assertEqual(hooks, [])

    def test_exception_does_not_leak_candidate_or_moments(self):
        p, opt, rows = self.make(True, lambda p, v: dict(objective=0., teacher_rmse=.01, teacher_max_abs=.1))
        opt.step()
        before, params = copy.deepcopy(opt.state_dict()), p.detach().clone()
        def record(row):
            if row['status'] == 'COMMITTED':
                raise OSError('Injected receipt failure')
        opt.record_candidate = record
        with self.assertRaises(OSError):
            opt.step()
        self.assertTrue(equal(before, opt.state_dict()))
        self.assertTrue(torch.equal(params, p))


if __name__ == '__main__':
    unittest.main()
