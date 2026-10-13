"""AdamW proposals on isolated tensors; only an accepted finite candidate commits."""

import copy

import torch

from script.a3.finite_retention_protocol import SCALES, accept


class CandidateRejected(RuntimeError):
    pass


class FiniteAdamW(torch.optim.AdamW):
    """Keep nominal Adam moments; damp the actual rounded parameter displacement.

    The evaluator observes detached candidate tensors, never live parameter writes.
    Live hooks fire once on commit, and no post-step hook fires on rejection.
    """

    def __init__(self, params, *, evaluate=None, record=None):
        super().__init__(params)
        self.evaluate_candidate = evaluate
        self.record_candidate = record

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None or self.evaluate_candidate is None or self.record_candidate is None:
            raise ValueError('Finite Adam requires its fixed evaluator and no closure')
        if getattr(self, 'grad_scale', None) is not None or getattr(self, 'found_inf', None) is not None:
            raise ValueError('Mixed precision is not part of this transaction')
        groups, live, starts, shadows = [], [], [], []
        for group in self.param_groups:
            if any(group.get(k, False) for k in ('amsgrad', 'maximize', 'capturable', 'differentiable')):
                raise ValueError('Unsupported AdamW mode')
            if not group.get('decoupled_weight_decay', True):
                raise ValueError('AdamW decay must stay decoupled')
            clones = []
            for param in group['params']:
                if param.dtype != torch.float32 or param.grad is None or not torch.isfinite(param.grad).all():
                    raise ValueError('Missing or nonfinite FP32 full-model gradient')
                if not torch.isfinite(param).all():
                    raise ValueError('Nonfinite live parameter')
                clone = torch.nn.Parameter(param.detach().clone(), requires_grad=True)
                clone.grad = param.grad.detach().clone()
                clones.append(clone)
                live.append(param)
                starts.append(param.detach().clone())
                shadows.append(clone)
            groups.append(dict(group, params=clones))
        if len({id(p) for p in live}) != len(live):
            raise ValueError('Duplicate optimizer parameter')
        before = copy.deepcopy(self.state_dict())
        shadow = torch.optim.AdamW(groups)
        shadow.load_state_dict(copy.deepcopy(before))
        baseline = self.evaluate_candidate(live, starts)
        if baseline['teacher_rmse'] > .05 or baseline['teacher_max_abs'] > .5:
            raise ValueError('Live state already violates original retention protection')
        accept(baseline['objective'], baseline)  # Finiteness even before proposal.
        shadow.step()  # Exactly one native AdamW proposal, never a live update.
        proposal = [p.detach().clone() for p in shadows]
        if any(not torch.isfinite(p).all() for p in proposal):
            raise ValueError('Nonfinite native proposal')
        rows, accepted = [], None
        try:
            for scale in SCALES:
                values = proposal if scale == 1. else [torch.lerp(a, b, scale) for a, b in zip(starts, proposal)]
                if all(torch.equal(a, b) for a, b in zip(starts, values)):
                    rows.append(dict(scale=scale, zero_displacement=True, accepted=False))
                    continue
                metrics = self.evaluate_candidate(live, values)
                row = dict(metrics, scale=scale, accepted=accept(baseline['objective'], metrics))
                rows.append(row)
                if row['accepted']:
                    accepted = values
                    break
            if accepted is None:
                self.record_candidate(dict(status='REJECTED_NO_LIVE_UPDATE', before=baseline, candidates=rows))
                raise CandidateRejected('All fixed finite candidates rejected; stop without another batch')
            # Persist decision before any commit; disk errors leave the live state intact.
            self.record_candidate(dict(status='ACCEPTED_PENDING_COMMIT', before=baseline, candidates=rows))
            next_state = copy.deepcopy(shadow.state_dict())
            self.load_state_dict(next_state)
            for param, value in zip(live, accepted):
                param.copy_(value)
            self.record_candidate(dict(status='COMMITTED', before=baseline, candidates=rows))
        except BaseException:
            for param, value in zip(live, starts):
                param.copy_(value)
            self.load_state_dict(before)
            raise
        return None
