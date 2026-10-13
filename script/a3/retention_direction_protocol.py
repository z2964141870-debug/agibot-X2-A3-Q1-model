"""Fixed limits and scalar interpretation for the E46 offline direction probe."""

import math


class Budget:
    LIMITS = {'policy_states': 2, 'actor_forwards': 24, 'autograd_grad_calls': 12}

    def __init__(self):
        self.counts = {name: 0 for name in self.LIMITS}

    def take(self, name):
        if self.counts[name] >= self.LIMITS[name]:
            raise RuntimeError('Registered diagnostic budget exhausted: ' + name)
        self.counts[name] += 1


def direction(dot, left_norm, right_norm):
    if any(not math.isfinite(v) for v in (dot, left_norm, right_norm)):
        raise ValueError('Nonfinite gradient geometry')
    if min(left_norm, right_norm) < 0:
        raise ValueError('Negative gradient norm')
    scale = left_norm * right_norm
    if scale == 0:
        if dot != 0:
            raise ValueError('Nonzero dot with zero gradient')
        return dict(dot=dot, cosine=None, teacher_change_along_descent=-dot,
                    interpretation='UNDEFINED_ZERO_GRADIENT')
    cosine = dot / scale
    if abs(cosine) > 1 + 1e-6:
        raise ValueError('Invalid Cauchy bound')
    label = ('LOCAL_TEACHER_INCREASE' if cosine < -1e-6 else
             'LOCAL_TEACHER_DECREASE' if cosine > 1e-6 else 'NEAR_ORTHOGONAL')
    return dict(dot=dot, cosine=cosine, teacher_change_along_descent=-dot,
                interpretation=label)
