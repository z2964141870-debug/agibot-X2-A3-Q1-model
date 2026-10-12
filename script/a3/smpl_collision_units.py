"""Adapt the installed Mink collision velocity bound to its displacement QP."""

import mink
import numpy as np
from mink.limits import Constraint, Limit

from script.a3.smpl_base_retarget import ContinuousBaseA3Retargeter


class DisplacementCollisionLimit(Limit):
    def __init__(self, velocity_limit):
        if velocity_limit.bound_relaxation != 0:
            raise ValueError('This adapter requires the registered zero relaxation')
        self.velocity_limit = velocity_limit

    def compute_qp_inequalities(self, configuration, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Need a finite positive IK step')
        constraint = self.velocity_limit.compute_qp_inequalities(configuration, dt)
        # build_ik optimizes Delta q; the installed collision h is in m/s.
        return Constraint(G=constraint.G, h=constraint.h * dt)


class UnitA3Retargeter(ContinuousBaseA3Retargeter):
    def __init__(self):
        super().__init__()
        matches = [i for i, limit in enumerate(self.limits)
                   if isinstance(limit, mink.CollisionAvoidanceLimit)]
        if len(matches) != 1:
            raise ValueError('Expected exactly the registered foot/floor limit')
        index = matches[0]
        self.limits[index] = DisplacementCollisionLimit(self.limits[index])
