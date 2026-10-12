"""Require existing foot/floor penetration to shrink, in displacement units."""

import mujoco

from mink.limits import Constraint
from script.a3.smpl_collision_units import DisplacementCollisionLimit, UnitA3Retargeter


class RecoveringFloorLimit(DisplacementCollisionLimit):
    def compute_qp_inequalities(self, configuration, dt):
        constraint = super().compute_qp_inequalities(configuration, dt)
        raw = self.velocity_limit
        bound = constraint.h.copy()
        for index, (first, second) in enumerate(raw.geom_id_pairs):
            distance = mujoco.mj_geomDistance(raw.model, configuration.data,
                first, second, raw.collision_detection_distance, None)
            if distance < raw.minimum_distance_from_collisions:
                bound[index] = raw.gain * (distance - raw.minimum_distance_from_collisions)
        return Constraint(G=constraint.G, h=bound)


class RecoveringA3Retargeter(UnitA3Retargeter):
    def __init__(self):
        super().__init__()
        matches = [i for i, limit in enumerate(self.limits)
                   if isinstance(limit, DisplacementCollisionLimit)]
        if len(matches) != 1:
            raise ValueError('Expected one displacement foot/floor limit')
        index = matches[0]
        self.limits[index] = RecoveringFloorLimit(self.limits[index].velocity_limit)
