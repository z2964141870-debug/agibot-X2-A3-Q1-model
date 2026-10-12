"""Constrain a conservative eight-corner envelope for every actual foot geom."""

from itertools import product

import mujoco
import numpy as np
from mink.limits import Constraint, Limit

from script.a3.smpl_checked_integration import CheckedA3Retargeter
from script.a3.smpl_collision_recovery import RecoveringFloorLimit


class FootEnvelopeLimit(Limit):
    def __init__(self, model, foot_geometries, minimum_distance=.001, gain=.85):
        self.model, self.minimum_distance, self.gain = model, minimum_distance, gain
        self.envelopes = []
        for side in foot_geometries:
            for geom, vertices in side:
                bounds = np.array([vertices.min(0), vertices.max(0)])
                corners = np.array([bounds[np.asarray(selector), np.arange(3)]
                                    for selector in product((0, 1), repeat=3)])
                self.envelopes.append((geom, corners))
        self.jacobian_position = np.zeros((3, model.nv))
        self.jacobian_rotation = np.zeros((3, model.nv))

    def compute_qp_inequalities(self, configuration, dt):
        del dt  # QP decision variable is displacement.
        data, rows, bounds = configuration.data, [], []
        for geom, corners in self.envelopes:
            offsets = corners @ data.geom_xmat[geom].reshape(3, 3).T
            points = offsets + data.geom_xpos[geom]
            mujoco.mj_jac(self.model, data, self.jacobian_position,
                self.jacobian_rotation, data.geom_xpos[geom], self.model.geom_bodyid[geom])
            jacobian_z = (self.jacobian_position[2]
                + offsets[:, 1, None] * self.jacobian_rotation[0]
                - offsets[:, 0, None] * self.jacobian_rotation[1])
            rows.append(-jacobian_z)
            bounds.append(self.gain * (points[:, 2] - self.minimum_distance))
        return Constraint(G=np.vstack(rows), h=np.concatenate(bounds))


class EnvelopeA3Retargeter(CheckedA3Retargeter):
    def __init__(self):
        super().__init__()
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if (floor < 0 or not np.allclose(self.configuration.data.geom_xpos[floor, 2], 0.)
                or not np.allclose(self.configuration.data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1])):
            raise ValueError('Requires the pinned horizontal world-zero floor')
        indices = [i for i, value in enumerate(self.limits) if isinstance(value, RecoveringFloorLimit)]
        if len(indices) != 1:
            raise ValueError('Expected the previous foot/floor constraint')
        self.limits[indices[0]] = FootEnvelopeLimit(self.model, self.foot_geometries)
