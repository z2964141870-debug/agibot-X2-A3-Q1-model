"""Use only actual mesh hull vertices in the foot/floor displacement QP."""

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from script.a3.smpl_collision_recovery import RecoveringFloorLimit
from script.a3.smpl_vertex_floor import EnvelopeA3Retargeter, FootEnvelopeLimit


def support_vertices(vertices):
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('Requires finite nonempty 3D geometry')
    points = np.unique(points, axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points - points[0]) < 3:
        return points, 'ALL_ACTUAL_VERTICES_DEGENERATE'
    try:
        hull = ConvexHull(points)
    except QhullError:
        return points, 'ALL_ACTUAL_VERTICES_QHULL_FALLBACK'
    return points[hull.vertices], 'SCIPY_CONVEX_HULL_ACTUAL_VERTICES'


class FootHullLimit(FootEnvelopeLimit):
    def __init__(self, model, foot_geometries, minimum_distance=.001, gain=.85):
        self.model, self.minimum_distance, self.gain = model, minimum_distance, gain
        self.envelopes, self.geometry = [], []
        for side in foot_geometries:
            for geom, vertices in side:
                hull, method = support_vertices(vertices)
                self.envelopes.append((geom, hull))
                self.geometry.append(dict(geom_id=int(geom),
                    geom_name=mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                    source_vertices=len(vertices), plane_rows=len(hull), method=method))
        self.jacobian_position = np.zeros((3, model.nv))
        self.jacobian_rotation = np.zeros((3, model.nv))


class HullA3Retargeter(EnvelopeA3Retargeter):
    def __init__(self):
        super().__init__()
        indices = [i for i, value in enumerate(self.limits) if isinstance(value, FootEnvelopeLimit)]
        if len(indices) != 1 or any(isinstance(value, RecoveringFloorLimit) for value in self.limits):
            raise ValueError('Expected exactly the previous plane constraint')
        self.hull_limit = FootHullLimit(self.model, self.foot_geometries)
        self.limits[indices[0]] = self.hull_limit

    def geometry_contract(self):
        return dict(geometries=self.hull_limit.geometry,
            foot_geometries=len(self.hull_limit.geometry),
            plane_rows=sum(row['plane_rows'] for row in self.hull_limit.geometry),
            artificial_vertices=False, minimum_distance_m=self.hull_limit.minimum_distance,
            gain=self.hull_limit.gain)

    def verify_floor_support(self):
        data = self.configuration.data
        hulls = dict(self.hull_limit.envelopes)
        differences = []
        for side in self.foot_geometries:
            for geom, vertices in side:
                direction = data.geom_xmat[geom].reshape(3, 3)[2]
                differences.append(abs(float((vertices @ direction).min())
                    - float((hulls[geom] @ direction).min())))
        error = max(differences)
        if error > 1e-10:
            raise ValueError('Hull does not preserve actual mesh floor support')
        return error
