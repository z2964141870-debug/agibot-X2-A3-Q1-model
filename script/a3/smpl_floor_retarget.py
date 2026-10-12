"""Keep swing targets above the floor and measure only supported intervals."""

import mink
import mujoco
import numpy as np

from script.a3.smpl_contact_retarget import ContactA3Retargeter, contact_pair_quality
from script.a3.smpl_corpus_qualify import contact_metrics


class FloorA3Retargeter(ContactA3Retargeter):
    def __init__(self):
        super().__init__()
        floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        if floor < 0:
            raise ValueError('Pinned A3 floor geometry missing')
        feet = [geom for selected in self.foot_geometries for geom, _ in selected]
        floor_limit = mink.CollisionAvoidanceLimit(self.model, [(feet, [floor])],
            gain=.85, minimum_distance_from_collisions=.001,
            collision_detection_distance=.15, bound_relaxation=0.)
        if not floor_limit.geom_id_pairs:
            raise ValueError('No collision-enabled foot/floor geometry pairs')
        self.floor_collision_pairs = len(floor_limit.geom_id_pairs)
        self.limits.append(floor_limit)

    def targets(self, source, scale):
        positions, rotations = super().targets(source, scale)
        original = self.root_offset.copy()
        bounded = original.copy()
        for frame in range(1, len(original)):
            delta = original[frame] - bounded[frame - 1]
            norm = np.linalg.norm(delta)
            bounded[frame] = bounded[frame - 1] + delta * min(1., (.5 / 30) / max(norm, 1e-12))
        for frame in range(len(positions)):
            difference = bounded[frame] - original[frame]
            positions[frame] += difference
            for foot, side in enumerate(('left', 'right')):
                ankle = self.names.index(side + '_ankle')
                if self.contact_mask[frame, foot]:
                    positions[frame, ankle] -= difference
                else:
                    bottom = float((self.ankle_vertices[foot] @ rotations[frame, ankle].T)[:, 2].min())
                    positions[frame, ankle, 2] = max(positions[frame, ankle, 2], .003 - bottom)
        self.root_offset = bounded
        return positions, rotations


def supported_interval_slip(joints, actual_positions, foot_minima, names):
    contact = contact_metrics(joints, actual_positions, foot_minima, names)['contact_mask']
    supported = contact[:-1] & contact[1:]
    ankles = actual_positions[:, [names.index('left_ankle'), names.index('right_ankle')], :2]
    speed = np.linalg.norm(np.diff(ankles, axis=0) * 30, axis=-1)
    counts = supported.sum(0)
    values = [float(np.quantile(speed[:, side][supported[:, side]], .95)) if counts[side] else None for side in range(2)]
    return dict(method='FORWARD_TIME_INTERVAL_BOTH_ENDPOINTS_SOURCE_CONTACT',
        supported_intervals_per_foot=counts.astype(int).tolist(),
        robot_contact_ankle_xy_speed_p95_m_s=values)


def floor_pair_quality(summary, trace, joints, retargeter):
    quality = contact_pair_quality(summary, trace, joints, retargeter)
    slip = supported_interval_slip(joints, trace['actual_position_m'], trace['foot_geometry_min_z_m'], retargeter.names)
    quality['contacts'].update(slip)
    quality['checks']['contact_coverage'] = min(slip['supported_intervals_per_foot']) >= 5
    quality['checks']['contact_slip'] = all(value is not None and value <= .25 for value in slip['robot_contact_ankle_xy_speed_p95_m_s'])
    quality['floor_collision_pairs'] = retargeter.floor_collision_pairs
    quality['passed'] = all(quality['checks'].values())
    return quality
