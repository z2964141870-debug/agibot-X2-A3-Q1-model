"""Map human limb directions to fixed A3 segment lengths before constrained IK."""

import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.mocap_adapter import A3Retargeter


def align_direction(before, after):
    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if (before.shape != (3,) or after.shape != (3,) or not np.isfinite(before).all()
            or not np.isfinite(after).all() or min(np.linalg.norm(before), np.linalg.norm(after)) < 1e-10):
        raise ValueError('Need two finite, nonzero segment directions')
    before, after = before / np.linalg.norm(before), after / np.linalg.norm(after)
    cross = np.cross(before, after)
    sine, cosine = np.linalg.norm(cross), float(np.clip(before @ after, -1, 1))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.eye(3)[int(np.argmin(np.abs(before)))]
        axis = np.cross(before, axis)
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()


class ChainA3Retargeter(A3Retargeter):
    def targets(self, source, scale):
        from q1_retarget import heading_rotation
        names = self.names
        index = {name: i for i, name in enumerate(names)}
        human = source.positions
        world = Rotation.from_quat(source.orientations.reshape(-1, 4), scalar_first=True).as_matrix().reshape(len(human), len(names), 3, 3)
        calibration = source.calibration_positions
        cal_rotation = Rotation.from_quat(source.calibration_orientations, scalar_first=True).as_matrix()
        alignment = self.rest_rotation[0] @ cal_rotation[0].T
        yaw = heading_rotation(self.rest_rotation[0]) @ heading_rotation(cal_rotation[0]).T
        root_rotation = alignment @ world[:, 0]
        positions = np.empty_like(human)
        positions[:, 0] = self.rest_position[0] + scale * (human[:, 0] - calibration[0]) @ yaw.T
        positions[:, index['spine3']] = positions[:, 0] + np.einsum('tij,j->ti', root_rotation,
            self.rest_rotation[0].T @ (self.rest_position[index['spine3']] - self.rest_position[0]))
        rotations = alignment @ world @ cal_rotation.transpose(0, 2, 1) @ self.rest_rotation
        for side in ('left', 'right'):
            hip, knee, ankle = [index[side + '_' + name] for name in ('hip', 'knee', 'ankle')]
            shoulder, elbow, wrist = [index[side + '_' + name] for name in ('shoulder', 'elbow', 'wrist')]
            positions[:, hip] = positions[:, 0] + np.einsum('tij,j->ti', root_rotation,
                self.rest_rotation[0].T @ (self.rest_position[hip] - self.rest_position[0]))
            torso = index['spine3']
            positions[:, shoulder] = positions[:, torso] + np.einsum('tij,j->ti', rotations[:, torso],
                self.rest_rotation[torso].T @ (self.rest_position[shoulder] - self.rest_position[torso]))
            for parent, child in ((hip, knee), (knee, ankle), (shoulder, elbow), (elbow, wrist)):
                direction = human[:, child] - human[:, parent]
                length = np.linalg.norm(direction, axis=-1)
                if not np.isfinite(length).all() or np.any(length < 1e-6):
                    raise ValueError('Zero or nonfinite human limb segment')
                robot_vector = self.rest_position[child] - self.rest_position[parent]
                positions[:, child] = positions[:, parent] + (direction / length[:, None] @ alignment.T) * np.linalg.norm(robot_vector)
                local_vector = self.rest_rotation[parent].T @ robot_vector
                current = np.einsum('tij,j->ti', rotations[:, parent], local_vector)
                desired = positions[:, child] - positions[:, parent]
                correction = np.stack([align_direction(before, after)
                    for before, after in zip(current, desired, strict=True)])
                rotations[:, parent] = correction @ rotations[:, parent]
            rotations[:, wrist] = correction @ rotations[:, wrist]
        return positions, rotations
