"""Contact-aware root and foot targets with bounded, joint-limited A3 IK."""

import csv
import json

import mink
import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.fullchain_support import write_json
from script.a3.smpl_chain_retarget import ChainA3Retargeter
from script.a3.smpl_corpus_qualify import contact_metrics


class ContactA3Retargeter(ChainA3Retargeter):
    def __init__(self):
        super().__init__()
        self.source_joints = None
        self.ankle_vertices = []
        for side, selected in enumerate(self.foot_geometries):
            ankle = self.names.index(('left', 'right')[side] + '_ankle')
            vertices = []
            for geom, local in selected:
                world = local @ self.configuration.data.geom_xmat[geom].reshape(3, 3).T + self.configuration.data.geom_xpos[geom]
                vertices.append((world - self.rest_position[ankle]) @ self.rest_rotation[ankle])
            self.ankle_vertices.append(np.concatenate(vertices))

    def set_human_source(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.ndim != 3 or joints.shape[1:] != (24, 3) or not np.isfinite(joints).all():
            raise ValueError('Need finite original full human joints for contact inference')
        self.source_joints = joints.copy()

    def targets(self, source, scale):
        positions, rotations = super().targets(source, scale)
        if self.source_joints is None or len(self.source_joints) != len(positions):
            raise ValueError('Contact targets require aligned original human toe positions')
        contact = contact_metrics(self.source_joints, positions, np.zeros((len(positions), 2)), self.names)['contact_mask']
        corrected, rotation_out = positions.copy(), rotations.copy()
        offsets = np.zeros((len(positions), 3))
        anchors = np.zeros((2, 2))
        previous, offset = np.zeros(2, dtype=bool), np.zeros(3)
        for frame in range(len(positions)):
            support_positions = {}
            for foot, side in enumerate(('left', 'right')):
                ankle = self.names.index(side + '_ankle')
                if not contact[frame, foot]:
                    continue
                if not previous[foot]:
                    anchors[foot] = positions[frame, ankle, :2] + offset[:2]
                # A planted sole uses world-up roll/pitch while retaining heading.
                rotation = rotations[frame, ankle]
                heading = np.arctan2(rotation[1, 0], rotation[0, 0])
                rotation_out[frame, ankle] = Rotation.from_euler('z', heading).as_matrix()
                lowest = float((self.ankle_vertices[foot] @ rotation_out[frame, ankle].T)[:, 2].min())
                support_positions[foot] = np.r_[anchors[foot], .001 - lowest]
            if support_positions:
                offset = np.mean([target - positions[frame, self.names.index(('left', 'right')[foot] + '_ankle')]
                    for foot, target in support_positions.items()], axis=0)
            corrected[frame] += offset
            offsets[frame] = offset
            for foot, target in support_positions.items():
                corrected[frame, self.names.index(('left', 'right')[foot] + '_ankle')] = target
            previous = contact[frame].copy()
        self.contact_mask, self.root_offset = contact, offsets
        return corrected, rotation_out

    def solve(self, source, scale, directory):
        positions, rotations = self.targets(source, scale)
        self.configuration.update(self.data.qpos.copy())
        qposes, actual, actual_rotations, foot_minima = [], [], [], []
        for frame in range(len(source.times)):
            q = self.configuration.data.qpos.copy()
            q[:3] = positions[frame, 0]
            q[3:7] = Rotation.from_matrix(rotations[frame, 0]).as_quat(scalar_first=True)
            self.configuration.update(q)
            self.root_task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotations[frame, 0]), positions[frame, 0]))
            for index, task in self.tasks:
                is_contact = index in (self.names.index('left_ankle'), self.names.index('right_ankle'))
                supported = is_contact and self.contact_mask[frame, int(index == self.names.index('right_ankle'))]
                task.set_position_cost(400. if supported else 20.)
                task.set_orientation_cost(40. if supported else 1.)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(rotations[frame, index]), positions[frame, index]))
            iterations = 120 if frame == 0 else 8
            dt = 1 / 120 if frame == 0 else 1 / 240
            for _ in range(iterations):
                velocity = mink.solve_ik(self.configuration, [task for _, task in self.tasks] + [self.posture],
                    dt, 'quadprog', damping=.001, safety_break=True, limits=self.limits, constraints=[self.root_task])
                self.configuration.integrate_inplace(velocity, dt)
            if not np.isfinite(self.configuration.data.qpos).all():
                raise ValueError('Nonfinite contact IK')
            qposes.append(self.configuration.data.qpos.copy())
            actual.append(self.configuration.data.xpos[self.bodies].copy())
            actual_rotations.append(self.configuration.data.xmat[self.bodies].reshape(-1, 3, 3).copy())
            foot_minima.append(self.foot_min_z())
        qposes, actual, foot_minima = map(np.asarray, (qposes, actual, foot_minima))
        joints = qposes[:, self.q_indices]
        error = np.linalg.norm(actual - positions, axis=-1)
        orientation = (Rotation.from_matrix(np.asarray(actual_rotations).reshape(-1,3,3)) * Rotation.from_matrix(rotations.reshape(-1,3,3)).inv()).magnitude()
        limits = self.model.jnt_range[self.joint_ids]
        violations = int(np.count_nonzero((joints < limits[:,0]-1e-6) | (joints > limits[:,1]+1e-6)))
        np.savez_compressed(directory / 'kinematic_trace.npz', time_s=source.times, qpos=qposes,
            joint_names=np.asarray(self.joint_names), target_position_m=positions, actual_position_m=actual,
            position_error_m=error, foot_geometry_min_z_m=foot_minima,
            inferred_contact=self.contact_mask, root_correction_m=self.root_offset,
            usage=np.asarray('CONTACT_PAIR_QUALIFICATION'), training_allowed=np.asarray(False))
        fields=['Frame','root_translateX','root_translateY','root_translateZ','root_rotateX','root_rotateY','root_rotateZ',
                *self.joint_names,'head_yaw_joint','head_pitch_joint']
        with (directory / 'reference.csv').open('w', newline='') as stream:
            writer=csv.writer(stream)
            writer.writerow(fields)
            euler=Rotation.from_quat(qposes[:,3:7],scalar_first=True).as_euler('xyz',degrees=True)
            for frame in range(len(qposes)):
                writer.writerow([frame,*(qposes[frame,:3]*100),*euler[frame],*np.rad2deg(joints[frame]),0,0])
        contacts=contact_metrics(self.source_joints, actual, foot_minima, self.names)
        contact_mask=contacts.pop('contact_mask')
        summary=dict(finite=True,frames=len(qposes),fps=30,duration_s=float(source.times[-1]),
            joint_limit_violations=violations, ik_position_p95_m=float(np.quantile(error,.95)),
            ik_position_max_m=float(error.max()),ik_orientation_p95_rad=float(np.quantile(orientation,.95)),
            joint_velocity_max_rad_s=float(np.abs(np.diff(joints,axis=0)*30).max()),
            foot_geometry_min_z_m=float(foot_minima.min()),
            contact_sole_height_p95_m=[float(np.quantile(np.abs(foot_minima[:,side][contact_mask[:,side]]),.95))
                if contact_mask[:,side].any() else None for side in range(2)],
            root_correction_max_m=float(np.linalg.norm(self.root_offset,axis=-1).max()),
            root_correction_speed_p99_m_s=float(np.quantile(np.linalg.norm(np.diff(self.root_offset,axis=0)*30,axis=-1),.99)),
            reference_root_speed_p99_m_s=float(np.quantile(np.linalg.norm(np.diff(qposes[:,:3],axis=0)*30,axis=-1),.99)),
            initial_ik_iterations=120,initialization_seconds_equivalent=1.,reference_ik_iterations=8,
            support_position_cost=400.,support_orientation_cost=40.,
            inferred_contact_not_sensor_truth=True,physical_balance_verified=False,
            usage='CONTACT_PAIR_QUALIFICATION',training_allowed=False)
        write_json(directory / 'retarget_summary.json',summary)
        print(json.dumps(summary),flush=True)
        if violations:
            raise ValueError('Contact IK violated joint limits')
        return summary


def contact_pair_quality(summary, trace, joints, retargeter):
    from script.a3.smpl_corpus_qualify import retarget_quality
    quality=retarget_quality(summary,trace,joints,retargeter)
    contact=contact_metrics(joints,trace['actual_position_m'],trace['foot_geometry_min_z_m'],retargeter.names)['contact_mask']
    sole=[float(np.quantile(np.abs(trace['foot_geometry_min_z_m'][:,side][contact[:,side]]),.95))
          if contact[:,side].any() else None for side in range(2)]
    root_offset=trace['root_correction_m']
    quality['contact_sole_height_p95_m']=sole
    quality['root_correction_max_m']=float(np.linalg.norm(root_offset,axis=-1).max())
    quality['root_correction_speed_p99_m_s']=float(np.quantile(np.linalg.norm(np.diff(root_offset,axis=0)*30,axis=-1),.99))
    quality['reference_root_speed_p99_m_s']=float(np.quantile(np.linalg.norm(np.diff(trace['qpos'][:,:3],axis=0)*30,axis=-1),.99))
    quality['checks'].update(contact_sole_height=all(value is not None and value <= .02 for value in sole),
        root_correction=quality['root_correction_max_m'] <= .4,
        root_correction_speed=quality['root_correction_speed_p99_m_s'] <= 1.,
        reference_root_speed=quality['reference_root_speed_p99_m_s'] <= 4.)
    quality['passed']=all(quality['checks'].values())
    return quality
