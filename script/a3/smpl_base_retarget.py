"""Continuous floating-base IK; floor constraints see every configuration move."""

import csv
import json

import mink
import numpy as np
from scipy.spatial.transform import Rotation

from script.a3.fullchain_support import write_json
from script.a3.smpl_floor_retarget import FloorA3Retargeter, floor_pair_quality
from script.a3.smpl_corpus_qualify import contact_metrics


class ContinuousBaseA3Retargeter(FloorA3Retargeter):
    def solve(self, source, scale, directory):
        positions, rotations = self.targets(source, scale)
        self.configuration.update(self.data.qpos.copy())
        self.root_task.set_position_cost(100.)
        self.root_task.set_orientation_cost(20.)
        qposes, actual, actual_rotations, foot_minima = [], [], [], []
        for frame in range(len(source.times)):
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
                velocity = mink.solve_ik(self.configuration, [self.root_task] + [task for _, task in self.tasks] + [self.posture],
                    dt, 'quadprog', damping=.001, safety_break=True, limits=self.limits)
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
            root_and_joints_solved_together=True,per_frame_root_assignment=False,
            root_position_cost=100.,root_orientation_cost=20.,
            inferred_contact_not_sensor_truth=True,physical_balance_verified=False,
            usage='CONTACT_PAIR_QUALIFICATION',training_allowed=False)
        write_json(directory / 'retarget_summary.json',summary)
        print(json.dumps(summary),flush=True)
        if violations:
            raise ValueError('Contact IK violated joint limits')
        return summary


def base_pair_quality(summary, trace, joints, retargeter):
    quality = floor_pair_quality(summary, trace, joints, retargeter)
    error = trace['position_error_m'][:, 0]
    quality['root_position_error_p95_m'] = float(np.quantile(error, .95))
    quality['root_position_error_max_m'] = float(error.max())
    quality['checks'].update(root_position_tracking=quality['root_position_error_p95_m'] <= .05
        and quality['root_position_error_max_m'] <= .15)
    quality['passed'] = all(quality['checks'].values())
    return quality

