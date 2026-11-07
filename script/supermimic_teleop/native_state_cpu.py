"""Inference-only Q1 state conversion using isolated MuJoCo kinematics."""

import mujoco
import numpy as np
import torch


def require(ok, message):
    if not ok:
        raise ValueError(message)


class KinematicState:
    def __init__(self, robot, spec):
        model = robot.model
        require((model.nq, model.nv, model.nu) == (29, 28, 22)
            and model.njnt == 23 and model.nmocap == 0
            and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
            and np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE)
            and model.jnt_bodyid[0] == 1,
            "Only the admitted single-root,22-hinge Q1 is supported")
        fields = dict(body_pos=model.body_pos, body_quat=model.body_quat,
            joint_axis=model.jnt_axis, joint_anchor=model.jnt_pos, default_q=model.qpos0[7:])
        require(all(np.array_equal(getattr(robot, k).detach().cpu().numpy(), v.astype(np.float32))
                    for k, v in fields.items()), "Robot buffers differ from the kinematic model")
        require(robot.parents == model.body_parentid.tolist()
            and robot.qindexes == (model.jnt_qposadr-7).tolist(), "Robot topology differs from the model")
        self.indices = np.asarray([model.body(name).id for name in spec.tracked_bodies])
        require(self.indices.shape == (14,) and len(set(self.indices)) == 14
            and np.all(self.indices > 0), "Require the exact14 tracked link origins")
        self.model, self.translator = model, spec.translator
        self.data = mujoco.MjData(model)
        self.velocity = np.empty(6, dtype=np.float64)
        self.body_velocity = np.empty((14, 6), dtype=np.float64)

    def __call__(self, value):
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.dtype == torch.float32 and value.shape == (1, 79)
            and not value.requires_grad, "Require inference-only CPU float32[1,79]")
        raw = value.detach().numpy()[0]
        require(np.isfinite(raw).all(), "Nonfinite Q1 input")
        norm = np.linalg.norm(raw[3:7])
        require(norm > 1e-8, "Degenerate root quaternion")
        data, model = self.data, self.model
        data.qpos[:] = raw[:29]
        data.qpos[3:7] = raw[3:7]/norm
        data.qvel[:3], data.qvel[6:] = raw[29:32], raw[35:57]
        inverse = data.qpos[3:7].copy()
        inverse[1:] *= -1
        # A free joint stores local angular velocity; the input stores world angular velocity.
        mujoco.mju_rotVecQuat(data.qvel[3:6], raw[32:35].astype(np.float64), inverse)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        for slot, body in enumerate(self.indices):
            # XBODY is the link-frame origin (xpos), whereas BODY uses the inertial origin (xipos).
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                     int(body), self.velocity, 0)
            self.body_velocity[slot] = self.velocity
        tr = self.translator
        return torch.cat((tr.to_canonical(value[:, 7:29]),
            tr.to_canonical_delta(value[:, 35:57]),
            torch.from_numpy(data.xpos[self.indices].astype(np.float32)).reshape(1, 42),
            torch.from_numpy(data.xquat[self.indices].astype(np.float32)).reshape(1, 56),
            torch.from_numpy(self.body_velocity[:, 3:].astype(np.float32)).reshape(1, 42),
            torch.from_numpy(self.body_velocity[:, :3].astype(np.float32)).reshape(1, 42),
            tr.to_canonical_delta(value[:, 57:79])), -1)
