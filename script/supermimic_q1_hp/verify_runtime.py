"""Check Q1 pose/velocity/gradient contracts independently against MuJoCo."""

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.rl.algos.supermimic.geometry import quat_to_mat

    model = mujoco.MjModel.from_xml_path(str(args.runtime / "q1_assets/scene.xml"))
    kin = Q1Kinematics(args.runtime / "q1_assets/scene.xml", "cpu")
    rng = np.random.default_rng(20261026)
    maxima = np.zeros(4)
    for _ in range(24):
        data = mujoco.MjData(model)
        data.qpos[:3] = rng.normal(size=3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        data.qpos[7:] = rng.uniform(kin.q_lo.numpy(), kin.q_hi.numpy())
        data.qvel[:] = rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        rotation = quat_to_mat(torch.tensor(data.qpos[3:7], dtype=torch.float32)[None])
        p = torch.tensor(data.qpos[:3], dtype=torch.float32)[None]
        v = torch.tensor(data.qvel[:3], dtype=torch.float32)[None]
        w = (rotation @ torch.tensor(data.qvel[3:6], dtype=torch.float32)[None, :, None]).squeeze(-1)
        q = torch.tensor(data.qpos[7:], dtype=torch.float32)[None]
        qd = torch.tensor(data.qvel[6:], dtype=torch.float32)[None]
        actual = [a[0].detach().numpy() for a in kin.forward(p, rotation, v, w, q, qd, all_bodies=True)]
        velocities = []
        for body in range(1, model.nbody):
            jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            # BODY object velocity is at COM; FK publishes link-origin velocity.
            mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body], body)
            velocities.append(np.r_[jacr @ data.qvel, jacp @ data.qvel])
        velocities = np.asarray(velocities)
        expected = [data.xpos[1:], data.xmat[1:].reshape(-1, 3, 3), velocities[:, 3:], velocities[:, :3]]
        maxima = np.maximum(maxima, [np.max(np.abs(a-b)) for a, b in zip(actual, expected)])
    if np.max(maxima) > 5e-5:
        raise AssertionError(f"Q1 FK differs from physical model: {maxima}")
    q.requires_grad_(True)
    body_p, body_r, body_v, body_w = kin.forward(p, rotation, v, w, q, qd)
    loss = body_p.square().sum() + .1*body_r.sum() + .01*body_v.square().sum() + .01*body_w.square().sum()
    gradient = torch.autograd.grad(loss, q)[0]
    if not torch.isfinite(gradient).all() or not gradient.abs().sum() > 0:
        raise AssertionError("Q1 FK gradient is unusable")
    if model.geom("floor").id < 0 or model.nu != 22:
        raise AssertionError("Q1 scene/control contract differs")
    result = {"status": "Q1_POSE_VELOCITY_GRADIENT_CONTRACT_PASSED", "random_states": 24, "body_count_including_fixed_wrists": kin.num_bodies, "tracked_bodies": len(kin.tracked_idx), "max_errors_position_rotation_linear_angular": maxima.tolist(), "finite_nonzero_joint_gradient": True, "model_dimensions": [model.nq, model.nv, model.nu], "optimizer_updates": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
