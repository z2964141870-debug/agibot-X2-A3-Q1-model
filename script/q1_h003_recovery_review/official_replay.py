"""Open-loop replay of recorded Q1 MC joint commands in the review plant, with no ROS."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np


def run(scene, record, step_ms):
    with np.load(record) as raw:
        a = {k: raw[k] for k in ("event_ns", "joint_names", "imu_stamp", "orientation_xyzw",
             "joint_state_stamp", "q", "joint_command_stamp", "command_position", "command_stiffness",
             "command_damping", "command_effort", "command_velocity")}
    transitions = json.loads(record.with_suffix(".json").read_text())["transitions"]
    start, stop = [next(t["t_rel_s"] for t in transitions if t["desc"] == desc)
                   for desc in ("STAND_UP", "BIPED_WALK_RUN")]
    epoch = a["event_ns"].item()
    # Select the latest sample available at each simulated instant.
    def indices(key, times):
        stamp = a[key]
        if (np.diff(stamp) < 0).any():
            raise ValueError(f"unsorted {key}")
        idx = np.searchsorted(stamp, epoch + np.rint(times * 1e9).astype(np.int64), side="right") - 1
        if np.min(idx) < 0 or np.max(times)*1e9 > stamp[-1]-epoch:
            raise ValueError(f"insufficient {key} coverage")
        return idx

    m = mujoco.MjModel.from_xml_path(str(scene))
    m.opt.timestep = step_ms / 1000.
    t = np.arange(start, stop + 1., m.opt.timestep)
    imu = indices("imu_stamp", t)
    measured = indices("joint_state_stamp", t)
    commands = indices("joint_command_stamp", t)
    assert np.array_equal(m.actuator_trnid[:, 0], np.arange(1, 23))
    assert np.all(m.actuator_gear[:, 0] == 1.) and not m.actuator_ctrllimited.any()
    order = [a["joint_names"].tolist().index(m.joint(j).name) for j in range(1, 23)]
    d = mujoco.MjData(m)
    d.qpos[7:] = a["q"][measured[0], order]
    d.qpos[3:7] = a["orientation_xyzw"][imu[0], [3, 0, 1, 2]]
    floor = m.geom("floor").id
    d.qpos[2] = -.25
    mujoco.mj_forward(m, d)
    low = min(c.dist for c in d.contact if floor in c.geom)
    d.qpos[2] -= low
    mujoco.mj_forward(m, d)
    rows = []
    for j, cmd, att in zip(measured, commands, imu):
        kp, kd = a["command_stiffness"][cmd, order], a["command_damping"][cmd, order]
        qref, vref = a["command_position"][cmd, order], a["command_velocity"][cmd, order]
        feed = a["command_effort"][cmd, order]
        target_q = a["q"][j, order]
        sensor = a["orientation_xyzw"][att]
        truth_cos = 1. - 2.*(sensor[0]**2 + sensor[1]**2)
        rows.append((d.qpos[2], 1.-2.*(d.qpos[4]**2 + d.qpos[5]**2), truth_cos,
                     np.mean((d.qpos[7:] - target_q)**2)))
        # Affine actuator force = kp*(reference-position) + kd*(vref-velocity) + feed.
        # Encode feed in the actuator bias so the model's actuator and joint force limits still apply.
        m.actuator_gainprm[:, 0] = kp
        m.actuator_biasprm[:, 0] = feed + kd * vref
        m.actuator_biasprm[:, 1] = -kp
        m.actuator_biasprm[:, 2] = -kd
        d.ctrl[:] = qref
        mujoco.mj_step(m, d)
    z, up, imu_up, mse = np.asarray(rows).T
    if not np.isfinite(rows).all():
        raise ValueError("nonfinite simulation")
    last = t >= stop
    return dict(step_ms=step_ms, start=start, walk_mode=stop, record=record.name,
                record_sha256=hashlib.sha256(record.read_bytes()).hexdigest(),
                initial_z_m=float(z[0]), terminal_z_m=float(z[-1]),
                last_second_stood=bool(np.all((z[last] >= .35) & (up[last] > .85))),
                last_second_sim_min_upright_cos=float(up[last].min()),
                last_second_measured_min_upright_cos=float(imu_up[last].min()),
                joint_rmse_rad=float(np.sqrt(mse.mean())))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scene", type=Path)
    p.add_argument("records", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    reports = [run(args.scene, args.records / name, dt) for dt in (5, 2)
               for name in ("q1_official_standup_from_mcap.npz", "q1_official_standup_faceup_20261120.npz")]
    with args.out.open("x") as f:
        json.dump(reports, f, indent=2)
