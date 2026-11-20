"""Evaluate exclusive ownership in the frozen Q1 MuJoCo plant (50 Hz control)."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent))
import bench
from bench import mujoco
from run_suite import BUNDLE, EXPERT, TRACKS, verify
from relay import CASES
from reference import HandoffReference
from q1_kinematics_np import quat_xyzw_to_mat, mat_to_quat_wxyz, yaw_angle_from_mat, rotz_np
from q1_autonomous_recovery import Q1AutonomousRecovery


def simulate(group, seed, out):
    verify(BUNDLE)
    out.mkdir(parents=True)
    model = bench.scene(out / "scene")
    ctrl = bench.Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    kin = bench.Q1KinematicsNP(bench.ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml")
    branch = Q1AutonomousRecovery(ctrl, EXPERT, BUNDLE / "robot_model/supermimic.xml")
    geom_floor = model.geom("floor").id
    ankles = {model.geom(s + "_ankle_roll_link_collision_1").id for s in ("left", "right")}
    tracks = [
        np.loadtxt(TRACKS / t, delimiter=",").astype(np.float32) for t in (CASES[0][0], CASES[2][0])
    ]
    poses = list(bench.poses(ctrl, seed))
    rng = np.random.default_rng(seed)
    static = group in ("static", "offline")
    count = {"static": 24, "offline": 3, "push": 10, "double": 5, "flat": 2}[group]
    reports = []
    for i in range(count):
        data = mujoco.MjData(model)
        track = tracks[(i % 2 if group == "flat" else int(i % 5 >= 2))]
        pos, rot, joints = track[:, :3], quat_xyzw_to_mat(track[:, 3:7]), track[:, 7:]
        j = i * 8 if group == "offline" else i
        data.qpos[:] = poses[j][1] if static else np.r_[pos[0], mat_to_quat_wxyz(rot[0]), joints[0]]
        data.ctrl[:] = data.qpos[7:]
        mujoco.mj_forward(model, data)
        if static:
            mujoco.mj_step(model, data, nstep=20)
        ctrl.reset(data.qpos[7:].astype(np.float32))
        branch.reset()
        est = bench.ContactGatedFootAnchorNP(kin)
        previous = np.zeros(22, np.float32)
        align = np.eye(3, dtype=np.float32)
        offset = np.zeros(3, np.float32)
        yaw = np.eye(3, dtype=np.float32)
        standmodal = bench.build_robot_pose_590_np(
            np.zeros((1, 3), np.float32), yaw[None], ctrl.default_q[None], 0
        )
        ref = None
        anchors = []
        captures = []
        releases = []
        kicks = []
        rows = []
        states = []
        seen_stable = None
        quiet_run = 0
        max_solo = 0
        wrong = 0
        pd_oob = 0
        angle = rng.uniform(-0.1, 0.1)
        c, s = np.cos(angle), np.sin(angle)
        kick = rng.uniform(0.92, 1.08) * (np.array([[c, -s], [s, c]]) @ CASES[i % 5][1])
        impact = int(rng.integers(80, 121))
        horizon = 1250 if group == "double" else 776 if group in ("static", "push") else 701
        for tick in range(horizon):
            if group in ("static", "push") and tick > 700 and not releases:
                break
            if group in ("push", "double") and tick == impact:
                data.qvel[:2] += kick
                kicks.append(tick)
            if (
                group == "double"
                and len(kicks) == 1
                and (
                    tick == (releases[0] + 60 if releases else -1)
                    or (not releases and tick == impact + 625)
                )
            ):
                data.qvel[:2] -= 1.2 * kick
                kicks.append(tick)
            R = bench.quat_wxyz_to_mat(data.qpos[3:7])
            q = data.qpos[7:].astype(np.float32)
            w = data.qvel[3:6].astype(np.float32)
            qd = data.qvel[6:].astype(np.float32)
            idx = min(tick, len(pos) - 1)

            def reanchor():
                nonlocal ref, align, offset, yaw
                anchors.append(tick)
                yaw = rotz_np(yaw_angle_from_mat(R))
                if static:
                    est.reset(R, w, q, qd, np.zeros(2, np.float32), R_ref=yaw)
                else:
                    align = rotz_np(yaw_angle_from_mat(R) - yaw_angle_from_mat(rot[idx]))
                    offset[:2] = -(align @ pos[idx])[:2]
                    ref = HandoffReference(align @ pos[idx] + offset, R, q)
                    est.reset(R, w, q, qd, ref.origin[:2], R_ref=ref.rotation)

            calls = 0

            def base():
                nonlocal calls
                calls += 1
                if static:
                    rp, rR, modal = np.zeros(3, np.float32), yaw, standmodal
                elif ref is None:
                    rp, rR, modal = (
                        pos[idx],
                        rot[idx],
                        bench.build_robot_pose_590_np(pos, rot, joints, idx),
                    )
                else:
                    rp, rR, _, modal = ref.step(
                        align @ pos[idx] + offset, align @ rot[idx], joints[idx]
                    )
                exy, _ = est.step(R, w, q, qd, rp[:2], R_ref=rR)
                obs = bench.build_obs_80_np(
                    q=q,
                    qd=qd,
                    default_q=ctrl.default_q,
                    R_imu=R,
                    w_body=w,
                    a_prev=previous,
                    R_ref=rR,
                    cgfa_exy=exy,
                )
                return ctrl.step(obs, modal, q, qd)

            now = tick * 0.02
            # Static tests require a full second of independent standing, then connect the operator.
            online = not static or (
                group == "static" and seen_stable is not None and tick >= seen_stable + 50
            )
            source_stamp = now if online else None
            if tick == 0:
                est.reset(
                    R,
                    w,
                    q,
                    qd,
                    np.zeros(2, np.float32) if static else pos[0, :2],
                    R_ref=yaw if static else rot[0],
                )
            before = branch.owner
            target, tau, g = branch.step(
                R,
                w,
                q,
                qd,
                previous,
                base,
                reanchor,
                sample_time_s=now,
                teleop_stamp_s=source_stamp,
            )
            assert g in (0.0, 1.0) and calls == int(g == 0)
            if anchors and anchors[-1] == tick:
                assert calls == 1 and before == "recovery" and online
            if g == 1 and before == "teleop":
                captures.append(tick)
            if g == 0 and before == "recovery":
                releases.append(tick)
            if branch.standing_confirmed and seen_stable is None:
                seen_stable = tick
            foot = 0.0
            body = False
            for ci, ct in enumerate(data.contact):
                if geom_floor not in (ct.geom1, ct.geom2):
                    continue
                other = ct.geom2 if ct.geom1 == geom_floor else ct.geom1
                if other in ankles:
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, force)
                    foot += max(0.0, force[0])
                else:
                    body = True
            upright = bool(data.qpos[2] >= 0.35 and R[2, 2] > 0.92 and foot > 10 and not body)
            calm = bool(
                upright and np.linalg.norm(data.qvel[:2]) < 0.20 and np.linalg.norm(w) < 0.80
            )
            if releases and releases[-1] == tick:
                wrong += int(not calm)
            quiet_run = quiet_run + 1 if g == 1 and calm else 0
            max_solo = max(max_solo, quiet_run)
            pd_oob += int(np.sum(abs(tau) > ctrl.eff + 1e-4))
            rows.append(
                [
                    data.qpos[2],
                    R[2, 2],
                    foot,
                    body,
                    g,
                    calm,
                    calls,
                    online,
                    branch.standing_confirmed,
                ]
            )
            previous = (target - ctrl.default_q) / ctrl.action_scale
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=4)
            states.append(data.qpos.copy())
            if (
                releases
                and tick >= releases[-1] + 75
                and (
                    group in ("static", "push")
                    or (group == "double" and len(kicks) == 2 and releases[-1] > kicks[-1])
                )
            ):
                break
        trace = np.array(rows)
        horizon = len(trace)

        def events(seq, first, last):
            return [t for t in seq if first <= t < last]

        phases = []
        for start, stop in zip(kicks or [0], (kicks[1:] + [horizon]) if kicks else [horizon]):
            caps = events(captures, start, stop)
            grants = events(releases, start, stop)
            span = trace[start:stop]
            fell = bool(np.any((span[:, 3] > 0) & ((span[:, 0] < 0.22) | (span[:, 1] < 0.65))))
            post = bool(
                grants
                and grants[0] + 50 <= stop
                and np.all(trace[grants[0] : grants[0] + 50, 4] == 0)
                and np.all(trace[grants[0] : grants[0] + 50, 0] >= 0.35)
                and np.all(trace[grants[0] : grants[0] + 50, 1] > 0.85)
            )
            phases.append(dict(fell=fell, captures=caps, releases=grants, post_1s=post))
        assert releases == anchors
        r = dict(
            trial=i,
            pose=poses[j][0] if static else None,
            phases=phases,
            kicks=kicks,
            releases=releases,
            takeovers=captures,
            wrong_release=wrong,
            solo_calm_s=max_solo * 0.02,
            first_confirmation_s=None if seen_stable is None else seen_stable * 0.02,
            terminal_standing=bool(trace[-1, 5]),
            pd_oob=pd_oob,
            frames=horizon,
        )
        if group == "offline":
            assert not releases and not np.any(trace[:, 6])
        if group == "flat":
            assert not captures
        np.savez_compressed(out / f"trial_{i:02d}.npz", trace=trace, qpos=states)
        reports.append(r)
    (out / "results.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("dev", "test"))
    parser.add_argument("group", nargs="?")
    parser.add_argument("seed", type=int, nargs="?")
    a = parser.parse_args()
    if a.group:
        simulate(a.group, a.seed, D / "runs" / a.suite / str(a.seed) / a.group)
    else:
        seeds = [251201] if a.suite == "dev" else [251233, 251257]
        jobs = [
            (seed, g) for seed in seeds for g in ("static", "offline", "push", "double", "flat")
        ]

        def run(job):
            seed, g = job
            subprocess.run([sys.executable, __file__, a.suite, g, str(seed)], check=True)

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(run, jobs))
