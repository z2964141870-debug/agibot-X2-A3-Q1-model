"""Evaluate exclusive ownership in the frozen Q1 MuJoCo plant (50 Hz control)."""

import argparse
from collections import deque
import hashlib
import os
import json
import sys
from pathlib import Path
import numpy as np

D = Path(os.environ["Q1_AUTONOMOUS_ROOT"]).resolve()
sys.path.insert(0, str(D))
sys.path.insert(0, str(D.parent))
import bench
from bench import mujoco
from run_suite import BUNDLE, EXPERT, TRACKS, verify
from relay import CASES
from reference import HandoffReference
from q1_kinematics_np import quat_xyzw_to_mat, mat_to_quat_wxyz, yaw_angle_from_mat, rotz_np
from q1_autonomous_recovery import Q1AutonomousRecovery


def simulate(group, seed, out, expert, cadence, delay_ms, box, physics_ms):
    verify(BUNDLE)
    out.mkdir(parents=True)
    model = bench.scene(out / "scene")
    # Static starts settle for 100 ms with the same 2 ms physics.
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    files = [expert, D / "q1_autonomous_recovery.py", Path(__file__),
             BUNDLE / "robot_model/supermimic.xml", out / "scene/fall_hard.xml",
             out / "scene/supermimic_abs.xml",
             *(TRACKS / name for name in {CASES[0][0], CASES[2][0]})]
    manifest = dict(files={str(p): digest(p) for p in files}, cadence=cadence,
                    delay_ms=delay_ms, seed=seed, group=group, step_s=physics_ms / 1000., box=box,
                    mujoco=mujoco.__version__, bundle=str(BUNDLE))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ctrl = bench.Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    kin = bench.Q1KinematicsNP(bench.ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml")
    branch = Q1AutonomousRecovery(ctrl, expert, BUNDLE / "robot_model/supermimic.xml")
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
        model.opt.timestep = .002
        data = mujoco.MjData(model)
        track = tracks[(i % 2 if group == "flat" else int(i % 5 >= 2))]
        pos, rot, joints = track[:, :3], quat_xyzw_to_mat(track[:, 3:7]), track[:, 7:]
        j = i * 8 if group == "offline" else i
        data.qpos[:] = poses[j][1] if static else np.r_[pos[0], mat_to_quat_wxyz(rot[0]), joints[0]]
        data.ctrl[:] = data.qpos[7:]
        mujoco.mj_forward(model, data)
        if static:
            mujoco.mj_step(model, data, nstep=50)
        ctrl.reset(data.qpos[7:].astype(np.float32))
        branch.reset()
        model.opt.timestep = physics_ms / 1000.
        sent = data.qpos[7:].astype(np.float32).copy()
        initial_qpos = data.qpos.copy()
        initial_ground=max([max(0.,-float(c.dist)) for c in data.contact if geom_floor in c.geom],default=0.)
        initial_self=max([max(0.,-float(c.dist)) for c in data.contact if geom_floor not in c.geom],default=0.)
        samples = deque([np.r_[data.qpos, data.qvel]] * 7, maxlen=7)
        peak_rows = []
        max_step = 0.
        pre_impact_fall = False
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
        horizon = 1250 if group == "double" else 1101 if group == "push" else 951 if group == "static" else 701
        for tick in range(horizon):
            if group in ("static", "push") and captures and tick > captures[0]+850 and not releases:
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
            true_R = R
            true_w = data.qvel[3:6].copy()
            # Seven 2 ms samples include the present and the two bracketing -11 ms.
            samples[-1] = np.r_[data.qpos, data.qvel]
            measured = samples[-1]
            if delay_ms == 11:
                older, newer = samples[0], samples[1]
                measured = (older + newer) * .5
                orientation = newer[3:7] if np.dot(older[3:7], newer[3:7]) >= 0 else -newer[3:7]
                measured[3:7] = (older[3:7] + orientation) * .5
            R = bench.quat_wxyz_to_mat(measured[3:7])
            q = measured[7:model.nq].astype(np.float32)
            w = measured[model.nq + 3:model.nq + 6].astype(np.float32)
            qd = measured[model.nq + 6:].astype(np.float32)
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
            upright = bool(data.qpos[2] >= 0.35 and true_R[2, 2] > 0.92 and foot > 10 and not body)
            calm = bool(
                upright and np.linalg.norm(data.qvel[:2]) < 0.20 and np.linalg.norm(true_w) < 0.80
            )
            if releases and releases[-1] == tick:
                wrong += int(not calm)
            quiet_run = quiet_run + 1 if g == 1 and calm else 0
            max_solo = max(max_solo, quiet_run)
            pd_oob += int(np.sum(abs(tau) > ctrl.eff + 1e-4))
            rows.append(
                [
                    data.qpos[2],
                    true_R[2, 2],
                    foot,
                    body,
                    g,
                    calm,
                    calls,
                    online,
                    branch.standing_confirmed,
                ]
            )
            if group == "push" and tick < impact and (captures or (body and true_R[2, 2] < .65)):
                pre_impact_fall = True
            desired = np.clip(target, ctrl.default_q - 2., ctrl.default_q + 2.) if box else target
            interval_speed = interval_depth = interval_leg = interval_vz = 0.
            interval_ground = interval_self = interval_effort = 0.
            joint = deepest_geom = -1
            for substep in range(20 // physics_ms):
                if cadence == 500 or substep == 0:
                    updated = desired if cadence == 0 else sent + np.clip(desired - sent, -.2, .2)
                    max_step = max(max_step, float(np.max(abs(updated - sent))))
                    sent = updated
                data.ctrl[:] = sent
                mujoco.mj_step(model, data)
                samples.append(np.r_[data.qpos, data.qvel])
                speed = np.abs(data.qvel[6:])
                if speed.max() > interval_speed:
                    joint = int(speed.argmax())
                    interval_speed = float(speed[joint])
                interval_leg = max(interval_leg, float(speed[:12].max()))
                interval_vz = max(interval_vz, float(data.qvel[2]))
                interval_effort = max(interval_effort,float(np.max(abs(data.actuator_force)-ctrl.eff)))
                for contact in data.contact:
                    depth = max(0.,-float(contact.dist))
                    if geom_floor in contact.geom:
                        interval_ground = max(interval_ground,depth)
                    else:
                        interval_self = max(interval_self,depth)
                contacts = data.contact
                if len(contacts):
                    ci = int(np.argmin(contacts.dist))
                    depth = -contacts[ci].dist
                    if depth > interval_depth:
                        interval_depth = depth
                        a, b = contacts[ci].geom
                        deepest_geom = int(b if a == geom_floor else a)
            if not np.isfinite(data.qpos).all() or (cadence and max_step > .200001):
                raise ValueError("invalid state or target increment")
            previous = (sent - ctrl.default_q) / ctrl.action_scale
            peak_rows.append([interval_speed, interval_leg, interval_depth, joint, deepest_geom, interval_vz,interval_ground,interval_self,interval_effort])
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
            initial_ground_mm=initial_ground*1000,initial_self_mm=initial_self*1000,
            legal_settled_start=bool(initial_ground<=.010 and initial_self<=.002),
        )
        if group == "offline":
            assert not releases and not np.any(trace[:, 6])
        if group == "flat":
            assert not captures
        peaks = np.asarray(peak_rows)
        r.update(trial_peak_qd=float(peaks[:,0].max()),trial_ground_mm=float(peaks[:,6].max()*1000),trial_self_mm=float(peaks[:,7].max()*1000))
        owned = trace[:, 4] == 1
        rows_owned = np.flatnonzero(owned)
        if len(rows_owned):
            peak_index = rows_owned[peaks[owned, 0].argmax()]
            depth_index = rows_owned[peaks[owned, 2].argmax()]
            geom = int(peaks[depth_index, 4])
            r.update(peak_qd=float(peaks[peak_index, 0]),
                     peak_qd_joint=model.joint(1 + int(peaks[peak_index, 3])).name,
                     peak_leg_qd=float(peaks[owned, 1].max()),
                     penetration_mm=float(peaks[depth_index, 2] * 1000.),
                     deep_geom=model.geom(geom).name if geom >= 0 else None,
                     up_speed=float(peaks[owned, 5].max()))
        if len(rows_owned):
            r.update(ground_mm=float(peaks[owned,6].max()*1000),self_mm=float(peaks[owned,7].max()*1000),
                     actual_effort_excess=float(peaks[owned,8].max()))
        within_time=seen_stable is not None and (seen_stable-(captures[0] if captures else 0))*.02<=15.+1e-8
        r.update(pre_impact_fall=pre_impact_fall, max_command_step=max_step,
                 success=bool(within_time and not pre_impact_fall and phases[0]["fell"] and len(captures) == 1
                              and len(releases) == 1 and wrong == 0 and phases[0]["post_1s"]))
        np.savez_compressed(out / f"trial_{i:02d}.npz", trace=trace, qpos=states,
                            initial=initial_qpos, peaks=peaks)
        reports.append(r)
    (out / "results.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=["static", "push"])
    parser.add_argument("expert", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--cadence", type=int, choices=[0, 50, 500], default=500)
    parser.add_argument("--delay-ms", type=int, choices=[0, 11], default=11)
    parser.add_argument("--no-box", action="store_true", help="omit the bridge's default pose +/-2 rad bound")
    parser.add_argument("--physics-ms", choices=[2, 5], type=int, default=2)
    args = parser.parse_args()
    if args.physics_ms == 5 and (args.delay_ms or args.cadence == 500):
        parser.error("5 ms reference only supports immediate state and a held target")
    simulate(args.group, args.seed, args.out.resolve(), args.expert.resolve(), args.cadence, args.delay_ms, not args.no_box, args.physics_ms)
