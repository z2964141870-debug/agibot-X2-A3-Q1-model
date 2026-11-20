"""Continuous walking perturbation, recovery and tracking handoff at 50 Hz."""

from bench import *
from q1_kinematics_np import quat_xyzw_to_mat, mat_to_quat_wxyz, yaw_angle_from_mat, rotz_np
from recovery import Recovery
from reference import HandoffReference
from restart_profile import PROFILES, RestartProfile

CASES = (
    ("straight_walk_50hz.csv", (0, 2.7), 100),
    ("straight_walk_50hz.csv", (0, -2.7), 100),
    ("loop_figure8_walk.csv", (-2.7, 0), 100),
    ("loop_figure8_walk.csv", (2.7, 0), 100),
    ("loop_figure8_walk.csv", (1.91, 1.91), 100),
)


def run(
    path,
    out,
    public_runtime=False,
    data_dir=Path("/tmp/q1_v2b_narrow_tracks"),
    cases=CASES,
    reference_factory=None,
    profile=PROFILES["published"],
    repeat_kick=False,
):
    from functools import partial

    make_reference = (
        partial(HandoffReference, resume_ticks=profile.resume_ticks)
        if reference_factory is None
        else reference_factory
    )
    out.mkdir(exist_ok=False, parents=True)
    ctrl = Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    model = scene(out / "xml")
    kin = Q1KinematicsNP(ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml")
    report = []
    for i, (track, kick, kick_tick) in enumerate(cases):
        clip = np.loadtxt(data_dir / track, delimiter=",").astype(np.float32)
        pos = clip[:, :3]
        rot = quat_xyzw_to_mat(clip[:, 3:7])
        joints = clip[:, 7:]
        data = mujoco.MjData(model)
        data.qpos[:] = np.r_[pos[0], mat_to_quat_wxyz(rot[0]), joints[0]]
        data.ctrl[:] = joints[0]
        mujoco.mj_forward(model, data)
        if public_runtime:
            from q1_recovery_runtime import Q1RecoveryBranch

            c = Q1RecoveryBranch(ctrl, path, BUNDLE / "robot_model/supermimic.xml")
        else:
            c = Recovery(ctrl, path)
        est = ContactGatedFootAnchorNP(kin)
        ctrl.reset(joints[0])
        a_prev = np.zeros(22, np.float32)
        ref_window = None
        idx = 0
        align = np.eye(3, dtype=np.float32)
        offset = np.zeros(3, np.float32)
        display_offset = np.zeros(3, np.float32)
        fallen = False
        handed = False
        walked = 0
        released = False
        refall = False
        states = []
        ghosts = []
        trace = []
        contact_after = False
        loads = []
        body_floor = []
        impacts = []
        reanchors = []
        first_release = None
        second_kick = None
        target_offsets = []
        ankle_ids = [
            model.geom(side + "_ankle_roll_link_collision_1").id for side in ("left", "right")
        ]
        floor_id = model.geom("floor").id
        # Reserve 3 s of walking after a handoff within the 12 s recovery observation.
        for tick in range(1400 if repeat_kick else 850):
            if tick == kick_tick:
                data.qvel[:2] += kick
                impacts.append(tick)
            elif repeat_kick and first_release is not None and tick == first_release + 100:
                data.qvel[:2] -= kick
                second_kick = tick
                impacts.append(tick)
            applied_offset = 0.0
            R = quat_wxyz_to_mat(data.qpos[3:7])
            q = data.qpos[7:].astype(np.float32)
            w = data.qvel[3:6].astype(np.float32)
            qd = data.qvel[6:].astype(np.float32)
            soles, feet, _, fw = est.compute_sole_fk(R, w, q, qd)

            def reanchor():
                nonlocal align, offset, ref_window
                reanchors.append(tick)
                align = rotz_np(yaw_angle_from_mat(R) - yaw_angle_from_mat(rot[idx]))
                offset[:2] = -(align @ pos[idx])[:2]
                # Recorded world position is used only to place the visual reference.
                display_offset[:2] = data.qpos[:2]
                ref_window = make_reference(align @ pos[idx] + offset, R, q)
                est.reset(R, w, q, qd, ref_window.origin[:2], R_ref=ref_window.rotation)

            goal_p, goal_R, goal_q = align @ pos[idx] + offset, align @ rot[idx], joints[idx]

            def observation(exy):
                return build_obs_80_np(
                    q=q,
                    qd=qd,
                    default_q=ctrl.default_q,
                    R_imu=R,
                    w_body=w,
                    a_prev=a_prev,
                    R_ref=goal_R,
                    cgfa_exy=exy,
                )

            def base():
                nonlocal goal_p, goal_R, goal_q, applied_offset
                correction = None
                if ref_window is not None:
                    correction = profile.ankle_offset(ref_window.tick, float(w[0]), soles[:, 2])
                    if correction is not None:
                        applied_offset = float(abs(correction).max())
                    goal_p, goal_R, goal_q, modal = ref_window.step(
                        align @ pos[idx] + offset, align @ rot[idx], joints[idx]
                    )
                else:
                    modal = build_robot_pose_590_np(pos, rot, joints, idx)
                exy, _ = est.step(R, w, q, qd, goal_p[:2], R_ref=goal_R)
                return ctrl.step(observation(exy), modal, q, qd, xi_joint_offset=correction)

            lowest = soles[:, 2].argmin()
            valid = float(feet[lowest, 2, 2] > 0.65)
            height = float(np.clip(-soles[lowest, 2], 0, 0.65)) * valid
            if public_runtime:
                target, tau, g = c.step(R, w, q, qd, a_prev, base, reanchor)
            else:
                target, tau, g = c.step_measured(
                    observation(np.zeros(2, np.float32)), height, valid, fw * 0.01, base, reanchor
                )
            body_contact = False
            if tick > kick_tick:
                # Require a downed posture as well as body/ground contact outside the feet.
                for con in data.contact:
                    geom_names = [model.geom(int(j)).name or "" for j in (con.geom1, con.geom2)]
                    if "floor" in geom_names and any(
                        x in n
                        for x in ("pelvis", "elbow", "wrist", "torso", "shoulder")
                        for n in geom_names
                    ):
                        body_contact = True
                        contact_after = True
                fallen |= bool(contact_after and (R[2, 2] < 0.65 or data.qpos[2] < 0.22))
                if fallen and g > 0:
                    handed = True
                if released and g == 1:
                    refall = True
                if handed and g == 0:
                    released = True
                    if first_release is None:
                        first_release = tick
                if handed and g == 0 and data.qpos[2] >= 0.35 and R[2, 2] > 0.85:
                    walked += 1
                else:
                    walked = 0
            action = (target - ctrl.default_q) / ctrl.action_scale
            change = abs(action - a_prev).mean()
            a_prev = action
            ghosts.append(np.r_[goal_p + display_offset, mat_to_quat_wxyz(goal_R), goal_q])
            if g < 1.0 and (ref_window is None or ref_window.tick >= 30):
                idx = min(idx + 1, len(pos) - 1)
            data.ctrl[:] = target
            mujoco.mj_step(model, data, nstep=4)
            depth = max(0.0, -min((ct.dist for ct in data.contact), default=0.0))
            force = np.zeros(6)
            pair = [0.0, 0.0]
            for ci, ct in enumerate(data.contact):
                if ct.geom1 != floor_id and ct.geom2 != floor_id:
                    continue
                other = ct.geom2 if ct.geom1 == floor_id else ct.geom1
                if other in ankle_ids:
                    mujoco.mj_contactForce(model, data, ci, force)
                    pair[ankle_ids.index(other)] += max(float(force[0]), 0.0)
            body_floor.append(body_contact)
            loads.append(pair)
            target_offsets.append(applied_offset)
            trace.append(
                [
                    data.qpos[2],
                    quat_wxyz_to_mat(data.qpos[3:7])[2, 2],
                    abs(tau).max(),
                    np.count_nonzero(abs(tau) > ctrl.eff + 1e-4),
                    change,
                    g,
                    depth,
                    fw,
                    idx,
                ]
            )
            states.append(data.qpos.copy())
            if repeat_kick:
                if second_kick is not None and tick >= second_kick + 400 and walked >= 150:
                    break
                if tick >= 950 and first_release is None:
                    break
            elif walked == 150 or (tick >= 700 and not released):
                break
        trace = np.asarray(trace)
        width = trace[-150:, 7].mean() if walked == 150 else None
        disp = float(np.linalg.norm(states[-1][:2] - states[-150][:2])) if walked == 150 else 0.0
        row = dict(
            trial=i,
            track=track,
            kick=kick,
            fallen=fallen,
            took_control=handed,
            walking_ticks=walked,
            foot_width_cm=width,
            walk_disp_m=disp,
            pd_oob=int(trace[:, 3].sum()),
            refall=refall,
            success=bool(
                fallen
                and handed
                and not refall
                and walked == 150
                and width <= 22
                and disp > 0.25
                and trace[:, 3].sum() == 0
            ),
        )
        if repeat_kick:
            from repeated import score_repeated

            row = dict(
                trial=i,
                track=track,
                kick=kick,
                **score_repeated(
                    trace, np.asarray(states), np.asarray(body_floor), impacts, reanchors
                ),
            )
        report.append(row)
        np.savez_compressed(
            out / f"relay_{i}.npz",
            qpos=states,
            reference=ghosts,
            trace=trace,
            outcome_json=json.dumps(row),
            loads=loads,
            body_floor=body_floor,
            kicks=impacts,
            reanchors=reanchors,
            ankle_offset=target_offsets,
        )
        (out / "relay.json").write_text(json.dumps(report, indent=2))
        print(row, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("expert", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--public-runtime", action="store_true")
    ap.add_argument("--tracks", type=Path, default=Path("/tmp/q1_v2b_narrow_tracks"))
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--gait-profile", choices=PROFILES, default="published")
    group.add_argument(
        "--resume-ticks", type=int, help="Walking reference fade length after handback (50 Hz)."
    )
    a = ap.parse_args()
    if a.resume_ticks is not None and a.resume_ticks <= 0:
        ap.error("--resume-ticks must be positive")
    profile = PROFILES[a.gait_profile] if a.resume_ticks is None else RestartProfile(a.resume_ticks)
    run(a.expert, a.out.resolve(), a.public_runtime, a.tracks, profile=profile)
