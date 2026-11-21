"""Build exact 2 ms collision-enabled training & unseen validation pools and verify plant parity."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

RUN_V1 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_20261121")
RISE_DIR = Path("/home/yu/projects/SuperMimicHP/runs/q1_supported_rise_20261121")
AUTON_ROOT = Path("/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery")
for p in (str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from common import BUNDLE, CONFIG, EFF, HI, KD, KP, LO, Plant, SCALE, STAND, mj
from limbs import add_limb_contacts
from q1_kinematics_np import quat_wxyz_to_mat
from q1_recovery_runtime import Q1RecoveryBranch
from q1_v2b_runtime import Q1V2bOnnxController
from recorded_rise import load_track


def collect_legal_starts(model, refs, seeds, scene_arm_pairs, geom_floor):
    """Collect legal settled 79D starts [qpos(29), qvel(28), sent(22)] using exact evaluate.py logic."""
    d = mj.MjData(model)
    settled = mj.MjData(model)
    by_kind = {0: [], 1: [], 2: []}
    side_min_d = []
    seg = np.zeros(6)
    pelvis_id = model.body("pelvis").id

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for i in range(24):
            kind = i // 8
            accepted = False
            for attempt in range(300):
                phase = int(rng.integers(20))
                base = refs[0 if kind == 0 else 1][0][phase]
                joints = np.clip(base + rng.uniform(-0.08, 0.08, 22), LO, HI)
                pitch = (np.pi / 2 if kind == 0 else -np.pi / 2 if kind == 1 else 0.0) + rng.uniform(-0.06, 0.06)
                roll = (0.0 if kind < 2 else (-1 if i % 2 else 1) * np.pi / 2) + rng.uniform(-0.04, 0.04)
                yaw = rng.uniform(-np.pi, np.pi)
                cr, sr = np.cos(roll / 2), np.sin(roll / 2)
                cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
                cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
                quat = [
                    cr * cp * cy + sr * sp * sy,
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                ]
                d.qpos[:] = np.r_[0.0, 0.0, -0.4, quat, joints]
                d.qvel[:] = 0.0
                mj.mj_forward(model, d)
                if any(c.dist < -0.0005 and geom_floor not in c.geom for c in d.contact):
                    continue
                contacts = [c.dist for c in d.contact if geom_floor in c.geom]
                if not contacts:
                    continue
                d.qpos[2] -= min(contacts) - 0.0002
                mj.mj_forward(model, d)
                mj.mj_resetData(model, settled)
                settled.qpos[:] = d.qpos
                settled.ctrl[:] = d.qpos[7:]
                mj.mj_forward(model, settled)
                mj.mj_step(model, settled, nstep=50)
                ground = max((-1000.0 * c.dist for c in settled.contact if geom_floor in c.geom), default=0.0)
                arm_leg = max(
                    (
                        -1000.0 * c.dist
                        for c in settled.contact
                        if geom_floor not in c.geom
                        and tuple(sorted((int(c.geom1), int(c.geom2)))) in scene_arm_pairs
                    ),
                    default=0.0,
                )
                if ground > 10.0 or arm_leg > 2.0:
                    continue
                accepted = True
                break
            if not accepted:
                raise RuntimeError(f"Cannot construct legal fall {seed}/{i}")
            row = np.r_[settled.qpos, settled.qvel, settled.qpos[7:]].astype(np.float64)
            by_kind[kind].append(row)
            if kind == 2:
                md = 999.0
                for a, b in scene_arm_pairs:
                    dist = mj.mj_geomDistance(model, settled, a, b, 0.25, seg)
                    if dist < md:
                        md = dist
                uy = float(settled.xmat[pelvis_id, 7])
                side_min_d.append((seed, i, "R_down" if uy > 0 else "L_down", round(md * 1000.0, 2)))
    return {k: np.stack(v) for k, v in by_kind.items()}, side_min_d


def rollout_prone_to_supine(env, prone_rows, roll_npz_path):
    """Roll prone starts using verified roll.npz and capture settled supine transition states."""
    roll = np.load(roll_npz_path)
    keys = roll["keys"].astype(np.float64)
    dur = float(roll["duration"])
    n = len(prone_rows)
    out_rows = []
    out_peaks = []
    reached = 0
    for start_idx in range(0, n, env.n):
        batch = prone_rows[start_idx : start_idx + env.n]
        bs = len(batch)
        padded = np.tile(batch[0], (env.n, 1))
        padded[:bs] = batch
        obs = env.reset(padded)
        init_q = padded[:, 7:29].copy()
        sent = padded[:, 7:29].astype(np.float32).copy()
        peaks = np.zeros((env.n, 3))
        quiet_cnt = np.zeros(env.n, dtype=np.int32)
        captured = [None] * bs
        kp32, kd32, eff32, stand32, lo32, hi32 = [
            x.astype(np.float32) for x in (KP, KD, EFF, STAND, LO, HI)
        ]
        margin = 0.5 * eff32 / kp32
        for tk in range(200):
            phase = min((tk + 1) / dur, 5.0)
            idx = min(int(phase), 4)
            origin = init_q if idx == 0 else np.broadcast_to(keys[idx - 1], (env.n, 22))
            blend = 0.5 - 0.5 * np.cos(np.pi * (phase - idx))
            target = np.clip(origin + blend * (keys[idx] - origin), lo32, hi32)
            # Replicate 500 Hz sender locally so captured `sent` is exact
            for j in range(bs):
                if captured[j] is None:
                    q_obs = obs[j, :22].astype(np.float32) + stand32
                    qd_obs = obs[j, 22:44].astype(np.float32) * 10.0
                    raw = np.clip(target[j].astype(np.float32), lo32 - margin, hi32 + margin)
                    tau = np.clip(kp32 * (raw - q_obs) - kd32 * qd_obs, -eff32, eff32)
                    des = np.clip(q_obs + (tau + kd32 * qd_obs) / kp32, stand32 - 2.0, stand32 + 2.0)
                    for _ in range(10):
                        sent[j] = sent[j] + np.clip(des - sent[j], -0.2, 0.2)
            obs, met = env.step(target)
            peaks = np.maximum(peaks, met[:, [4, 5, 6]])
            up_x = -obs[:, 47]
            w_norm = np.linalg.norm(obs[:, 44:47] * 4.0, axis=1)
            q_now = (up_x > 0.80) & (w_norm < 1.7)
            quiet_cnt = np.where(q_now, quiet_cnt + 1, 0)
            for j in range(bs):
                if captured[j] is None and quiet_cnt[j] >= 10:
                    captured[j] = np.r_[env.state[j], sent[j].astype(np.float64)].copy()
            if all(c is not None for c in captured):
                break
        for j in range(bs):
            out_peaks.append(peaks[j].tolist())
            if captured[j] is not None:
                reached += 1
                out_rows.append(captured[j])
    return np.stack(out_rows), np.array(out_peaks), reached, n


class DummySineExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float32))
        self.gain = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, rec_obs_hist, rec_state):
        x = rec_obs_hist[:, -1]
        age = rec_state[:, 0]
        cmd = self.stand[None] + 0.08 * torch.sin(age[:, None] * 0.15) + 0.0 * x[:, :22] + self.gain
        nxt = torch.cat((age[:, None] + 1.0, rec_state[:, 1:]), 1)
        return cmd, nxt


def verify_parity(out_dir: Path, start_row: np.ndarray):
    """Assert 74D obs and 57D physics state parity between plant.so and Q1RecoveryBranch."""
    onnx_path = out_dir / "parity_dummy.onnx"
    m = DummySineExpert().eval()
    sample = torch.zeros((1, 10, 74))
    sample[:, :, 49] = -1.0
    torch.onnx.export(
        m,
        (sample, torch.zeros(1, 26)),
        str(onnx_path),
        input_names=["rec_obs_hist", "rec_state"],
        output_names=["joint_target", "next_state"],
        opset_version=17,
        dynamo=False,
    )
    scene_dir = out_dir / "parity_scene"
    plant = Plant(1, scene_dir, delay=11)
    ctrl = Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    branch = Q1RecoveryBranch(ctrl, onnx_path, BUNDLE / "robot_model/supermimic.xml")
    branch.reset()
    branch.expert.gate.mode = "recover"
    ctrl.reset(start_row[7:29].astype(np.float32))

    # Setup independent MuJoCo matching eval_core.py
    mj_model = add_limb_contacts(scene_dir / "eval_ref")
    mj_model.opt.timestep = 0.002
    mj_data = mj.MjData(mj_model)
    mj_data.qpos[:] = start_row[:29]
    mj_data.qvel[:] = start_row[29:57]
    sent = start_row[57:].astype(np.float32).copy()
    mj_data.ctrl[:] = sent
    mj.mj_forward(mj_model, mj_data)
    ring = deque([np.r_[mj_data.qpos, mj_data.qvel]] * 7, maxlen=7)

    p_obs = plant.reset(start_row[None])
    max_obs_diff = 0.0
    max_state_diff = 0.0
    early_obs_diff = 0.0
    early_state_diff = 0.0
    for tk in range(45):
        older, newer = ring[0], ring[1]
        meas = 0.5 * (older + newer)
        ori = newer[3:7] if np.dot(older[3:7], newer[3:7]) >= 0 else -newer[3:7]
        meas[3:7] = 0.5 * (older[3:7] + ori)
        mj.mj_normalizeQuat(mj_model, meas[:29])
        R = quat_wxyz_to_mat(meas[3:7])
        q = meas[7:29].astype(np.float32)
        w = meas[29 + 3 : 29 + 6].astype(np.float32)
        qd = meas[29 + 6 :].astype(np.float32)
        prev = (sent - ctrl.default_q) / ctrl.action_scale

        # Evaluate identical ORT float32 target in both Q1RecoveryBranch and plant.so.
        obs_hist_pre = branch.expert.history
        state_pre = branch.expert.state.copy()
        target, _, _ = branch.step(R, w, q, qd, prev, lambda: None, lambda: None)
        branch_x = branch.expert.history[0, -1]
        cur_obs_diff = float(np.max(np.abs(branch_x - p_obs[0])))
        max_obs_diff = max(max_obs_diff, cur_obs_diff)
        if tk < 25:
            early_obs_diff = max(early_obs_diff, cur_obs_diff)

        raw_onnx = branch.expert.policy.run(
            ["joint_target"],
            {
                "rec_obs_hist": np.repeat(p_obs[0][None, None].astype(np.float32), 10, axis=1)
                if obs_hist_pre is None
                else branch.expert.history,
                "rec_state": state_pre,
            },
        )[0][0]

        des = np.clip(target, ctrl.default_q - 2.0, ctrl.default_q + 2.0)
        for _ in range(10):
            sent = sent + np.clip(des - sent, -0.2, 0.2)
            mj_data.ctrl[:] = sent
            mj.mj_step(mj_model, mj_data)
            ring.append(np.r_[mj_data.qpos, mj_data.qvel])

        p_obs, _ = plant.step(raw_onnx.astype(np.float64)[None])
        py_state = np.r_[mj_data.qpos, mj_data.qvel]
        cur_state_diff = float(np.max(np.abs(py_state - plant.state[0])))
        max_state_diff = max(max_state_diff, cur_state_diff)
        if tk < 25:
            early_state_diff = max(early_state_diff, cur_state_diff)

    plant.close()
    res = dict(
        control_steps=45,
        early25_obs_74d_max_abs_diff=early_obs_diff,
        early25_state_57d_max_abs_diff=early_state_diff,
        obs_74d_max_abs_diff=max_obs_diff,
        state_57d_max_abs_diff=max_state_diff,
    )
    assert early_obs_diff < 1e-6 and early_state_diff < 1e-5, res
    assert max_obs_diff < 1e-4 and max_state_diff < 1e-3, res
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/pools"),
    )
    p.add_argument(
        "--roll",
        type=Path,
        default=RUN_V1 / "joined_roll_dev/roll.npz",
    )
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)

    env = Plant(64, out / "scene", delay=11)
    model = env.model
    geom_floor = model.geom("floor").id
    arm_pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True)}
    files = [
        Path.home() / "projects/Q1_deploy/real_logs" / n
        for n in ("q1_official_standup_from_mcap.npz", "q1_official_standup_faceup_20261120.npz")
    ]
    refs = [load_track(f, CONFIG["joint_names"]) for f in files]

    train_seeds = list(range(71001, 71026))
    val_seeds = list(range(71026, 71030))
    train_by_kind, train_side_md = collect_legal_starts(model, refs, train_seeds, arm_pairs, geom_floor)
    val_by_kind, val_side_md = collect_legal_starts(model, refs, val_seeds, arm_pairs, geom_floor)

    p2s_train, p2s_train_peaks, tr_reach, tr_tot = rollout_prone_to_supine(env, train_by_kind[0], args.roll)
    p2s_val, p2s_val_peaks, va_reach, va_tot = rollout_prone_to_supine(env, val_by_kind[0], args.roll)
    env.close()

    parity_res = verify_parity(out, train_by_kind[1][0])

    np.savez(out / "prone_train.npz", rows=train_by_kind[0])
    np.savez(out / "supine_train.npz", rows=train_by_kind[1])
    np.savez(out / "side_train.npz", rows=train_by_kind[2])
    np.savez(out / "prone_val.npz", rows=val_by_kind[0])
    np.savez(out / "supine_val.npz", rows=val_by_kind[1])
    np.savez(out / "side_val.npz", rows=val_by_kind[2])
    np.savez(out / "prone_to_supine_train.npz", rows=p2s_train)
    np.savez(out / "prone_to_supine_val.npz", rows=p2s_val)

    train_mds = sorted(x[3] for x in train_side_md)
    val_mds = sorted(x[3] for x in val_side_md)
    manifest = dict(
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        counts=dict(
            prone_train=len(train_by_kind[0]),
            supine_train=len(train_by_kind[1]),
            side_train=len(train_by_kind[2]),
            prone_val=len(val_by_kind[0]),
            supine_val=len(val_by_kind[1]),
            side_val=len(val_by_kind[2]),
            prone_to_supine_train=f"{tr_reach}/{tr_tot}",
            prone_to_supine_val=f"{va_reach}/{va_tot}",
        ),
        prone_roll_train_peaks_max=p2s_train_peaks.max(0).tolist(),
        prone_roll_val_peaks_max=p2s_val_peaks.max(0).tolist(),
        side_train_min_d_mm=dict(min=train_mds[0], p5=float(np.percentile(train_mds, 5)), median=float(np.median(train_mds))),
        side_val_min_d_mm=dict(min=val_mds[0], p5=float(np.percentile(val_mds, 5)), median=float(np.median(val_mds))),
        parity=parity_res,
        roll_npz_sha256=hashlib.sha256(args.roll.read_bytes()).hexdigest(),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
