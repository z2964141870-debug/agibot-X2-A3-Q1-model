"""Stage C: Export a unified single stateful ONNX combining prone roll, SideV3Combined, supine rise, and soft landing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

RUN_V1 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_20261121")
RISE_DIR = Path("/home/yu/projects/SuperMimicHP/runs/q1_supported_rise_20261121")
AUTON_ROOT = Path("/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery")
SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from command_support import sample_record
from common import HI, LO, STAND
from train_side_v3 import SideV3Combined
from train_supine_v2 import SupineRiseNet


class UnifiedRecoveryV2(nn.Module):
    """Single stateful 50 Hz recovery expert without legacy 5 ms kick code."""

    def __init__(self, roll_fit, side_ckpt: dict, supine_state_dict: dict, faceup_cmd: np.ndarray):
        super().__init__()
        self.register_buffer("stand", torch.tensor(STAND, dtype=torch.float32))
        self.register_buffer("lo", torch.tensor(LO, dtype=torch.float32))
        self.register_buffer("hi", torch.tensor(HI, dtype=torch.float32))
        self.register_buffer("roll_keys", torch.tensor(roll_fit["keys"], dtype=torch.float32))
        self.roll_duration = float(roll_fit["duration"])

        bias = np.zeros(22, dtype=np.float32)
        bias[14] = -0.08
        bias[15] = 0.08
        bias[18] = -0.08
        bias[19] = -0.08
        self.register_buffer("arm_clearance_bias", torch.tensor(bias, dtype=torch.float32))

        self.side_net = SideV3Combined(faceup_cmd[0])
        self.side_net.landing.load_state_dict(side_ckpt["landing"])
        self.side_net.roll.load_state_dict(side_ckpt["roll"])

        self.supine_net = SupineRiseNet(faceup_cmd)
        self.supine_net.load_state_dict(supine_state_dict)

    def forward(self, rec_obs_hist: torch.Tensor, rec_state: torch.Tensor):
        x = rec_obs_hist[:, -1]
        q = x[:, :22] + self.stand
        qd = x[:, 22:44] * 10.0
        w_norm = torch.linalg.vector_norm(x[:, 44:47] * 4.0, dim=1)
        up_x = -x[:, 47]
        up_y = -x[:, 48]
        up_z = -x[:, 49]
        mode_in = rec_state[:, 1]
        zero_col = mode_in * 0.0

        # Dispatch strictly from mode == 0 using sensed gravity & body angular velocity.
        is_disp = mode_in == 0
        d_soft = is_disp & ((up_z > 0.35) | (w_norm > 2.5))
        d_side = is_disp & (~d_soft) & (up_y.abs() > 0.65) & (up_y.abs() > up_x.abs())
        d_prone = is_disp & (~d_soft) & (~d_side) & (up_x < -0.60)
        d_supine = is_disp & (~d_soft) & (~d_side) & (~d_prone)

        # Mode 12: Soft Landing (mid-air fall / violent tumble).
        act_12 = d_soft | (mode_in == 12)
        age_12 = torch.where(d_soft, zero_col, rec_state[:, 0])
        t_12 = (q - 0.04 * qd + self.arm_clearance_bias[None]).clamp(self.lo, self.hi)
        landed_12 = (up_z.abs() < 0.35) & (w_norm < 1.6)
        cnt_12 = torch.where(landed_12, torch.where(d_soft, zero_col, rec_state[:, 2]) + 1.0, zero_col)
        exit_12 = (cnt_12 >= 5.0) | (age_12 >= 40.0)
        state_12_hold = torch.cat((torch.stack((age_12 + 1.0, zero_col + 12.0, cnt_12, zero_col), 1), rec_state[:, 4:]), 1)
        state_12 = torch.where(exit_12[:, None], torch.zeros_like(rec_state), state_12_hold)

        # Mode 10: Decoupled SideV3Combined -> SupineRiseNet (mode 6) or roll.npz (mode 8) if facedown at tick 12.
        act_10 = d_side | (mode_in == 10)
        age_10 = torch.where(d_side, zero_col, rec_state[:, 0])
        side_10 = torch.where(d_side, (x[:, 48] < 0).float(), rec_state[:, 2]).clamp(0.0, 1.0)
        orig_10 = torch.where((d_side | (age_10 == 12.0))[:, None], q, rec_state[:, 4:])
        t_10 = self.side_net.step_components(rec_obs_hist, age_10, side_10, orig_10)
        prone_fb_10 = (age_10 >= 12.0) & (up_x < -0.75)
        quiet_10 = (age_10 >= 12.0) & (~prone_fb_10) & (up_x > 0.84) & (w_norm < 1.45)
        qcnt_10 = torch.where(quiet_10, torch.where(d_side | (age_10 == 12.0), zero_col, rec_state[:, 3]) + 1.0, zero_col)
        done_10 = qcnt_10 >= 8.0
        timeout_10 = age_10 >= 150.0
        state_10_roll = torch.cat((torch.stack((age_10 + 1.0, zero_col + 10.0, side_10, qcnt_10), 1), orig_10), 1)
        state_10_prone = torch.cat((torch.stack((zero_col, zero_col + 8.0, zero_col, zero_col), 1), q), 1)
        state_10_supine = torch.cat((torch.stack((zero_col, zero_col + 6.0, zero_col + 1.0, zero_col), 1), q), 1)
        state_10 = torch.where(
            prone_fb_10[:, None],
            state_10_prone,
            torch.where(
                done_10[:, None],
                state_10_supine,
                torch.where(timeout_10[:, None], torch.zeros_like(rec_state), state_10_roll),
            ),
        )

        # Mode 8: Verified Prone Roll -> SupineRiseNet (mode 6).
        act_8 = d_prone | (mode_in == 8)
        tick_8 = torch.where(d_prone, zero_col, rec_state[:, 3])
        orig_8 = torch.where(d_prone[:, None], q, rec_state[:, 4:])
        phase_8 = ((tick_8 + 1.0) / self.roll_duration).clamp(0.0, 5.0)
        idx_8 = phase_8.long().clamp(0, 4)
        prev_wp_8 = torch.where((idx_8 == 0)[:, None], orig_8, self.roll_keys[(idx_8 - 1).clamp(0, 4)])
        blend_8 = 0.5 - 0.5 * torch.cos(torch.pi * (phase_8 - idx_8.float()))
        t_8 = (prev_wp_8 + blend_8[:, None] * (self.roll_keys[idx_8] - prev_wp_8)).clamp(self.lo, self.hi)
        quiet_8 = (up_x > 0.80) & (w_norm < 1.7)
        qcnt_8 = torch.where(quiet_8, torch.where(d_prone, zero_col, rec_state[:, 2]) + 1.0, zero_col)
        done_8 = qcnt_8 >= 10.0
        timeout_8 = tick_8 >= 180.0
        state_8_roll = torch.cat((torch.stack((tick_8 + 1.0, zero_col + 8.0, qcnt_8, tick_8 + 1.0), 1), orig_8), 1)
        state_8_supine = torch.cat((torch.stack((zero_col, zero_col + 6.0, zero_col + 1.0, zero_col), 1), q), 1)
        state_8 = torch.where(
            done_8[:, None],
            state_8_supine,
            torch.where(timeout_8[:, None], torch.zeros_like(rec_state), state_8_roll),
        )

        # Mode 6: Supine Rise (Stage B Closed-loop Residual Policy).
        pt_6 = torch.where(d_supine, zero_col, rec_state[:, 3])
        orig_6 = torch.where(d_supine[:, None], q, rec_state[:, 4:])
        t_6 = self.supine_net.target_from_action(self.supine_net.raw_mean(rec_obs_hist, pt_6, orig_6), pt_6, orig_6)
        next_pt_6 = (pt_6 + 1.0).clamp(max=10000.0)
        retry_6 = (pt_6 >= 500.0) & (up_z < 0.50)
        state_6_run = torch.cat((torch.stack((next_pt_6, zero_col + 6.0, zero_col + 1.0, next_pt_6), 1), orig_6), 1)
        state_6 = torch.where(retry_6[:, None], torch.zeros_like(rec_state), state_6_run)

        target = torch.where(
            act_12[:, None],
            t_12,
            torch.where(act_10[:, None], t_10, torch.where(act_8[:, None], t_8, t_6)),
        )
        next_state = torch.where(
            act_12[:, None],
            state_12,
            torch.where(act_10[:, None], state_10, torch.where(act_8[:, None], state_8, state_6)),
        )
        return target.clamp(self.lo, self.hi), next_state


def verify_onnx_and_benchmark(model: UnifiedRecoveryV2, onnx_path: Path):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    # Probe dispatch + ongoing across all 4 modes including SideLanding (age<12) and SideSettledRoll (age>=12) and prone fallback.
    probes = [
        ("soft_landing", -0.1, 0.0, -0.8, 0.0, [0.0] * 26),
        ("side_right", 0.0, -0.98, 0.0, 0.0, [0.0] * 26),
        ("side_left", 0.0, 0.98, 0.0, 0.0, [0.0] * 26),
        ("side_tk12_ok", 0.0, -0.90, 0.0, 0.0, [12.0, 10.0, 1.0, 0.0] + list(STAND)),
        ("side_tk12_fd", 0.80, -0.50, 0.0, 0.0, [12.0, 10.0, 1.0, 0.0] + list(STAND)),
        ("side_cont", -0.2, 0.90, 0.0, 0.0, [20.0, 10.0, 0.0, 0.0] + list(STAND)),
        ("prone", 0.98, 0.0, 0.0, 0.0, [0.0] * 26),
        ("prone_cont", -0.90, 0.0, 0.0, 0.0, [40.0, 8.0, 10.0, 40.0] + list(STAND)),
        ("supine", -0.98, 0.0, 0.0, 0.0, [0.0] * 26),
        ("supine_mid", 0.90, 0.0, -0.3, 0.0, [260.0, 6.0, 1.0, 260.0] + list(STAND)),
    ]
    max_diff = 0.0
    for name, o47, o48, o49, w_val, st in probes:
        h = torch.zeros((1, 10, 74), dtype=torch.float32)
        h[0, :, 44] = w_val
        h[0, :, 47] = o47
        h[0, :, 48] = o48
        h[0, :, 49] = o49
        s = torch.tensor([st], dtype=torch.float32)
        with torch.no_grad():
            pt_t, pt_s = model(h, s)
        ox_t, ox_s = sess.run(["joint_target", "next_state"], {"rec_obs_hist": h.numpy(), "rec_state": s.numpy()})
        diff = max(float(np.max(np.abs(pt_t.numpy() - ox_t))), float(np.max(np.abs(pt_s.numpy() - ox_s))))
        max_diff = max(max_diff, diff)
        print(f"  parity[{name:12s}] -> mode={int(ox_s[0,1]):2d} max_abs_diff={diff:.3e}")
    assert max_diff < 1e-5, f"Unified ONNX parity exceeded 1e-5: {max_diff}"

    h_np = np.zeros((1, 10, 74), dtype=np.float32)
    h_np[0, :, 49] = -1.0
    s_np = np.array([[260.0, 6.0, 1.0, 260.0] + list(STAND)], dtype=np.float32)
    for _ in range(50):
        sess.run(["joint_target", "next_state"], {"rec_obs_hist": h_np, "rec_state": s_np})
    times_ms = []
    for _ in range(500):
        t0 = time.perf_counter()
        sess.run(["joint_target", "next_state"], {"rec_obs_hist": h_np, "rec_state": s_np})
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    return dict(
        max_abs_diff=max_diff,
        cpu_1t_p50_ms=float(np.percentile(times_ms, 50)),
        cpu_1t_p95_ms=float(np.percentile(times_ms, 95)),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roll", type=Path, default=RUN_V1 / "joined_roll_dev/roll.npz")
    p.add_argument(
        "--side-ckpt",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/side_v2_04/best_side_v3.pt"),
    )
    p.add_argument(
        "--supine-ckpt",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/supine_v2_01/best.pt"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/unified_v2"),
    )
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    roll_fit = np.load(args.roll)
    side_ckpt = torch.load(args.side_ckpt, map_location="cpu", weights_only=False)
    supine_ckpt = torch.load(args.supine_ckpt, map_location="cpu", weights_only=False)
    faceup_cmd = sample_record("q1_official_standup_faceup_20261120")["command_position"]

    model = UnifiedRecoveryV2(roll_fit, side_ckpt, supine_ckpt["actor"], faceup_cmd).cpu().eval()
    onnx_path = out / "expert.onnx"
    sample_h = torch.zeros((1, 10, 74), dtype=torch.float32)
    sample_h[:, :, 49] = -1.0
    torch.onnx.export(
        model,
        (sample_h, torch.zeros((1, 26), dtype=torch.float32)),
        str(onnx_path),
        input_names=["rec_obs_hist", "rec_state"],
        output_names=["joint_target", "next_state"],
        opset_version=17,
        dynamo=False,
    )
    stats = verify_onnx_and_benchmark(model, onnx_path)
    meta = dict(
        onnx=str(onnx_path),
        onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        roll_npz=str(args.roll),
        roll_sha256=hashlib.sha256(args.roll.read_bytes()).hexdigest(),
        side_ckpt=str(args.side_ckpt),
        side_ckpt_sha256=hashlib.sha256(args.side_ckpt.read_bytes()).hexdigest(),
        supine_ckpt=str(args.supine_ckpt),
        supine_ckpt_sha256=hashlib.sha256(args.supine_ckpt.read_bytes()).hexdigest(),
        verification=stats,
    )
    (out / "export_manifest.json").write_text(json.dumps(meta, indent=2))
    print("UNIFIED V2 EXPORTED:", json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
