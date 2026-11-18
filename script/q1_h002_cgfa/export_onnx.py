"""Export M3 (H002 Full) Q1 causal teleop policy to single-file ONNX & verify CPU parity/latency."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cgfa_estimator import ContactGatedFootAnchor
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import GOAL_FRAME_DIM, NUM_J


class CausalTeleopActorONNX(nn.Module):
    def __init__(self, policy: Policy):
        super().__init__()
        self.obs_norm = policy.obs_norm
        self.goal_norm = policy.goal_norm
        self.enc = policy.enc
        self.dec = policy.dec

    def forward(self, obs_hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        z = torch.tanh(self.enc(self.goal_norm(goal)))
        obs = self.obs_norm(obs_hist).flatten(-2)
        return self.dec(torch.cat((z, obs), dim=-1))


def export_and_verify(ckpt_path: Path, onnx_path: Path, out_json: Path) -> dict:
    networks.AMP_DTYPE = None
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    args = ck["args"]
    goal_frames = int(args["goal_frames"])
    hist = int(args["hist"])
    z_dim = int(args["z_dim"])
    hidden = tuple(args.get("test_hidden_dims") or args["policy_hidden"])
    obs_dim = int(ck["pi"]["obs_norm.mean"].numel())
    goal_dim = GOAL_FRAME_DIM * goal_frames

    pi = Policy(obs_dim, goal_dim, num_joints=NUM_J, hist=hist, z_dim=z_dim, hidden=hidden)
    pi.load_state_dict(ck["pi"])
    pi.eval()

    actor = CausalTeleopActorONNX(pi).eval()
    dummy_obs = torch.zeros((1, hist, obs_dim), dtype=torch.float32)
    dummy_goal = torch.zeros((1, goal_dim), dtype=torch.float32)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        (dummy_obs, dummy_goal),
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["obs_hist", "goal"],
        output_names=["action"],
        dynamic_axes={"obs_hist": {0: "batch"}, "goal": {0: "batch"}, "action": {0: "batch"}},
        dynamo=False,
    )
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    test_obs = rng.normal(0.0, 0.5, size=(64, hist, obs_dim)).astype(np.float32)
    test_goal = rng.normal(0.0, 0.3, size=(64, goal_dim)).astype(np.float32)
    with torch.no_grad():
        pt_out = actor(torch.from_numpy(test_obs), torch.from_numpy(test_goal)).numpy()
    ort_out = sess.run(["action"], {"obs_hist": test_obs, "goal": test_goal})[0]
    max_abs_diff = float(np.max(np.abs(pt_out - ort_out)))
    mean_abs_diff = float(np.mean(np.abs(pt_out - ort_out)))
    assert max_abs_diff < 1e-5, f"ONNX parity failed: max_abs_diff={max_abs_diff}"

    # Benchmark CPU latency (pure ONNX + full ONNX+CGFA step)
    kin = Q1Kinematics(str(ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"), "cpu")
    est = ContactGatedFootAnchor(kin, 1, dt=0.02, device="cpu")
    R_dummy = torch.eye(3).unsqueeze(0)
    w_dummy = torch.zeros((1, 3))
    q_dummy = torch.zeros((1, NUM_J))
    qd_dummy = torch.zeros((1, NUM_J))
    p_ref_dummy = torch.zeros((1, 2))
    est.reset(R_dummy, w_dummy, q_dummy, qd_dummy, p_ref_dummy)

    obs_1 = test_obs[:1].copy()
    goal_1 = test_goal[:1].copy()
    pure_ms, full_ms = [], []
    with torch.no_grad():
        for i in range(2200):
            t0 = time.perf_counter()
            est_xy, _ = est.step(R_dummy, w_dummy, q_dummy, qd_dummy, p_ref_dummy, mode="cgfa")
            obs_1[0, -1, 78:80] = est_xy[0].numpy()
            t1 = time.perf_counter()
            act = sess.run(["action"], {"obs_hist": obs_1, "goal": goal_1})[0]
            _ = np.clip(act, -4.0, 4.0)
            t2 = time.perf_counter()
            if i >= 200:
                pure_ms.append((t2 - t1) * 1000.0)
                full_ms.append((t2 - t0) * 1000.0)

    report = {
        "checkpoint": str(ckpt_path),
        "onnx_path": str(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "parity_max_abs_diff": max_abs_diff,
        "parity_mean_abs_diff": mean_abs_diff,
        "cpu_onnx_only_p50_ms": float(np.percentile(pure_ms, 50)),
        "cpu_onnx_only_p95_ms": float(np.percentile(pure_ms, 95)),
        "cpu_onnx_only_p99_ms": float(np.percentile(pure_ms, 99)),
        "cpu_full_cgfa_onnx_p50_ms": float(np.percentile(full_ms, 50)),
        "cpu_full_cgfa_onnx_p95_ms": float(np.percentile(full_ms, 95)),
        "cpu_full_cgfa_onnx_p99_ms": float(np.percentile(full_ms, 99)),
        "budget_50hz_ms": 20.0,
    }
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ONNX_EXPORT_OK:", json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parent
    parser.add_argument("--ckpt", type=Path, default=base / "checkpoints/m3_h002_full_0ms.pt")
    parser.add_argument("--onnx", type=Path, default=base / "q1_h002_0ms.onnx")
    parser.add_argument("--out-json", type=Path, default=base / "onnx_verify.json")
    args = parser.parse_args()
    export_and_verify(args.ckpt.resolve(), args.onnx.resolve(), args.out_json.resolve())


if __name__ == "__main__":
    main()
