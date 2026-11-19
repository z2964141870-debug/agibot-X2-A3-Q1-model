"""Export H002-SONIC v3 5-file ONNX Suite (1 Shared TokenHead + 4 Wide Causal Tokenizers) & verify parity/latency."""

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
from multimodal_features import MODALITY_DIMS
from train_h002_sonic_v3_mjwarp import ModalityTokenizer
from unisupermimic.rl.algos.supermimic import networks
from unisupermimic.rl.algos.supermimic.networks import Policy
from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
from unisupermimic.tasks.motion_tracking.q1.supermimic.mdp import GOAL_FRAME_DIM, NUM_J


class SharedTokenHeadONNX(nn.Module):
    def __init__(self, pi: Policy):
        super().__init__()
        self.obs_norm = pi.obs_norm
        self.dec = pi.dec

    def forward(self, obs_hist: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        obs = self.obs_norm(obs_hist).flatten(-2)
        return self.dec(torch.cat((token, obs), dim=-1))


class RobotPoseTokenizerONNX(nn.Module):
    def __init__(self, pi: Policy):
        super().__init__()
        self.goal_norm = pi.goal_norm
        self.enc = pi.enc

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.enc(self.goal_norm(goal)))


def _export_one(mod: nn.Module, dummy: tuple[torch.Tensor, ...], in_names: list[str], out_names: list[str], path: Path, meta: dict[str, str]) -> None:
    dyn = {k: {0: "batch"} for k in in_names + out_names}
    torch.onnx.export(
        mod, dummy, str(path), export_params=True, opset_version=17, do_constant_folding=True,
        input_names=in_names, output_names=out_names, dynamic_axes=dyn, dynamo=False,
    )
    m = onnx.load(str(path))
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key = k
        p.value = str(v)
    onnx.save(m, str(path))
    onnx.checker.check_model(onnx.load(str(path)))


def export_all(bundle_path: Path, onnx_dir: Path, out_json: Path) -> dict:
    networks.AMP_DTYPE = None
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    args = bundle["args"]
    goal_frames = int(args["goal_frames"])
    hist = int(bundle["hist"])
    z_dim = int(bundle["z_dim"])
    obs_dim = int(bundle["obs_dim"])
    hidden = tuple(bundle["hidden"])
    fp = str(bundle["codec_fingerprint"])

    pi = Policy(obs_dim, GOAL_FRAME_DIM * goal_frames, num_joints=NUM_J, hist=hist, z_dim=z_dim, hidden=hidden)
    pi.load_state_dict(bundle["token_head_pi"])
    pi.eval()

    onnx_dir.mkdir(parents=True, exist_ok=True)
    meta_common = {"codec_fingerprint": fp, "ori_mode": "heading", "ctrl_hz": "50", "lookahead_ms": "0", "version": "v3"}

    head_mod = SharedTokenHeadONNX(pi).eval()
    head_path = onnx_dir / "q1_h002_v3_token_head.onnx"
    _export_one(
        head_mod,
        (torch.zeros((1, hist, obs_dim)), torch.zeros((1, z_dim))),
        ["obs_hist", "token"], ["action"], head_path, meta_common,
    )

    tok_mods: dict[str, tuple[nn.Module, int, Path]] = {
        "robot_pose": (RobotPoseTokenizerONNX(pi).eval(), MODALITY_DIMS["robot_pose"], onnx_dir / "q1_v3_tok_robot_pose.onnx"),
    }
    for mname, minfo in bundle["modal_tokenizers"].items():
        indim = int(minfo["in_dim"])
        mhid = tuple(minfo["hidden"])
        tm = ModalityTokenizer(indim, mhid, z_dim=z_dim)
        tm.load_state_dict(minfo["state_dict"])
        tm.eval()
        tok_mods[mname] = (tm, indim, onnx_dir / f"q1_v3_tok_{mname}.onnx")

    for mname, (tmod, indim, tpath) in tok_mods.items():
        _export_one(tmod, (torch.zeros((1, indim)),), ["modal_input"], ["token"], tpath, {**meta_common, "modality": mname})

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess_head = ort.InferenceSession(str(head_path), sess_options=opts, providers=["CPUExecutionProvider"])
    sess_toks = {m: ort.InferenceSession(str(p), sess_options=opts, providers=["CPUExecutionProvider"]) for m, (_, _, p) in tok_mods.items()}

    rng = np.random.default_rng(42)
    test_obs = rng.normal(0.0, 0.4, size=(64, hist, obs_dim)).astype(np.float32)
    report: dict = {"codec_fingerprint": fp, "onnx_dir": str(onnx_dir), "models": {}}

    kin = Q1Kinematics(str(ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"), "cpu")
    est = ContactGatedFootAnchor(kin, 1, dt=0.02, device="cpu")
    R_d = torch.eye(3).unsqueeze(0); w_d = torch.zeros((1, 3)); q_d = torch.zeros((1, NUM_J)); qd_d = torch.zeros((1, NUM_J)); p_d = torch.zeros((1, 2))
    est.reset(R_d, w_d, q_d, qd_d, p_d)
    obs_1 = test_obs[:1].copy()

    for mname, (tmod, indim, tpath) in tok_mods.items():
        test_in = rng.normal(0.0, 0.3, size=(64, indim)).astype(np.float32)
        with torch.no_grad():
            pt_z = tmod(torch.from_numpy(test_in))
            pt_a = head_mod(torch.from_numpy(test_obs), pt_z).numpy()
        ort_z = sess_toks[mname].run(["token"], {"modal_input": test_in})[0]
        ort_a = sess_head.run(["action"], {"obs_hist": test_obs, "token": ort_z})[0]
        max_diff = float(np.max(np.abs(pt_a - ort_a)))
        assert max_diff < 1e-5, f"Parity failed {mname}: {max_diff}"

        in_1 = test_in[:1]
        ms_full = []
        with torch.no_grad():
            for i in range(1200):
                t0 = time.perf_counter()
                exy, _ = est.step(R_d, w_d, q_d, qd_d, p_d, R_ref=R_d, mode="cgfa")
                obs_1[0, -1, 78:80] = exy[0].numpy()
                tok_out = sess_toks[mname].run(["token"], {"modal_input": in_1})[0]
                _ = sess_head.run(["action"], {"obs_hist": obs_1, "token": tok_out})[0]
                if i >= 200:
                    ms_full.append((time.perf_counter() - t0) * 1000.0)
        report["models"][mname] = {
            "tokenizer_onnx": str(tpath),
            "tokenizer_bytes": tpath.stat().st_size,
            "parity_action_max_abs_diff": max_diff,
            "cpu_cgfa_tok_head_p50_ms": float(np.percentile(ms_full, 50)),
            "cpu_cgfa_tok_head_p99_ms": float(np.percentile(ms_full, 99)),
        }
    report["token_head"] = {"onnx": str(head_path), "bytes": head_path.stat().st_size}
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SONIC_V3_ONNX_OK:", json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=base / "checkpoints/h002_sonic_v3_bundle.pt")
    parser.add_argument("--onnx-dir", type=Path, default=base / "onnx")
    parser.add_argument("--out-json", type=Path, default=base / "onnx_v3_verify.json")
    args = parser.parse_args()
    export_all(args.bundle.resolve(), args.onnx_dir.resolve(), args.out_json.resolve())


if __name__ == "__main__":
    main()
