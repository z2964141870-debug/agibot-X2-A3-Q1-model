"""Build self-contained zero-PyTorch/zero-MuJoCo Q1 FGP Control V2 Deploy Addon Bundle & Run Isolated Smoke Tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import onnxruntime as ort

SMHP_ROOT = Path("/home/yu/projects/SuperMimicHP")
REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_CANDIDATE = Path("/home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122")
RUN_FGP_V2 = SMHP_ROOT / "runs/q1_fgp_human_smpl_v2_20261122"
SCRIPT_DIR = Path(__file__).resolve().parent

SMOKE_TEST_SRC = '''"""Isolated zero-PyTorch, zero-MuJoCo smoke & regression verification for Q1 FGP Control V2 addon."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from q1_fgp_client_v2 import FGPHttpSourceV2, Q1FGPFallbackGuard
from q1_fgp_smpl_v2 import FGP_TO_Q1, FGPSMPLStreamV2, so3_exp_np
from q1_v2b_runtime import Q1V2bOnnxController


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _make_synthetic_fgp_response(
    seq: int,
    stamp_ns: int,
    *,
    pitch_deg: float = 0.0,
    drift_x: float = 0.0,
    status: str = "tracking",
    realtime_valid: bool = True,
) -> dict:
    # Build upright Q1-aligned SMPL24 rest pose then rotate by FGP_TO_Q1.T
    q1_to_fgp = FGP_TO_Q1.T
    joints_q1 = np.zeros((24, 3), dtype=np.float32)
    joints_q1[0] = [drift_x, 0.0, 0.94]     # pelvis
    joints_q1[1] = [drift_x, 0.09, 0.88]    # L_hip
    joints_q1[2] = [drift_x, -0.09, 0.88]   # R_hip
    joints_q1[4] = [drift_x, 0.11, 0.48]    # L_knee
    joints_q1[5] = [drift_x, -0.11, 0.48]   # R_knee
    joints_q1[7] = [drift_x, 0.12, 0.07]    # L_ankle (drifts together with pelvis in floor-skating bias!)
    joints_q1[8] = [drift_x, -0.12, 0.07]   # R_ankle
    joints_q1[9] = [drift_x, 0.0, 1.25]     # spine3 -> torso
    joints_q1[16] = [drift_x, 0.18, 1.38]   # L_shoulder
    joints_q1[18] = [drift_x, 0.22, 1.12]   # L_elbow
    joints_q1[20] = [drift_x + 0.10, 0.20, 0.92]  # L_wrist
    joints_q1[17] = [drift_x, -0.18, 1.38]  # R_shoulder
    joints_q1[19] = [drift_x, -0.22, 1.12]  # R_elbow
    joints_q1[21] = [drift_x + 0.10, -0.20, 0.92] # R_wrist

    R_pitch = so3_exp_np([0.0, math.radians(pitch_deg), 0.0])
    pel = joints_q1[0:1]
    joints_q1 = pel + (joints_q1 - pel) @ R_pitch.T
    joints_fgp = joints_q1 @ q1_to_fgp.T

    poses_q1 = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], 24, axis=0)
    poses_q1[0] = R_pitch
    poses_fgp = np.einsum("ij,njk,kl->nil", q1_to_fgp, poses_q1, FGP_TO_Q1)
    return {
        "status": status,
        "realtime_valid": realtime_valid,
        "session_id": "smoke_fgp_v2",
        "calibration_id": 1,
        "error": None,
        "sample": {
            "session_id": "smoke_fgp_v2",
            "calibration_id": 1,
            "sequence": seq,
            "source_ns": stamp_ns,
            "received_ns": stamp_ns,
            "output_ns": stamp_ns + 25_000_000,
            "floor_y_m": 0.0,
            "joints_m": joints_fgp.tolist(),
            "pose_mat": poses_fgp.tolist(),
        },
    }


def main() -> None:
    assert "torch" not in sys.modules, "Unexpected torch import"
    assert "mujoco" not in sys.modules, "Unexpected mujoco import"

    manifest_path = ROOT / "SHA256.json"
    require_manifest = "--require-manifest" in sys.argv
    if require_manifest:
        assert manifest_path.is_file(), f"Missing SHA256.json in {ROOT}"
    hashes = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for rel, exp in hashes.items():
        fp = ROOT / rel
        assert fp.is_file(), f"Missing bundle file: {fp}"
        assert _sha256(fp) == exp, f"SHA-256 mismatch on {rel}"

    cfg = json.loads((ROOT / "config/q1_deploy_config.json").read_text(encoding="utf-8"))
    for fname in ("q1_h002_token_head.onnx", "q1_tok_human_smpl.onnx", "q1_tok_planner_cmd.onnx", "q1_tok_robot_pose.onnx"):
        spec = cfg["onnx_files"][fname]
        fp = ROOT / "onnx" / fname
        assert _sha256(fp) == spec["sha256"], f"Config SHA-256 mismatch for {fname}"

    # 1. Runtime latency + OOB check for human_smpl (FGP-V2) & planner_cmd (1-thread CPU ORT)
    ctrl = Q1V2bOnnxController(ROOT, modalities=("human_smpl", "planner_cmd"))
    eff = np.array(cfg["effort_limit_nm"], dtype=np.float32)
    q0 = ctrl.default_q.copy()
    qd0 = np.zeros(22, dtype=np.float32)
    rng = np.random.default_rng(20261122)

    latencies = {}
    oob_total = 0
    for mod in ("human_smpl", "planner_cmd"):
        ctrl.reset(q0)
        dim = int(cfg["modality_input_dims"][mod])
        ms = []
        for si in range(250):
            obs80 = rng.normal(0.0, 0.15, size=(80,)).astype(np.float32)
            m_in = rng.normal(0.0, 0.25, size=(dim,)).astype(np.float32)
            t0 = time.perf_counter()
            q_safe, tau_est, _ = ctrl.step(obs80, m_in, q0, qd0, modality=mod)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            assert np.isfinite(q_safe).all() and np.isfinite(tau_est).all()
            oob_total += int(np.sum((np.abs(tau_est) - eff) > 1e-4))
            if si >= 50:
                ms.append(dt_ms)
        latencies[mod] = {
            "p50_ms": round(float(np.percentile(ms, 50)), 4),
            "p95_ms": round(float(np.percentile(ms, 95)), 4),
        }
    assert oob_total == 0, f"PD OOB detected: {oob_total}"

    # 2. Functional checks for FGPSMPLStreamV2 + FGPHttpSourceV2 + Q1FGPFallbackGuard
    src_startup = FGPHttpSourceV2("http://127.0.0.1:8090", 0.02)
    t_now_ns = time.monotonic_ns()
    for k in range(15):
        resp = _make_synthetic_fgp_response(seq=k + 1, stamp_ns=t_now_ns - (15 - k) * 33_333_333)
        src_startup.ingest(resp, arrival_ns=t_now_ns - (15 - k) * 20_000_000)
    assert src_startup.status()["fresh"] is True, "status().fresh must be True before first pop_paced()"

    src = FGPHttpSourceV2("http://127.0.0.1:8090", 0.02)
    base_ns = 1_000_000_000_000
    emitted = []
    outage_inject_idx = {25, 65, 105}
    for k in range(140):
        t_ns = base_ns + k * 33_333_333
        # Inject constant +11.0 deg FGP tilt bias and 0.12 m/s pure floor-skating drift
        resp = _make_synthetic_fgp_response(
            seq=k + 1, stamp_ns=t_ns, pitch_deg=11.0, drift_x=0.12 * (k * 0.033333),
        )
        src.ingest(resp, arrival_ns=t_ns)
        if k in outage_inject_idx:
            flicker = dict(resp, status="outage", realtime_valid=False)
            src.ingest(flicker, arrival_ns=t_ns + 10_000_000)
        # Consume at ~50Hz cadence
        now_p = t_ns
        tk_q, _, _ = src.pop_paced(now_ns=now_p)
        if tk_q is not None:
            emitted.append(tk_q.tick)

    assert src.stream.segment == 1, f"Transient outages broke segment: {src.stream.segment}"
    assert src.stream.events["ignored_transient_outages"] == 3, src.stream.events

    last_f = emitted[-1].feature.reshape(10, 43)[-1]
    pitch_after = math.degrees(math.asin(float(np.clip(-emitted[-1].root_rotation[2, 0], -1, 1))))
    torso_x_after = float(last_f[16])
    v_x_after = float(last_f[10] * 0.4213)
    assert abs(pitch_after) < 1.5, f"Tilt debias failed: pitch_after={pitch_after}"
    assert abs(torso_x_after) < 0.04, f"Torso debias failed: torso_x={torso_x_after}"
    assert abs(v_x_after) < 0.03, f"Anti-skating failed: v_x={v_x_after}"

    # Verify hold + fallback guard semantics across >220ms real disconnect
    guard = Q1FGPFallbackGuard()
    guard.reset("planner")
    b1, r1 = guard.resolve_branch("human_smpl", True)
    assert (b1, r1) == ("human_smpl", False)
    # Drain remaining buffered ticks (50 Hz consumption) within 220ms hold grace -> pop_paced enters decayed hold
    t_hold_ns = base_ns + 139 * 33_333_333 + 120_000_000
    hold_q, is_h = None, False
    for _ in range(src.max_buffer_ticks + 1):
        hold_q, _, is_h = src.pop_paced(now_ns=t_hold_ns)
        if is_h:
            break
    assert hold_q is not None and is_h is True
    # Beyond 220ms grace -> pop_paced returns None and guard triggers one-shot planner reanchor
    t_expired_ns = base_ns + 139 * 33_333_333 + 260_000_000
    exp_q, _, _ = src.pop_paced(now_ns=t_expired_ns)
    assert exp_q is None
    b2, r2 = guard.resolve_branch("human_smpl", False)
    assert (b2, r2) == ("planner", True)
    b3, r3 = guard.resolve_branch("human_smpl", False)
    assert (b3, r3) == ("planner", False)

    assert "torch" not in sys.modules
    assert "mujoco" not in sys.modules
    print(
        "FGP_V2_ADDON_SMOKE_PASS:",
        json.dumps(
            {
                "verified_bundle_files": len(hashes),
                "torch_loaded": False,
                "mujoco_loaded": False,
                "oob_total": oob_total,
                "latencies": latencies,
                "transient_outages_ignored": int(src.stream.events["ignored_transient_outages"]),
                "final_segment": int(src.stream.segment),
                "debiased_pitch_deg": round(pitch_after, 3),
                "debiased_torso_x": round(torso_x_after, 4),
                "antiskate_vx_m_s": round(v_x_after, 4),
                "fallback_guard_verified": True,
            }
        ),
    )


if __name__ == "__main__":
    main()
'''


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RUN_FGP_V2 / "deploy_bundle_fgp_v2_20261122")
    ap.add_argument("--zip-path", type=Path, default=RUN_FGP_V2 / "q1_fgp_control_addon_v2_20261122.zip")
    ap.add_argument("--smpl-onnx", type=Path, default=RUN_FGP_V2 / "onnx/q1_tok_human_smpl.onnx")
    ap.add_argument("--smpl-variant", type=str, default="fgp_v2_warmstart")
    ap.add_argument(
        "--handoff-md",
        type=Path,
        default=REPO_ROOT / "reports/DEPLOY_HANDOFF_Q1_FGP_CONTROL_V2_20261122.md",
    )
    ap.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "data/manifests/q1_fgp_deploy_addon_v2_manifest_20261122.json",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    zip_path = args.zip_path.resolve()
    assert args.smpl_onnx.is_file(), args.smpl_onnx
    assert args.handoff_md.is_file(), args.handoff_md

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("code", "config", "onnx", "robot_model"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy shared byte-identical runtime modules from Q1_deploy candidate + upgraded FGP V2 modules
    for fn in ("q1_v2b_runtime.py", "q1_cgfa_obs_np.py", "q1_kinematics_np.py"):
        shutil.copy2(DEPLOY_CANDIDATE / "code" / fn, out_dir / "code" / fn)
    shutil.copy2(SCRIPT_DIR / "q1_fgp_smpl_v2.py", out_dir / "code/q1_fgp_smpl_v2.py")
    shutil.copy2(SCRIPT_DIR / "q1_fgp_client_v2.py", out_dir / "code/q1_fgp_client_v2.py")
    (out_dir / "code/smoke_test_fgp_v2_addon.py").write_text(SMOKE_TEST_SRC, encoding="utf-8")

    # Copy frozen ONNX siblings + drop-in new human_smpl ONNX + robot XML
    for fn in ("q1_h002_token_head.onnx", "q1_tok_robot_pose.onnx", "q1_tok_planner_cmd.onnx"):
        shutil.copy2(DEPLOY_CANDIDATE / "onnx" / fn, out_dir / "onnx" / fn)
    shutil.copy2(args.smpl_onnx, out_dir / "onnx/q1_tok_human_smpl.onnx")
    shutil.copy2(DEPLOY_CANDIDATE / "robot_model/supermimic.xml", out_dir / "robot_model/supermimic.xml")

    cfg = json.loads((DEPLOY_CANDIDATE / "config/q1_deploy_config.json").read_text(encoding="utf-8"))
    smpl_sha = sha256_file(out_dir / "onnx/q1_tok_human_smpl.onnx")
    smpl_bytes = (out_dir / "onnx/q1_tok_human_smpl.onnx").stat().st_size
    cfg["onnx_files"]["q1_tok_human_smpl.onnx"]["bytes"] = smpl_bytes
    cfg["onnx_files"]["q1_tok_human_smpl.onnx"]["sha256"] = smpl_sha
    _sess = ort.InferenceSession(str(out_dir / "onnx/q1_tok_human_smpl.onnx"), providers=["CPUExecutionProvider"])
    cfg["onnx_files"]["q1_tok_human_smpl.onnx"]["metadata"] = dict(_sess.get_modelmeta().custom_metadata_map)
    cfg["fgp_v2"] = {
        "stream_class": "FGPSMPLStreamV2",
        "client_class": "FGPHttpSourceV2",
        "fallback_guard_class": "Q1FGPFallbackGuard",
        "smpl_onnx_variant": args.smpl_variant,
        "outage_grace_ms": 220.0,
        "hold_grace_ms": 220.0,
        "max_buffer_ticks": 3,
        "enable_tilt_debias": True,
        "tilt_ema_tau_s": 5.0,
        "max_pelvis_pitch_deg": 12.0,
        "max_pelvis_roll_deg": 10.0,
        "enable_vel_conditioning": True,
        "vel_stance_beta": 0.60,
        "vel_deadband_lo_m_s": 0.025,
        "vel_deadband_hi_m_s": 0.060,
        "sender_ramp_hz": 500,
        "sender_max_increment_rad_per_2ms": 0.2,
        "source": zip_path.name,
    }
    (out_dir / "config/q1_deploy_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.handoff_md, out_dir / "DEPLOY_HANDOFF.md")

    for d in list(out_dir.rglob("__pycache__")):
        shutil.rmtree(d, ignore_errors=True)
    rel_files = sorted(
        p.relative_to(out_dir).as_posix()
        for p in out_dir.rglob("*")
        if p.is_file()
        and p.name != "SHA256.json"
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    )
    hashes = {rel: sha256_file(out_dir / rel) for rel in rel_files}
    (out_dir / "SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=5) as zf:
        for rel in [*rel_files, "SHA256.json"]:
            zf.write(out_dir / rel, rel)
    zip_sha = sha256_file(zip_path)
    zip_bytes = zip_path.stat().st_size
    zip_path.with_suffix(".zip.sha256").write_text(f"{zip_sha}  {zip_path.name}\n", encoding="utf-8")

    # Isolated python -I check on extracted bundle (strict SHA256.json) + merged sim check (without SHA256.json)
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(zip_path, "r") as zf:
            assert zf.testzip() is None
            zf.extractall(tmp)
        raw_out = subprocess.check_output(
            [sys.executable, "-I", "code/smoke_test_fgp_v2_addon.py", "--require-manifest"],
            cwd=tmp,
            text=True,
        )
        os.remove(Path(tmp) / "SHA256.json")
        merged_out = subprocess.check_output(
            [sys.executable, "-I", "code/smoke_test_fgp_v2_addon.py"],
            cwd=tmp,
            text=True,
        )

    smoke_standalone = json.loads([ln for ln in raw_out.splitlines() if ln.startswith("FGP_V2_ADDON_SMOKE_PASS:")][-1].split(":", 1)[1].strip())
    smoke_merged = json.loads([ln for ln in merged_out.splitlines() if ln.startswith("FGP_V2_ADDON_SMOKE_PASS:")][-1].split(":", 1)[1].strip())

    manifest = {
        "schema_version": 1,
        "date": "2026-11-22",
        "smpl_onnx_variant": args.smpl_variant,
        "bundle_dir": str(out_dir),
        "zip_path": str(zip_path),
        "zip_bytes": zip_bytes,
        "zip_sha256": zip_sha,
        "zip_sha256_file": str(zip_path.with_suffix(".zip.sha256")),
        "components": hashes,
        "isolated_python_I_smoke_standalone": smoke_standalone,
        "isolated_python_I_smoke_merged_no_sha_json": smoke_merged,
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
