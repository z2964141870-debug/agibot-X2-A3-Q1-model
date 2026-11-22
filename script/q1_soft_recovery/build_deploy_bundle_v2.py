"""Build, verify (parity + isolated python -I smoke + real-log audit), and zip the Q1 V2 deploy addon."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
Q1_DEPLOY = Path("/home/yu/projects/Q1_deploy")
RUN_V2 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121")
AUTON_ROOT = Path("/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery")
RUN_V1 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_20261121")
RISE_DIR = Path("/home/yu/projects/SuperMimicHP/runs/q1_supported_rise_20261121")

PLANNER_V2_ONNX = Path(
    "/home/yu/projects/SuperMimicHP/runs/h003_v2b_planner_cal_strafe_v2_20261121/onnx/q1_tok_planner_cmd.onnx"
)
RECOVERY_V2_ONNX = RUN_V2 / "unified_v3_08/expert.onnx"
OWNER_PY = SCRIPT_DIR / "q1_soft_recovery_v2_owner.py"

EXPECTED_PLANNER_V2_SHA = "ce0aa0a49666515e0b47dd200995fecc8980cd3bcdd41149fbc4824fb92679e5"
EXPECTED_RECOVERY_V2_SHA = "5d38cfcb3c31ad1103bd8897d4c8fad97472a04d5e258106732dd3a409b842ee"
EXPECTED_AUTON_OWNER_SHA = "5c99519c69d28e648fe47e84e18c63fdb9f35022c0bd2e30e7b25fb95be3a378"

SMOKE_TEST_SRC = '''"""Isolated zero-PyTorch/MuJoCo smoke test for Q1 Deploy V2 Addon (Lateral Strafe + Recovery V2)."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from q1_soft_recovery_v2_owner import DEFAULT_BLEND_TICKS, Q1SoftRecoveryV2Owner, check_v2_ownership
from q1_v2b_runtime import Q1V2bOnnxController


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    assert "torch" not in sys.modules, "Unexpected torch import in deploy runtime"
    assert "mujoco" not in sys.modules, "Unexpected mujoco import in deploy runtime"

    manifest_path = ROOT / "SHA256.json"
    require_manifest = "--require-manifest" in sys.argv
    if require_manifest:
        assert manifest_path.is_file(), f"Missing SHA256.json in {ROOT}"
    hashes = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for rel, exp in hashes.items():
        fp = ROOT / rel
        assert fp.is_file(), f"Missing bundle file: {fp}"
        got = _sha256(fp)
        assert got == exp, f"SHA-256 mismatch on {rel}: {got} != {exp}"

    cfg = json.loads((ROOT / "config/q1_deploy_config.json").read_text(encoding="utf-8"))
    for fname in ("q1_h002_token_head.onnx", "q1_tok_robot_pose.onnx", "q1_tok_planner_cmd.onnx"):
        spec = cfg["onnx_files"][fname]
        fp = ROOT / "onnx" / fname
        assert _sha256(fp) == spec["sha256"], f"Config SHA mismatch for {fname}"

    rec_cfg = cfg["recovery"]
    assert rec_cfg["enabled_by_default"] is False, "recovery.enabled_by_default must remain False"
    expert_path = ROOT / "onnx" / rec_cfg["expert_file"]
    assert _sha256(expert_path) == rec_cfg["expert_sha256"], "Recovery V2 expert SHA mismatch"

    # 1. PlannerCmd v2 runtime smoke test (250 steps, 1-thread CPU ORT)
    tracker = Q1V2bOnnxController(ROOT, modalities=("robot_pose", "planner_cmd"))
    eff = np.array(cfg["effort_limit_nm"], dtype=np.float32)
    q0 = tracker.default_q.copy()
    qd0 = np.zeros(22, dtype=np.float32)
    rng = np.random.default_rng(20261122)

    tracker.reset(q0)
    planner_lat = []
    planner_oob = 0
    for step_i in range(250):
        obs80 = rng.normal(0.0, 0.15, size=(80,)).astype(np.float32)
        m_in = rng.normal(0.0, 0.25, size=(cfg["modality_input_dims"]["planner_cmd"],)).astype(np.float32)
        t0 = time.perf_counter()
        q_safe, tau_est, _ = tracker.step(obs80, m_in, q0, qd0, modality="planner_cmd")
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert np.isfinite(q_safe).all() and np.isfinite(tau_est).all()
        planner_oob += int(np.sum((np.abs(tau_est) - eff) > 1e-4))
        if step_i >= 50:
            planner_lat.append(dt_ms)
    assert planner_oob == 0

    # 2. Deterministic ownership + timestamp/nonfinite/stale/offline regression checks (11/11)
    xml_path = ROOT / "robot_model/supermimic.xml"
    own_res = check_v2_ownership(tracker, expert_path, xml_path, blend_ticks=DEFAULT_BLEND_TICKS)
    assert all(bool(v) for v in own_res.values()), own_res

    # 3. Single-thread CPU latency & safety check for Q1SoftRecoveryV2Owner (teleop & recovery paths)
    owner = Q1SoftRecoveryV2Owner(tracker, expert_path, xml_path, blend_ticks=DEFAULT_BLEND_TICKS)
    obs80 = np.zeros(80, dtype=np.float32)
    modal_p = np.zeros(100, dtype=np.float32)
    prev = np.zeros(22, dtype=np.float32)
    R_up = np.eye(3, dtype=np.float32)
    R_sup = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    w0 = np.zeros(3, dtype=np.float32)

    def base_step():
        return tracker.step(obs80, modal_p, q0, qd0, modality="planner_cmd")

    def anchor_cb():
        tracker.reset(q0)

    owner.reset()
    tracker.reset(q0)
    teleop_lat = []
    rec_lat = []
    owner_oob = 0
    for k in range(250):
        t_s = (k + 1) * 0.02
        t0 = time.perf_counter()
        tgt, tau, ow = owner.step(R_up, w0, q0, qd0, prev, base_step, anchor_cb, sample_time_s=t_s, teleop_stamp_s=t_s)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert ow == 0.0 and np.isfinite(tgt).all() and np.isfinite(tau).all()
        owner_oob += int(np.sum((np.abs(tau) - eff) > 1e-4))
        if k >= 50:
            teleop_lat.append(dt_ms)

    owner.reset()
    tracker.reset(q0)
    for k in range(250):
        t_s = (k + 1) * 0.02
        t0 = time.perf_counter()
        tgt, tau, ow = owner.step(R_sup, w0, q0, qd0, prev, base_step, anchor_cb, sample_time_s=t_s, teleop_stamp_s=None)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        assert ow == 1.0 and np.isfinite(tgt).all() and np.isfinite(tau).all()
        owner_oob += int(np.sum((np.abs(tau) - eff) > 1e-4))
        if k >= 50:
            rec_lat.append(dt_ms)
    assert owner_oob == 0

    assert "torch" not in sys.modules, "Unexpected torch import after execution"
    assert "mujoco" not in sys.modules, "Unexpected mujoco import after execution"

    out = {
        "verified_bundle_files": len(hashes),
        "torch_loaded": "torch" in sys.modules,
        "mujoco_loaded": "mujoco" in sys.modules,
        "planner_cmd_v2": {
            "p50_ms": round(float(np.percentile(planner_lat, 50)), 4),
            "p95_ms": round(float(np.percentile(planner_lat, 95)), 4),
            "oob": planner_oob,
        },
        "recovery_owner_v2_teleop_mode": {
            "p50_ms": round(float(np.percentile(teleop_lat, 50)), 4),
            "p95_ms": round(float(np.percentile(teleop_lat, 95)), 4),
            "oob": owner_oob,
        },
        "recovery_owner_v2_active_mode": {
            "p50_ms": round(float(np.percentile(rec_lat, 50)), 4),
            "p95_ms": round(float(np.percentile(rec_lat, 95)), 4),
            "oob": owner_oob,
        },
        "ownership_checks": own_res,
    }
    print("ADDON_SMOKE_ALL_PASS:", json.dumps(out))


if __name__ == "__main__":
    main()
'''


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_parity_with_eval_v2(bundle_dir: Path) -> dict:
    """Assert 0.0 bit-for-bit parity between standalone Q1SoftRecoveryV2Owner and eval_v2.make_calm_controller(16)."""
    for p in reversed((str(SCRIPT_DIR), str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT))):
        if p not in sys.path:
            sys.path.insert(0, p)
    import eval_v2
    from eval_v2 import eval_core

    sys.path.insert(0, str(bundle_dir / "code"))
    from q1_soft_recovery_v2_owner import DEFAULT_BLEND_TICKS, Q1SoftRecoveryV2Owner
    from q1_v2b_runtime import Q1V2bOnnxController

    expert = bundle_dir / "onnx/q1_recovery_expert_v2.onnx"
    xml = bundle_dir / "robot_model/supermimic.xml"

    t_ref = Q1V2bOnnxController(bundle_dir, ("robot_pose", "planner_cmd"))
    t_new = Q1V2bOnnxController(bundle_dir, ("robot_pose", "planner_cmd"))
    ref_cls = eval_v2.make_calm_controller(blend_ticks=DEFAULT_BLEND_TICKS)
    # Restore eval_core cls after obtaining ref_cls just in case
    c_ref = ref_cls(t_ref, expert, xml)
    c_new = Q1SoftRecoveryV2Owner(t_new, expert, xml, blend_ticks=DEFAULT_BLEND_TICKS)

    q0 = t_ref.default_q.copy()
    zero = np.zeros_like(q0)
    w0 = np.zeros(3, dtype=np.float32)
    R_up = np.eye(3, dtype=np.float32)
    R_sup = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    R_tilt = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    modal_p = np.zeros(100, dtype=np.float32)
    obs80 = np.zeros(80, dtype=np.float32)

    max_tgt_diff = 0.0
    max_tau_diff = 0.0
    max_ow_diff = 0.0

    # Sequence exercising: teleop -> severe fall into protect -> protect exit -> rise (expert.onnx) ->
    # CalmUprightGate handoff into track -> 16-tick blend + leg speed braking (> 8.5 rad/s) -> handback
    c_ref.reset()
    c_new.reset()
    t_ref.reset(q0)
    t_new.reset(q0)
    prev_ref = np.zeros(22, dtype=np.float32)
    prev_new = np.zeros(22, dtype=np.float32)
    steps_tested = 0
    saw_protect = False
    saw_rise = False
    saw_blend_brake = False
    saw_handback = False

    def _run_tick(R, w, q, qd, t_s, stamp_s):
        nonlocal max_tgt_diff, max_tau_diff, max_ow_diff, prev_ref, prev_new, steps_tested
        tr, taur, owr = c_ref.step(
            R, w, q, qd, prev_ref,
            lambda: t_ref.step(obs80, modal_p, q, qd, modality="planner_cmd"),
            lambda: t_ref.reset(q),
            sample_time_s=t_s,
            teleop_stamp_s=stamp_s,
        )
        tn, taun, own = c_new.step(
            R, w, q, qd, prev_new,
            lambda: t_new.step(obs80, modal_p, q, qd, modality="planner_cmd"),
            lambda: t_new.reset(q),
            sample_time_s=t_s,
            teleop_stamp_s=stamp_s,
        )
        max_tgt_diff = max(max_tgt_diff, float(np.max(np.abs(tr - tn))))
        max_tau_diff = max(max_tau_diff, float(np.max(np.abs(taur - taun))))
        max_ow_diff = max(max_ow_diff, abs(float(owr) - float(own)))
        prev_ref = (tr - q0) / t_ref.action_scale
        prev_new = (tn - q0) / t_new.action_scale
        steps_tested += 1

    # 1. Upright teleop
    for k in range(5):
        t_s = k * 0.02
        _run_tick(R_up, w0, q0, zero, t_s, t_s)
        assert c_new.owner == "teleop"

    # 2. Sudden fall -> protect (age 0..0.36 s -> ticks k=5..23)
    for k in range(5, 23):
        t_s = k * 0.02
        _run_tick(R_tilt, np.array([1.5, 0.2, -0.3], np.float32), q0, zero, t_s, t_s)
        saw_protect = saw_protect or c_new.protecting

    # 3. Exit protect & run expert.onnx rise for 12 ticks
    for k in range(23, 36):
        t_s = k * 0.02
        _run_tick(R_sup, w0, q0, zero, t_s, t_s)
        saw_rise = saw_rise or (c_new.owner == "recovery" and c_new.rise.expert.gate.mode == "recover")

    # 4. Present calm upright posture -> CalmUprightGate fires after 4 ticks, then blend + leg brake!
    for k in range(36, 95):
        t_s = k * 0.02
        qd_k = zero.copy()
        if c_new.rise.expert.gate.mode == "track" and c_new.blend_index < 16 and not saw_blend_brake:
            qd_k[0] = 10.5
            qd_k[3] = -9.8
            saw_blend_brake = True
        _run_tick(R_up, w0, q0, qd_k, t_s, t_s)
        if c_new.owner == "teleop":
            saw_handback = True

    assert saw_protect and saw_rise and saw_blend_brake and saw_handback, (
        saw_protect,
        saw_rise,
        saw_blend_brake,
        saw_handback,
    )
    assert max_tgt_diff == 0.0 and max_tau_diff == 0.0 and max_ow_diff == 0.0, (
        max_tgt_diff,
        max_tau_diff,
        max_ow_diff,
    )
    return dict(
        steps_tested=steps_tested,
        phases_verified=dict(protect=saw_protect, rise=saw_rise, blend_and_leg_brake=saw_blend_brake, handback=saw_handback),
        max_target_abs_diff=max_tgt_diff,
        max_tau_abs_diff=max_tau_diff,
        max_owner_abs_diff=max_ow_diff,
    )


def audit_real_logs(bundle_dir: Path) -> dict:
    """Run read-only zero-false-trigger audit of Q1SoftRecoveryV2Owner on real hardware upright segments."""
    if str(bundle_dir / "code") not in sys.path:
        sys.path.insert(0, str(bundle_dir / "code"))
    from q1_kinematics_np import quat_xyzw_to_mat
    from q1_soft_recovery_v2_owner import DEFAULT_BLEND_TICKS, Q1SoftRecoveryV2Owner
    from q1_v2b_runtime import Q1V2bOnnxController

    tracker = Q1V2bOnnxController(bundle_dir, ("robot_pose", "planner_cmd"))
    owner = Q1SoftRecoveryV2Owner(
        tracker,
        bundle_dir / "onnx/q1_recovery_expert_v2.onnx",
        bundle_dir / "robot_model/supermimic.xml",
        blend_ticks=DEFAULT_BLEND_TICKS,
    )
    cfg = json.loads((bundle_dir / "config/q1_deploy_config.json").read_text(encoding="utf-8"))
    cfg_names = list(cfg["joint_names"])

    def dummy_base():
        return tracker.default_q.copy(), np.zeros(22, np.float32), {}

    def dummy_anchor():
        return None

    # 1. Real planner keyboard monitor hardware log (171.12 s, 8554 samples at ~50 Hz)
    jsonl_path = Q1_DEPLOY / "real_logs/q1_planner_keyboard_monitor_20261121_194500.jsonl"
    owner.reset()
    prev = np.zeros(22, dtype=np.float32)
    last_t = -1.0
    n_samples = 0
    takeovers_jsonl = 0
    max_fall_count = 0
    max_imminent_count = 0
    min_up_z = 1.0
    with jsonl_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            rec = json.loads(line)
            by_name = {j["name"]: j for j in rec["joints"]}
            q = np.array([by_name[n]["position"] for n in cfg_names], dtype=np.float32)
            qd = np.array([by_name[n]["velocity"] for n in cfg_names], dtype=np.float32)
            R = quat_xyzw_to_mat(np.array(rec["orientation_xyzw"], dtype=np.float32))
            w = np.array(rec["angular_velocity"], dtype=np.float32)
            t_s = float(rec["elapsed_s"])
            if t_s <= last_t:
                t_s = last_t + 0.001
            last_t = t_s
            _, _, ow = owner.step(R, w, q, qd, prev, dummy_base, dummy_anchor, sample_time_s=t_s, teleop_stamp_s=t_s)
            n_samples += 1
            min_up_z = min(min_up_z, float(R[2, 2]))
            max_fall_count = max(max_fall_count, int(owner.fall_count))
            max_imminent_count = max(max_imminent_count, int(owner.imminent_ticks))
            if ow != 0.0:
                takeovers_jsonl += 1
    assert takeovers_jsonl == 0, f"Unexpected false trigger in planner monitor log: {takeovers_jsonl}"

    # 2. Upright BIPED_WALK_RUN tails of both official real standup logs (sampled at 50 Hz)
    npz_audits = {}
    for stem in ("q1_official_standup_faceup_20261120", "q1_official_standup_from_mcap"):
        npz_p = Q1_DEPLOY / "real_logs" / f"{stem}.npz"
        meta = json.loads(npz_p.with_suffix(".json").read_text(encoding="utf-8"))
        d = np.load(npz_p)
        walk_t = next(t["t_rel_s"] for t in meta["transitions"] if t["desc"] == "BIPED_WALK_RUN")
        t_j = (d["joint_state_stamp"] - d["event_ns"].item()) * 1e-9
        t_i = (d["imu_stamp"] - d["event_ns"].item()) * 1e-9
        order = [d["joint_names"].tolist().index(n) for n in cfg_names]
        t_end = min(float(t_j[-1]), float(t_i[-1]))
        grid = np.arange(walk_t, t_end, 0.02)
        owner.reset()
        prev = np.zeros(22, dtype=np.float32)
        tks = 0
        min_uz = 1.0
        for t_s in grid:
            q = np.array([np.interp(t_s, t_j, d["q"][:, i]) for i in order], dtype=np.float32)
            qd = np.array([np.interp(t_s, t_j, d["qd"][:, i]) for i in order], dtype=np.float32)
            idx_i = int(np.clip(np.searchsorted(t_i, t_s), 0, len(t_i) - 1))
            R = quat_xyzw_to_mat(np.array(d["orientation_xyzw"][idx_i], dtype=np.float32))
            w = np.array(d["angular_velocity"][idx_i], dtype=np.float32)
            _, _, ow = owner.step(
                R, w, q, qd, prev, dummy_base, dummy_anchor, sample_time_s=float(t_s), teleop_stamp_s=float(t_s)
            )
            min_uz = min(min_uz, float(R[2, 2]))
            if ow != 0.0:
                tks += 1
        assert tks == 0, f"Unexpected false trigger in {stem} BIPED_WALK_RUN: {tks}"
        npz_audits[stem] = dict(samples_50hz=len(grid), start_s=walk_t, end_s=round(float(grid[-1]), 3), min_up_z=round(min_uz, 4), false_takeovers=tks)

    return {
        "q1_planner_keyboard_monitor_20261121_194500": {
            "samples": n_samples,
            "duration_s": round(last_t, 3),
            "min_up_z": round(min_up_z, 4),
            "max_fall_count": max_fall_count,
            "max_imminent_ticks": max_imminent_count,
            "false_takeovers": takeovers_jsonl,
        },
        "official_biped_walk_run_segments": npz_audits,
    }


def main() -> None:
    sys.dont_write_bytecode = True
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=RUN_V2 / "deploy_bundle_v2_20261122")
    ap.add_argument("--zip-path", type=Path, default=RUN_V2 / "q1_deploy_addon_lateral_and_recovery_v2_20261122.zip")
    ap.add_argument(
        "--handoff-md",
        type=Path,
        default=REPO_ROOT / "reports/DEPLOY_HANDOFF_Q1_LATERAL_AND_RECOVERY_V2_20261122.md",
    )
    ap.add_argument(
        "--manifest-out",
        type=Path,
        default=REPO_ROOT / "data/manifests/q1_deploy_addon_v2_manifest_20261122.json",
    )
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    zip_path = args.zip_path.resolve()
    assert out_dir.parent == RUN_V2.resolve(), out_dir

    # Verify source SHA-256 before copying
    assert sha256_file(PLANNER_V2_ONNX) == EXPECTED_PLANNER_V2_SHA
    assert sha256_file(RECOVERY_V2_ONNX) == EXPECTED_RECOVERY_V2_SHA
    auton_owner_src = AUTON_ROOT / "q1_autonomous_recovery.py"
    assert sha256_file(auton_owner_src) == EXPECTED_AUTON_OWNER_SHA
    assert OWNER_PY.is_file()
    assert args.handoff_md.is_file()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for sub in ("code", "config", "onnx", "robot_model"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    # Copy byte-identical runtime modules from existing Q1_deploy + AUTON owner + new V2 owner
    deploy_shared = (
        "q1_v2b_runtime.py",
        "q1_recovery_runtime.py",
        "recovery.py",
        "leg_height.py",
        "q1_cgfa_obs_np.py",
        "q1_kinematics_np.py",
    )
    for fn in deploy_shared:
        shutil.copy2(Q1_DEPLOY / "code" / fn, out_dir / "code" / fn)
    shutil.copy2(auton_owner_src, out_dir / "code/q1_autonomous_recovery.py")
    shutil.copy2(OWNER_PY, out_dir / "code/q1_soft_recovery_v2_owner.py")
    (out_dir / "code/smoke_test_deploy_v2_addon.py").write_text(SMOKE_TEST_SRC, encoding="utf-8")

    # Copy ONNX models + robot XML (keeping /home/yu/projects/Q1_deploy untouched)
    shutil.copy2(Q1_DEPLOY / "onnx/q1_h002_token_head.onnx", out_dir / "onnx/q1_h002_token_head.onnx")
    shutil.copy2(Q1_DEPLOY / "onnx/q1_tok_robot_pose.onnx", out_dir / "onnx/q1_tok_robot_pose.onnx")
    shutil.copy2(PLANNER_V2_ONNX, out_dir / "onnx/q1_tok_planner_cmd.onnx")
    shutil.copy2(RECOVERY_V2_ONNX, out_dir / "onnx/q1_recovery_expert_v2.onnx")
    shutil.copy2(Q1_DEPLOY / "robot_model/supermimic.xml", out_dir / "robot_model/supermimic.xml")

    # Import GATE_CONFIG defaults directly from bundled V2 owner module
    sys.path.insert(0, str(out_dir / "code"))
    from q1_soft_recovery_v2_owner import (
        DEFAULT_BLEND_TICKS,
        DEFAULT_LEG_BRAKE_GAIN,
        DEFAULT_LEG_SPEED_CAP,
        GATE_CONFIG,
    )

    cfg = json.loads((Q1_DEPLOY / "config/q1_deploy_config.json").read_text(encoding="utf-8"))
    cfg["onnx_files"]["q1_tok_planner_cmd.onnx"]["bytes"] = PLANNER_V2_ONNX.stat().st_size
    cfg["onnx_files"]["q1_tok_planner_cmd.onnx"]["sha256"] = EXPECTED_PLANNER_V2_SHA
    cfg["recovery"] = {
        "enabled_by_default": False,
        "expert_file": "q1_recovery_expert_v2.onnx",
        "expert_sha256": EXPECTED_RECOVERY_V2_SHA,
        "obs_dim": 74,
        "state_dim": 26,
        "owner_class": "Q1SoftRecoveryV2Owner",
        "calm_upright_gate": GATE_CONFIG,
        "blend_ticks": DEFAULT_BLEND_TICKS,
        "leg_speed_cap": DEFAULT_LEG_SPEED_CAP,
        "leg_brake_gain": DEFAULT_LEG_BRAKE_GAIN,
        "sender_ramp_hz": 500,
        "sender_max_increment_rad_per_2ms": 0.2,
        "fall_gate_ticks": 3,
        "stand_dwell_ticks": 25,
        "wide_stand_dwell_ticks": 40,
        "handoff_ticks": 30,
        "source": zip_path.name,
        "strict_acceptance_note": (
            "15/18 strict gates passed (unseen static 116/120 timely & 114/120 soft, "
            "unseen push 30/30 recovered); push impact [cap,cap+40) peaks exceed 12/10/2; "
            "keep enabled_by_default=false"
        ),
    }
    (out_dir / "config/q1_deploy_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(args.handoff_md, out_dir / "DEPLOY_HANDOFF.md")

    # Remove any Python bytecode caches and hash only real bundle files (except SHA256.json itself)
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

    # Build zip deterministically
    with ZipFile(zip_path, "w", ZIP_DEFLATED, compresslevel=5) as zf:
        for rel in [*rel_files, "SHA256.json"]:
            zf.write(out_dir / rel, rel)
    zip_sha = sha256_file(zip_path)
    zip_bytes = zip_path.stat().st_size
    zip_path.with_suffix(".zip.sha256").write_text(f"{zip_sha}  {zip_path.name}\n", encoding="utf-8")

    # Verification 1: Bit-for-bit parity vs eval_v2.make_calm_controller(16)
    parity_res = verify_parity_with_eval_v2(out_dir)

    # Verification 2: Isolated extraction + python -I execution of smoke_test_deploy_v2_addon.py
    with tempfile.TemporaryDirectory() as tmp:
        with ZipFile(zip_path, "r") as zf:
            assert zf.testzip() is None
            zf.extractall(tmp)
        raw_out = subprocess.check_output(
            [sys.executable, "-I", "code/smoke_test_deploy_v2_addon.py", "--require-manifest"],
            cwd=tmp,
            text=True,
        )
    smoke_line = [ln for ln in raw_out.splitlines() if ln.startswith("ADDON_SMOKE_ALL_PASS:")][-1]
    isolated_smoke = json.loads(smoke_line.split("ADDON_SMOKE_ALL_PASS:", 1)[1].strip())

    # Verification 3: Read-only real hardware upright logs zero-false-trigger audit
    real_audit = audit_real_logs(out_dir)

    manifest = {
        "schema_version": 1,
        "date": "2026-11-22",
        "bundle_dir": str(out_dir),
        "zip_path": str(zip_path),
        "zip_bytes": zip_bytes,
        "zip_sha256": zip_sha,
        "zip_sha256_file": str(zip_path.with_suffix(".zip.sha256")),
        "components": hashes,
        "bit_for_bit_parity_vs_eval_v2": parity_res,
        "isolated_python_I_smoke": isolated_smoke,
        "real_log_zero_false_trigger_audit": real_audit,
    }
    (out_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
