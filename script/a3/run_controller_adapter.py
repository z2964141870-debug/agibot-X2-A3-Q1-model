"""Guarded E20 paired supervised adapters and matched causal simulation."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from script.a3.autoresume import monitor_child
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.controller_adapter_trial import DEVICE, EXPERIMENT, OUTPUT
from script.a3.corrected_trial import OFFICIAL
from script.a3.evaluate_mujoco import evaluate_explicit
from script.a3.evaluation_common import checkpoint_metadata, preflight
from script.a3.fullchain_support import DATA, ROOT, VENDOR


LOGS = ROOT / f"logs/a3_controller_adapter_20261010_{EXPERIMENT}"
BASELINE = ROOT / "data/experiments/a3_causal_reference_20261010_E18"
REFERENCES = ROOT / "data/experiments/a3_joint_ablation_20261010_E16/references"


def register():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / "split.json").read_text())
    if len(split["train"]) != 16 or len(split["heldout"]) != 4:
        raise ValueError("Unexpected fixed motion split")
    if any(sha256(Path(row["path"])) != row["sha256"] for row in split["train"]+split["heldout"]):
        raise ValueError("Source motion identity changed")
    spec = dict(experiment=EXPERIMENT, training_device=str(DEVICE),
        type="PAIRED_SUPERVISED_REFERENCE_ADAPTER_NOT_PPO", base_policy_updates=0,
        base_policy_sha256=checkpoint_metadata(OFFICIAL)["sha256"], split_sha256=sha256(DATA / "split.json"),
        train_motions=split["train"], diagnostic_motions=split["heldout"], teacher_diagnostic_e19_used=False,
        code_sha256={name: sha256(ROOT / "script/a3" / name) for name in
            ("controller_adapter_trial.py", "run_controller_adapter.py", "verify_controller_adapter.py",
             "joint_forecast_replay.py", "causal_orientation.py", "sim_entry.py", "evaluate_mujoco.py")},
        baseline_sha256=sha256(BASELINE / "result.json"), objectives=["mse", "latent"],
        architecture="ZERO_LINEAR_CV_RESIDUAL_145_TO_290", steps=200, smoke_steps=2,
        seed=0, batch_size=256, learning_rate=1e-4, optimizer="AdamW", weight_decay=0, gradient_clip=1,
        noise_std_rad=[0., .01], history_frames=6, horizons_ms=list(range(20, 201, 20)),
        train_anchor="ARRIVED_REFERENCE_PELVIS_NOT_ROLLOUT_STATE", teacher_orientation="SAME_CAUSAL_AS_STUDENT",
        gate=dict(no_new_falls=True, each_motion_joint_no_worse=True, mean_joint_improvement_min_percent=1.,
            each_motion_body_and_root_regression_max_percent=5., action_delta_regression_max_percent=5.,
            both_noise_conditions_required=True, scope="REPEATED_DIAGNOSTIC_ONLY_NOT_FINAL_ACCEPTANCE"),
        initialization="MATCHED_OFFLINE_OFFICIAL_RESET_CAUSAL_INITIALIZATION_UNVERIFIED",
        max_attempts=None, expires_at=None, max_no_progress=2, cooldown_seconds=300,
        max_cpu_c=90, max_gpu_c=85, sample_seconds=5, cpu_quota_percent=100, backup_status="LOCAL_ONLY")
    path = OUTPUT / "job.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Adapter registration changed")
    if not path.exists():
        atomic_json(path, spec)
    return spec


def progress():
    checkpoints = sum(len(list((OUTPUT / objective).glob("adapter_step_*.json"))) for objective in ("mse", "latent"))
    motions = sum(sum(row.get("status") == "passed" for row in json.loads(path.read_text())["runs"].values())
                 for path in OUTPUT.glob("*/explicit_manifest.json"))
    return checkpoints+motions+int((OUTPUT / "training_windows.json").exists())


def run_child(module, arguments, name, ordinal, guard):
    preflight()
    directory = LOGS / name / f"attempt_{ordinal:03d}"
    directory.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    with (directory / "console.log").open("w") as stream:
        child = subprocess.Popen([str(ROOT / "data/environments/a3-sonic/bin/python"), "-m", module]+arguments,
            cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        code, reason = monitor_child(child, directory, guard)
    if code != 0 or reason is not None:
        raise RuntimeError(f"{name}: returncode={code}, guard={reason}; see {directory}")


def args_for(objective, sigma, smoke=False):
    key = f"smoke_{objective}" if smoke else f"{objective}_noise{sigma:g}_seed0"
    return SimpleNamespace(output=OUTPUT / key, logs=LOGS / key, checkpoint=OFFICIAL,
        motion=VENDOR / "a3_data/agibot_a3/001_walk_front_slow.csv" if smoke else REFERENCES,
        reference_fps=30., frame_stride=4, max_policy_steps=10 if smoke else None,
        action_delay_ms=0., reference_buffer_ms=0., capture_inputs=True, capture_count=100000,
        summarize_only=False, joint_reference_mode="cv_residual",
        joint_predictor=OUTPUT / objective / f"adapter_step_{0 if smoke else 200:06d}.npz",
        joint_noise_std=sigma, joint_noise_seed=0, causal_orientation=True)


def summarize():
    results = {}
    baseline_results = json.loads((BASELINE / "result.json").read_text())["results"]
    for objective in ("mse", "latent"):
        for sigma in (0., .01):
            key = f"{objective}_noise{sigma:g}_seed0"
            rows = json.loads((OUTPUT / key / "explicit_summary.json").read_text())["motions"]
            baseline = baseline_results[f"cv_noise{sigma:g}_seed0"]
            total, count, motions, reasons = 0., 0, [], []
            for row in rows:
                name = Path(row["motion"]).stem
                previous = next(m for m in baseline["motions"] if Path(m["motion"]).stem == name)
                metric, old = row["windows"]["full_rollout"], previous["metrics"]
                if not row["full_replay_completed"] or metric["policy_steps"] != old["policy_steps"]:
                    raise ValueError("Incomplete or unmatched rollout")
                n = metric["policy_steps"]
                total += metric["joint_rmse_rad"]**2*n
                count += n
                if row["fall"] and not previous["fall"]:
                    reasons.append(f"{name}:new_fall")
                if metric["joint_rmse_rad"] > old["joint_rmse_rad"]+1e-12:
                    reasons.append(f"{name}:joint_regression")
                for field in ("root_pos_error_m_mean", "root_quat_error_deg_mean", "torso_world_mean_m",
                              "wrists_world_mean_m", "legs_world_mean_m", "torso_anchor_local_mean_m",
                              "wrists_anchor_local_mean_m", "legs_anchor_local_mean_m"):
                    if metric[field] > old[field]*1.05+1e-12:
                        reasons.append(f"{name}:{field}_regression")
                with (OUTPUT / key / f"{name}.timeseries.json").open() as stream:
                    trace = json.load(stream)
                with (BASELINE / f"cv_noise{sigma:g}_seed0" / f"{name}.timeseries.json").open() as stream:
                    old_trace = json.load(stream)
                action = np.asarray(trace["motions"][0]["raw_action_29"], dtype=float)
                old_action = np.asarray(old_trace["motions"][0]["raw_action_29"], dtype=float)
                delta = float(np.sqrt(np.mean(np.diff(action, axis=0)**2)))
                old_delta = float(np.sqrt(np.mean(np.diff(old_action, axis=0)**2)))
                if delta > old_delta*1.05+1e-12:
                    reasons.append(f"{name}:action_delta_regression")
                motions.append(dict(motion=row["motion"], fall=row["fall"], first_fall_s=row["first_fall_s"],
                    metrics=metric, windows=row["windows"], baseline_metrics=old,
                    joint_change_percent=100*(metric["joint_rmse_rad"]/old["joint_rmse_rad"]-1),
                    action_delta_rms=delta, baseline_action_delta_rms=old_delta))
            rmse = float(np.sqrt(total/count))
            change = 100*(rmse/baseline["joint_rmse_rad"]-1)
            if change > -1.:
                reasons.append("mean_joint_improvement_below_1_percent")
            if len(motions) != 4:
                raise ValueError("Expected four diagnostic motions")
            results[key] = dict(joint_rmse_rad=rmse, joint_change_percent=change,
                falls=sum(m["fall"] for m in motions), diagnostic_gate_passed=not reasons,
                gate_failure_reasons=reasons, motions=motions)
    receipt = dict(experiment=EXPERIMENT, training_device=str(DEVICE), results=results,
        objective_passed={objective: all(results[f"{objective}_noise{s:g}_seed0"]["diagnostic_gate_passed"]
            for s in (0., .01)) for objective in ("mse", "latent")},
        base_policy_updates=0, adapter_updates_each=200,
        independent_verification_sha256=sha256(OUTPUT / "independent_verification.json"),
        scope="REPEATED_FOUR_DIAGNOSTIC_MOTIONS_NOT_FRESH_FINAL_TEST", backup_status="LOCAL_ONLY")
    atomic_json(OUTPUT / "result.json", receipt)
    atomic_json(ROOT / f"data/manifests/a3_controller_adapter_20261010_{EXPERIMENT}.json", receipt)
    print(json.dumps({k: {field: v[field] for field in
        ("joint_rmse_rad", "joint_change_percent", "falls", "diagnostic_gate_passed")}
        for k, v in results.items()}, indent=2), flush=True)


def main(action):
    spec = register()
    if action == "register":
        return 0
    if action == "summarize":
        summarize()
        return 0
    with (DATA / "evaluation_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = OUTPUT / "state.json"
        state = json.loads(path.read_text()) if path.exists() else dict(status="pending", attempts=[], no_progress=0)
        if state["status"] == "complete":
            return 0
        if state["status"] == "blocked":
            return 2
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for old in state["attempts"]:
            if old["status"] == "running":
                if old["boot_id"] == boot:
                    raise RuntimeError("Same-boot running attempt requires inspection")
                old.update(status="interrupted", reason="boot_changed", progress_after=progress())
                state["no_progress"] = state["no_progress"]+1 if old["progress_after"] <= old["progress_before"] else 0
        if state["no_progress"] >= 2:
            state.update(status="blocked", reason="two_attempts_without_saved_progress")
            atomic_json(path, state)
            return 2
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        if uptime < spec["cooldown_seconds"]:
            state.update(status="cooldown", cooldown_remaining_seconds=300-uptime)
            atomic_json(path, state)
            return 75
        attempt = dict(status="running", boot_id=boot, progress_before=progress(), started_at=time.time())
        state["attempts"].append(attempt)
        state["status"] = "running"
        atomic_json(path, state)
        ordinal = len(state["attempts"])
        guard = {key: spec[key] for key in ("max_cpu_c", "max_gpu_c", "sample_seconds", "expires_at")}
        try:
            run_child("script.a3.controller_adapter_trial", ["prepare"], "prepare", ordinal, guard)
            for limit in (2, 200):
                for objective in ("mse", "latent"):
                    run_child("script.a3.controller_adapter_trial", ["train", "--objective", objective, "--limit", str(limit)],
                        f"{objective}_{limit}", ordinal, guard)
            run_child("script.a3.verify_controller_adapter", [], "independent_verification", ordinal, guard)
            for objective in ("mse", "latent"):
                if evaluate_explicit(args_for(objective, 0, smoke=True)) != 0:
                    raise RuntimeError("Zero adapter causal smoke failed")
                for sigma in (0., .01):
                    if evaluate_explicit(args_for(objective, sigma)) != 0:
                        raise RuntimeError(f"Adapter closed-loop condition failed: {objective}/{sigma}")
            summarize()
        except Exception as error:
            attempt.update(status="failed", reason=str(error), progress_after=progress(), finished_at=time.time())
            state["no_progress"] = state["no_progress"]+1 if attempt["progress_after"] <= attempt["progress_before"] else 0
            state.update(status="blocked", reason=str(error))
            atomic_json(path, state)
            raise
        attempt.update(status="complete", progress_after=progress(), finished_at=time.time())
        state.update(status="complete", no_progress=0)
        atomic_json(path, state)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "summarize"), nargs="?", default="run")
    raise SystemExit(main(parser.parse_args().action))
