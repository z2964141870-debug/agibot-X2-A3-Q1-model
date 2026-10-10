"""E18 full policy-reference causality, paired with E16 oracle orientation."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import OFFICIAL
from script.a3.evaluate_mujoco import evaluate_explicit
from script.a3.evaluation_common import checkpoint_metadata
from script.a3.fullchain_support import DATA, ROOT, VENDOR


OUTPUT = ROOT / "data/experiments/a3_causal_reference_20261010_E18"
LOGS = ROOT / "logs/a3_causal_reference_20261010_E18"
E16 = ROOT / "data/experiments/a3_joint_ablation_20261010_E16"
CONDITIONS = [(mode, sigma) for sigma in (0., .01) for mode in ("cv", "E14")]


def label(mode, sigma):
    return f"{mode}_noise{sigma:g}_seed0"


def args_for(mode, sigma, smoke=False):
    key = "smoke" if smoke else label(mode, sigma)
    model = ROOT / "data/experiments/a3_future_reference_20261010_E14/ridge_joint_future.npz" if mode == "E14" else None
    return SimpleNamespace(output=OUTPUT / key, logs=LOGS / key, checkpoint=OFFICIAL,
        motion=VENDOR / "a3_data/agibot_a3/001_walk_front_slow.csv" if smoke else E16 / "references",
        reference_fps=30., frame_stride=4, max_policy_steps=10 if smoke else None,
        action_delay_ms=0., reference_buffer_ms=0., capture_inputs=True, capture_count=100000, summarize_only=False,
        joint_reference_mode=mode, joint_predictor=model, joint_noise_std=sigma, joint_noise_seed=0,
        causal_orientation=True)


def register():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / "split.json").read_text())
    if len(split["heldout"]) != 4 or any(sha256(Path(r["path"])) != r["sha256"] for r in split["heldout"]):
        raise ValueError("Reference split changed")
    spec = dict(experiment="E18", checkpoint_sha256=checkpoint_metadata(OFFICIAL)["sha256"],
        split_sha256=sha256(DATA / "split.json"), motions=split["heldout"],
        conditions=[dict(mode=m, noise_std_rad=s, label=label(m, s)) for m, s in CONDITIONS],
        code_sha256={name: sha256(ROOT / "script/a3" / name) for name in
                     ("causal_orientation.py", "joint_forecast_replay.py", "run_causal_reference.py", "sim_entry.py", "evaluate_mujoco.py")},
        predictor_sha256=sha256(args_for("E14", 0).joint_predictor), comparator_e16_result_sha256=sha256(E16 / "result.json"),
        main_variable="REPLACE_ORACLE_FUTURE_PELVIS_ORIENTATION_WITH_CAUSAL_SO3_ANGULAR_VELOCITY",
        anchor="pelvis_link", root_translation="NOT_IN_A3_FAST_POLICY_REFERENCE_TOKENS",
        joint_noise_only=True, root_quaternion_noise_std_rad=0, raw_reference_fps=30, policy_fps=50,
        max_attempts=None, expires_at=None, max_no_progress=2, cooldown_seconds=300,
        max_cpu_c=90, max_gpu_c=85, sample_seconds=5, cpu_quota_percent=100,
        policy_updates=0, scope="FULL_POLICY_REFERENCE_CAUSAL_SIMULATED_ARRIVALS_NOT_LIVE_CLOCKS",
        validation="FOUR_REPEATED_DIAGNOSTIC_MOTIONS_NOT_FRESH_FINAL_TEST", backup_status="LOCAL_ONLY")
    path = OUTPUT / "job.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("E18 protocol changed")
    if not path.exists():
        atomic_json(path, spec)
    return spec


def progress():
    return sum(sum(r.get("status") == "passed" for r in json.loads(p.read_text())["runs"].values())
               for p in OUTPUT.glob("*/explicit_manifest.json"))


def summarize():
    job = json.loads((OUTPUT / "job.json").read_text())
    baseline = json.loads((E16 / "result.json").read_text())["results"]
    results = {}
    for mode, sigma in CONDITIONS:
        key = label(mode, sigma)
        rows = json.loads((OUTPUT / key / "explicit_summary.json").read_text())["motions"]
        counts, sums, ori_sq, ori_count, ages, motions = [], [], 0., 0., [], []
        if len(rows) != 4 or any(not row["full_replay_completed"] for row in rows):
            raise ValueError("Incomplete full causal condition")
        for row in rows:
            stem = Path(row["motion"]).stem
            n = row["windows"]["full_rollout"]["policy_steps"]
            trace_path = OUTPUT / key / f"{stem}.joint.npz"
            with np.load(trace_path, allow_pickle=False) as stored:
                trace, orientation = stored["rows"], stored["orientation_errors"]
                if (trace.shape != (n, 7) or orientation.shape != (n, 3) or
                        not np.isfinite(trace).all() or not np.isfinite(orientation).all() or
                        np.any(trace[:, 3] > trace[:, 0]+1e-9) or
                        np.any(trace[trace[:, 5] == 0, 6] > .20+1e-9)):
                    raise ValueError("Actual causal joint/root arrival trace failed")
                ages.extend(trace[:, 2].tolist())
                ori_sq += orientation[:, 0].sum()
                ori_count += orientation[:, 1].sum()
            capture_path = OUTPUT / key / f"{stem}.inputs.npz"
            with np.load(capture_path, allow_pickle=False) as stored:
                if stored["observations"].shape != (n, 1570) or stored["actions"].shape != (n, 29):
                    raise ValueError("Actual policy input capture ABI failed")
                if not np.isfinite(stored["observations"]).all() or not np.isfinite(stored["actions"]).all():
                    raise ValueError("Nonfinite actual policy input/actions")
            metric = row["windows"]["full_rollout"]
            previous = next(m for m in baseline[key]["motions"] if m["motion"] == row["motion"])
            if previous["metrics"]["policy_steps"] != n:
                raise ValueError("Paired oracle/causal replay durations differ")
            counts.append(n)
            sums.append(metric["joint_rmse_rad"]**2*n)
            motions.append(dict(motion=row["motion"], fall=row["fall"], first_fall_s=row["first_fall_s"], metrics=metric,
                before_fall=row["windows"]["before_first_fall"],
                oracle_joint_rmse_rad=previous["metrics"]["joint_rmse_rad"],
                joint_change_percent=(metric["joint_rmse_rad"]/previous["metrics"]["joint_rmse_rad"]-1)*100,
                trace_sha256=sha256(trace_path), inputs_sha256=sha256(capture_path)))
        metrics = {key_: sum(m["metrics"][key_]*count for m, count in zip(motions, counts))/sum(counts)
                   for key_ in motions[0]["metrics"] if key_ not in ("joint_rmse_rad", "policy_steps")}
        rmse = math.sqrt(sum(sums)/sum(counts))
        results[key] = dict(falls=sum(m["fall"] for m in motions), joint_rmse_rad=rmse, metrics=metrics,
            oracle_joint_rmse_rad=baseline[key]["joint_rmse_rad"],
            joint_change_percent=(rmse/baseline[key]["joint_rmse_rad"]-1)*100,
            reference_age_ms=dict(min=1000*min(ages), max=1000*max(ages)),
            future_orientation_rmse_deg=math.sqrt(ori_sq/ori_count)*180/math.pi,
            motions=motions, full_policy_reference_causal_guard_passed=True,
            actual_policy_inputs_captured=sum(counts), policy_steps=sum(counts))
    receipt = dict(experiment="E18", status="complete", job_sha256=sha256(OUTPUT / "job.json"), results=results,
        scope=job["scope"], policy_updates=0, checkpoint_sha256_after=sha256(OFFICIAL),
        policy_unchanged=sha256(OFFICIAL) == job["checkpoint_sha256"], backup_status="LOCAL_ONLY",
        effect_acceptance="NO_GENERAL_POLICY_UPGRADE_CLAIM; FIXED_FOUR_MOTION_SEED0_DIAGNOSTIC")
    atomic_json(OUTPUT / "result.json", receipt)
    atomic_json(ROOT / "data/manifests/a3_causal_reference_20261010_E18.json", receipt)
    print(json.dumps({key: {k: v[k] for k in ("falls", "joint_rmse_rad", "joint_change_percent",
                    "future_orientation_rmse_deg")} for key, v in results.items()}, indent=2), flush=True)


def main(action="run"):
    register()
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
                    raise RuntimeError("Existing same-boot attempt requires inspection")
                old.update(status="interrupted", reason="boot_changed", progress_after=progress())
                state["no_progress"] = state["no_progress"]+1 if old["progress_after"] <= old["progress_before"] else 0
        if state["no_progress"] >= 2:
            state.update(status="blocked", reason="two_attempts_without_saved_progress")
            atomic_json(path, state)
            return 2
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        if uptime < 300:
            state.update(status="cooldown", cooldown_remaining_seconds=300-uptime)
            atomic_json(path, state)
            return 75
        attempt = dict(status="running", boot_id=boot, progress_before=progress(), started_at=time.time())
        state["attempts"].append(attempt)
        state["status"] = "running"
        atomic_json(path, state)
        try:
            if evaluate_explicit(args_for("cv", 0, smoke=True)) != 0:
                raise RuntimeError("Full causal smoke failed")
            for mode, sigma in CONDITIONS:
                if evaluate_explicit(args_for(mode, sigma)) != 0:
                    raise RuntimeError(f"Full causal condition failed: {label(mode, sigma)}")
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
