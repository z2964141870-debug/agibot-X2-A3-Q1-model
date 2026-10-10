"""Registered E16, motion-resumable closed-loop joint forecast screening."""

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
from script.a3.joint_forecast_replay import MODES


OUTPUT = ROOT / "data/experiments/a3_joint_ablation_20261010_E16"
LOGS = ROOT / "logs/a3_joint_ablation_20261010_E16"


def conditions():
    return [("oracle_aligned", 0.)] + [(mode, sigma) for sigma in (0., .01) for mode in MODES[1:]]


def label(mode, sigma):
    return f"{mode}_noise{sigma:g}_seed0"


def predictor(mode):
    if mode not in ("E12", "E14"):
        return None
    return ROOT / "data/experiments" / f"a3_future_reference_20261010_{mode}" / "ridge_joint_future.npz"


def register():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    split = json.loads((DATA / "split.json").read_text())
    sources = split["heldout"]
    if len(sources) != 4:
        raise ValueError("Diagnostic motion coverage changed")
    references = OUTPUT / "references"
    references.mkdir(exist_ok=True)
    for row in sources:
        path = Path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Reference identity changed")
        target = references / path.name
        if not target.exists():
            target.symlink_to(path)
        if target.resolve() != path.resolve():
            raise ValueError("Existing registered reference link changed")
    spec = dict(experiment="E16", checkpoint_sha256=checkpoint_metadata(OFFICIAL)["sha256"],
        split_sha256=sha256(DATA / "split.json"), motions=sources, conditions=[dict(mode=m, noise_std_rad=s,
        label=label(m, s), predictor_sha256=sha256(predictor(m)) if predictor(m) else None) for m, s in conditions()],
        code_sha256={name: sha256(ROOT / "script/a3" / name) for name in
                     ("joint_forecast_replay.py", "run_joint_ablation.py", "sim_entry.py", "evaluate_mujoco.py")},
        scope="JOINT_ONLY_FUTURE_ORIENTATION_ORACLE_NOT_FULL_CAUSAL_TELEOP",
        metrics_target="CLEAN_REFERENCE_AT_COMMON_ARRIVED_SAMPLE_ALIGNED_TARGET",
        data_note="FOUR_REPEATED_DIAGNOSTIC_VALIDATION_MOTIONS_NOT_FRESH_FINAL_TEST",
        arrival_fps=30, policy_fps=50, action_delay_ms=0, seed=0, max_attempts=None, expires_at=None,
        max_no_progress=2, cooldown_seconds=300, max_cpu_c=90, max_gpu_c=85, sample_seconds=5,
        optimizer_updates=0, backup_status="LOCAL_ONLY",
        gate="No extra falls; aggregate and every motion joint RMSE below both hold and smooth_cv; body/root not worse by >5%")
    path = OUTPUT / "job.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Registered E16 protocol changed")
    if not path.exists():
        atomic_json(path, spec)
    return spec


def args_for(mode, sigma, smoke=False):
    output = OUTPUT / ("smoke" if smoke else label(mode, sigma))
    return SimpleNamespace(output=output, logs=LOGS / ("smoke" if smoke else label(mode, sigma)),
        checkpoint=OFFICIAL, motion=(VENDOR / "a3_data/agibot_a3/001_walk_front_slow.csv" if smoke
                                    else OUTPUT / "references"),
        reference_fps=30., frame_stride=4, max_policy_steps=10 if smoke else None,
        action_delay_ms=0., reference_buffer_ms=0., capture_inputs=smoke, summarize_only=False,
        joint_reference_mode=mode, joint_predictor=predictor(mode), joint_noise_std=sigma, joint_noise_seed=0)


def progress():
    return sum(sum(row.get("status") == "passed" for row in json.loads(path.read_text())["runs"].values())
               for path in OUTPUT.glob("*/explicit_manifest.json"))


def summarize():
    spec = json.loads((OUTPUT / "job.json").read_text())
    results = {}
    for item in spec["conditions"]:
        directory = OUTPUT / item["label"]
        summary = json.loads((directory / "explicit_summary.json").read_text())
        rows = summary["motions"]
        if len(rows) != 4 or any(not row["full_replay_completed"] for row in rows):
            raise ValueError("Incomplete condition")
        totals, counts, ages, pred_error, per_motion = [], [], [], [], []
        for row in rows:
            path = directory / f"{Path(row['motion']).stem}.joint.npz"
            with np.load(path, allow_pickle=False) as stored:
                trace, errors = stored["rows"], stored["errors"]
                n = row["windows"]["full_rollout"]["policy_steps"]
                if (trace.shape != (n, 7) or errors.shape != (n, 3) or
                        not np.isfinite(trace).all() or not np.isfinite(errors).all() or
                        np.any(trace[:, 3] > trace[:, 0] + 1e-9) or
                        np.any(trace[trace[:, 5] == 0, 6] > .20 + 1e-9)):
                    raise ValueError("Actual policy joint-arrival trace failed")
                if not np.allclose(trace[:, 0], np.arange(n) / 50, atol=1e-9, rtol=0):
                    raise ValueError("Missing or duplicated policy input windows")
                ages.extend(trace[:, 2].tolist())
                pred_error.extend(errors.tolist())
            metric = row["windows"]["full_rollout"]
            totals.append(metric["joint_rmse_rad"]**2 * n)
            counts.append(n)
            per_motion.append(dict(motion=row["motion"], fall=row["fall"], first_fall_s=row["first_fall_s"],
                metrics=metric, before_fall=row["windows"]["before_first_fall"], trace_sha256=sha256(path)))
        weighted = {key: sum(row["metrics"][key] * count for row, count in zip(per_motion, counts)) / sum(counts)
                    for key in per_motion[0]["metrics"] if key not in ("policy_steps", "joint_rmse_rad")}
        results[item["label"]] = dict(falls=sum(row["fall"] for row in rows), policy_steps=sum(counts),
            joint_rmse_rad=math.sqrt(sum(totals)/sum(counts)), metrics=weighted, motions=per_motion,
            commanded_future_joint_rmse_rad=math.sqrt(sum(row[0] for row in pred_error)/sum(row[1] for row in pred_error)),
            reference_age_ms=dict(min=1000*min(ages), max=1000*max(ages)))
    gates = []
    for sigma in (0., .01):
        baselines = [results[label(mode, sigma)] for mode in ("hold", "smooth_cv")]
        for mode in ("cv", "E12", "E14"):
            candidate = results[label(mode, sigma)]
            paired = []
            for motion in candidate["motions"]:
                baseline_rows = [next(row for row in b["motions"] if row["motion"] == motion["motion"])
                                 for b in baselines]
                joint_ok = all(motion["metrics"]["joint_rmse_rad"] < b["metrics"]["joint_rmse_rad"]
                               for b in baseline_rows)
                falls_ok = all(not motion["fall"] or b["fall"] for b in baseline_rows)
                body_ok = all(motion["metrics"][key] <= b["metrics"][key]*1.05 + 1e-9
                              for b in baseline_rows for key in motion["metrics"]
                              if key.endswith("_mean_m") or key == "root_pos_error_m_mean")
                paired.append(dict(motion=motion["motion"], joint_ok=joint_ok, falls_ok=falls_ok, body_ok=body_ok))
            accepted = (all(candidate["joint_rmse_rad"] < b["joint_rmse_rad"] and candidate["falls"] <= b["falls"]
                            for b in baselines) and all(r["joint_ok"] and r["falls_ok"] and r["body_ok"] for r in paired))
            gates.append(dict(mode=mode, noise_std_rad=sigma, diagnostic_gate=accepted, per_motion=paired))
    receipt = dict(experiment="E16", status="complete", job_sha256=sha256(OUTPUT / "job.json"),
        results=results, gates=gates, policy_updates=0, policy_sha256_after=sha256(OFFICIAL),
        policy_unchanged=sha256(OFFICIAL) == spec["checkpoint_sha256"], scope=spec["scope"],
        independent_policy_improvement_proven=False, backup_status="LOCAL_ONLY")
    atomic_json(OUTPUT / "result.json", receipt)
    atomic_json(ROOT / "data/manifests/a3_joint_ablation_20261010_E16.json", receipt)
    print(json.dumps({"status": "complete", "results": {k: {x: v[x] for x in
                      ("falls", "joint_rmse_rad", "commanded_future_joint_rmse_rad")} for k,v in results.items()},
                      "gates": gates}), flush=True)
    return receipt


def main(action="run"):
    spec = register()
    if action == "register":
        print(json.dumps(spec, indent=2))
        return 0
    if action == "summarize":
        summarize()
        return 0
    with (DATA / "evaluation_supervisor.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = OUTPUT / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(status="pending", attempts=[], no_progress=0)
        if state["status"] == "complete":
            summarize()
            return 0
        if state["status"] == "blocked":
            return 2
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        for attempt in state["attempts"]:
            if attempt["status"] == "running":
                if attempt["boot_id"] == boot:
                    raise RuntimeError("Existing running attempt requires inspection")
                attempt.update(status="interrupted", reason="boot_changed", progress_after=progress())
                state["no_progress"] = state["no_progress"] + 1 if attempt["progress_after"] <= attempt["progress_before"] else 0
        if state["no_progress"] >= 2:
            state.update(status="blocked", reason="two_attempts_without_saved_progress")
            atomic_json(state_path, state)
            return 2
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        if uptime < 300:
            state.update(status="cooldown", cooldown_remaining_seconds=300-uptime)
            atomic_json(state_path, state)
            return 75
        attempt = dict(status="running", boot_id=boot, started_at=time.time(), progress_before=progress())
        state["attempts"].append(attempt)
        state["status"] = "running"
        atomic_json(state_path, state)
        try:
            if evaluate_explicit(args_for("cv", 0, smoke=True)) != 0:
                raise RuntimeError("Ten-step smoke did not complete")
            for mode, sigma in conditions():
                if evaluate_explicit(args_for(mode, sigma)) != 0:
                    raise RuntimeError(f"Condition interrupted: {label(mode, sigma)}")
            summarize()
        except Exception as error:
            attempt.update(status="failed", reason=str(error), progress_after=progress(), finished_at=time.time())
            state["no_progress"] = state["no_progress"] + 1 if attempt["progress_after"] <= attempt["progress_before"] else 0
            # Runtime/thermal failures are recorded for inspection; no automatic relaunch.
            state.update(status="blocked", reason=str(error))
            atomic_json(state_path, state)
            raise
        attempt.update(status="complete", progress_after=progress(), finished_at=time.time())
        state.update(status="complete", no_progress=0)
        atomic_json(state_path, state)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "summarize"), nargs="?", default="run")
    raise SystemExit(main(parser.parse_args().action))
