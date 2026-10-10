"""Evaluate saved R05 checkpoints without training or changing prior ledgers."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from script.a3.checkpoint_store import atomic_json, sha256, validate_payload
from script.a3.fullchain_support import ROOT, DATA, VENDOR


OUTPUT = ROOT / "data/experiments/a3_regression_20261010_E01"
LOGS = ROOT / "logs/a3_regression_20261010_E01"
TRAINING = ROOT / "data/training/a3_finetune_20261010"
CHECKPOINTS = {
    0: "R05_001_s0", 2: "R05_001_s0", 10: "R05_002_s2",
    20: "R05_003_s10", 40: "R05_004_s30", 80: "R05_007_s50",
    120: "R05_009_s80", 200: "R05_010_s120",
}
PROTOCOL_FIELDS = ("motions", "reference_fps_after_stride", "frame_stride",
                   "max_policy_steps", "action_delay_ms", "encoder", "vendor_commit")


def read(path):
    return json.loads(path.read_text())


def verify():
    import torch
    from script.a3.evaluation_common import checkpoint_metadata
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    official_path = ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
    official_metadata = checkpoint_metadata(official_path)
    official = torch.load(official_path, map_location="cpu", weights_only=False)
    validate_payload(official)
    records = []
    for step, directory in CHECKPOINTS.items():
        path = TRAINING / directory / f"model_step_{step:06d}.pt"
        metadata = checkpoint_metadata(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if validate_payload(payload) != step or metadata["step"] != step:
            raise ValueError("Checkpoint step disagrees")
        optimizer = payload["optimizer_state_dict"]
        counters = sorted({int(s["step"]) for s in optimizer["state"].values() if "step" in s})
        if counters != ([step * 20] if step else []):
            raise ValueError("Optimizer counter disagrees with R05 update convention")
        if payload["state"].cur_episode_length.numel() != 16:
            raise ValueError("Wrong environment count")
        if step == 0:
            for name in ("policy_state_dict", "value_state_dict"):
                if official[name].keys() != payload[name].keys() or not all(
                        torch.equal(t, payload[name][key]) for key, t in official[name].items()):
                    raise ValueError("Zero-update weights differ from official")
        row = dict(step=step, checkpoint=str(path), bytes=metadata["bytes"],
                   sha256=metadata["sha256"], config_sha256=sha256(path.parent / "config.yaml"),
                   cpu_reload=True, networks_finite=True, optimizer_finite=True,
                   optimizer_steps=counters,
                   optimizer_learning_rates=[group["lr"] for group in optimizer["param_groups"]],
                   zero_update_matches_official=True if step == 0 else None)
        records.append(row)
        del payload
        print(json.dumps(row), flush=True)
    plan = dict(checkpoints=records, official_sha256=official_metadata["sha256"],
                steps_to_evaluate=[0, 2, 10, 20, 40, 80, 120],
                reused_results={"official": str(DATA / "official_baseline"),
                                "200": str(DATA / "finetuned_evaluation")},
                training_allowed=False, backup_status="LOCAL_ONLY")
    path = OUTPUT / "plan.json"
    if path.exists() and read(path) != plan:
        raise ValueError("Registered plan changed")
    atomic_json(path, plan)
    del official
    return plan


def summary_directory(label):
    if label == "official":
        return DATA / "official_baseline"
    if label == "200":
        return DATA / "finetuned_evaluation"
    return OUTPUT / f"step{int(label):03d}"


def report():
    from script.a3.evaluate_mujoco import summarize_explicit
    from script.a3.official_results import aggregate
    plan = read(OUTPUT / "plan.json")
    labels = ["official"] + [str(s) for s in plan["steps_to_evaluate"]] + ["200"]
    summaries = {label: (read(summary_directory(label) / "explicit_summary.json")
                         if label in ("official", "200") else summarize_explicit(summary_directory(label)))
                 for label in labels}
    base = summaries["official"]
    expected_sha = {str(row["step"]): row["sha256"] for row in plan["checkpoints"]}
    for label, payload in summaries.items():
        for field in PROTOCOL_FIELDS:
            if payload["protocol"][field] != base["protocol"][field]:
                raise ValueError(f"Different evaluation conditions: {label}/{field}")
        if payload["protocol"]["checkpoint_sha256"] != (
                plan["official_sha256"] if label == "official" else expected_sha[label]):
            raise ValueError("Evaluated wrong checkpoint")
        if len(payload["motions"]) != 20 or not all(r["full_replay_completed"] for r in payload["motions"]):
            raise ValueError("Incomplete selected20 coverage")
        manifest = read(summary_directory(label) / "explicit_manifest.json")
        for motion in payload["motions"]:
            stem = Path(motion["motion"]).stem
            directory = summary_directory(label)
            if (sha256(directory / f"{stem}.metrics.json") != manifest["runs"][stem]["metrics_sha256"] or
                    sha256(directory / f"{stem}.timeseries.json") != motion["timeseries_sha256"]):
                raise ValueError("Evaluation artifact changed since publication")
    split = read(DATA / "split.json")
    groups = {role: {row["name"] for row in split[role]} for role in ("train", "heldout")}
    rows = []
    for label, payload in summaries.items():
        motions = payload["motions"]
        row = dict(label=label, all=aggregate(motions),
                   groups={role: aggregate([m for m in motions if m["motion"] in names])
                           for role, names in groups.items()},
                   falls=[dict(motion=m["motion"], first_fall_s=m["first_fall_s"])
                          for m in motions if m["fall"]])
        rows.append(row)
    common = {label: [] for label in labels}
    for motion in base["motions"]:
        name = motion["motion"]
        traces = {}
        horizon = motion["expected_full_policy_steps"]
        for label in labels:
            directory = summary_directory(label)
            metric = read(directory / f"{Path(name).stem}.metrics.json")["motions"][0]
            horizon = min(horizon, metric["fall_tick"] if metric["fall"] else metric["num_policy_steps"])
            traces[label] = read(directory / f"{Path(name).stem}.timeseries.json")["motions"][0]
        for label, trace in traces.items():
            delta = np.asarray(trace["q_state_29"][:horizon]) - np.asarray(trace["reference_q_29"][:horizon])
            item = dict(motion=name, policy_steps=horizon,
                        joint_rmse_rad=float(np.sqrt(np.mean(delta ** 2))) if horizon else None,
                        root_pos_error_m_mean=float(np.mean(trace["root_pos_error_m"][:horizon])) if horizon else None)
            names = trace["body_names"]
            error = np.linalg.norm(np.asarray(trace["sim_body_pos_anchor"][:horizon]) -
                                   np.asarray(trace["ref_body_pos_anchor"][:horizon]), axis=-1)
            for group, indices in {
                "wrists": [names.index(n) for n in ("left_wrist_yaw_Link", "right_wrist_yaw_Link")],
                "legs": [i for i, n in enumerate(names) if any(t in n for t in ("hip_", "knee_", "ankle_"))],
            }.items():
                item[group + "_anchor_local_mean_m"] = float(error[:, indices].mean()) if horizon else None
            common[label].append(item)
    for row in rows:
        metrics = common[row["label"]]
        count = sum(m["policy_steps"] for m in metrics)
        row["common_prefix"] = {"policy_steps": count}
        for metric in ("joint_rmse_rad", "root_pos_error_m_mean", "wrists_anchor_local_mean_m", "legs_anchor_local_mean_m"):
            power = 2 if metric == "joint_rmse_rad" else 1
            total = sum(m["policy_steps"] * (m[metric] or 0) ** power for m in metrics)
            row["common_prefix"][metric] = (total / count) ** (1 / power) if count else None
    result = dict(rows=rows, common_prefix_by_motion=common, plan=plan,
                  scope="SINGLE_R05_TRAJECTORY_SIM2SIM_DIAGNOSTIC_NOT_CAUSAL_ATTRIBUTION",
                  heldout_scope="WITHHELD_FROM_R05_ONLY_NOW_USED_FOR_DIAGNOSIS",
                  generated_utc=datetime.now(timezone.utc).isoformat())
    for name in ("update_audit", "fixed_input_diagnostic"):
        path = OUTPUT / f"{name}.json"
        if path.exists():
            result[name] = read(path)
    result["state"] = read(OUTPUT / "state.json")
    result["artifacts"] = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                                 sha256=sha256(p), backup_status="LOCAL_ONLY")
                           for p in sorted(OUTPUT.rglob("*"))
                           if p.is_file() and p.name != "comparison.json"]
    result["code"] = [dict(path=str(p.relative_to(ROOT)), sha256=sha256(p)) for p in (
        ROOT / "script/a3/regression_diagnostic.py", ROOT / "script/a3/audit_finetune_updates.py",
        ROOT / "script/a3/fixed_input_diagnostic.py", ROOT / "script/a3/plot_regression.py",
        ROOT / "script/a3/a3-regression-e01.service")]
    health, rejected_health = [], []
    for path in LOGS.rglob("health.jsonl"):
        for ordinal, line in enumerate(path.read_text().splitlines(), 1):
            try:
                sample = json.loads(line)
                if not isinstance(sample, dict) or not all(
                        key in sample for key in ("boot_id", "temperatures_c", "gpus", "memory_bytes")):
                    raise ValueError("Missing health fields")
                health.append(sample)
            except (ValueError, TypeError):
                rejected_health.append(dict(path=str(path), line=ordinal))
    result["health"] = dict(samples=len(health), boot_ids=sorted({h["boot_id"] for h in health}),
                           cpu_max_c=max((h["temperatures_c"].get("x86_pkg_temp", 0) for h in health), default=None),
                           gpu_max_c=max((g["temperature.gpu"] for h in health for g in h["gpus"]), default=None),
                           ram_available_min_bytes=min((h["memory_bytes"]["MemAvailable"] for h in health), default=None),
                           rejected_records=rejected_health,
                           scope="RECORDED_SAMPLES_ONLY_NOT_REBOOT_CAUSE_OR_HARDWARE_ACCEPTANCE")
    atomic_json(OUTPUT / "comparison.json", result)
    atomic_json(ROOT / "data/manifests/a3_regression_20261010_E01.json", result)
    print(json.dumps(rows, indent=2), flush=True)
    return result


def run():
    from script.a3.evaluate_mujoco import evaluate_explicit
    from script.a3.recover_evaluation import recover
    from script.a3.evaluation_common import checkpoint_metadata
    plan = read(OUTPUT / "plan.json")
    state_path = OUTPUT / "state.json"
    with (DATA / "evaluation_supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if float(Path("/proc/uptime").read_text().split()[0]) < 300:
            return 75
        state = read(state_path) if state_path.exists() else dict(status="pending", completed=[], attempts=[])
        if state["status"] == "blocked":
            return 2
        if state["status"] == "complete":
            return 0
        current = {row["step"]: row for row in plan["checkpoints"]}
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        saved = sum(sum(r["status"] == "passed" for r in read(p)["runs"].values())
                    for p in OUTPUT.glob("step*/explicit_manifest.json"))
        previous = state["attempts"]
        if len(previous) >= 2 and all(a["start_saved"] >= saved for a in previous[-2:]):
            state.update(status="blocked", reason="two_attempts_without_saved_motion_progress")
            atomic_json(state_path, state)
            return 2
        state["attempts"].append(dict(boot_id=boot, start_saved=saved))
        state.update(status="running", current_boot=boot)
        atomic_json(state_path, state)
        for step in plan["steps_to_evaluate"]:
            output = summary_directory(str(step))
            output.mkdir(parents=True, exist_ok=True)
            if (output / "explicit_manifest.json").exists():
                recover(output)
            checkpoint = Path(current[step]["checkpoint"])
            if checkpoint_metadata(checkpoint)["sha256"] != current[step]["sha256"]:
                raise ValueError("Checkpoint changed after independent verification")
            state["current_step"] = step
            atomic_json(state_path, state)
            args = SimpleNamespace(output=output, logs=LOGS / f"step{step:03d}", checkpoint=checkpoint,
                motion=VENDOR / "a3_data/agibot_a3", reference_fps=30, frame_stride=4,
                max_policy_steps=None, action_delay_ms=0, capture_inputs=False,
                summarize_only=False, reference_buffer_ms=0)
            code = evaluate_explicit(args)
            if code:
                state.update(status="blocked", reason="evaluation_error_or_protection", returncode=code)
                atomic_json(state_path, state)
                return 2
            if step not in state["completed"]:
                state["completed"].append(step)
            atomic_json(state_path, state)
        report()
        state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "run", "report"))
    action = parser.parse_args().action
    if action == "verify":
        verify()
    elif action == "report":
        report()
    else:
        raise SystemExit(run())
