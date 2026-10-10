"""Publish evidence and comparisons for the independently resumed official stage."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from script.a3.fullchain_support import DATA, LOGS, ROOT, VENDOR, digest, mark, write_json
from script.a3.host_health import sample


def aggregate(rows):
    result = {"motions": len(rows), "falls": sum(r["fall"] for r in rows),
              "completed_without_fall": sum(r["motion_completed_without_fall"] for r in rows)}
    for window in ("before_first_fall", "full_rollout"):
        n = sum(r["windows"][window]["policy_steps"] for r in rows)
        result[window] = {"policy_steps": n}
        for metric in ("joint_rmse_rad", "root_pos_error_m_mean", "root_quat_error_deg_mean"):
            power = 2 if metric == "joint_rmse_rad" else 1
            weighted = sum(r["windows"][window]["policy_steps"] *
                           (r["windows"][window][metric] or 0) ** power for r in rows)
            result[window][metric] = (weighted / n) ** (1 / power) if n else None
    return result


def baseline():
    path = DATA / "official_baseline/explicit_summary.json"
    payload = json.loads(path.read_text())
    if len(payload["motions"]) != 20 or not all(r["full_replay_completed"] for r in payload["motions"]):
        raise ValueError("Official baseline is incomplete")
    write_json(DATA / "official_baseline_aggregate.json", aggregate(payload["motions"]))
    mark("mujoco_baseline", "passed", evidence=str(path),
         result=aggregate(payload["motions"]), scope="SELECTED20_SIM2SIM_ONLY")


def verify_trial():
    import torch
    from script.a3.checkpoint_store import select_checkpoint, validate_payload
    from script.a3.finetune_job import select_trial_checkpoint
    torch.set_num_threads(1)
    state_path = DATA / "finetune_R05/state.json"
    state = json.loads(state_path.read_text())
    step, checkpoint = select_trial_checkpoint(state["attempts"], "R05")
    if checkpoint is None:
        raise ValueError("No independently reloadable trial checkpoint")
    path, payload, rejected = select_checkpoint(checkpoint.parent)
    if path != checkpoint or validate_payload(payload) != step:
        raise ValueError("Independent final reload disagrees")
    if state["status"] == "complete" and step != 200:
        raise ValueError("Complete trial has the wrong final counter")
    optimizer = payload["optimizer_state_dict"]["state"]
    optimizer_steps = sorted({int(row["step"]) for row in optimizer.values() if "step" in row})
    if step > 0 and optimizer_steps != [step * 20]:
        raise ValueError("16env 4minibatch 5epoch optimizer counter mismatch")
    if payload["state"].cur_episode_length.numel() != 16:
        raise ValueError("Final checkpoint environment count changed")
    receipt = json.loads((LOGS / "finetune/R05_001_s0/warm_start_loaded.json").read_text())
    if receipt["loaded_global_step"] != 0 or receipt["optimizer_state_entries"] != 0:
        raise ValueError("Initial counter/optimizer did not reset")
    health = []
    for attempt in state["attempts"]:
        health_path = Path(attempt["logs"]) / "health.jsonl"
        if health_path.exists():
            for line in health_path.read_text().splitlines():
                try:
                    health.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    cpu = [row["temperatures_c"]["x86_pkg_temp"] for row in health
           if "x86_pkg_temp" in row["temperatures_c"]]
    gpu = [g["temperature.gpu"] for row in health for g in row["gpus"]
           if g.get("temperature.gpu") is not None]
    report = {"checkpoint": str(path), "sha256": digest(path), "bytes": path.stat().st_size,
              "verified_step": step, "target": 200, "target_completed": step == 200,
              "cpu_reload": True, "networks_finite": True, "optimizer_finite": True,
              "optimizer_steps": optimizer_steps, "num_envs": 16, "warm_start": receipt,
              "rejected_candidates": rejected, "attempts": state["attempts"],
              "health_samples": len(health), "sampled_cpu_max_c": max(cpu) if cpu else None,
              "sampled_gpu_max_c": max(gpu) if gpu else None,
              "boot_ids": sorted({row["boot_id"] for row in state["attempts"]}),
              "backup_status": "LOCAL_ONLY", "scope": "TRAINING_CHAIN_NOT_POLICY_EFFECT"}
    write_json(DATA / "finetune_verification.json", report)
    del payload
    print(json.dumps({key: report[key] for key in (
        "checkpoint", "verified_step", "sha256", "bytes", "optimizer_steps")}, indent=2))
    return report


def compare(partial=False):
    base = json.loads((DATA / "official_baseline/explicit_summary.json").read_text())
    tuned_dir = "finetuned_partial_evaluation" if partial else "finetuned_evaluation"
    tuned = json.loads((DATA / tuned_dir / "explicit_summary.json").read_text())
    from script.a3.evaluation_common import checkpoint_metadata
    metadata = checkpoint_metadata(Path(tuned["protocol"]["checkpoint"]))
    step = metadata["step"]
    if not partial and step != 200:
        raise ValueError("Formal comparison requires the independently verified step200 checkpoint")
    for field in ("motions", "reference_fps_after_stride", "frame_stride", "max_policy_steps",
                  "action_delay_ms", "encoder", "vendor_commit"):
        if base["protocol"][field] != tuned["protocol"][field]:
            raise ValueError(f"Paired conditions differ: {field}")
    split = json.loads((DATA / "split.json").read_text())
    before = {r["motion"]: r for r in base["motions"]}
    after = {r["motion"]: r for r in tuned["motions"]}
    if before.keys() != after.keys():
        raise ValueError("Paired motion coverage differs")
    pairs = []
    for name in before:
        if before[name]["expected_full_policy_steps"] != after[name]["expected_full_policy_steps"]:
            raise ValueError("Paired horizon differs")
        pairs.append({"motion": name, "official": before[name], "finetuned": after[name]})
    groups = {}
    for role in ("train", "heldout"):
        names = [r["name"] for r in split[role]]
        groups[role] = {"official": aggregate([before[n] for n in names]),
                        "finetuned": aggregate([after[n] for n in names])}
    report = {"split_sha256": digest(DATA / "split.json"), "groups": groups, "motions": pairs,
              "scope": "SINGLE_SEED_SHORT_TRIAL_SIM2SIM_NOT_DEPLOYMENT_ACCEPTANCE",
              "trial_step": step, "target_step": 200, "target_completed": step == 200,
              "official_checkpoint": base["protocol"]["checkpoint_sha256"],
              "finetuned_checkpoint": tuned["protocol"]["checkpoint_sha256"]}
    path = DATA / ("paired_partial_comparison.json" if partial else "paired_comparison.json")
    if partial:
        report["scope"] = "PARTIAL_CHECKPOINT_DIAGNOSTIC_TARGET200_NOT_COMPLETED"
    write_json(path, report)
    mark("paired_evaluation", "deferred" if partial else "passed", evidence=str(path),
         reason="target200_not_completed" if partial else None, scope=report["scope"])
    return report


def buffer():
    report = []
    for name in ("official_buffer_selected20", "official_buffer_mocap"):
        directory = DATA / name
        summary = directory / "explicit_summary.json"
        if not summary.exists():
            continue
        payload = json.loads(summary.read_text())
        rows = []
        for motion in payload["motions"]:
            path = directory / (Path(motion["motion"]).stem + ".buffer.npz")
            with np.load(path, allow_pickle=False) as trace:
                values = trace["rows"]
                if len(values) != motion["windows"]["full_rollout"]["policy_steps"]:
                    raise ValueError("Missing policy buffer evidence")
                if not np.isfinite(values).all() or np.any(values[:, 4] > values[:, 1] + 1e-9):
                    raise ValueError("Causal source-read violation")
                rows.append({"motion": motion["motion"], "policy_windows": len(values),
                             "age_min_ms": float(values[:, 3].min() * 1000),
                             "age_max_ms": float(values[:, 3].max() * 1000),
                             "startup_prefill_ms": float(trace["startup_prefill_ms"]),
                             "trace_sha256": digest(path), "future_reads": 0})
        offline_name = "official_baseline" if name.endswith("selected20") else "official_mocap"
        offline = json.loads((DATA / offline_name / "explicit_summary.json").read_text())
        if payload["protocol"]["motions"] != offline["protocol"]["motions"]:
            raise ValueError("Offline/buffer reference mismatch")
        report.append({"buffer": name, "windows": rows, "offline": aggregate(offline["motions"]),
                       "buffered": aggregate(payload["motions"]),
                       "scope": "SIMULATED_ARRIVALS_DELAYED_TARGET_METRICS_NOT_LIVE_END_TO_END"})
    if not report:
        raise ValueError("No policy buffer comparison completed")
    write_json(DATA / "policy_buffer_comparison.json", report)
    mark("reference_buffer_policy", "passed", evidence=str(DATA / "policy_buffer_comparison.json"),
         scope="SIMULATED_REPLAY_WITH_200MS_VELOCITY_PREFILL")


def snapshot(final=False):
    tasks = json.loads((DATA / "tasks.json").read_text())
    if final and any(row["status"] in ("pending", "running") for row in tasks["tasks"].values()):
        raise ValueError("Finalization would omit pending tasks")
    artifacts = []
    for directory in (DATA, LOGS, ROOT / "data/training/a3_finetune_20261010"):
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name.endswith(".lock"):
                continue
            artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                              "sha256": digest(path), "backup_status": "LOCAL_ONLY"})
    report = {"updated_utc": datetime.now(timezone.utc).isoformat(),
              "stage": "OFFICIAL_FULLCHAIN_IN_NEW_A3_WORKSPACE",
              "status": "DONE_OR_DOCUMENTED_BLOCKERS" if final else "IN_PROGRESS",
              "tasks": tasks["tasks"], "artifacts": artifacts, "host": sample(),
              "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
              "old_lineages_restarted": False, "hardware_control": False,
              "git_parent": __import__("subprocess").check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "assets": {str(p.relative_to(ROOT)): digest(p) for p in (
                  VENDOR / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml",)}}
    write_json(ROOT / "data/manifests/a3_official_stage_20261010.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("baseline", "verify-trial", "compare", "buffer", "snapshot", "final"))
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.action == "baseline":
        baseline()
    elif args.action == "verify-trial":
        verify_trial()
    elif args.action == "compare":
        print(json.dumps(compare(partial=args.partial)["groups"], indent=2))
    elif args.action == "buffer":
        buffer()
    else:
        snapshot(final=args.action == "final")


if __name__ == "__main__":
    main()
