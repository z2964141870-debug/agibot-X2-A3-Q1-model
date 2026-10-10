"""Independent corrected two-update integration trials and selected20 evaluation."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from script.a3.autoresume import monitor_child
from script.a3.checkpoint_store import atomic_json, sha256, validate_payload
from script.a3.evaluation_common import checkpoint_metadata, preflight
from script.a3.finetune_job import recovery_remaining, select_trial_checkpoint
from script.a3.fullchain_support import DATA, ROOT, VENDOR, write_json


OUTPUT = ROOT / "data/experiments/a3_corrected_20261010_E03"
LOGS = ROOT / "logs/a3_corrected_20261010_E03"
TRAINING = ROOT / "data/training/a3_finetune_20261010"
OFFICIAL = ROOT / "data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt"
TRIAL = "R06"


def read(path):
    return json.loads(path.read_text())


def configuration():
    return dict(source=str(OFFICIAL), source_sha256=checkpoint_metadata(OFFICIAL)["sha256"],
                target_step=2, num_envs=16, num_mini_batches=4, num_learning_epochs=5,
                configured_num_learning_iterations=200, save_frequency=1,
                max_attempts=None, expires_at=None, max_no_progress=2,
                max_cpu_c=90, max_gpu_c=85, sample_seconds=5, cooldown_seconds=300,
                dataset=str(DATA / "motionlib/train"), split_sha256=sha256(DATA / "split.json"),
                trainer="script.a3.verified_finetune.VerifiedFineTuneTrainer",
                trainer_sha256=sha256(ROOT / "script/a3/verified_finetune.py"),
                job_sha256=sha256(Path(__file__)), old_lineages_allowed=False,
                seed=0, backup_status="LOCAL_ONLY")


def train():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        config = configuration()
        path = OUTPUT / "job.json"
        if path.exists() and read(path) != config:
            raise ValueError("Registered trial changed; do not rewrite its ledger")
        write_json(path, config)
        state_path = OUTPUT / "state.json"
        state = read(state_path) if state_path.exists() else dict(status="pending", attempts=[])
        if state["status"] in ("blocked", "complete"):
            return 0 if state["status"] == "complete" else 2
        step, checkpoint = select_trial_checkpoint(state["attempts"], TRIAL)
        if step > config["target_step"]:
            raise ValueError("Checkpoint exceeded this short trial")
        if step == config["target_step"]:
            state.update(status="complete", verified_step=step, checkpoint=str(checkpoint))
            atomic_json(state_path, state)
            return 0
        count = 0
        for previous in reversed(state["attempts"]):
            if step > previous["start_step"]:
                break
            count += 1
        if count >= config["max_no_progress"]:
            state.update(status="blocked", reason="two_attempts_without_saved_progress")
            atomic_json(state_path, state)
            return 2
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        uptime = float(Path("/proc/uptime").read_text().split()[0])
        delay = recovery_remaining(state, config, boot, uptime, time.time())
        if delay > 0:
            state.update(status="cooldown", cooldown_remaining_seconds=delay, boot_id=boot)
            atomic_json(state_path, state)
            return 75
        # Occupancy is a wait condition; it does not spend a training attempt.
        query = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                               text=True, capture_output=True, check=True)
        if query.stdout.strip():
            return 75
        preflight()
        if os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize < 20 * 1024 ** 3:
            raise RuntimeError("Less than 20 GiB free")
        run_dir = TRAINING / f"{TRIAL}_{len(state['attempts']) + 1:03d}_s{step}"
        logs = LOGS / run_dir.name
        run_dir.mkdir(parents=True, exist_ok=False)
        logs.mkdir(parents=True, exist_ok=False)
        for previous in state["attempts"]:
            if previous["status"] == "running":
                previous.update(status="interrupted", reason="supervisor_or_host_interrupted")
        row = dict(start_step=step, target=config["target_step"], run_dir=str(run_dir),
                   logs=str(logs), boot_id=boot, started_at=time.time(), status="running",
                   checkpoint=str(checkpoint or OFFICIAL), full_state_resume=checkpoint is not None)
        state["attempts"].append(row)
        state.update(status="running", boot_id=boot)
        atomic_json(state_path, state)
        environment = os.environ.copy()
        environment.update(ISAAC_PYTHON=str(ROOT / "data/environments/a3-sonic/bin/python"),
                           REPO_DIR=str(VENDOR.resolve()), MOTION_FILE=config["dataset"],
                           EXPERIMENT_DIR=str(run_dir), CHECKPOINT=row["checkpoint"],
                           NUM_ENVS="16", NUM_MINI_BATCHES="4", NUM_LEARNING_ITERATIONS="200",
                           NUM_PROCESSES="1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", WANDB_MODE="disabled", PYTHONPATH=str(ROOT),
                           YUANQI_RESUME_LOG_DIR=str(logs))
        argv = ["bash", str(VENDOR / "train_a3_035_fromscratch.sh"), "seed=0",
                "trainer._target_=" + config["trainer"],
                "callbacks.model_save._target_=script.a3.resumable_sonic.DurableModelSaveCallback",
                "callbacks.model_save.save_frequency=1", "+callbacks.model_save.save_last_frequency=1",
                "+callbacks.model_save.target_global_step=2"]
        if checkpoint:
            argv += ["++resume=true", "++resume_in_place=false"]
        row["argv"] = argv
        atomic_json(state_path, state)
        with (logs / "console.log").open("w") as stream:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code, reason = monitor_child(child, logs, config)
        new_step, saved = select_trial_checkpoint(state["attempts"], TRIAL)
        row.update(status="finished", returncode=code, reason=reason, verified_step=new_step,
                   finished_at=time.time())
        if code == 0 and reason is None and new_step == 2:
            state.update(status="complete", verified_step=new_step, checkpoint=str(saved))
            result = 0
        elif reason is None and code in (-9, 137, 75):
            state.update(status="interrupted", verified_step=new_step)
            result = 75
        else:
            state.update(status="blocked", reason=reason or f"training_exit_{code}", verified_step=new_step)
            result = 2
        atomic_json(state_path, state)
        print(json.dumps(state), flush=True)
        return result


def verify():
    import torch
    import yaml
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    torch.set_num_threads(1)
    state = read(OUTPUT / "state.json")
    if state["status"] != "complete" or state["verified_step"] != 2:
        raise ValueError("Training smoke has not completed")
    official = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
    records, audit, scalar_records, receipts, undefined_statistics = {}, [], [], [], []
    for attempt in state["attempts"]:
        directory, logs = Path(attempt["run_dir"]), Path(attempt["logs"])
        config = yaml.safe_load((directory / "config.yaml").read_text())
        if (config["trainer"]["_target_"] != configuration()["trainer"] or
                config["algo"]["config"]["num_learning_epochs"] != 5 or
                not config["algo"]["config"]["compute_aux_loss"]):
            raise ValueError("Full trainer config does not match the registered objective")
        completed_episode_counts = {}
        for path in sorted(directory.glob("model_step_*.pt")):
            metadata = checkpoint_metadata(path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            step = validate_payload(payload)
            if step not in (0, 1, 2) or payload["state"].cur_episode_length.numel() != 16:
                raise ValueError("Checkpoint escaped the short-trial lineage")
            optimizer = payload["optimizer_state_dict"]
            completed_episode_counts[step] = len(payload["state"].lenbuffer)
            counters = sorted({int(s["step"]) for s in optimizer["state"].values() if "step" in s})
            if counters != ([step * 20] if step else []):
                raise ValueError("Optimizer count disagrees with update count")
            rates = [float(g["lr"]) for g in optimizer["param_groups"]]
            args_lr = float(payload["args"].learning_rate)
            last_lr = payload["lr_scheduler_state_dict"]["_last_lr"]
            if not math.isfinite(args_lr) or any(rate != args_lr for rate in rates + last_lr):
                raise ValueError("Saved optimizer, controller and scheduler learning rates differ")
            if step == 0 and any(not torch.equal(t, payload[name][key])
                                 for name in ("policy_state_dict", "value_state_dict")
                                 for key, t in official[name].items()):
                raise ValueError("Initial weights differ from official")
            records[str(step)] = dict(checkpoint=str(path), **metadata,
                                      config_sha256=sha256(directory / "config.yaml"), cpu_reload=True,
                                      networks_finite=True, optimizer_finite=True,
                                      optimizer_steps=counters, adaptive_args_lr=args_lr,
                                      optimizer_lrs=rates, scheduler_last_lrs=last_lr)
            del payload
        step_path = logs / "actual_optimizer_steps.jsonl"
        rows = [json.loads(line) for line in step_path.read_text().splitlines()] if step_path.exists() else []
        for row in rows:
            if any(not math.isfinite(rate) or rate != row["adaptive_args_lr"]
                   for rate in row["applied_group_lrs"]):
                raise ValueError("Actual optimizer step learning rate mismatch")
        audit.extend(rows)
        event = EventAccumulator(str(directory / "tensorboard"), size_guidance={"scalars": 0})
        event.Reload()
        tags = event.Tags()["scalars"]
        scalars = {tag: [dict(step=e.step, value=e.value) for e in event.Scalars(tag)] for tag in tags}
        for tag, values in scalars.items():
            for entry in values:
                if math.isfinite(entry["value"]):
                    continue
                if (tag == "Objective/length" and math.isnan(entry["value"]) and
                        completed_episode_counts.get(entry["step"]) == 0):
                    undefined_statistics.append(dict(run=str(directory), tag=tag, step=entry["step"],
                        reason="NO_COMPLETED_EPISODES_CHECKPOINT_LENBUFFER_EMPTY"))
                    entry["value"] = None
                else:
                    raise ValueError(f"Nonfinite training scalar: {tag}/{entry['step']}")
        scalar_records.append(dict(run=str(directory), scalars=scalars))
        for name in ("warm_start_loaded.json", "resume_loaded.json", "verified_resume_lr.json"):
            if (logs / name).exists():
                receipts.append(dict(path=str(logs / name), receipt=read(logs / name)))
    if not all(str(step) in records for step in (0, 1, 2)):
        raise ValueError("Missing independently verified initial/first/second update")
    completed_audit = [row for row in audit if row["trainer_global_step"] <= 2]
    if len(completed_audit) < 40:
        raise ValueError("Missing actual optimizer step audit")
    aux_tags = {tag for record in scalar_records for tag in record["scalars"] if "aux" in tag.lower()}
    if not any("total_aux_loss" in tag for tag in aux_tags):
        raise ValueError("Official auxiliary loss was not logged by the live trainer")
    warm_starts = [r["receipt"] for r in receipts if r["path"].endswith("warm_start_loaded.json")]
    if not warm_starts or any(r["loaded_global_step"] != 0 or r["optimizer_state_entries"] != 0 or
                            not all(r["networks_match_source"].values()) for r in warm_starts):
        raise ValueError("Missing verified official warm start with reset optimizer")
    result = dict(checkpoints=records, actual_optimizer_steps=audit, scalar_records=scalar_records,
                  auxiliary_tags=sorted(aux_tags), receipts=receipts,
                  undefined_episode_statistics=undefined_statistics,
                  source_sha256=sha256(OFFICIAL), status="passed", backup_status="LOCAL_ONLY",
                  scope="FULL_TRAINER_INTEGRATION_AND_NUMERICAL_INTEGRITY_NOT_POLICY_IMPROVEMENT")
    atomic_json(OUTPUT / "verification.json", result)
    print(json.dumps({k: result[k] for k in ("status", "checkpoints", "auxiliary_tags")}), flush=True)
    return result


def evaluate():
    from script.a3.evaluate_mujoco import evaluate_explicit
    from script.a3.recover_evaluation import recover
    plan = read(OUTPUT / "verification.json")
    with (DATA / "evaluation_supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if float(Path("/proc/uptime").read_text().split()[0]) < 300:
            return 75
        state_path = OUTPUT / "evaluation_state.json"
        state = read(state_path) if state_path.exists() else dict(status="pending", attempts=[], completed=[])
        if state["status"] in ("blocked", "complete"):
            return 2 if state["status"] == "blocked" else 0
        saved = sum(sum(r["status"] == "passed" for r in read(p)["runs"].values())
                    for p in OUTPUT.glob("step*/explicit_manifest.json"))
        if len(state["attempts"]) >= 2 and all(a["start_saved"] >= saved for a in state["attempts"][-2:]):
            state.update(status="blocked", reason="two_attempts_without_saved_motion_progress")
            atomic_json(state_path, state)
            return 2
        state["attempts"].append(dict(start_saved=saved,
            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip()))
        state["status"] = "running"
        atomic_json(state_path, state)
        for step in (1, 2):
            output = OUTPUT / f"step{step:03d}"
            output.mkdir(parents=True, exist_ok=True)
            if (output / "explicit_manifest.json").exists():
                recover(output)
            checkpoint = Path(plan["checkpoints"][str(step)]["checkpoint"])
            if checkpoint_metadata(checkpoint)["sha256"] != plan["checkpoints"][str(step)]["sha256"]:
                raise ValueError("Checkpoint changed after verification")
            state["current_step"] = step
            atomic_json(state_path, state)
            args = SimpleNamespace(output=output, logs=LOGS / f"step{step:03d}", checkpoint=checkpoint,
                motion=VENDOR / "a3_data/agibot_a3", reference_fps=30, frame_stride=4,
                max_policy_steps=None, action_delay_ms=0, capture_inputs=False,
                summarize_only=False, reference_buffer_ms=0)
            if evaluate_explicit(args):
                state.update(status="blocked", reason="evaluation_error_or_protection")
                atomic_json(state_path, state)
                return 2
            if step not in state["completed"]:
                state["completed"].append(step)
            atomic_json(state_path, state)
        state.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(state_path, state)
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "verify", "evaluate"))
    parser.add_argument("--trial", choices=("R06", "R07"), default="R06")
    args = parser.parse_args()
    action = args.action
    if args.trial == "R07":
        TRIAL = "R07"
        OUTPUT = ROOT / "data/experiments/a3_corrected_20261010_E04"
        LOGS = ROOT / "logs/a3_corrected_20261010_E04"
    if action == "verify":
        verify()
    else:
        raise SystemExit(train() if action == "train" else evaluate())
