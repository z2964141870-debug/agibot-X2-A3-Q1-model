"""Validate a replacement GPU, then start one fresh, fully audited shared run."""

import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys

from run_native import ROOT, identity, process_identity, save
from train_native import OPTIMIZERS, validate_plan


HERE = Path(__file__).resolve().parent
COUNTS = dict(zip(OPTIMIZERS, (80, 8, 160, 8, 160)))


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def build_plans(config, template, commit):
    require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
            "Require a full published commit")
    profile = copy.deepcopy(template)
    profile.update(output=str(Path(config["output"]) / "profile"), code_commit=commit)
    require(profile["robots"] == ["g1", "q1"] and profile["cfg"]["iterations"] == 8,
            "Require the paired eight-iteration profile")
    require(profile["cfg"]["num_envs"] == 2048, "Keep the approved high environment count")
    formal = copy.deepcopy(profile)
    formal.update(purpose="train", output=str(Path(config["output"]) / "train"), receipt_every=50)
    formal["cfg"].update(config["formal_cfg_overrides"])
    require(formal["cfg"]["iterations"] == 80000 and formal["cfg"]["wm_only_iters"] == 1000,
            "Preserve the full exposure target")
    require(formal["cfg"]["ckpt_every"] == 8000, "Keep bounded checkpoint storage and the 40000 snapshot")
    for plan in (profile, formal):
        require(not any(k in plan["cfg"] for k in ("resume", "checkpoint", "resume_from")),
                "Formal training must initialize from scratch")
        validate_plan(plan)
    return profile, formal


def check_phase(root, purpose, commit):
    status = json.loads((root / "status.json").read_text())
    cleanup = status.get("guard", {}).get("cleanup", {})
    require(status.get("status") == "COMPLETED" and status.get("purpose") == purpose
            and status.get("code_commit") == commit
            and status.get("guard", {}).get("status") == "WORKER_COMPLETED"
            and cleanup.get("worker_reaped") is True
            and cleanup.get("worker_return_code") == 0,
            "The entire guarded phase must complete and reap its worker")
    return status


def check_native(result, robot):
    require(result.get("status") == "PASSED" and result.get("robot") == robot
            and result.get("environments") == 16 and result.get("control_steps") == 24
            and result.get("environment_transitions") == 384
            and result.get("new_optimizer_updates") == 0
            and result.get("batched_physics_steps") == (240 if robot == "q1" else 96),
            "Missing or incomplete native execution verification")
    for name in ("reward_max_abs_error", "observation_max_abs_error", "target_max_abs_error"):
        require(0 <= result[name] <= (1e-6 if name.startswith("target") else 3e-5),
                "Native command/reward/observation contract failed")
    if robot == "q1":
        require(result["torque_callback"]["calls"] >= 240
                and 0 <= result["torque_callback"]["max_abs_error_nm"] <= 1e-4,
                "Q1 torque callbacks did not pass")


def check_profile(workers, profile):
    require(len(workers) == 2, "Both native robots must finish")
    for robot, worker in zip(profile["robots"], workers):
        require(worker.get("status") == "COMPLETED" and worker.get("robot") == robot
                and worker.get("code_commit") == profile["code_commit"]
                and worker.get("formal_training") is False
                and worker.get("iterations_completed") == 8
                and worker.get("reference_clips") == 559
                and len(worker.get("common_clip_names", [])) == 559
                and worker.get("collected_environment_transitions") == 450560
                and worker.get("optimizer_updates") == COUNTS
                and worker.get("native_rank_checks_required") is True
                and worker.get("native_rank_checks_passed") is True
                and len(worker.get("peer_audit", [])) == 2,
                "Shared profile is missing a complete native audit")
        require(all(worker["resolved_cfg"].get(k) == v for k, v in profile["cfg"].items()),
                "The tested settings differ from the frozen profile")
        require(worker.get("nonfinite_updates") and not any(worker["nonfinite_updates"].values()),
                "Nonfinite updates invalidate capacity verification")
        require(worker.get("optimizer_adam_steps") == {k: [v] for k, v in COUNTS.items()},
                "Five Adam counters must agree with the complete update hooks")
        require(worker.get("latent_draws") and worker.get("last_latent_draw")
                and worker.get("shared_parameter_sha")
                and len(worker.get("iteration_seconds", [])) == 8
                and all(t > 0 for t in worker["iteration_seconds"]),
                "Profile lacks actual paired draws, shared parameters or measured timing")
    for key in ("common_clip_names", "shared_parameter_sha", "latent_draws", "last_latent_draw"):
        require(workers[0][key] == workers[1][key], "Shared native ranks diverged: " + key)
    require(workers[0]["initial_buffers"]["joint_presence"] !=
            workers[1]["initial_buffers"]["joint_presence"], "Robot presence buffers were overwritten")


def audit_saved_profile(profile):
    import torch

    require(not torch.cuda.is_initialized(), "Checkpoint audit must remain on CPU")
    records = []
    names = dict(zip(OPTIMIZERS, ("wm", "policy", "critic", "kp", "critic_kp")))
    for robot in profile["robots"]:
        path = Path(profile["output"]) / "run" / robot / "ckpt_8.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        require(checkpoint["iteration"] == 8 and checkpoint["schedule"]["step"] == 8
                and checkpoint["schedule"]["world_size"] == 2
                and checkpoint["native_q1_contract"]["code_commit"] == profile["code_commit"],
                "Saved checkpoint identity or iteration differs")
        require(all(checkpoint["cfg"].get(k) == v for k, v in profile["cfg"].items()),
                "Saved checkpoint cfg differs from the tested profile")
        steps = {}
        for hook, name in names.items():
            states = checkpoint["optim"][name]["state"].values()
            steps[hook] = sorted({int(s["step"].item()) for s in states if "step" in s})
        require(steps == {k: [v] for k, v in COUNTS.items()}, "Saved Adam counts differ")
        tensors = []
        for module in ("wm", "policy", "critic", "critic_kp", "obs_norm", "goal_norm",
                       "human_norm", "heading_norm"):
            tensors.extend(t for t in checkpoint[module].values() if isinstance(t, torch.Tensor))
        require(tensors and all(torch.isfinite(t).all().item() for t in tensors),
                "Saved model or normalizer contains nonfinite values")
        records.append(dict(robot=robot, checkpoint=identity(path), adam_steps=steps,
                            finite_tensor_count=len(tensors)))
        del checkpoint, tensors, states
        gc.collect()
    require(not torch.cuda.is_initialized(), "CPU audit created a CUDA context")
    return records


def execute(config, config_path, commit):
    output = Path(config["output"])
    record = dict(status="INITIALIZING", process=process_identity(), started_utc=now(),
                  code_commit=commit, config=identity(config_path), stages=[],
                  automatic_retry=False, reuse_profile_weights=False, user_accepted_checkpoint=None)
    child = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def phase(name, purpose, command, seconds, extra=()):
        nonlocal child
        require(not stopped, "Pipeline was stopped; do not start another stage")
        root = output / name
        record.update(status="TRAINING" if purpose == "train" else "VALIDATING", phase=name)
        save(output / "pipeline.json", record)
        cmd = [sys.executable, "-u", str(HERE / "run_native.py"), "--output", str(root),
               "--source", profile["source"], "--purpose", purpose, "--code-commit", commit,
               "--wall-seconds", str(seconds), "--initial-admission-gib", "12",
               "--expected-boot", config["expected_boot"], "--expected-gpu", config["expected_gpu"]]
        for path in sorted({*inputs, *map(str, extra)}):
            cmd.extend(("--input", path))
        with (output / (name + "_controller.log")).open("x") as stream:
            child = subprocess.Popen([*cmd, "--", *command], cwd=ROOT,
                                     stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            code = child.wait()
        child = None
        require(code == 0 and not stopped, "Guarded stage failed: " + name)
        check_phase(root, purpose, commit)
        record["stages"].append(dict(name=name, status="COMPLETED", ended_utc=now()))
        save(output / "pipeline.json", record)

    try:
        template_path = Path(config["profile_config"])
        profile, formal = build_plans(config, json.loads(template_path.read_text()), commit)
        baseline = json.loads(Path(config["baseline_input_receipt"]).read_text())
        # The failed run supplies only immutable input identities, never a success gate or weights.
        require(len(baseline["inputs"]) == 1820, "Pinned input closure changed")
        for item in baseline["inputs"]:
            require(identity(Path(item["path"])) == item, "Pinned input changed: " + item["path"])
        inputs = [x["path"] for x in baseline["inputs"]]
        inputs.extend((str(config_path), str(template_path)))
        record["verified_baseline_inputs"] = len(baseline["inputs"])
        for purpose, plan in (("profile", profile), ("train", formal)):
            save(output / (purpose + "_plan.json"), plan)
        for robot, subset in zip(profile["robots"], profile["subsets"]):
            phase(robot + "_preflight", "preflight", [sys.executable, "-u",
                  str(HERE / "verify_native_gpu.py"), "--source", profile["source"],
                  "--subset", subset, "--robot", robot, "--output",
                  str(output / (robot + "_preflight") / "result.json")],
                  config["native_wall_seconds"])
            check_native(json.loads((output / (robot + "_preflight") / "result.json").read_text()), robot)
        phase("profile", "profile", [sys.executable, "-u", str(HERE / "train_native.py"),
              "--plan", str(output / "profile_plan.json")], config["profile_wall_seconds"],
              [output / "profile_plan.json"])
        workers = [json.loads((output / "profile/workers" / (r + ".json")).read_text())
                   for r in profile["robots"]]
        check_profile(workers, profile)
        record["checkpoint_audit"] = audit_saved_profile(profile)
        record["steady_iteration_seconds"] = {
            w["robot"]: statistics.median(w["iteration_seconds"][-6:]) for w in workers}
        record["validation_passed_utc"] = now()
        record["formal_iteration_only_hours_estimate"] = (
            max(record["steady_iteration_seconds"].values()) * 80000 / 3600)
        save(output / "validation.json", record)
        phase("train", "train", [sys.executable, "-u", str(HERE / "train_native.py"),
              "--plan", str(output / "train_plan.json")], config["train_wall_seconds"],
              [output / "train_plan.json", output / "validation.json"])
        record.update(status="TRAINING_COMPLETED_PENDING_PHYSICAL_EVALUATION", ended_utc=now())
    except BaseException as error:
        record.update(status="FAILED", ended_utc=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        save(output / "pipeline.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--controller", action="store_true")
    args = parser.parse_args()
    path = args.config.resolve(strict=True)
    config = json.loads(path.read_text())
    output = Path(config["output"]).resolve()
    require(path.is_relative_to(ROOT) and output.is_relative_to(ROOT / "data/supermimic_for_q1"),
            "Use the sole project root")
    build_plans(config, json.loads(Path(config["profile_config"]).read_text()), args.code_commit)
    if not args.controller:
        output.mkdir(parents=True, exist_ok=False)
    if args.detach:
        forwarded = [v for v in sys.argv[1:] if v != "--detach"]
        with (output / "pipeline_controller.log").open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--controller", *forwarded], stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True, cwd=ROOT)
        launch = dict(controller_pid=child.pid, code_commit=args.code_commit,
                      output=str(output), started_utc=now())
        save(output / "launch.json", launch)
        print(json.dumps(launch), flush=True)
    else:
        execute(config, path, args.code_commit)


if __name__ == "__main__":
    main()
