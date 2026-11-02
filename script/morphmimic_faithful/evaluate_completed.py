"""Audit the completed G1 run and call its pinned upstream deployment pipeline."""

import argparse
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
SOURCE = ROOT / "script/third_party/morphmimic_aa40fe46"
PYTHON = ROOT / "data/environments/morphmimic_original_20261029/bin/python"
TREE = ROOT / "data/references/morphmimic_review_20261029/tree.json"
sys.path.insert(0, str(HERE))
from supervise import Q1Budget, guarded_run, process_identity, save, verify_upstream


def now():
    return datetime.now(timezone.utc).isoformat()


def identity(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=value.hexdigest())


def optimizer_steps(iteration, warmup=2000):
    policy = max(0, iteration - warmup)
    return dict(wm=iteration * 10, policy=policy, critic=policy * 20,
                kp=policy, critic_kp=policy * 20)


def completed_training(experiment):
    states = {name: json.loads((experiment / name).read_text()) for name in
              ("status.json", "guard.json", "run/progress.json")}
    status, guard, progress = (states[n] for n in states)
    if (status["status"] != "COMPLETED" or guard["status"] != "WORKER_COMPLETED"
            or guard["cleanup"]["worker_return_code"] != 0
            or not guard["cleanup"]["worker_reaped"]
            or progress["status"] != "TRAINING_COMPLETED"
            or progress["iteration"] != 160000 or progress["nonfinite_steps"] != 0):
        raise ValueError("Original full training has not completed cleanly")
    for pid in (362416, 362421):
        if Path(f"/proc/{pid}").exists():
            raise RuntimeError("Original training PID is still present")
    observed = {key.removesuffix("_opt"): value
                for key, value in progress["optimizer_steps"].items()}
    if observed != optimizer_steps(160000):
        raise ValueError("Observed five optimizer totals differ from the recipe")
    return states, progress


def audit_checkpoints(run, progress, output):
    import torch

    torch.set_num_threads(4)
    baseline = None
    results = []
    for path in sorted(run.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[-1])):
        iteration = int(path.stem.split("_")[-1])
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["iteration"] != iteration or checkpoint["schedule"]["step"] != iteration:
            raise ValueError("Checkpoint iteration or schedule differs: " + path.name)
        expected = optimizer_steps(iteration)
        if set(checkpoint["optim"]) != set(expected):
            raise ValueError("Missing optimizer state: " + path.name)
        steps = {}
        for name, optimizer in checkpoint["optim"].items():
            values = sorted({int(v["step"].item()) for v in optimizer["state"].values()})
            if values != ([expected[name]] if expected[name] else []):
                raise ValueError("Saved Adam steps differ: " + path.name + " " + name)
            if any(not torch.isfinite(t).all().item()
                   for state in optimizer["state"].values() for t in state.values()
                   if isinstance(t, torch.Tensor) and t.is_floating_point()):
                raise ValueError("Nonfinite Adam state: " + path.name)
            steps[name] = values
        entries = ("wm", "policy", "critic", "critic_kp", "obs_norm", "goal_norm", "kp_norm")
        for name in entries:
            if any(not torch.isfinite(t).all().item() for t in checkpoint[name].values()
                   if isinstance(t, torch.Tensor) and t.is_floating_point()):
                raise ValueError("Nonfinite saved tensor: " + path.name + " " + name)
        if iteration == 0:
            baseline = {name: {k: v.clone() for k, v in checkpoint[name].items()}
                        for name in ("wm", "policy", "critic", "critic_kp")}
        changes = {}
        if iteration in (40000, 160000):
            for name, initial in baseline.items():
                changes[name] = max(float((v.float() - initial[k].float()).abs().max())
                                    for k, v in checkpoint[name].items() if v.numel())
            if not all(v > 0 for v in changes.values()):
                raise ValueError("Unchanged network in trained checkpoint")
        result = dict(**identity(path), iteration=iteration, optimizer_steps=steps,
                      network_and_normalizer_tensors_finite=True,
                      adam_state_tensors_finite=True, max_change_from_initial=changes)
        results.append(result)
        save(output / "checkpoint_audit.json", dict(status="RUNNING", checkpoints=results))
        del checkpoint
        gc.collect()
    if {r["iteration"] for r in results} != {0, 16000, 32000, 40000, 48000, 64000,
                                             80000, 96000, 112000, 128000, 144000, 160000}:
        raise ValueError("Checkpoint inventory differs from the completed recipe")
    latest = identity(run / "latest.pt")
    if latest["sha256"] != results[-1]["sha256"]:
        raise ValueError("latest.pt is not the final checkpoint")
    save(output / "checkpoint_audit.json", dict(status="COMPLETED", checkpoints=results,
          latest=latest, optimizer_updates_during_audit=0,
          observation_totals=progress["optimizer_steps"]))
    del baseline
    gc.collect()


def worker(args, plan):
    states, progress = completed_training(Path(plan["training_root"]))
    run = Path(progress["run_dir"])
    output = args.output / "run"
    output.mkdir()
    record = dict(status="RUNNING", started_utc=now(), phase="checkpoint_audit",
                  plan=plan, commands=[], replays=[], optimizer_updates=0,
                  user_accepted_checkpoint=None, original_training=states)

    def update():
        save(output / "progress.json", record)

    def command(label, argv, timeout):
        item = dict(label=label, argv=[str(x) for x in argv], timeout_seconds=timeout,
                    started_utc=now())
        record["phase"] = label
        record["commands"].append(item)
        update()
        started = time.monotonic()
        with (output / (label + ".log")).open("x") as stream:
            child = subprocess.Popen(item["argv"], cwd=SOURCE, stdout=stream,
                                     stderr=subprocess.STDOUT)
            try:
                item["returncode"] = child.wait(timeout=timeout)
            finally:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                item.update(ended_utc=now(), wall_seconds=time.monotonic() - started)
                update()
        if item.get("returncode") != 0:
            raise RuntimeError(label + " failed; its original log is retained")
        print("completed: " + label, flush=True)

    try:
        update()
        audit_checkpoints(run, progress, output)
        import yaml
        train_subset = Path(progress["config"]["subset_yaml"])
        train_names = {Path(p).stem for p in yaml.safe_load(train_subset.read_text())["clips"]}
        evaluation_subset = Path(plan["asset_subset_yaml"])
        data = yaml.safe_load(evaluation_subset.read_text())
        by_name = {Path(p).stem: Path(data["root"]) / p for p in data["clips"]}
        import numpy as np
        clips = []
        for requested in plan["clips"]:
            name = requested["name"]
            path = by_name[name]
            frames = len(np.load(path)["joint_pos"])
            if frames != requested["frames"]:
                raise ValueError("Fixed clip length differs: " + name)
            clips.append(dict(**requested, in_training_subset=name in train_names,
                              source=identity(path)))
        record["clips"] = clips
        record["input_identities"] = [identity(p) for p in (train_subset, evaluation_subset)]
        update()
        command("assets", [PYTHON, "-m", "deploy.sim2sim.export_assets", "--robot", "g1",
                "--subset-yaml", evaluation_subset, "--clips", *[c["name"] for c in clips],
                "--out", output / "assets"], 1800)
        for iteration in plan["deploy_iterations"]:
            exported = output / ("onnx_" + str(iteration))
            command("export_" + str(iteration), [PYTHON, "-m", "deploy.export_onnx",
                    run / f"ckpt_{iteration}.pt", exported], 900)
            interfaces = ["motion"] if iteration == 0 else ["motion", "keypoint"]
            for interface in interfaces:
                for clip in clips:
                    label = f"it{iteration}_{interface}_{clip['name']}"
                    summary = output / (label + ".json")
                    argv = [PYTHON, "-m", "deploy.run", "--deploy", exported, "--assets",
                            output / "assets", "--robot", "g1", "--clip", clip["name"],
                            "--interface", interface, "--summary", summary]
                    video = output / (label + ".mp4")
                    if iteration in plan["video_iterations"]:
                        argv += ["--video", video]
                    command(label, argv, min(6000, max(1800, 300 + clip["frames"] * 5)))
                    result = dict(iteration=iteration, interface=interface,
                                  in_training_subset=clip["in_training_subset"],
                                  upstream_summary=json.loads(summary.read_text()),
                                  summary=identity(summary))
                    if video.exists():
                        result["video"] = identity(video)
                    record["replays"].append(result)
                    update()
        command("curves", [PYTHON, HERE.parent / "morphmimic_repro/summarize_original.py",
                "--run", run, "--output", output / "curves", "--label",
                "MorphMimic G1 | 1024 envs | WM batch 512 | imag/KP batch 1024 | H32",
                "--wm-only-iters", "2000", "--smooth-iters", "2000"], 600)
        for item in record["input_identities"] + [c["source"] for c in clips]:
            if identity(Path(item["path"])) != item:
                raise ValueError("Evaluation input changed")
        record.update(status="EVALUATION_COMPLETED", phase="completed")
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["ended_utc"] = now()
        update()


class EvaluationBudget(Q1Budget):
    def check(self):
        result = super().check()
        if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != self.args.expected_boot:
            raise RuntimeError("Original boot identity changed")
        if result["gpu"]["uuid"] != self.args.expected_gpu:
            raise RuntimeError("Original GPU identity changed")
        if time.monotonic() - self.started > self.args.whole_wall_seconds:
            raise RuntimeError("Evaluation whole-wall limit reached")
        return result


def controller(args, plan):
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    status = dict(status="INITIALIZING", started_utc=now(), process=process_identity(),
                  code_commit=args.code_commit, automatic_retry=False,
                  goal_complete=False, training_restarted=False, optimizer_updates=0)
    lock = (ROOT / "data/supermimic_for_q1/.gait_goal01.gpu.lock").open("a")
    save(args.output / "status.json", status)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed_training(Path(plan["training_root"]))
        compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                           "--format=csv,noheader"], text=True)
        if compute.strip():
            raise RuntimeError("GPU has another compute process")
        memory = Path("/sys/fs/cgroup/memory")
        if int((memory / "memory.usage_in_bytes").read_text()) > 24 * 1024**3:
            raise RuntimeError("Original24GiB initial admission not satisfied")
        status["source_blobs_verified"] = verify_upstream(SOURCE, TREE)
        inputs = [args.plan, Path(__file__), HERE / "supervise.py",
                  HERE.parent / "morphmimic_repro/summarize_original.py"]
        status["entry_identities"] = [identity(p) for p in inputs]
        os.environ.update(PYTHONPATH=str(SOURCE / "src") + os.pathsep + str(SOURCE),
            GIT_PYTHON_GIT_EXECUTABLE=str(ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"),
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", TORCHINDUCTOR_COMPILE_THREADS="1",
            MUJOCO_GL="egl", PYTHONUNBUFFERED="1")
        command = [str(PYTHON), "-u", str(Path(__file__)), "--plan", str(args.plan),
                   "--output", str(args.output), "--code-commit", args.code_commit, "--worker"]
        budget_args = SimpleNamespace(output=args.output, runtime=SOURCE, mode="independent",
            expected_oom_kill=0, expected_boot=plan["expected_boot"],
            expected_gpu=plan["expected_gpu"], whole_wall_seconds=plan["whole_wall_seconds"])
        status.update(status="RUNNING", command=command)
        save(args.output / "status.json", status)
        result = guarded_run(command, ROOT, EvaluationBudget(budget_args),
            args.output / "guard.json", ROOT / "logs" / (args.output.name + ".log"),
            check_interval_s=2, grace_s=10, stop_requested=lambda: requested)
        if result["status"] != "WORKER_COMPLETED" or result["cleanup"]["worker_return_code"]:
            raise RuntimeError("Evaluation worker did not complete cleanly")
        status["source_blobs_unchanged"] = verify_upstream(SOURCE, TREE)
        if [identity(p) for p in inputs] != status["entry_identities"]:
            raise ValueError("Evaluation entries changed during execution")
        status["status"] = "COMPLETED"
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["ended_utc"] = now()
        save(args.output / "status.json", status)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--controller", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.plan = args.plan.resolve(strict=True)
    if not args.output.is_relative_to(ROOT / "data/supermimic_for_q1"):
        parser.error("Evaluation must use a new directory in the sole project root")
    if len(args.code_commit) != 40 or any(c not in "0123456789abcdef" for c in args.code_commit):
        parser.error("An independently published full40 commit is required")
    plan = json.loads(args.plan.read_text())
    if not args.worker and not args.controller:
        args.output.mkdir(exist_ok=False)
    if args.worker:
        worker(args, plan)
    elif args.detach:
        flags = [x for x in sys.argv[1:] if x != "--detach"]
        with (ROOT / "logs" / (args.output.name + "_controller.log")).open("x") as stream:
            child = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), *flags,
                "--controller"], stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / "launch.json", dict(controller_pid=child.pid,
             code_commit=args.code_commit, started_utc=now()))
        print(json.dumps(dict(controller_pid=child.pid, output=str(args.output))))
    else:
        controller(args, plan)


if __name__ == "__main__":
    main()
