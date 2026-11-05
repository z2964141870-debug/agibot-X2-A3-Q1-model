"""One bounded Q1 intermediate review; preserve the separate final-review gate."""

import argparse
import gc
import importlib.metadata
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from contracts import valid_commit, validate_frozen
from common import FIXED, ROOT, identity, now, require, save
from delivery import read
from memory_watch import memory, process, subtree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "supermimic_q1_sources"))
from runtime_guard import allocated_tree, guarded_run

GIB = 1024**3
TRAINING = ROOT / "data/supermimic_for_q1/MORPHMIMIC_Q1_CONDITIONAL_SHARED01_V04_20261104"
COMMIT = "4bef63517164537a9074b7ce58f28e0da35ac8b5"
HANDLES = {54868: "7431283", 55672: "7449427", 55677: "7449543",
           55678: "7449555", 55679: "7449555"}


def resource_limits(sample, gpu, initial=False):
    require(sample["container_limit_bytes"] > 30 * GIB, "Container capacity changed")
    require(sample["oom_kill"] == 0 and sample["under_oom"] == 0, "Cgroup OOM changed")
    require(sample["total_bytes"] < (22 if initial else 24) * GIB,
            "Intermediate review memory reserve reached; stop only this review")
    require(gpu["used_mib"] < 18 * 1024 and gpu["temperature_c"] < 80
            and gpu["recovery"] == "None", "Intermediate graphics reserve reached")


def fixed_rows(frozen):
    rows = {Path(row["q1_primebot"]["path"]).stem: row for row in frozen["rows"]}
    require(set(FIXED) <= rows.keys(), "Missing fixed intermediate clips")
    return [rows[name] for name in FIXED]


def live(config):
    require(Path.cwd().resolve() == ROOT, "Require the sole physical server root")
    require(Path("/proc/sys/kernel/random/boot_id").read_text().strip() == config["expected_boot"],
            "Server boot changed")
    for pid, ticks in HANDLES.items():
        row = process(Path("/proc"), pid)
        require(row is not None and row["start_ticks"] == ticks, "Original training PID identity changed")
    require(read(TRAINING / "pipeline.json")["status"] == "TRAINING"
            and read(TRAINING / "e1024/train/guard.json")["status"] == "RUNNING",
            "Require the original live training and guard")
    require(any(" WRITE 55672 " in line for line in Path("/proc/locks").read_text().splitlines()),
            "Original training GPU lock is no longer held")


def gpu_snapshot(config):
    text = subprocess.check_output(["nvidia-smi",
        "--query-gpu=uuid,memory.used,temperature.gpu,gpu_recovery_action",
        "--format=csv,noheader,nounits"], text=True, timeout=10).strip().split(",")
    gpu, used, temperature, recovery = [value.strip() for value in text]
    require(gpu == config["expected_gpu"], "GPU identity changed")
    return dict(uuid=gpu, used_mib=int(used), temperature_c=int(temperature), recovery=recovery)


class ReviewBudget:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.started = time.monotonic()
        self.last_gpu, self.gpu_time = None, 0.
        self.peak_total, self.peak_rss, self.peak_gpu = 0, 0, 0

    def check(self):
        live(self.config)
        sample = memory(Path("/sys/fs/cgroup/memory"))
        owner = process(Path("/proc"), os.getpid())
        rows = subtree(Path("/proc"), owner)
        require(not (set(HANDLES) & {row["pid"] for row in rows}), "Training entered the review subtree")
        own_rss = sum(row.get("VmRSS_bytes", 0) for row in rows)
        current = time.monotonic()
        if self.last_gpu is None or current - self.gpu_time >= 2:
            self.last_gpu, self.gpu_time = gpu_snapshot(self.config), current
        require(current - self.started < 1800, "Intermediate review reached30min wall bound")
        tree = allocated_tree(self.output)
        require(tree["unique_allocated_bytes"] < 3 * GIB
                and shutil.disk_usage(self.output).free >= 100 * GIB,
                "Intermediate review storage reserve reached")
        self.peak_total = max(self.peak_total, sample["total_bytes"])
        self.peak_rss = max(self.peak_rss, own_rss)
        self.peak_gpu = max(self.peak_gpu, self.last_gpu["used_mib"])
        row = dict(memory=sample, review_subtree_rss_bytes=own_rss, gpu=self.last_gpu,
            maximum_total_bytes=self.peak_total, maximum_review_rss_bytes=self.peak_rss,
            maximum_gpu_used_mib=self.peak_gpu, storage=tree, training_signals_sent=0,
            memory_reclaim_attempted=False, gpu_lock_owner=55672,
            note="RSS can count shared mappings repeatedly; limits stop only the review.")
        save(self.output / "resources.json", row)
        resource_limits(sample, self.last_gpu)
        return row


def preflight(args):
    require(valid_commit(args.evaluation_commit), "Require independently published review code")
    require(args.iteration == 16000, "This authorization covers the first trained16000 checkpoint")
    require(args.output.is_absolute() and args.output.resolve() == args.output
            and args.output.is_relative_to(ROOT / "data/supermimic_for_q1")
            and not args.output.is_relative_to(TRAINING) and not args.output.exists(),
            "Use a new independent review root outside all training outputs")
    config, runtime = read(args.config), read(args.runtime_config)
    require(Path(config["output"]) == TRAINING, "Not the authorized formal experiment")
    live(config)
    resource_limits(memory(Path("/sys/fs/cgroup/memory")), gpu_snapshot(config), initial=True)
    require(shutil.disk_usage(ROOT).free >= 103 * GIB, "Require review budget plus100GiB")
    pipeline = read(TRAINING / "pipeline.json")
    plan_path = TRAINING / "e1024_train_plan.json"
    plan = read(plan_path)
    require(pipeline["formal_plan"] == identity(plan_path) and plan["code_commit"] == COMMIT
            and pipeline["code_commit"] == COMMIT and plan["purpose"] == "train"
            and plan["cfg"]["iterations"] == 160000 and plan["robots"] == ["g1", "q1"],
            "Formal training plan identity changed")
    require(runtime["morph_source"] == plan["source"]
            and runtime["morph_python"] == config["morph_python"], "Runtime and training plan differ")
    frozen_path = Path(config["frozen_data"])
    require(identity(frozen_path)["sha256"] == config["frozen_data_sha256"], "Frozen data changed")
    frozen = read(frozen_path)
    validate_frozen(frozen)
    for robot in ("g1", "q1"):
        worker = read(TRAINING / "e1024/train/workers" / (robot + ".json"))
        require(worker["iterations_completed"] >= args.iteration
                and not any(worker["nonfinite_updates"].values()), "Training has not reached a finite checkpoint")
        checkpoint = Path(plan["output"]) / "run" / robot / (f"ckpt_{args.iteration}.pt")
        require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Missing trained checkpoint")
    inputs = read(TRAINING / "e1024/train/status.json")["inputs"]
    require(str(Path(__file__).resolve()) not in {row["path"] for row in inputs},
            "Intermediate entry must not be a frozen training input")
    source_inputs = [row for row in inputs if Path(row["path"]).suffix in (".py", ".xml")]
    require(source_inputs and all(identity(row["path"]) == row for row in source_inputs),
            "Frozen training source or assets changed")
    return config, runtime, plan, frozen, source_inputs


def reusable_audit(path, plan, iteration):
    value = read(path)
    authorization = read(path.parent / "authorization.json")
    guard = read(path.parent / "audit.guard.json")
    require(authorization["original_training_commit"] == COMMIT
            and authorization["plan"] == identity(TRAINING / "e1024_train_plan.json")
            and valid_commit(authorization["evaluation_commit"]), "Reused audit has another provenance")
    require(guard["status"] == "WORKER_COMPLETED"
            and guard["cleanup"]["worker_reaped"] is True
            and guard["cleanup"]["worker_return_code"] == 0
            and value["status"] == "PASSED" and value["cuda_initialized"] is False
            and value["iteration"] == iteration, "Reuse requires a clean completed CPU audit")
    require({row["robot"] for row in value["records"]} == {"g1", "q1"}
            and len(value["records"]) == 2, "Reuse requires both robot audits")
    for row in value["records"]:
        expected = Path(plan["output"]) / "run" / row["robot"] / f"ckpt_{iteration}.pt"
        require(row["iteration"] == iteration and row["finite_tensors"] == 330
                and row["inference_metadata"]["code_commit"] == COMMIT
                and Path(row["checkpoint"]["path"]) == expected
                and identity(expected) == row["checkpoint"], "Audited checkpoint changed before reuse")
    return value


def validate_saved_case(row, item, directory):
    name = Path(item["q1_primebot"]["path"]).stem
    require(row["clip"] == name and row["seed"] == 0 and row["interface"] == "motion"
            and row["evaluation_randomization"] is False and row["complete"] is True
            and row["failure_frame"] is None and row["valid_steps"] > 0
            and row["valid_steps"] == row["attempted_steps"] == row["reference_steps"]
            and Path(row["trace"]) == directory / "rollout.npz"
            and row["reference_sha256"] == item["q1_primebot"]["sha256"]
            and row["native_reference_sha256"] == item["q1_morph"]["sha256"]
            and row["human_sha256"] == item["human"]["sha256"],
            "Saved intermediate case has another reference or incomplete provenance")


def reusable_cases(path, audit, sources, frozen):
    if path is None:
        return []
    require(path.is_absolute() and path.resolve() == path
            and path.is_relative_to(ROOT / "data/supermimic_for_q1")
            and not path.is_relative_to(TRAINING), "Reuse only an independent review root")
    authorization, guard = read(path / "authorization.json"), read(path / "replay.guard.json")
    require(authorization["stage"] == "INTERMEDIATE_ONLY"
            and authorization["original_training_commit"] == COMMIT
            and valid_commit(authorization["evaluation_commit"])
            and authorization["plan"] == identity(TRAINING / "e1024_train_plan.json")
            and authorization["source_inputs"] == sources
            and read(path / "checkpoint_audit.json") == audit,
            "Saved review has another checkpoint, plan or frozen source")
    require(guard["status"] == "FAILED" and guard["cleanup"]["worker_reaped"] is True
            and guard["cleanup"]["worker_return_code"] == -15
            and guard["error"] == "RuntimeError: Intermediate review memory reserve reached; stop only this review",
            "Reuse only the preserved, reaped intermediate memory-limit exit")
    cases = []
    for item in fixed_rows(frozen):
        directory = path / "review" / (Path(item["q1_primebot"]["path"]).stem + "_s0")
        if not (directory / "result.json").is_file():
            continue
        row = read(directory / "result.json")
        validate_saved_case(row, item, directory)
        files = [directory / name for name in
                 ("result.json", "rollout.npz", "foot_posture.csv", "upper_timing.csv")]
        if (directory / "tracking.mp4").is_file():
            files += [directory / "tracking.mp4"] + sorted(directory.glob("frame_*.png"))
        require(all(p.is_file() and not p.is_symlink() and 0 < p.stat().st_size < 8 * 1024**2
                    for p in files), "Reuse requires bounded regular case files")
        cases.append(dict(case=row, files=[identity(p) for p in files],
                          source_authorization=identity(path / "authorization.json"),
                          source_guard=identity(path / "replay.guard.json")))
    require(cases, "No completed physical intermediate case to reuse")
    return cases


def copy_saved_cases(cases, output):
    for saved in cases:
        row = saved["case"]
        directory = output / "review" / (row["clip"] + "_s0")
        directory.mkdir(parents=True, exist_ok=False)
        for original in saved["files"]:
            require(identity(original["path"]) == original, "Saved review changed before copy")
            destination = directory / Path(original["path"]).name
            shutil.copyfile(original["path"], destination)
            copied = identity(destination)
            require((copied["bytes"], copied["sha256"]) == (original["bytes"], original["sha256"]),
                    "Saved review copy differs")
        row.update(trace=str(directory / "rollout.npz"), reused_physical_case=saved["files"],
                   source_authorization=saved["source_authorization"], source_guard=saved["source_guard"])
        save(directory / "result.json", row)


def audit(args, config, plan):
    import torch
    torch.set_num_threads(1)
    from controller import audit_checkpoint
    from inference import validate_metadata
    value = audit_checkpoint(plan, args.iteration)
    for row in value["records"]:
        saved = torch.load(row["checkpoint"]["path"], map_location="cpu", weights_only=False, mmap=True)
        require(saved["native_q1_contract"]["plan"] == plan, "Saved checkpoint has another training plan")
        row["inference_metadata"] = validate_metadata(saved, row["robot"], COMMIT)
        del saved
        gc.collect()
    require(not torch.cuda.is_initialized(), "Checkpoint audit initialized CUDA")
    save(args.output / "checkpoint_audit.json", value)


def load_physical_robot(checkpoint):
    import torch
    from supermimic.robot import Q1Robot
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    settings, model_id = payload["config"], payload["identity"]
    robot = Q1Robot(Path(settings["robot_xml"]),
                    modeled_action_delay=settings.get("modeled_action_delay", False))
    require(robot.asset_sha256 == model_id["asset_sha256"], "Common physical assets differ")
    robot.load_state_dict(payload["robot"])
    del payload
    gc.collect()
    return robot, settings


def replay(args, config, runtime, plan, frozen):
    import torch
    torch.set_num_threads(1)
    require(Path(sys.executable).absolute() == Path(runtime["primebot_python"])
            and importlib.metadata.version("torch").split("+")[0] == "2.7.0"
            and importlib.metadata.version("mujoco") == "3.6.0", "Common physics runtime changed")
    sys.path[:0] = [str(Path(runtime["adapted_source"]) / "src"), str(HERE.parent / "primebot_q1_review")]
    from policy_process import PolicyProcess
    from q1_native_review import rollout
    audited = read(args.output / "checkpoint_audit.json")
    require(audited["status"] == "PASSED" and audited["iteration"] == args.iteration,
            "Require the trained checkpoint CPU audit")
    checkpoint = next(row["checkpoint"] for row in audited["records"] if row["robot"] == "q1")
    require(identity(checkpoint["path"]) == checkpoint, "Checkpoint changed after audit")
    baseline = read(Path(config["primebot_review"]) / "checkpoint_audit.json")["final"]["checkpoint"]
    require(identity(baseline["path"]) == baseline, "Original baseline checkpoint changed")
    output = args.output / "review"
    output.mkdir(exist_ok=True)
    command = [runtime["morph_python"], "-u", str(HERE / "policy_process.py")]
    for key, value in {"plan": TRAINING / "e1024_train_plan.json", "checkpoint": checkpoint["path"],
            "frozen": config["frozen_data"], "primebot-checkpoint": baseline["path"],
            "morph-source": plan["source"], "primebot-source": runtime["adapted_source"]}.items():
        command.extend(("--" + key, str(value)))
    result = dict(status="RUNNING", stage="INTERMEDIATE_ONLY", iteration=args.iteration,
        interface="motion", seed=0, cases=[], checkpoint=checkpoint, optimizer_updates=0,
        evaluation_commit=args.evaluation_commit, final_acceptance=False, g1_quality_assessed=False,
        user_accepted_checkpoint=None, rendering="GPU EGL; original-speed libx264 video",
        physics="Locked Primebot CPU Torch2.7/MuJoCo3.6", training_set_only=True)
    actor = None
    try:
        actor = PolicyProcess(command, output / "policy_worker.log")
        require(actor.ready["checkpoint"] == checkpoint
                and actor.ready["metadata"]["iteration"] == args.iteration, "Wrong isolated native policy")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5, "Common physical action scale changed")
        result.update(policy_runtime=actor.ready, asset_sha256=robot.asset_sha256,
                      termination=robot.reward_identity(), render_xml=identity(settings["robot_xml"]),
                      unused_primebot_actor_constructed=False)
        require(Path(settings["robot_xml"]) == Path(runtime["xml"]), "Common rendering XML changed")
        for item in fixed_rows(frozen):
            directory = output / (Path(item["q1_primebot"]["path"]).stem + "_s0")
            if (directory / "result.json").is_file():
                row = read(directory / "result.json")
                validate_saved_case(row, item, directory)
            else:
                row = rollout(robot, actor, settings, item, "motion", 0, directory)
            result["cases"].append(row)
            save(output / "progress.json", result)
        actor.close()
        require(not torch.cuda.is_initialized(), "Common CPU policy or physics initialized CUDA")
        require(identity(checkpoint["path"]) == checkpoint, "Checkpoint changed during replay")
        result.update(status="PHYSICS_PASSED", complete_cases=sum(row["complete"] for row in result["cases"]),
            policy_worker_reaped=True, policy_worker_return_code=actor.child.returncode,
            cuda_initialized=False, ended_utc=now())
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        if actor is not None and not actor.closed:
            actor.close(graceful=False)
        save(output / "result.json", result)


def media(args, config, runtime):
    import mujoco
    from OpenGL import GL
    from video import decoded_metadata, render_case
    output = args.output / "review"
    result = read(output / "result.json")
    require(result["status"] == "PHYSICS_PASSED" and result["policy_worker_reaped"] is True
            and result["policy_worker_return_code"] == 0 and "torch" not in sys.modules,
            "Render only after both policy and physics runtimes have exited")
    xml = result["render_xml"]
    require(identity(xml["path"]) == xml and Path(xml["path"]) == Path(runtime["xml"]),
            "Saved physical rendering asset changed")
    context = mujoco.GLContext(1, 1)
    try:
        context.make_current()
        graphics = {key: GL.glGetString(value).decode() for key, value in
                    (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER), ("version", GL.GL_VERSION))}
        require("NVIDIA" in graphics["vendor"], "Intermediate video requires the requested NVIDIA EGL GPU")
    finally:
        context.free()
    result["graphics"] = graphics
    for row in result["cases"]:
        directory = Path(row["trace"]).parent
        video = directory / "tracking.mp4"
        if video.exists():
            hashes = subprocess.check_output([runtime["ffmpeg"], "-nostdin", "-v", "error",
                "-threads", "1", "-i", str(video), "-map", "0:v:0", "-pix_fmt", "rgb24",
                "-f", "framehash", "-hash", "sha256", "pipe:1"], text=True)
            row["video"] = dict(status="PASSED", **identity(video),
                probe=decoded_metadata(hashes, row["reference_steps"] + 1), original_speed=True,
                actual_frames_before_hold=row["attempted_steps"] + 1,
                failure_preserved=True, reused_encoded_video=True)
        else:
            row["video"] = render_case(Path(xml["path"]), row, video)
        frames = [directory / f"frame_{i:05d}.png" for i in
                  (0, (row["reference_steps"] + 1) // 2, row["reference_steps"])]
        require(all(p.is_file() for p in frames), "Missing original-speed inspection frames")
        row["inspection_frames"] = [identity(p) for p in frames]
        save(directory / "result.json", row)
        save(output / "progress.json", result)
    result.update(status="PASSED", media_ended_utc=now(), media_torch_imported=False)
    save(output / "result.json", result)


def execute(args):
    if args.child:
        config, runtime = read(args.config), read(args.runtime_config)
        plan = read(TRAINING / "e1024_train_plan.json")
        if args.child == "audit":
            return audit(args, config, plan)
        if args.child == "media":
            return media(args, config, runtime)
        return replay(args, config, runtime, plan, read(config["frozen_data"]))
    config, runtime, plan, frozen, sources = preflight(args)
    reused = reusable_audit(args.reuse_audit, plan, args.iteration) if args.reuse_audit else None
    require(args.reuse_review is None or reused is not None, "Saved cases require the completed checkpoint audit")
    cases = reusable_cases(args.reuse_review, reused, sources, frozen)
    args.output.mkdir(exist_ok=False)
    save(args.output / "authorization.json", dict(stage="INTERMEDIATE_ONLY", started_utc=now(),
        evaluation_commit=args.evaluation_commit, original_training_commit=COMMIT,
        plan=identity(TRAINING / "e1024_train_plan.json"), source_inputs=sources,
        cases=[dict(clip=name, interface="motion", seed=0) for name in FIXED],
        memory_admission_bytes=22 * GIB, memory_exit_bytes=24 * GIB, subtree_rss_is_diagnostic=True,
        gpu_used_exit_mib=18 * 1024, gpu_temperature_exit_c=80,
        training_guard_pressure_bytes=30 * GIB, training_guard_gpu_lock_owner=55672,
        concurrent_graphics_requested_by_user=True, training_inputs_modified=False,
        reused_checkpoint_audit=identity(args.reuse_audit) if args.reuse_audit else None,
        reused_physical_cases=cases, sequential_physics_then_gpu_media=True,
        evaluation_allocator=dict(MALLOC_ARENA_MAX="1", MALLOC_TRIM_THRESHOLD_="131072",
                                  MALLOC_MMAP_THRESHOLD_="131072"),
        final_acceptance=False, g1_quality_assessed=False, user_accepted_checkpoint=None))
    if reused is not None:
        save(args.output / "checkpoint_audit.json", reused)
    copy_saved_cases(cases, args.output)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="1", MALLOC_TRIM_THRESHOLD_="131072",
        MALLOC_MMAP_THRESHOLD_="131072", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless",
        Q1_REVIEW_FFMPEG=runtime["ffmpeg"],
        __EGL_VENDOR_LIBRARY_FILENAMES=str(ROOT / "script/morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
        LD_LIBRARY_PATH=str(Path(runtime["graphics_dir"]) / "lib")
            + (os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
        GIT_PYTHON_GIT_EXECUTABLE=str(ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"))
    requested = False

    def stop(signum, frame):
        nonlocal requested
        requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    budget = ReviewBudget(config, args.output)
    previous = dict(os.environ)
    try:
        os.environ.update(env)
        for name, python in (("audit", runtime["morph_python"]), ("replay", runtime["primebot_python"]),
                             ("media", runtime["primebot_python"])):
            if name == "audit" and reused is not None:
                continue
            command = [python, "-u", str(Path(__file__).resolve()), "--child", name,
                "--config", str(args.config), "--runtime-config", str(args.runtime_config),
                "--output", str(args.output), "--iteration", str(args.iteration),
                "--evaluation-commit", args.evaluation_commit]
            result = guarded_run(command, ROOT, budget, args.output / (name + ".guard.json"),
                args.output / (name + ".log"), check_interval_s=.5, grace_s=5,
                stop_requested=lambda: requested)
            require(result["status"] == "WORKER_COMPLETED"
                    and result["cleanup"]["worker_reaped"] is True
                    and result["cleanup"]["worker_return_code"] == 0, "Intermediate phase failed; no retry")
        require(all(identity(row["path"]) == row for row in sources), "Frozen source changed during review")
        save(args.output / "source_integrity.json", dict(status="PASSED", source_inputs=sources,
            original_training_inputs_modified=False, training_signals_sent=0))
        return read(args.output / "review/result.json")
    finally:
        os.environ.clear()
        os.environ.update(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("config", "runtime-config", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--reuse-audit", type=Path)
    parser.add_argument("--reuse-review", type=Path)
    parser.add_argument("--child", choices=("audit", "replay", "media"))
    execute(parser.parse_args())
