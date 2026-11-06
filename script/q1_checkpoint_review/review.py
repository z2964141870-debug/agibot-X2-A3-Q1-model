"""Bounded saved-checkpoint comparisons, independent of formal final acceptance."""

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import FIXED, ROOT, identity, now, require, save
from contracts import valid_commit, validate_frozen
from delivery import read
from interim_review import (COMMIT, GIB, HANDLES, TRAINING, ReviewBudget, audit,
    fixed_rows, gpu_snapshot, live, load_physical_robot, resource_limits, reusable_audit)
from memory_watch import memory, process, subtree
from runtime_guard import allocated_tree, guarded_run

ITERATIONS = (16000, 32000, 48000)
INTERFACES = ("motion", "human")
OLD_REVIEW = ROOT / "data/supermimic_for_q1/Q1_SHARED_INTERIM_CKPT16000_LEAN_20261105"
OLD_AUDIT = ROOT / "data/supermimic_for_q1/Q1_SHARED_INTERIM_CKPT16000_20261105/checkpoint_audit.json"


def check_iteration(value):
    require(type(value) is int and value in ITERATIONS, "Only the three authorized saved checkpoints")
    return value


def checkpoint_audit(output, iteration):
    return output / "checkpoints" / str(check_iteration(iteration)) / "checkpoint_audit.json"


def audit_record(output, iteration, robot):
    value = read(checkpoint_audit(output, iteration))
    require(value["status"] == "PASSED" and value["cuda_initialized"] is False
            and value["iteration"] == iteration, "Require a completed CPU checkpoint audit")
    rows = value["records"]
    require(len(rows) == 2 and {r["robot"] for r in rows} == {"g1", "q1"}, "Require both robot audits")
    selected = next(row for row in rows if row["robot"] == robot)
    require(identity(selected["checkpoint"]["path"]) == selected["checkpoint"], "Audited checkpoint changed")
    return selected["checkpoint"]


def preflight(args):
    require(valid_commit(args.evaluation_commit), "Publish the independent review code first")
    require(args.output.is_absolute() and args.output.resolve() == args.output
            and args.output.is_relative_to(ROOT / "data/supermimic_for_q1")
            and not args.output.is_relative_to(TRAINING) and not args.output.exists(),
            "Require a new output outside the frozen training tree")
    config, runtime = read(args.config), read(args.runtime_config)
    live(config)
    resource_limits(memory(Path("/sys/fs/cgroup/memory")), gpu_snapshot(config), initial=True)
    require(shutil.disk_usage(ROOT).free >= 103 * GIB, "Keep output budget plus100GiB free")
    plan_path = TRAINING / "e1024_train_plan.json"
    plan, pipeline = read(plan_path), read(TRAINING / "pipeline.json")
    require(Path(config["output"]) == TRAINING and plan["code_commit"] == pipeline["code_commit"] == COMMIT
            and pipeline["formal_plan"] == identity(plan_path) and plan["purpose"] == "train"
            and plan["cfg"]["iterations"] == 160000 and plan["robots"] == ["g1", "q1"],
            "Require the original unmodified live shared plan")
    require(runtime["morph_source"] == plan["source"]
            and runtime["morph_python"] == config["morph_python"], "Pinned runtime changed")
    require(identity(config["frozen_data"])["sha256"] == config["frozen_data_sha256"], "Frozen data changed")
    frozen = read(config["frozen_data"])
    validate_frozen(frozen)
    for robot in plan["robots"]:
        worker = read(TRAINING / "e1024/train/workers" / (robot + ".json"))
        require(worker["iterations_completed"] >= max(ITERATIONS)
                and not any(worker["nonfinite_updates"].values()), "Missing finite trained checkpoint")
    inputs = read(TRAINING / "e1024/train/status.json")["inputs"]
    require(not any(Path(r["path"]).is_relative_to(HERE) for r in inputs), "New code entered frozen inputs")
    sources = [r for r in inputs if Path(r["path"]).suffix in (".py", ".xml")]
    require(sources and all(identity(r["path"]) == r for r in sources), "Frozen source or assets changed")
    reused = reusable_audit(OLD_AUDIT, plan, 16000)
    old = read(OLD_REVIEW / "review/result.json")
    require(old["status"] == "PASSED" and old["iteration"] == 16000 and old["interface"] == "motion"
            and old["complete_cases"] == 3 and old["checkpoint"] == next(
                r["checkpoint"] for r in reused["records"] if r["robot"] == "q1"), "Wrong saved16k review")
    require({r["clip"] for r in old["cases"]} == set(FIXED)
            and all(r["seed"] == 0 and r["complete"] for r in old["cases"]), "Incomplete saved16k cases")
    for row, item in zip(sorted(old["cases"], key=lambda r: r["clip"]),
                         sorted(fixed_rows(frozen), key=lambda r: Path(r["q1_primebot"]["path"]).stem)):
        require(row["reference_sha256"] == item["q1_primebot"]["sha256"]
                and row["native_reference_sha256"] == item["q1_morph"]["sha256"]
                and row["human_sha256"] == item["human"]["sha256"], "Saved16k references changed")
    return config, runtime, plan, frozen, sources, reused


class Budget(ReviewBudget):
    def __init__(self, config, output):
        super().__init__(config, output)
        self.phase_started = self.started

    def check(self):
        live(self.config)
        sample = memory(Path("/sys/fs/cgroup/memory"))
        rows = subtree(Path("/proc"), process(Path("/proc"), os.getpid()))
        require(not (set(HANDLES) & {r["pid"] for r in rows}), "Training entered the review subtree")
        current = time.monotonic()
        require(current - self.started < 7200 and current - self.phase_started < 1800,
                "Review whole or phase wall-time limit reached")
        if self.last_gpu is None or current - self.gpu_time >= 2:
            self.last_gpu, self.gpu_time = gpu_snapshot(self.config), current
        storage = allocated_tree(self.output)
        require(storage["unique_allocated_bytes"] < 3 * GIB and shutil.disk_usage(self.output).free >= 100 * GIB,
                "Review storage reserve reached")
        self.peak_total = max(self.peak_total, sample["total_bytes"])
        self.peak_gpu = max(self.peak_gpu, self.last_gpu["used_mib"])
        self.peak_rss = max(self.peak_rss, sum(r.get("VmRSS_bytes", 0) for r in rows))
        result = dict(memory=sample, gpu=self.last_gpu, storage=storage, maximum_total_bytes=self.peak_total,
            maximum_gpu_used_mib=self.peak_gpu, maximum_subtree_rss_bytes=self.peak_rss,
            training_signals_sent=0, memory_reclaim_attempted=False, gpu_lock_owner=55672)
        save(self.output / "resources.json", result)
        resource_limits(sample, self.last_gpu)
        return result


def common_q1(args, config, runtime, plan, frozen):
    import torch
    torch.set_num_threads(1)
    require(Path(sys.executable).absolute() == Path(runtime["primebot_python"])
            and importlib.metadata.version("mujoco") == "3.6.0"
            and importlib.metadata.version("torch").split("+")[0] == "2.7.0", "Original common plant runtime changed")
    sys.path[:0] = [str(Path(runtime["adapted_source"]) / "src"), str(HERE.parent / "primebot_q1_review")]
    from policy_process import PolicyProcess
    from q1_native_review import rollout
    checkpoint = audit_record(args.output, args.iteration, "q1")
    baseline = read(Path(config["primebot_review"]) / "checkpoint_audit.json")["final"]["checkpoint"]
    require(identity(baseline["path"]) == baseline, "Original common plant checkpoint changed")
    directory = args.output / "common_q1" / str(args.iteration) / args.interface
    directory.mkdir(parents=True, exist_ok=False)
    command = [runtime["morph_python"], "-u", str(HERE.parent / "q1_five_stage/policy_process.py")]
    for key, value in {"plan": TRAINING / "e1024_train_plan.json", "checkpoint": checkpoint["path"],
        "frozen": config["frozen_data"], "primebot-checkpoint": baseline["path"],
        "morph-source": plan["source"], "primebot-source": runtime["adapted_source"]}.items():
        command.extend(("--" + key, str(value)))
    result = dict(status="RUNNING", iteration=args.iteration, interface=args.interface, robot="q1",
        plant="common_primebot_cpu", checkpoint=checkpoint, cases=[], final_acceptance=False,
        training_set_only=True, optimizer_updates=0, user_accepted_checkpoint=None)
    actor = None
    try:
        actor = PolicyProcess(command, directory / "policy.log")
        require(actor.ready["checkpoint"] == checkpoint and actor.ready["metadata"]["iteration"] == args.iteration,
                "Wrong isolated policy checkpoint")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5 and Path(settings["robot_xml"]) == Path(runtime["xml"]),
                "Common physical settings changed")
        result["render_xml"] = identity(settings["robot_xml"])
        for item in fixed_rows(frozen):
            name = Path(item["q1_primebot"]["path"]).stem
            result["cases"].append(rollout(robot, actor, settings, item, args.interface, 0, directory / (name + "_s0")))
            save(directory / "progress.json", result)
        actor.close()
        require(not torch.cuda.is_initialized(), "Common replay initialized CUDA")
        require(identity(checkpoint["path"]) == checkpoint, "Policy checkpoint changed during replay")
        result.update(status="PASSED", complete_cases=sum(r["complete"] for r in result["cases"]),
                      cuda_initialized=False, policy_worker_return_code=actor.child.returncode)
    finally:
        if actor is not None and not actor.closed:
            actor.close(graceful=False)
        result["ended_utc"] = now()
        save(directory / "result.json", result)


def g1_cpu(args, config, runtime, plan, frozen):
    import torch
    torch.set_num_threads(1)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "G1 CPU isolation required")
    import warp as wp
    wp.config.enable_cuda = False
    wp.init()
    require(all(device.is_cpu for device in wp.get_devices()), "Warp exposed an accelerator")
    sys.path[:0] = [str(Path(plan["source"]) / "src"), str(HERE.parent / "morphmimic_q1")]
    from bootstrap import install
    spec = install("g1")["robot"]
    from inference import load_policy
    from morphmimic.eval.benchmark import make_eval_env
    from native_cpu import rollout
    import mujoco
    checkpoint = audit_record(args.output, args.iteration, "g1")
    actor, metadata = load_policy(checkpoint["path"], spec, COMMIT, "cpu", expected_plan=plan)
    directory = args.output / "g1_native_cpu" / str(args.iteration)
    directory.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", iteration=args.iteration, robot="g1", plant="native_mjlab_cpu",
        checkpoint=checkpoint, metadata=metadata, cases=[], optimizer_updates=0,
        final_acceptance=False, training_set_only=True, user_accepted_checkpoint=None,
        backend_note="Same pinned native scene across checkpoints; CPU Warp, not a GPU/CPU parity claim")
    env = None
    try:
        env = make_eval_env(plan["subsets"][0], 1, "cpu", seed=0, robot="g1",
            human_root=plan["human_root"], goal_anchor="reference")
        names = validate_frozen(frozen)
        require(set(env.command_manager.get_term("motion").library.clip_names) == set(names), "Native library changed")
        scene = directory / "scene.mjb"
        mujoco.mj_saveModel(env.sim.mj_model, str(scene), None)
        result["scene"] = identity(scene)
        for interface in INTERFACES:
            for item in fixed_rows(frozen):
                name = Path(item["g1"]["path"]).stem
                for key in ("g1", "human"):
                    require(identity(item[key]["path"]) == item[key], "G1 reference changed")
                result["cases"].append(rollout(env, actor, spec, item, interface, 0, directory / interface / (name + "_s0")))
                save(directory / "progress.json", result)
        require(not torch.cuda.is_initialized() and identity(checkpoint["path"]) == checkpoint, "G1 isolation or identity changed")
        result.update(status="PASSED", complete_cases=sum(r["complete"] for r in result["cases"]), cuda_initialized=False)
    finally:
        if env is not None:
            env.close()
        result["ended_utc"] = now()
        save(directory / "result.json", result)


def render(args, runtime):
    import mujoco
    from OpenGL import GL
    require("torch" not in sys.modules, "Render after policy processes have exited")
    context = mujoco.GLContext(1, 1)
    try:
        context.make_current()
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        require("NVIDIA" in vendor, "Use the requested GPU EGL renderer")
    finally:
        context.free()
    if args.child == "media_q1":
        sys.path.insert(0, str(HERE.parent / "primebot_q1_review"))
        from video import render_case
        targets = [args.output / "common_q1/48000" / i / "result.json" for i in INTERFACES]
    else:
        from native_video import render_case
        targets = [args.output / "g1_native_cpu/48000/result.json"]
    for target in targets:
        result = read(target)
        require(result["status"] == "PASSED", "Render only saved completed replay processes")
        scene = result.get("scene", result.get("render_xml"))
        require(identity(scene["path"]) == scene, "Rendering scene changed")
        for row in result["cases"]:
            row["video"] = render_case(Path(scene["path"]), row, Path(row["trace"]).with_name("tracking.mp4"))
            save(Path(row["trace"]).with_name("result.json"), row)
        result["graphics_vendor"] = vendor
        save(target, result)


def read_groups(output):
    groups = {}
    for robot, interfaces in (("q1", INTERFACES), ("g1", INTERFACES)):
        for interface in interfaces:
            records = []
            for iteration in ITERATIONS:
                if robot == "q1":
                    path = OLD_REVIEW / "review/result.json" if (iteration, interface) == (16000, "motion") else (
                        output / "common_q1" / str(iteration) / interface / "result.json")
                else:
                    path = output / "g1_native_cpu" / str(iteration) / "result.json"
                if not path.is_file():
                    continue
                value = read(path)
                if value.get("status") != "PASSED":
                    continue
                cases = [r for r in value["cases"] if r["interface"] == interface]
                require({r["clip"] for r in cases} == set(FIXED) and len(cases) == 3
                        and all(r["seed"] == 0 for r in cases), "Missing fixed comparison case")
                records.append(dict(iteration=iteration, source=identity(path), cases=cases))
            groups[robot + "/" + interface] = records
    return groups


def observation(config):
    live(config)
    state = dict(checked_utc=now(), pipeline=read(TRAINING / "pipeline.json")["status"],
        guard=read(TRAINING / "e1024/train/guard.json"),
        memory=memory(Path("/sys/fs/cgroup/memory")), gpu=gpu_snapshot(config), workers={},
        identities=[process(Path("/proc"), pid) for pid in HANDLES],
        locks=[line for line in Path("/proc/locks").read_text().splitlines() if " WRITE 55672 " in line],
        memory_summary=read(TRAINING / "e1024/train/memory_watch_summary.json")
            if (TRAINING / "e1024/train/memory_watch_summary.json").is_file() else None,
        checkpoint_stats=[], logs={}, reclaim_records=[])
    for robot in ("g1", "q1"):
        worker = read(TRAINING / "e1024/train/workers" / (robot + ".json"))
        state["workers"][robot] = {k: worker[k] for k in ("iterations_completed", "optimizer_updates",
            "nonfinite_updates", "collected_environment_transitions")}
        log = TRAINING / "e1024/train/workers" / (robot + ".log")
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 12000))
            state["logs"][robot] = stream.read().decode(errors="replace")
        for p in sorted((TRAINING / "e1024/train/run" / robot).glob("*.pt")):
            st = p.stat()
            state["checkpoint_stats"].append(dict(robot=robot, name=p.name, bytes=st.st_size, mtime=st.st_mtime))
    for path in (TRAINING / "e1024/train").glob("*reclaim*.json"):
        state["reclaim_records"].append(dict(source=identity(path), value=read(path)))
    return state


def execute(args):
    if args.child:
        check_iteration(args.iteration)
        config, runtime = read(args.config), read(args.runtime_config)
        plan, frozen = read(TRAINING / "e1024_train_plan.json"), read(config["frozen_data"])
        if args.child == "audit":
            path = args.output / "checkpoints" / str(args.iteration)
            path.mkdir(parents=True, exist_ok=False)
            return audit(SimpleNamespace(output=path, iteration=args.iteration), config, plan)
        if args.child == "q1":
            return common_q1(args, config, runtime, plan, frozen)
        if args.child == "g1":
            return g1_cpu(args, config, runtime, plan, frozen)
        if args.child == "curves":
            from plots import generate
            return generate(args.output, plan, config, runtime)
        return render(args, runtime)
    config, runtime, plan, frozen, sources, reused = preflight(args)
    args.output.mkdir(exist_ok=False)
    save(args.output / "training_before.json", observation(config))
    path = checkpoint_audit(args.output, 16000)
    path.parent.mkdir(parents=True)
    save(path, reused)
    save(args.output / "authorization.json", dict(started_utc=now(), evaluation_commit=args.evaluation_commit,
        original_training_commit=COMMIT, plan=identity(TRAINING / "e1024_train_plan.json"), source_inputs=sources,
        iterations=ITERATIONS, interfaces=INTERFACES, clips=FIXED, seed=0, new_training=False,
        reused16k_audit=identity(OLD_AUDIT), reused16k_motion=identity(OLD_REVIEW / "review/result.json"),
        memory_admission_bytes=22 * GIB, memory_exit_bytes=24 * GIB, gpu_used_exit_mib=18 * 1024,
        gpu_temperature_exit_c=80, maximum_wall_seconds=7200, phase_wall_seconds=1800,
        maximum_output_bytes=3 * GIB, training_signals_sent=0, gpu_compute_requested=False,
        comparison_rule="Same clips, full valid motion if complete; failed cases retained, no successful-case selection",
        practical_plateau_hint=dict(maximum_tracking_gain_fraction=.02, maximum_slip_change_fraction=.10,
            maximum_sole_change_m=.001, maximum_tilt_change_rad=.01, maximum_root_change_m=.01,
            scope="Exploratory recommendation only; inspect each clip and both interfaces; never an automatic stop gate"),
        final_acceptance=False, user_accepted_checkpoint=None))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", MALLOC_ARENA_MAX="1", MALLOC_TRIM_THRESHOLD_="131072", MALLOC_MMAP_THRESHOLD_="131072",
        MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless", Q1_REVIEW_FFMPEG=runtime["ffmpeg"],
        __EGL_VENDOR_LIBRARY_FILENAMES=str(ROOT / "script/morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
        LD_LIBRARY_PATH=str(Path(runtime["graphics_dir"]) / "lib") + (
            os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
        GIT_PYTHON_GIT_EXECUTABLE=str(ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"))
    jobs = [("audit", i, "motion") for i in ITERATIONS[1:]] + [("curves", 48000, "motion")]
    jobs += [("q1", i, face) for i in ITERATIONS for face in INTERFACES if (i, face) != (16000, "motion")]
    jobs += [("media_q1", 48000, "motion")]
    jobs += [("g1", i, "motion") for i in ITERATIONS] + [("media_g1", 48000, "motion")]
    state = dict(status="RUNNING", started_utc=now(), phases=[], optimizer_updates=0,
                 training_signals_sent=0, final_acceptance=False, user_accepted_checkpoint=None)
    requested = False
    def stop(signum, frame):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    budget, previous = Budget(config, args.output), dict(os.environ)
    try:
        os.environ.update(env)
        for job, iteration, interface in jobs:
            label = f"{job}_{iteration}_{interface}"
            state["phase"] = label
            save(args.output / "status.json", state)
            python = runtime["primebot_python"] if job in ("q1", "media_q1") else runtime["morph_python"]
            command = [python, "-u", str(Path(__file__).resolve()), "--child", job, "--iteration", str(iteration),
                "--interface", interface, "--config", str(args.config), "--runtime-config", str(args.runtime_config),
                "--output", str(args.output), "--evaluation-commit", args.evaluation_commit]
            budget.phase_started = time.monotonic()
            result = guarded_run(command, ROOT, budget, args.output / (label + ".guard.json"),
                args.output / (label + ".log"), check_interval_s=.5, grace_s=5, stop_requested=lambda: requested)
            state["phases"].append(dict(name=label, guard=identity(args.output / (label + ".guard.json"))))
            require(result["status"] == "WORKER_COMPLETED" and result["cleanup"]["worker_reaped"] is True
                    and result["cleanup"]["worker_return_code"] == 0, "Review phase failed: " + label)
        require(all(identity(r["path"]) == r for r in sources), "Frozen input changed during review")
        require(identity(OLD_REVIEW / "review/result.json") == read(args.output / "authorization.json")["reused16k_motion"],
                "Reused physical review changed during comparisons")
        save(args.output / "comparisons.json", dict(groups=read_groups(args.output), final_acceptance=False,
            user_accepted_checkpoint=None, scope="Three training motions; deterministic CPU replay, not full-library or robot acceptance"))
        state.update(status="COMPLETED_PENDING_LOCAL_REVIEW", ended_utc=now())
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(args.output / "status.json", state)
        save(args.output / "source_integrity.json", dict(passed=all(identity(r["path"]) == r for r in sources),
            frozen_sources=len(sources), original_training_inputs_modified=False, training_signals_sent=0))
        save(args.output / "training_after.json", observation(config))
        os.environ.clear()
        os.environ.update(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--child", choices=("audit", "q1", "g1", "curves", "media_q1", "media_g1"))
    parser.add_argument("--iteration", type=int, choices=ITERATIONS, default=48000)
    parser.add_argument("--interface", choices=INTERFACES, default="motion")
    execute(parser.parse_args())
