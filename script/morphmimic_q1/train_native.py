"""Launch native Q1 or same-GPU G1/Q1 Trainers with the original shared updates."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time


ROOT = Path("/opt/data/private/ZHY")
OPTIMIZERS = ("wm_opt", "policy_opt", "critic_opt", "kp_opt", "critic_kp_opt")


def wait_for_workers(children):
    if not children:
        raise ValueError("At least one native worker is required")
    while True:
        codes = [child.poll() for child in children]
        if any(code not in (None, 0) for code in codes):
            raise RuntimeError("A native rank failed; preserve every rank's logs")
        if all(code == 0 for code in codes):
            return
        time.sleep(1)


def validate_plan(plan):
    commit = plan["code_commit"]
    if (plan["robots"] not in (["q1"], ["g1", "q1"]) or
            len(plan["subsets"]) != len(plan["robots"]) or
            plan["purpose"] not in ("profile", "train") or
            len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("Invalid frozen native plan")
    cfg = plan["cfg"]
    for key in ("num_envs", "iterations", "wm_batch", "imag_batch", "kp_imag_batch"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError("Require positive integer native batch/iteration settings")
    if not cfg.get("kp_interface") or cfg.get("policy_goal", "motion") != "motion":
        raise ValueError("This entry audits the motion policy and return KP student")
    if cfg.get("kp_student", "return") != "return":
        raise ValueError("Five-optimizer audit requires the return KP student")
    if plan["purpose"] == "profile" and (cfg["iterations"] > 8 or cfg["wm_only_iters"] != 0):
        raise ValueError("Resource profiles require at most eight full-update iterations")
    for key in ("physics_handoff_min_free_bytes", "learner_handoff_min_free_bytes"):
        handoff = plan.get(key, 0)
        if type(handoff) is not int or handoff < 0:
            raise ValueError("VRAM handoff requires a nonnegative integer byte threshold")
    for value in (plan["source"], plan["output"], plan["human_root"], *plan["subsets"]):
        path = Path(value)
        if not path.is_absolute() or not path.resolve().is_relative_to(ROOT):
            raise ValueError("Keep native phase paths inside the sole project root")


def release_inactive_cuda_cache(cuda, minimum_free_bytes, record, phase="physics"):
    if not minimum_free_bytes:
        return
    free, total = cuda.mem_get_info()
    key = "physics_vram_handoff" if phase == "physics" else "learner_vram_handoff"
    state = record.setdefault(key, dict(
        minimum_free_bytes=minimum_free_bytes, checks=0, cache_releases=0,
        total_seconds=0., events=[], by_phase={}))
    state["checks"] += 1
    phase_stats = state["by_phase"].setdefault(phase, dict(checks=0, cache_releases=0))
    phase_stats["checks"] += 1
    state["last_checked_free_bytes"] = int(free)
    if free >= minimum_free_bytes:
        return
    started = time.monotonic()
    cuda.synchronize()
    before = dict(free_bytes=int(free), total_bytes=int(total),
        allocated_bytes=int(cuda.memory_allocated()), reserved_bytes=int(cuda.memory_reserved()))
    # Return inactive pages before an external allocator or a peer rank needs them.
    cuda.empty_cache()
    free_after, total_after = cuda.mem_get_info()
    after = dict(free_bytes=int(free_after), total_bytes=int(total_after),
        allocated_bytes=int(cuda.memory_allocated()), reserved_bytes=int(cuda.memory_reserved()))
    elapsed = time.monotonic() - started
    event = dict(check=state["checks"], phase=phase, before=before, after=after, seconds=elapsed,
        global_free_change_bytes=after["free_bytes"] - before["free_bytes"],
        torch_reserved_decrease_bytes=before["reserved_bytes"] - after["reserved_bytes"])
    state["cache_releases"] += 1
    phase_stats["cache_releases"] += 1
    state["total_seconds"] += elapsed
    if len(state["events"]) < 8:
        state["events"].append(event)
    state["last_event"] = event


def wrap_learner_handoffs(trainer, cuda, minimum_free_bytes, record):
    if not minimum_free_bytes:
        return

    def wrap(original, phase):
        def step(*args, **kwargs):
            release_inactive_cuda_cache(cuda, minimum_free_bytes, record, phase)
            return original(*args, **kwargs)
        return step

    for name, phase in (("fit_world_model", "world_model"),
                        ("improve_policy", "motion"), ("improve_policy_kp", "return_kp")):
        setattr(trainer, name, wrap(getattr(trainer, name), phase))


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def tensor_sha(value):
    import torch
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def parameter_sha(module):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode("ascii"))
        digest.update(tensor_sha(value).encode("ascii"))
    return digest.hexdigest()


def run_rank(plan, rank):
    import numpy as np
    import torch
    import torch.distributed as dist

    source = Path(plan["source"]).resolve(strict=True)
    output = Path(plan["output"]).resolve(strict=True)
    robots = tuple(plan["robots"])
    robot = robots[rank]
    sys.path.insert(0, str(source / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "morphmimic_cross_robot"))
    from bootstrap import install
    adapter = install(robot)
    from mjlab.utils.torch import configure_torch_backends
    from collective_transport import installed_transport
    from morphmimic.train.trainer import TrainCfg, Trainer
    configure_torch_backends()
    seed = plan["cfg"].get("seed", 0) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.set_device(0)
    shared = len(robots) > 1
    if shared:
        dist.init_process_group("gloo", init_method="file://" + str(output / "collectives.store"),
            rank=rank, world_size=len(robots), timeout=timedelta(minutes=20))
    modules = [importlib.import_module(name) for name in (
        "morphmimic.train.distributed", "morphmimic.train.multi_robot",
        "morphmimic.train.normalizers", "morphmimic.train.trainer")]
    transport = installed_transport(modules, dist) if shared else nullcontext(None)
    record = dict(status="INITIALIZING", robot=robot, rank=rank, pid=os.getpid(),
        contracts=adapter["contracts"], code_commit=plan["code_commit"],
        optimizer_updates={key: 0 for key in OPTIMIZERS}, iterations_completed=0,
        formal_training=plan["purpose"] == "train", latent_draws=[],
        nonfinite_updates={}, collected_environment_transitions=0,
        user_accepted_checkpoint=None)
    receipt = output / "workers" / (robot + ".json")
    trainer = None
    try:
        with transport as bridge:
            values = dict(plan["cfg"], robot=robot, morphology=True,
                human_root=plan["human_root"], log_dir=str(output), run_name="run")
            if shared:
                values.update(robots=robots, subset_yamls=tuple(plan["subsets"]), share="all")
            else:
                values.update(subset_yaml=plan["subsets"][0], robots=None, subset_yamls=None,
                    decoder_film=True, latent_align_weight=0.)
            cfg = TrainCfg(**values)
            trainer = Trainer(cfg, "cuda:0")
            record["resolved_cfg"] = asdict(trainer.cfg)
            record["status"] = "RUNNING"
            record["native_joint_names"] = list(adapter["robot"].joint_names)
            record["reference_clips"] = trainer.library.num_clips
            record["initial_buffers"] = dict(morph=tensor_sha(trainer.wm.morph),
                joint_presence=tensor_sha(trainer.wm.joint_presence),
                joint_limits=tensor_sha(trainer.wm.joint_limits),
                default_joint_pos=tensor_sha(trainer.wm.default_joint_pos),
                state_mean=tensor_sha(trainer.wm.normalizer.mean),
                state_std=tensor_sha(trainer.wm.normalizer.std))
            handles = []
            for name in OPTIMIZERS:
                opt = getattr(trainer, name, None)
                if opt is None:
                    raise RuntimeError("Full native integration requires five optimizers: " + name)

                def post_step(opt, args, kwargs, key=name):
                    record["optimizer_updates"][key] += 1

                handles.append(opt.register_step_post_hook(post_step))
            sampler = trainer._align_sampler
            if shared:
                if sampler is None or len(sampler.names) != trainer.library.num_clips:
                    raise RuntimeError("Shared Trainer needs the actual common-frame sampler")
                original_draw = sampler.draw

                def draw(it, n):
                    clip, offset = original_draw(it, n)
                    row = dict(it=it, n=n, clip_sha=tensor_sha(clip), offset_sha=tensor_sha(offset))
                    if len(record["latent_draws"]) < 8:
                        record["latent_draws"].append(row)
                    record["last_latent_draw"] = row
                    return clip, offset

                sampler.draw = draw
                record["common_clip_names"] = sampler.names
            original_checkpoint = trainer.checkpoint

            def checkpoint(*args, **kwargs):
                state = original_checkpoint(*args, **kwargs)
                state["native_q1_contract"] = dict(contracts=adapter["contracts"],
                    code_commit=plan["code_commit"], plan=plan, buffers=record["initial_buffers"])
                return state

            trainer.checkpoint = checkpoint
            original_iteration = trainer.iteration
            original_collect = trainer.collector.step
            last_started = time.monotonic()
            durations = []
            transitions = 0

            def collect_steps(*args, **kwargs):
                release_inactive_cuda_cache(torch.cuda,
                    plan.get("physics_handoff_min_free_bytes", 0), record)
                stats = original_collect(*args, **kwargs)
                record["collected_environment_transitions"] += int(stats["steps"])
                return stats

            trainer.collector.step = collect_steps
            wrap_learner_handoffs(trainer, torch.cuda,
                plan.get("learner_handoff_min_free_bytes", 0), record)
            if plan.get("passive_checkpoint_audit") is True:
                sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "q1_five_stage"))
                from snapshots import install as install_snapshot_audit
                install_snapshot_audit(trainer, plan, record)

            def iteration():
                nonlocal last_started, transitions
                started = time.monotonic()
                metrics = original_iteration()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - started)
                collect = metrics.get("_collect")
                if collect:
                    transitions += int(collect["steps"])
                record["iterations_completed"] = trainer.it
                record["iteration_transitions"] = transitions
                for key, value in metrics.items():
                    if key.endswith("nonfinite_steps"):
                        record["nonfinite_updates"][key] = (
                            record["nonfinite_updates"].get(key, 0) + int(value))
                if any(record["nonfinite_updates"].values()):
                    raise RuntimeError("Native optimizer reported a nonfinite gradient update")
                if trainer.it % plan.get("receipt_every", 50) == 0 or trainer.it == cfg.iterations:
                    record["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                    record["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                    record["last_iteration_seconds"] = durations[-1]
                    record["elapsed_seconds"] = time.monotonic() - last_started
                    save(receipt, record)
                return metrics

            trainer.iteration = iteration
            if plan["purpose"] == "train":
                record["initial_checkpoint"] = str(trainer.save_checkpoint())
            save(receipt, record)
            trainer.train()
            record["optimizer_adam_steps"] = {}
            for name in OPTIMIZERS:
                opt = getattr(trainer, name)
                steps = sorted({int(s["step"].item()) for s in opt.state.values() if "step" in s})
                record["optimizer_adam_steps"][name] = steps
                if not steps or max(steps) != record["optimizer_updates"][name]:
                    raise RuntimeError("Optimizer hook/checkpoint state count mismatch: " + name)
            record["shared_parameter_sha"] = {name: parameter_sha(getattr(trainer, name))
                for name in ("wm", "policy", "critic", "critic_kp")}
            record["replay_sample_sha"] = tensor_sha(trainer.replay.state[:, :min(8, trainer.replay.cfg.capacity)])
            record["native_rank_checks_required"] = shared
            record["native_rank_checks_passed"] = not shared or trainer._check_ranks_identical()
            record["iteration_seconds"] = durations if plan["purpose"] == "profile" else durations[-20:]
            if not record["native_rank_checks_passed"]:
                raise RuntimeError("Native shared parameter/normalizer checks failed")
            if bridge is not None:
                record["host_collectives"] = dict(calls=bridge.calls,
                    staged_input_bytes=bridge.staged_input_bytes)
                peers = [None] * len(robots)
                dist.all_gather_object(peers, {k: record[k] for k in (
                    "robot", "shared_parameter_sha", "initial_buffers", "replay_sample_sha",
                    "latent_draws", "last_latent_draw", "optimizer_updates")})
                if any(p["shared_parameter_sha"] != peers[0]["shared_parameter_sha"] for p in peers):
                    raise RuntimeError("Shared native parameters diverged by full SHA")
                if any(p["latent_draws"] != peers[0]["latent_draws"] or
                       p["last_latent_draw"] != peers[0]["last_latent_draw"] for p in peers):
                    raise RuntimeError("Shared latent draws differ in actual Trainer calls")
                if len({p["initial_buffers"]["joint_presence"] for p in peers}) != len(robots):
                    raise RuntimeError("Robot-specific morphology buffers were overwritten")
                record["peer_audit"] = peers
            record["status"] = "COMPLETED"
            save(receipt, record)
            for handle in handles:
                handle.remove()
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        save(receipt, record)
        raise
    finally:
        if trainer is not None:
            trainer.close()
        if shared and dist.is_initialized():
            dist.destroy_process_group()


def launch(plan, path):
    output = Path(plan["output"]).resolve(strict=True)
    workers = output / "workers"
    workers.mkdir(exist_ok=False)
    children, streams = [], []
    try:
        for rank, robot in enumerate(plan["robots"]):
            stream = (workers / (robot + ".log")).open("x")
            streams.append(stream)
            children.append(subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()),
                "--plan", str(path), "--rank", str(rank)], cwd=ROOT,
                stdout=stream, stderr=subprocess.STDOUT))
        save(output / "workers_launch.json", dict(pids=[p.pid for p in children], plan=plan))
        wait_for_workers(children)
        print(json.dumps(dict(status="COMPLETED", robots=plan["robots"])), flush=True)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    path = args.plan.resolve(strict=True)
    plan = json.loads(path.read_text())
    validate_plan(plan)
    if args.rank is None:
        launch(plan, path)
    else:
        run_rank(plan, args.rank)


if __name__ == "__main__":
    main()
