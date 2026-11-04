"""Apply the complete local gate, then validate and train one fresh shared model."""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys

from contracts import (MODULES, REFERENCE_KEYS, allow_capacity_fallback, build_plans,
                       check_profile, disk_budget, expected_transitions, primebot_gate,
                       valid_commit, validate_frozen)
from delivery import artifact, load_manifest, read
from common import MORPH_UPSTREAM, ROOT, identity, now, require, save

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent / "morphmimic_q1"
OPTIMIZERS = dict(wm_opt="wm", policy_opt="policy", critic_opt="critic",
                  kp_opt="kp", critic_kp_opt="critic_kp")


def rooted(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def final_gate(config, expected_names):
    review = Path(config["primebot_review"])
    manifest = load_manifest(config["delivery_manifest"])
    require(Path(manifest["review_root"]) == review, "Delivery manifest belongs to another evaluation")
    require(all(artifact(review, row["relative"]) == row for row in manifest["artifacts"]),
            "Server delivery changed after Mac verification")
    return primebot_gate(
        server=read(review / "result.json"), pipeline=read(config["primebot_pipeline"]),
        pairs=read(review / "paired_fixed_metrics.json")["pairs"],
        fixed={s: read(review / (s + "_fixed/result.json")) for s in ("initial", "final")},
        libraries={s: read(review / (s + "_library/result.json")) for s in ("initial", "final")},
        expected_names=expected_names, audit=read(review / "checkpoint_audit.json"),
        mac=read(config["mac_receipt"]), manifest=manifest)


def updates(cfg, iteration):
    actor = max(0, iteration - cfg["wm_only_iters"])
    return dict(wm_opt=10 * iteration, policy_opt=actor, critic_opt=20 * actor,
                kp_opt=actor, critic_kp_opt=20 * actor)


def audit_checkpoint(plan, iteration):
    import torch
    require(not torch.cuda.is_initialized(), "Shared checkpoint audit must stay on CPU")
    records = []
    expected = updates(plan["cfg"], iteration)
    transitions = 0 if iteration == 0 else expected_transitions(dict(plan["cfg"], iterations=iteration))
    for robot in plan["robots"]:
        path = Path(plan["output"]) / "run" / robot / f"ckpt_{iteration}.pt"
        value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        require(value["iteration"] == iteration and value["schedule"]["step"] == iteration
                and value["schedule"]["world_size"] == 2
                and value["native_q1_contract"]["code_commit"] == plan["code_commit"],
                "Shared saved identity or schedule mismatch")
        require(all(value["cfg"].get(k) == v for k, v in plan["cfg"].items()), "Saved cfg changed")
        actual = value.get("actual_training", {})
        require(actual.get("iteration") == iteration and actual.get("optimizer_updates") == expected
                and actual.get("collected_environment_transitions") == transitions
                and actual.get("code_commit") == plan["code_commit"]
                and actual.get("fresh_initialization") is True and actual.get("profile_weights_reused") is False,
                "Saved actual update/exposure metadata mismatch")
        counters, tensors = {}, []
        for hook, key in OPTIMIZERS.items():
            states = value["optim"][key]["state"].values()
            counters[hook] = sorted({int(v["step"].item()) for v in states if "step" in v})
            require(counters[hook] == ([expected[hook]] if expected[hook] else []), "Saved Adam counter mismatch")
            for state in states:
                tensors.extend(v for v in state.values() if isinstance(v, torch.Tensor))
        for name in ("wm", "policy", "critic", "critic_kp", "obs_norm", "goal_norm", "human_norm", "heading_norm"):
            tensors.extend(v for v in value[name].values() if isinstance(v, torch.Tensor))
        require(tensors and all(torch.isfinite(v).all().item() for v in tensors), "Nonfinite saved tensor")
        records.append(dict(robot=robot, checkpoint=identity(path), iteration=iteration,
                            optimizer_adam_steps=counters, finite_tensors=len(tensors), actual_training=actual))
        del value, tensors, states
        gc.collect()
    require(not torch.cuda.is_initialized(), "Audit created a CUDA context")
    return dict(status="PASSED", iteration=iteration, records=records, cuda_initialized=False)


def check_formal(workers, plan, names):
    expected = updates(plan["cfg"], plan["cfg"]["iterations"])
    for robot, worker in zip(plan["robots"], workers):
        require(worker.get("status") == "COMPLETED" and worker.get("robot") == robot
                and worker.get("formal_training") is True and worker.get("code_commit") == plan["code_commit"]
                and worker.get("iterations_completed") == plan["cfg"]["iterations"]
                and worker.get("collected_environment_transitions") == expected_transitions(plan["cfg"])
                and worker.get("optimizer_updates") == expected
                and worker.get("optimizer_adam_steps") == {k: [v] for k, v in expected.items()}
                and worker.get("native_rank_checks_required") is True
                and worker.get("native_rank_checks_passed") is True
                and len(worker.get("peer_audit", [])) == 2
                and set(worker.get("shared_parameter_sha", {})) == MODULES,
                "Formal shared run is incomplete")
        require(set(worker.get("common_clip_names", [])) == set(names)
                and len(worker["common_clip_names"]) == len(names)
                and all(worker.get("resolved_cfg", {}).get(k) == v for k, v in plan["cfg"].items()),
                "Formal shared data or cfg changed")
        zero = worker.get("zero_snapshot_audit", {})
        require(zero.get("status") == "PASSED" and zero.get("parameters_unchanged") is True
                and zero.get("rng_unchanged") is True, "Missing passive initial save audit")
        require(worker.get("nonfinite_updates") and not any(worker["nonfinite_updates"].values()), "Nonfinite shared update")
    require(len(workers) == 2, "Need both shared ranks")
    for key in ("shared_parameter_sha", "latent_draws", "last_latent_draw"):
        require(workers[0].get(key) and workers[0][key] == workers[1].get(key), "Shared rank disagreement: " + key)
    require(workers[0]["initial_buffers"]["joint_presence"] != workers[1]["initial_buffers"]["joint_presence"],
            "Shared training overwrote robot buffers")


def choose_profile(attempt):
    """The only automatic retry is a new smaller capacity profile after proven OOM."""
    for environments in (1024, 512):
        phase, logs, plan = attempt(environments)
        if phase.get("status") == "COMPLETED":
            guard = phase.get("guard", {})
            cleanup = guard.get("cleanup", {})
            require(guard.get("status") == "WORKER_COMPLETED" and cleanup.get("worker_reaped") is True
                    and cleanup.get("worker_return_code") == 0 and phase.get("inputs_unchanged") is True,
                    "Whole shared profile did not finish cleanly")
            return plan
        require(environments == 1024 and allow_capacity_fallback(phase, logs),
                "Shared failure does not authorize another capacity attempt")
    raise RuntimeError("No approved shared capacity passed")


def oom_count():
    rows = dict(line.split() for line in Path("/sys/fs/cgroup/memory/memory.oom_control").read_text().splitlines())
    return int(rows["oom_kill"])


def recovery():
    return subprocess.check_output(["nvidia-smi", "--query-gpu=gpu_recovery_action", "--format=csv,noheader"], text=True).strip()


def tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 1024**2))
        return stream.read().decode(errors="replace")


class Runner:
    def __init__(self, config, output, inputs):
        self.config, self.output, self.inputs = config, output, inputs
        self.child = None
        self.stopped = False
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self.stop)

    def stop(self, signum, frame):
        self.stopped = True
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)

    def phase(self, name, command, seconds, maximum_bytes, extra=()):
        require(not self.stopped, "Controller stopped; no next stage")
        output = self.output / name
        inputs = sorted({*self.inputs, *map(str, extra)})
        before = [identity(path) for path in inputs]
        oom = oom_count()
        require(oom == 0, "Do not reset the original OOM baseline")
        cmd = [self.config["morph_python"], "-u", str(HERE.parent / "primebot_q1/guard.py"),
            "--config", str(self.output / "guard_config.json"), "--output", str(output),
            "--seconds", str(seconds), "--maximum-bytes", str(maximum_bytes)]
        if name.rsplit("/", 1)[-1] == "profile" and self.config.get("profile_maximum_initial_cache_bytes") is not None:
            cmd.extend(("--maximum-initial-cache-bytes", str(self.config["profile_maximum_initial_cache_bytes"])))
        for path in inputs:
            cmd.extend(("--input", path))
        env = dict(os.environ, GIT_PYTHON_GIT_EXECUTABLE=str(
            ROOT / "data/environments/morphmimic_git_tools/usr/bin/git"))
        with (self.output / (name.replace("/", "_") + "_controller.log")).open("x") as log:
            self.child = subprocess.Popen([*cmd, "--", *command], cwd=ROOT, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, env=env)
            code = self.child.wait()
        self.child = None
        value = read(output / "status.json")
        value.update(inputs_unchanged=before == [identity(path) for path in inputs],
                     cgroup_oom_kill_delta=oom_count() - oom, gpu_recovery_action=recovery())
        require((value.get("status") == "COMPLETED") == (code == 0), "Guard exit/status disagreement")
        require(not self.stopped, "Controller stopped after a phase; do not advance")
        logs = tail(output / "stdout.log")
        for path in (output / "workers").glob("*.log"):
            logs += tail(path)
        save(output / "transition_receipt.json", value)
        return value, logs


def execute(config, config_path, commit):
    require(valid_commit(commit), "Require an independently verified full published commit")
    require(config["morph_upstream"] == MORPH_UPSTREAM, "Use the approved MorphMimic v0.1 source")
    require(type(config.get("memory_pressure_reclaim", False)) is bool, "Invalid memory pressure policy")
    require(not (config.get("memory_pressure_reclaim", False)
                 and config.get("profile_maximum_initial_cache_bytes") is not None),
            "Total-memory pressure policy cannot use an independent cache admission")
    output = Path(config["output"])
    require(output.is_absolute() and output.is_relative_to(ROOT / "data/supermimic_for_q1")
            and not output.exists(), "Use a new shared experiment root")
    frozen_path = Path(config["frozen_data"])
    require(identity(frozen_path)["sha256"] == config["frozen_data_sha256"], "Frozen dataset identity changed")
    frozen = read(frozen_path)
    names = validate_frozen(frozen)
    gate = final_gate(config, names)
    require(gate["passed"], "Full Primebot/local delivery gate failed: " + ",".join(gate["gate_reason"]))
    cache_admission = None
    if config.get("profile_maximum_initial_cache_bytes") is not None:
        from guard import profile_cache_admission
        cache_admission = profile_cache_admission(config["profile_maximum_initial_cache_bytes"])
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="PREPARING", started_utc=now(), code_commit=commit, gate=gate,
        automatic_formal_retry=False, profile_weights_reused=False, user_accepted_checkpoint=None)
    if cache_admission is not None:
        state["initial_cache_admission"] = cache_admission
    try:
        review = Path(config["primebot_review"])
        inputs = [str(config_path), str(frozen_path), config["profile_template"], config["pinned_input_receipt"],
                  config["primebot_pipeline"], config["delivery_manifest"], config["mac_receipt"],
                  *map(str, (review / n for n in ("result.json", "paired_fixed_metrics.json", "checkpoint_audit.json"))),
                  *map(str, (review / (s + "_" + m + "/result.json") for s in ("initial", "final") for m in ("fixed", "library"))),
                  *map(str, HERE.glob("*.py")),
                  *map(str, NATIVE.glob("*.py")), *map(str, (HERE.parent / "morphmimic_cross_robot").glob("*.py"))]
        for row in frozen["rows"]:
            for key in REFERENCE_KEYS:
                require(identity(row[key]["path"]) == row[key], "Frozen paired reference changed")
                inputs.append(row[key]["path"])
        subsets = []
        for key in ("g1_subset", "q1_subset"):
            item = frozen[key]
            path = rooted(item["path"])
            require(identity(path) == dict(item, path=str(path)), "Frozen subset identity changed")
            subsets.append(path)
            inputs.append(str(path))
        template = read(config["profile_template"])
        source = Path(template["source"])
        original = read(config["pinned_input_receipt"])["inputs"]
        asset_root = rooted(frozen["xml"]["path"]).parent.parent
        pinned = [r for r in original if Path(r["path"]).is_relative_to(source)
                  or Path(r["path"]).is_relative_to(asset_root)]
        require(any(Path(r["path"]).is_relative_to(source) and Path(r["path"]).suffix == ".py" for r in pinned)
                and any(Path(r["path"]).suffix == ".xml" for r in pinned), "Pinned source/assets are missing")
        for item in pinned:
            require(identity(item["path"]) == item, "Pinned upstream source or asset changed")
            inputs.append(item["path"])
        sys.path.insert(0, str(NATIVE))
        from train_native import validate_plan
        from validate_and_train import check_native
        for robot, path in config["native_results"].items():
            check_native(read(path), robot)
            inputs.append(path)
        require(set(config["native_results"]) == {"g1", "q1"}, "Both existing native control checks required")
        guard_config = dict(adapted_source=str(source), expected_boot=config["expected_boot"], expected_gpu=config["expected_gpu"],
                            maximum_source_bytes=config.get("maximum_source_bytes", 64 * 1024**2))
        if config.get("record_memory_breakdown", False):
            guard_config["record_memory_breakdown"] = config["record_memory_breakdown"]
        if config.get("memory_pressure_reclaim", False):
            guard_config["memory_pressure_reclaim"] = True
            inputs.extend((str(HERE / "memory_pressure.py"),
                           str(HERE.parent / "project_tools/reclaim_q1_file_cache.py")))
        save(output / "guard_config.json", guard_config)
        state["source_storage_budget_bytes"] = guard_config["maximum_source_bytes"]
        runner = Runner(config, output, inputs)
        formal_plans = {}

        def attempt(envs):
            prefix = f"e{envs}"
            profile, formal = build_plans(template, output / prefix, subsets, config["human_root"], commit, envs)
            validate_plan(profile)
            validate_plan(formal)
            for label, plan in (("profile", profile), ("train", formal)):
                save(output / (prefix + "_" + label + "_plan.json"), plan)
            formal_plans[envs] = formal
            state.update(status="VALIDATING", environments_per_robot=envs)
            save(output / "pipeline.json", state)
            path = output / (prefix + "_profile_plan.json")
            phase, logs = runner.phase(prefix + "/profile", [config["morph_python"], "-u", str(NATIVE / "train_native.py"),
                "--plan", str(path)], 10800, 8 * 1024**3, [path])
            return phase, logs, profile

        profile = choose_profile(attempt)
        envs = profile["cfg"]["num_envs"]
        formal = formal_plans[envs]
        workers = [read(Path(profile["output"]) / "workers" / (r + ".json")) for r in profile["robots"]]
        check_profile(workers, profile, names)
        audit_path = output / "profile_checkpoint_audit.json"
        profile_path = output / f"e{envs}_profile_plan.json"
        phase, _ = runner.phase("profile_audit", [config["morph_python"], "-u", str(HERE / "controller.py"),
            "audit", "--plan", str(profile_path), "--iteration", "8", "--output", str(audit_path)], 900, 1024**3, [profile_path])
        require(phase["status"] == "COMPLETED" and read(audit_path)["status"] == "PASSED", "Shared CPU save audit failed")
        budget = disk_budget({r["robot"]: r["checkpoint"]["bytes"] for r in read(audit_path)["records"]}, formal["cfg"])
        require(shutil.disk_usage(ROOT).free >= budget["maximum_experiment_bytes"] + budget["required_free_reserve_bytes"],
                "Whole shared formal recipe does not fit the measured disk reserve")
        steady = max(statistics.median(w["iteration_seconds"][2:]) for w in workers)
        seconds = min(30 * 86400, max(86400, math.ceil(steady * formal["cfg"]["iterations"] * 2.5 + 14400)))
        state.update(status="TRAINING", formal_plan=identity(output / f"e{envs}_train_plan.json"),
            profile_audit=identity(audit_path), disk_budget=budget, steady_iteration_seconds=steady,
            iteration_only_hours_estimate=steady * formal["cfg"]["iterations"] / 3600)
        save(output / "pipeline.json", state)
        formal_path = output / f"e{envs}_train_plan.json"
        phase, _ = runner.phase(f"e{envs}/train", [config["morph_python"], "-u", str(NATIVE / "train_native.py"),
            "--plan", str(formal_path)], seconds, budget["maximum_experiment_bytes"], [formal_path, audit_path])
        require(phase["status"] == "COMPLETED", "Formal shared training failed; preserve it without retry")
        check_formal([read(Path(formal["output"]) / "workers" / (r + ".json")) for r in formal["robots"]], formal, names)
        for iteration in (0, formal["cfg"]["iterations"]):
            audit_path = output / f"formal_checkpoint_audit_{iteration}.json"
            phase, _ = runner.phase(f"formal_audit_{iteration}", [config["morph_python"], "-u", str(HERE / "controller.py"),
                "audit", "--plan", str(formal_path), "--iteration", str(iteration), "--output", str(audit_path)],
                900, 1024**3, [formal_path])
            require(phase["status"] == "COMPLETED" and read(audit_path)["status"] == "PASSED", "Formal CPU save audit failed")
        state.update(status="SHARED_TRAINING_COMPLETED_PENDING_EVALUATION", formal_training_completed=True,
                     next_action="Q1 motion/KP and G1 initial/final evaluation, local media and comprehensive comparison")
    except BaseException as error:
        state.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["ended_utc"] = now()
        save(output / "pipeline.json", state)
    if config.get("evaluation_runtime_config"):
        from shared_review import execute as evaluate
        try:
            state["evaluation"] = evaluate(config_path, output / "pipeline.json",
                Path(config["evaluation_runtime_config"]), output / "evaluation", commit)
            state.update(status=state["evaluation"]["status"], next_action="Mac shared media and data acceptance; then review comprehensive conclusions")
        except BaseException as error:
            state["evaluation"] = dict(status="FAILED", error=f"{type(error).__name__}: {error}", automatic_retry=False)
            state.update(status="SHARED_TRAINING_COMPLETED_EVALUATION_FAILED",
                         next_action="Preserve formal training and failed evaluation; no automatic retry")
            raise
        finally:
            save(output / "pipeline.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--code-commit", required=True)
    audit = sub.add_parser("audit")
    for key in ("plan", "output"):
        audit.add_argument("--" + key, type=Path, required=True)
    audit.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "audit":
        save(args.output, audit_checkpoint(read(args.plan), args.iteration))
    else:
        execute(read(args.config), args.config.resolve(), args.code_commit)


if __name__ == "__main__":
    main()
