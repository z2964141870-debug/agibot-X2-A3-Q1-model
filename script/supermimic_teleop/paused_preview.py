"""One bounded32k CUDA human-preview campaign after the verified user pause."""

import argparse
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import FIXED, ROOT, identity, now, require, save
from contracts import valid_commit, validate_frozen
from delivery import read
from interim_review import COMMIT, GIB, HANDLES, TRAINING, fixed_rows, gpu_snapshot, load_physical_robot, media, resource_limits
from memory_watch import memory, process, subtree
from paused_runtime import assert_owner, compute_pids, own_gpu, paused_state
from preview_review import case_folder, environment, policy_command, require_plant, runtime_paths, validate_code_manifest, validate_stream
from runtime_guard import allocated_tree, guarded_run

ITERATION = 32000
METHODS = ("masked", "hold", "damped_cv")
HORIZONS = (10, 5, 4, 2, 1)
NEW_FILES = ("paused_preview.py", "paused_runtime.py", "gpu_preview_worker.py")


def cases():
    return [(method, horizon, clip) for method in METHODS
            for horizon in (HORIZONS if method == "masked" else (5, 4)) for clip in FIXED]


def checkpoint(args, config):
    audited = read(args.output / "checkpoint_audit.json")
    require(audited.get("status") == "PASSED" and audited.get("iteration") == ITERATION,
            "This campaign requires the reused32k CPU save audit")
    selected = next(row["checkpoint"] for row in audited["records"] if row["robot"] == "q1")
    baseline = read(Path(config["primebot_review"]) / "checkpoint_audit.json")["final"]["checkpoint"]
    require(all(identity(row["path"]) == row for row in (selected, baseline)), "Saved policy or physical baseline changed")
    return selected, baseline


def preflight(args):
    require(valid_commit(args.evaluation_commit), "Publish the independent campaign before execution")
    require(args.output.is_absolute() and args.output.resolve() == args.output and not args.output.exists()
            and args.output.is_relative_to(ROOT / "data/supermimic_for_q1")
            and not args.output.is_relative_to(TRAINING), "Require one new independent output root")
    config, runtime, receipt = read(args.config), read(args.runtime_config), read(args.pause_receipt)
    paused_state(config, receipt)
    resource_limits(memory(Path("/sys/fs/cgroup/memory")), gpu_snapshot(config), initial=True)
    require(shutil.disk_usage(ROOT).free >= 103 * GIB, "Keep100GiB reserve plus bounded campaign output")
    plan_path = TRAINING / "e1024_train_plan.json"
    plan, pipeline = read(plan_path), read(TRAINING / "pipeline.json")
    require(plan["code_commit"] == pipeline["code_commit"] == COMMIT and plan["purpose"] == "train"
            and pipeline["formal_plan"] == identity(plan_path) and plan["cfg"]["iterations"] == 160000
            and plan["robots"] == ["g1", "q1"], "Original frozen plan changed")
    require(runtime["morph_source"] == plan["source"] and runtime["morph_python"] == config["morph_python"],
            "Locked policy runtime differs from the original plan")
    require(identity(config["frozen_data"])["sha256"] == config["frozen_data_sha256"], "Frozen references changed")
    frozen = read(config["frozen_data"])
    validate_frozen(frozen)
    inputs = read(TRAINING / "e1024/train/status.json")["inputs"]
    changed = {str(HERE / name) for name in (*NEW_FILES, "preview_worker.py", "preview_review.py")}
    require(not changed & {row["path"] for row in inputs}, "New campaign code entered frozen training inputs")
    sources = [row for row in inputs if Path(row["path"]).suffix in (".py", ".xml")]
    require(sources and all(identity(row["path"]) == row for row in sources), "Frozen source or assets changed")
    authorization = read(args.audit_root / "authorization.json")
    audit = read(args.audit_root / "checkpoints/32000/checkpoint_audit.json")
    guard = read(args.audit_root / "audit_32000_motion.guard.json")
    require(authorization["original_training_commit"] == COMMIT
            and authorization["plan"] == identity(plan_path) and valid_commit(authorization["evaluation_commit"])
            and guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_reaped"] is True
            and guard["cleanup"]["worker_return_code"] == 0
            and audit["status"] == "PASSED" and audit["iteration"] == ITERATION
            and audit["cuda_initialized"] is False and len(audit["records"]) == 2
            and {row["robot"] for row in audit["records"]} == {"g1", "q1"}, "Missing clean32k audit provenance")
    for row in audit["records"]:
        require(row["iteration"] == ITERATION and row["finite_tensors"] == 330
                and row["inference_metadata"]["code_commit"] == COMMIT
                and row["inference_metadata"]["iteration"] == ITERATION
                and Path(row["checkpoint"]["path"]) == Path(plan["output"]) / "run" / row["robot"] / "ckpt_32000.pt"
                and identity(row["checkpoint"]["path"]) == row["checkpoint"], "Audited32k checkpoint changed")
    code = validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
    require({str(HERE / name) for name in NEW_FILES} <= {row["path"] for row in code}, "Missing published paused/GPU code")
    return config, runtime, plan, frozen, receipt, sources, audit, code


class Budget:
    def __init__(self, config, receipt, output, owner):
        self.config, self.receipt, self.output, self.owner = config, receipt, output, owner
        self.started = self.phase_started = time.monotonic()
        self.maximum_total = self.maximum_gpu = self.maximum_rss = 0
        self.last_gpu, self.gpu_time = None, 0.

    def check(self):
        paused_state(self.config, self.receipt)
        assert_owner(self.owner["pid"], self.owner["start_ticks"])
        rows = subtree(Path("/proc"), self.owner)
        pids = {row["pid"] for row in rows}
        require(not pids & set(HANDLES), "An original training PID entered the new campaign")
        current = time.monotonic()
        require(current-self.started < 7200 and current-self.phase_started < 1800,
                "Paused campaign whole/phase wall-time bound reached")
        if self.last_gpu is None or current-self.gpu_time >= 2:
            self.last_gpu, self.gpu_time = gpu_snapshot(self.config), current
            require(compute_pids() <= pids, "GPU has compute outside this owned campaign subtree")
        sample, storage = memory(Path("/sys/fs/cgroup/memory")), allocated_tree(self.output)
        resource_limits(sample, self.last_gpu)
        require(storage["unique_allocated_bytes"] < 3 * GIB and shutil.disk_usage(self.output).free >= 100 * GIB,
                "Paused campaign storage limit reached")
        self.maximum_total = max(self.maximum_total, sample["total_bytes"])
        self.maximum_gpu = max(self.maximum_gpu, self.last_gpu["used_mib"])
        self.maximum_rss = max(self.maximum_rss, sum(row.get("VmRSS_bytes", 0) for row in rows))
        record = dict(memory=sample, gpu=self.last_gpu, storage=storage,
            maximum_total_bytes=self.maximum_total, maximum_gpu_used_mib=self.maximum_gpu,
            maximum_subtree_rss_bytes=self.maximum_rss, gpu_lock_owner=self.owner,
            training_signals_sent=0, memory_reclaim_attempted=False)
        save(self.output / "resources.json", record)
        return record


def gpu_command(args, config, runtime, plan, selected, baseline, horizon=10, method="masked"):
    command = [runtime["morph_python"], "-u", str(HERE / "gpu_preview_worker.py")]
    values = {"plan": TRAINING / "e1024_train_plan.json", "checkpoint": selected["path"],
        "frozen": config["frozen_data"], "primebot-checkpoint": baseline["path"], "morph-source": plan["source"],
        "primebot-source": runtime["adapted_source"], "preview-frames": horizon, "tail-method": method,
        "expected-gpu": config["expected_gpu"], "gpu-owner-pid": args.gpu_owner_pid,
        "gpu-owner-ticks": args.gpu_owner_ticks}
    for key, value in values.items():
        command.extend(("--"+key, str(value)))
    return command


def equivalence(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    from gpu_preview_worker import GpuPolicyProcess
    from policy_process import PolicyProcess
    from supermimic.data import Q1Motions
    from unittest.mock import patch
    import mujoco
    selected, baseline = checkpoint(args, config)
    robot, _ = load_physical_robot(baseline["path"])
    original, records = {}, []
    for gpu in (False, True):
        command = (gpu_command(args, config, runtime, plan, selected, baseline) if gpu else
                   policy_command(runtime, config, plan, selected, baseline))
        log = args.output / ("gpu_h10_equivalence.log" if gpu else "cpu_native_equivalence.log")
        actor = GpuPolicyProcess(command, log, config["expected_gpu"]) if gpu else PolicyProcess(command, log)
        try:
            require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == ITERATION,
                    "Equivalence worker loaded another checkpoint")
            for item in fixed_rows(frozen):
                name = Path(item["q1_primebot"]["path"]).stem
                reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
                actor.start(item, 0, "human")
                actions, maximum = [], 0.
                with patch.object(mujoco, "mj_step", side_effect=AssertionError("No equivalence physics")), \
                        patch.object(mujoco, "mj_step1", side_effect=AssertionError("No equivalence physics")), \
                        patch.object(mujoco, "mj_step2", side_effect=AssertionError("No equivalence physics")):
                    for frame in range(len(reference)-1):
                        reply = actor.action(frame, reference[frame])
                        values = reply["canonical_action"] + reply["native_action"]
                        require(len(values) == 51 and all(math.isfinite(x) for x in values), "Invalid equivalence action")
                        if gpu:
                            validate_stream(reply["stream"], 10, frame, policy_device="cuda")
                            left, right = torch.tensor(original[name][frame]), torch.tensor(values)
                            require(torch.allclose(left, right, atol=2e-5, rtol=2e-5), "GPU H10 differs from native CPU math")
                            maximum = max(maximum, float((left-right).abs().max()))
                        else:
                            actions.append(values)
                if gpu:
                    records.append(dict(clip=name, frames=len(reference)-1, maximum_absolute_action_difference=maximum))
                else:
                    original[name] = actions
            actor.close()
            require(actor.child.returncode == 0, "Equivalence worker did not exit cleanly")
        finally:
            if not actor.closed:
                actor.close(graceful=False)
    save(args.output / "equivalence.json", dict(status="PASSED", iteration=ITERATION,
        checkpoint=selected, cases=records, atol=2e-5, rtol=2e-5, exact_equality_claimed=False,
        physical_transitions=0, optimizer_updates=0, parent_cuda_initialized=torch.cuda.is_initialized(),
        comparison="native CPU versus timestamped CUDA H10; both runtimes reaped",
        GPU_baseline_physics_reused_from_CPU=False))


def replay(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    from gpu_preview_worker import GpuPolicyProcess
    from q1_native_review import rollout
    selected, baseline = checkpoint(args, config)
    folder = case_folder(args.output, args.horizon, args.clip, args.tail_method)
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", iteration=ITERATION, interface="human", horizon_frames=args.horizon,
        tail_method=args.tail_method, checkpoint=selected, policy_device="cuda", cases=[],
        preprocessing="offline_unverified", final_acceptance=False, measured_end_to_end_latency_ns=None)
    telemetry = (folder / "stream_timing.jsonl").open("x")

    class RecordingProcess(GpuPolicyProcess):
        def action(self, frame, state):
            import json
            reply = super().action(frame, state)
            timing = validate_stream(reply["stream"], args.horizon, frame, args.tail_method, policy_device="cuda")
            telemetry.write(json.dumps(dict(frame=frame, **timing), allow_nan=False)+"\n")
            return reply

    actor = None
    try:
        actor = RecordingProcess(gpu_command(args, config, runtime, plan, selected, baseline,
            args.horizon, args.tail_method), folder / "policy_worker.log", config["expected_gpu"])
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == ITERATION,
                "Preview CUDA worker loaded another checkpoint")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5 and Path(settings["robot_xml"]) == Path(runtime["xml"]),
                "Common physical configuration changed")
        item = next(row for row in fixed_rows(frozen) if Path(row["q1_primebot"]["path"]).stem == args.clip)
        (folder / "review").mkdir()
        row = rollout(robot, actor, settings, item, "human", 0, folder / "review" / (args.clip+"_s0"))
        row.update(horizon_frames=args.horizon, tail_method=args.tail_method, policy_device="cuda",
                   preview_budget_ms=args.horizon*20, measured_end_to_end_latency_ns=None)
        save(Path(row["trace"]).parent / "result.json", row)
        actor.close()
        telemetry.close()
        require(not torch.cuda.is_initialized() and actor.child.returncode == 0, "CPU plant or worker isolation failed")
        result.update(status="PHYSICS_PASSED", cases=[row], render_xml=identity(settings["robot_xml"]),
            policy_worker_reaped=True, policy_worker_return_code=0, cuda_initialized=False,
            stream_telemetry=identity(folder / "stream_timing.jsonl"), asset_sha256=robot.asset_sha256,
            termination=robot.reward_identity(), ended_utc=now())
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        if actor is not None and not actor.closed:
            actor.close(graceful=False)
        telemetry.close()
        save(folder / "review/result.json", result)


def comparison(args, runtime):
    require_plant(runtime)
    from comparison import case_index, summary
    from metrics import compare

    def group(method, horizon):
        rows = []
        for clip in FIXED:
            value = read(case_folder(args.output, horizon, clip, method) / "review/result.json")
            require(value["status"] == "PASSED" and value["iteration"] == ITERATION
                    and value["policy_device"] == "cuda", "Missing completed CUDA campaign case")
            rows.extend(value["cases"])
        return case_index(dict(status="PASSED", cases=rows), FIXED, (0,), "human")

    baseline, rows = group("masked", 10), []
    for method in METHODS:
        for horizon in (HORIZONS[1:] if method == "masked" else (5, 4)):
            candidate = group(method, horizon)
            controls = [("masked", 10, baseline)]
            if method != "masked":
                controls.append(("masked", horizon, group("masked", horizon)))
            for control_method, control_horizon, control in controls:
                pairs = [compare(control[key], candidate[key]) for key in control]
                item = summary(control, candidate, pairs, FIXED)
                item.update(baseline_method=control_method, baseline_horizon=control_horizon,
                    candidate_method=method, candidate_horizon=horizon, pairs=pairs)
                rows.append(item)
    save(args.output / "comparison.json", dict(status="COMPLETED", iteration=ITERATION, comparisons=rows,
        policy_device="cuda", new_physical_cases=27, optimizer_updates=0,
        measured_end_to_end_latency_ms=None, natural_gait_qualified=False,
        scope="Three already-seen motions, deterministic nominal replay; no new abrupt-event acceptance",
        final_acceptance=False, user_accepted_checkpoint=None))


def execute(args):
    if args.child:
        assert_owner(args.gpu_owner_pid, args.gpu_owner_ticks)
        config, runtime = read(args.config), read(args.runtime_config)
        plan, frozen = read(TRAINING / "e1024_train_plan.json"), read(config["frozen_data"])
        if args.child == "equivalence":
            return equivalence(args, config, runtime, plan, frozen)
        if args.child == "replay":
            return replay(args, config, runtime, plan, frozen)
        if args.child == "comparison":
            return comparison(args, runtime)
        runtime_paths(runtime)
        return media(SimpleNamespace(output=case_folder(args.output, args.horizon, args.clip, args.tail_method)), config, runtime)
    config, runtime, plan, frozen, receipt, sources, audited, code = preflight(args)
    controls = [identity(path) for path in (args.config, args.runtime_config, args.pause_receipt, args.code_manifest,
        args.audit_root / "authorization.json", args.audit_root / "checkpoints/32000/checkpoint_audit.json")]
    with own_gpu() as owner:
        require(owner is not None, "Cannot identify the new GPU controller")
        args.gpu_owner_pid, args.gpu_owner_ticks = owner["pid"], owner["start_ticks"]
        args.output.mkdir(exist_ok=False)
        save(args.output / "checkpoint_audit.json", audited)
        save(args.output / "authorization.json", dict(stage="PAUSED_LOW_LATENCY32K_GPU_PREVIEW",
            original_training_commit=COMMIT, evaluation_commit=args.evaluation_commit,
            plan=identity(TRAINING / "e1024_train_plan.json"), controller_inputs=controls,
            training_source_inputs=sources, evaluation_sources=code, gpu_lock_owner=owner,
            cases=cases(), iteration=ITERATION, policy_device="cuda", physics="locked common Primebot CPU",
            maximum_wall_seconds=7200, phase_wall_seconds=1800, maximum_output_bytes=3*GIB,
            new_training_gpu_seconds_budget=14400, new_training_gpu_seconds_used=0,
            original_training_restarted=False, final_acceptance=False, user_accepted_checkpoint=None))
        result = dict(status="RUNNING", phases=[], started_utc=now(), controller=owner, iteration=ITERATION)
        budget, requested = Budget(config, receipt, args.output, owner), False

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        previous = dict(os.environ)
        try:
            os.environ.update(environment(runtime))
            base = [runtime["primebot_python"], "-u", str(Path(__file__).resolve()),
                "--config", str(args.config), "--runtime-config", str(args.runtime_config), "--output", str(args.output),
                "--evaluation-commit", args.evaluation_commit, "--gpu-owner-pid", str(owner["pid"]),
                "--gpu-owner-ticks", owner["start_ticks"]]
            jobs = [("equivalence", [*base, "--child", "equivalence"])]
            for method, horizon, clip in cases():
                extra = ["--tail-method", method, "--horizon", str(horizon), "--clip", clip]
                for phase in ("replay", "media"):
                    jobs.append((f"{method}_h{horizon}_{clip}_{phase}", [*base, "--child", phase, *extra]))
            jobs.append(("comparison", [*base, "--child", "comparison"]))
            for label, command in jobs:
                require(all(identity(row["path"]) == row for row in controls), "Campaign inputs changed")
                validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
                result["phase"] = label
                save(args.output / "result.json", result)
                budget.phase_started = time.monotonic()
                phase = guarded_run(command, ROOT, budget, args.output / (label+".guard.json"),
                    args.output / (label+".log"), check_interval_s=.5, grace_s=5, stop_requested=lambda: requested)
                result["phases"].append(dict(name=label, guard=identity(args.output / (label+".guard.json"))))
                require(phase["status"] == "WORKER_COMPLETED" and phase["cleanup"]["worker_reaped"] is True
                        and phase["cleanup"]["worker_return_code"] == 0, "Paused campaign phase failed: "+label)
            require(all(identity(row["path"]) == row for row in sources), "Frozen sources changed during campaign")
            result.update(status="COMPLETED", source_integrity="PASSED", comparison=identity(args.output / "comparison.json"),
                new_training_updates=0, original_training_restarted=False, user_accepted_checkpoint=None)
        except BaseException as error:
            result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            result["ended_utc"] = now()
            save(args.output / "result.json", result)
            os.environ.clear()
            os.environ.update(previous)
            for signum, handler in handlers.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("config", "runtime-config", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--pause-receipt", type=Path)
    parser.add_argument("--audit-root", type=Path)
    parser.add_argument("--code-manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--child", choices=("equivalence", "replay", "media", "comparison"))
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--gpu-owner-ticks")
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--tail-method", choices=METHODS, default="masked")
    parser.add_argument("--clip", choices=FIXED)
    options = parser.parse_args()
    if not options.child and not all((options.pause_receipt, options.audit_root, options.code_manifest, options.manifest_sha256)):
        parser.error("Controller needs the pause receipt, completed audit root and published code manifest/SHA")
    if options.child in ("replay", "media") and (options.horizon is None or options.clip is None):
        parser.error("A replay/media child needs a horizon and fixed clip")
    execute(options)
