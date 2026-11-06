"""Bounded synthetic-event CUDA replay after the completed paused32k campaign."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]
from common import FIXED, ROOT, identity, now, require, save
from delivery import read
from event_inputs import paired_reference, validate_event_timing
from gpu_preview_worker import GpuPolicyProcess
from interim_review import COMMIT, GIB, TRAINING, fixed_rows, load_physical_robot, media
from paused_preview import Budget, ITERATION, checkpoint, gpu_command, preflight
from paused_runtime import assert_owner, own_gpu
from preview_review import case_folder, environment, require_plant, runtime_paths, validate_code_manifest
from runtime_guard import guarded_run
from stream_events import EventSpec, EventTimeline

EVENTS = ("stop", "reverse", "dropout", "late_burst")
CONFIGS = (("masked", 10), ("masked", 5), ("masked", 4), ("hold", 5),
           ("hold", 4), ("damped_cv", 5), ("damped_cv", 4))
NEW_FILES = ("stream_events.py", "event_inputs.py", "event_gpu_worker.py", "event_review.py")


def cases():
    return [(event, method, horizon, clip) for event in EVENTS
            for method, horizon in CONFIGS for clip in FIXED]


def event_folder(args):
    return case_folder(args.output / args.event_kind, args.horizon, args.clip, args.tail_method)


def event_spec(args):
    return EventSpec(args.event_kind, first_frame=50, duration_frames=10, transport_frames=0)


def nominal_evidence(args, config, audit):
    root = args.nominal_root
    require(root.is_absolute() and root.resolve() == root and root != args.output
            and root.is_relative_to(ROOT / "data/supermimic_for_q1"), "Require the independent completed nominal root")
    result, auth, equality = (read(root / name) for name in ("result.json", "authorization.json", "equivalence.json"))
    require(result.get("status") == "COMPLETED" and result.get("source_integrity") == "PASSED"
            and auth.get("stage") == "PAUSED_LOW_LATENCY32K_GPU_PREVIEW" and auth.get("iteration") == ITERATION
            and auth.get("original_training_commit") == COMMIT and auth.get("policy_device") == "cuda"
            and auth.get("plan") == identity(TRAINING / "e1024_train_plan.json")
            and read(root / "checkpoint_audit.json") == audit and equality.get("status") == "PASSED"
            and equality.get("iteration") == ITERATION and result.get("new_training_updates") == 0,
            "Nominal CUDA campaign or checkpoint provenance is incomplete")
    require(all(identity(row["path"]) == row for row in auth["training_source_inputs"]+auth["evaluation_sources"]),
            "Completed nominal sources changed; do not overwrite its code snapshot")
    files = [root / name for name in ("result.json", "authorization.json", "equivalence.json", "checkpoint_audit.json")]
    selected = next(row["checkpoint"] for row in audit["records"] if row["robot"] == "q1")
    require(equality.get("checkpoint") == selected and len(equality.get("cases", [])) == 3,
            "Nominal CPU/CUDA equality used another policy")
    for method, horizon in CONFIGS:
        for clip in FIXED:
            folder = case_folder(root, horizon, clip, method)
            value = read(folder / "review/result.json")
            require(value.get("status") == "PASSED" and value.get("checkpoint") == selected
                    and value.get("policy_device") == "cuda" and len(value.get("cases", [])) == 1,
                    "Missing nominal media and physical case")
            row = value["cases"][0]
            require(row.get("clip") == clip and row.get("seed") == 0 and row.get("interface") == "human"
                    and Path(row["trace"]) == folder / "review" / (clip+"_s0") / "rollout.npz",
                    "Nominal trace identity differs")
            files.extend((folder / "review/result.json", Path(row["trace"])))
    return [identity(path) for path in files]


def command(args, config, runtime, plan, selected, baseline, *, nominal=False):
    values = gpu_command(args, config, runtime, plan, selected, baseline, args.horizon, args.tail_method)
    values[2] = str(HERE / "event_gpu_worker.py")
    values.extend(("--event-kind", "nominal" if nominal else args.event_kind, "--event-frame", "50",
                   "--event-duration", "10", "--transport-frames", "0"))
    return values


def equivalence(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    from supermimic.data import Q1Motions
    selected, baseline = checkpoint(args, config)
    robot, _ = load_physical_robot(baseline["path"])
    records = []
    for method, horizon in CONFIGS:
        args.tail_method, args.horizon = method, horizon
        original = {}
        for event in (False, True):
            argv = (command(args, config, runtime, plan, selected, baseline, nominal=True) if event else
                    gpu_command(args, config, runtime, plan, selected, baseline, horizon, method))
            actor = GpuPolicyProcess(argv, args.output / f"equivalence_{method}_h{horizon}_{event}.log", config["expected_gpu"])
            try:
                require(actor.ready["checkpoint"] == selected, "Event equivalence loaded another policy")
                for item in fixed_rows(frozen):
                    name = Path(item["q1_primebot"]["path"]).stem
                    reference = Q1Motions(robot, [Path(item["q1_primebot"]["path"])], device="cpu").load(0)
                    actor.start(item, 0, "human")
                    timeline = EventTimeline(len(reference), EventSpec("nominal", first_frame=50))
                    values, maximum = [], 0.
                    for frame in range(len(reference)-1):
                        reply = actor.action(frame, reference[frame])
                        action = torch.tensor(reply["canonical_action"]+reply["native_action"])
                        require(action.shape == (51,) and torch.isfinite(action).all().item(), "Invalid equality action")
                        if event:
                            validate_event_timing(reply["stream"], frame, horizon, method, timeline)
                            left = original[name][frame]
                            require(torch.allclose(left, action, atol=2e-5, rtol=2e-5), "Event nominal adapter changed actions")
                            maximum = max(maximum, float((left-action).abs().max()))
                        else:
                            values.append(action)
                    if event:
                        records.append(dict(clip=name, method=method, horizon=horizon, frames=len(reference)-1,
                                            maximum_absolute_action_difference=maximum))
                    else:
                        original[name] = values
                actor.close()
                require(actor.child.returncode == 0, "Equivalence policy did not exit cleanly")
            finally:
                actor.close(graceful=False)
    require(not torch.cuda.is_initialized(), "Event equivalence parent initialized CUDA")
    save(args.output / "equivalence.json", dict(status="PASSED", checkpoint=selected, cases=records,
        atol=2e-5, rtol=2e-5, physical_transitions=0, optimizer_updates=0,
        scope="new event adapter in nominal mode versus completed GPU preview adapter; original audit reused"))


def rollout(robot, actor, settings, item, args, directory):
    import numpy as np
    import torch
    from evaluate import diagnostics, loads
    from metrics import tracking
    from supermimic.data import Q1Motions
    from supermimic.simulator import Q1Simulator

    torch.manual_seed(0)
    for key in ("q1_primebot", "q1_morph", "human"):
        require(identity(item[key]["path"]) == item[key], "Frozen event references changed")
    path = Path(item["q1_primebot"]["path"])
    original = Q1Motions(robot, [path], device="cpu").load(0)
    derived, mapping = paired_reference(original.numpy(), event_spec(args), robot.model)
    target = torch.from_numpy(derived)
    started = actor.start(item, 0, "human")
    simulator = Q1Simulator(robot, 1, "cpu", settings["action_scale"],
        use_tn=settings.get("use_tn", False), physics_timestep=settings.get("physics_timestep"))
    simulator.reset(torch.tensor([0]), target[:1], randomize=False)
    state = simulator.state()
    actual, forces, actions = [state[0].numpy().copy()], [loads(simulator)], []
    failure = None
    with torch.inference_mode():
        for frame in range(len(target)-1):
            reply = actor.action(frame, state[0])
            native = torch.tensor(reply["native_action"], dtype=state.dtype)[None]
            action = np.asarray(reply["canonical_action"], dtype=np.float32)
            require(native.shape == (1, 22) and action.shape == (29,)
                    and torch.isfinite(native).all().item() and np.isfinite(action).all(), "Invalid event physical action")
            state = simulator.step(native)
            require(torch.isfinite(state).all().item(), "Nonfinite event physical state")
            actual.append(state[0].numpy().copy())
            forces.append(loads(simulator))
            actions.append(action.copy())
            if robot.terminated(state, target[frame+1:frame+2]).item():
                failure = frame+1
                break
    actual, forces = np.asarray(actual), np.asarray(forces)
    body, body_ref, height, slip, tilt, target_height, diagnosis = diagnostics(robot, actual, derived, forces)
    valid = len(actual)-1 if failure is None else max(0, failure-1)
    upper = [i for i, name in enumerate(robot.joint_names) if "shoulder" in name or "elbow" in name]
    row = dict(clip=path.stem, seed=0, interface="human", event=args.event_kind,
        horizon_frames=args.horizon, tail_method=args.tail_method, policy_device="cuda",
        complete=failure is None and len(actual) == len(target), failure_frame=failure,
        valid_steps=valid, attempted_steps=len(actual)-1, reference_steps=len(target)-1,
        reference_sha256=item["q1_primebot"]["sha256"], native_reference_sha256=item["q1_morph"]["sha256"],
        human_sha256=item["human"]["sha256"], derived_reference_sha256=hashlib.sha256(derived.tobytes()).hexdigest(),
        diagnostics=diagnosis, original_paired_reference=started["paired_reference"], event_mapping=mapping,
        reference_is_synthetic=True, evaluation_randomization=False, measured_end_to_end_latency_ns=None,
        trace=str(directory / "rollout.npz"), plant="locked common Primebot CPU", event_safety_qualified=False,
        upper_joint_indices=upper, upper_joint_names=[robot.joint_names[i] for i in upper])
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(row["trace"], actual=actual, reference=derived, body_actual=body, body_reference=body_ref,
        load=forces, sole_height=height, material_point_slip=slip, reference_sole_height=target_height,
        tilt=tilt, canonical_action=np.asarray(actions))
    if valid:
        row["valid_prefix_metrics"] = tracking(actual, derived, body, body_ref, valid)
    if row["complete"]:
        row["full_clip_metrics"] = row["valid_prefix_metrics"]
    with (directory / "foot_posture.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "time_s", "left_sole_m", "right_sole_m", "left_load_n", "right_load_n",
                         "left_slip_m_s", "right_slip_m_s", "tilt_rad"))
        writer.writerows((i, i*.02, *height[i], *forces[i], *slip[i], tilt[i]) for i in range(len(actual)))
    with (directory / "upper_timing.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s"]+[stage+"_"+robot.joint_names[i]
            for stage in ("actual", "reference") for i in upper])
        writer.writerows([i, i*.02]+actual[i, 7:29][upper].tolist()+derived[i, 7:29][upper].tolist()
                         for i in range(len(actual)))
    save(directory / "result.json", row)
    return row


def replay(args, config, runtime, plan, frozen):
    torch = require_plant(runtime)
    selected, baseline = checkpoint(args, config)
    folder = event_folder(args)
    folder.mkdir(parents=True, exist_ok=False)
    item = next(row for row in fixed_rows(frozen) if Path(row["q1_primebot"]["path"]).stem == args.clip)
    result = dict(status="RUNNING", iteration=ITERATION, checkpoint=selected, interface="human", cases=[],
        policy_device="cuda", event=args.event_kind, horizon_frames=args.horizon, tail_method=args.tail_method,
        final_acceptance=False, measured_end_to_end_latency_ns=None)
    actor, telemetry = None, (folder / "stream_timing.jsonl").open("x")
    timeline = None

    class RecordingProcess(GpuPolicyProcess):
        def start(self, item, seed, interface):
            nonlocal timeline
            value = super().start(item, seed, interface)
            timeline = EventTimeline(value["paired_reference"]["frames"], event_spec(args))
            return value

        def action(self, frame, state):
            reply = super().action(frame, state)
            value = validate_event_timing(reply["stream"], frame, args.horizon, args.tail_method, timeline)
            telemetry.write(json.dumps(value, allow_nan=False)+"\n")
            if frame % 50 == 0:
                telemetry.flush()
            return reply

    try:
        actor = RecordingProcess(command(args, config, runtime, plan, selected, baseline),
            folder / "policy_worker.log", config["expected_gpu"])
        require(actor.ready["checkpoint"] == selected and actor.ready["metadata"]["iteration"] == ITERATION,
                "Event worker loaded another policy")
        robot, settings = load_physical_robot(baseline["path"])
        require(settings["action_scale"] == .5 and settings["robot_xml"] == runtime["xml"], "Event plant changed")
        directory = folder / "review" / (args.clip+"_s0")
        row = rollout(robot, actor, settings, item, args, directory)
        actor.close()
        require(actor.child.returncode == 0 and not torch.cuda.is_initialized(), "Event runtime isolation failed")
        result.update(status="PHYSICS_PASSED", cases=[row], render_xml=identity(settings["robot_xml"]),
            policy_worker_reaped=True, policy_worker_return_code=0, cuda_initialized=False,
            asset_sha256=robot.asset_sha256, termination=robot.reward_identity())
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if actor is not None:
            actor.close(graceful=False)
        telemetry.close()
        result.update(ended_utc=now(), stream_telemetry=identity(folder / "stream_timing.jsonl"))
        save(folder / "review/result.json", result)


def comparison(args, runtime):
    import numpy as np
    require_plant(runtime)
    from metrics import compare
    from tracking_response import analyze_signal

    rows = []
    for event, method, horizon, clip in cases():
        candidate = read(case_folder(args.output / event, horizon, clip, method) / "review/result.json")
        baseline = read(case_folder(args.output / event, 10, clip, "masked") / "review/result.json")
        require(candidate["status"] == baseline["status"] == "PASSED", "Missing event physical/media result")
        row, control = candidate["cases"][0], baseline["cases"][0]
        require(row["derived_reference_sha256"] == control["derived_reference_sha256"]
                and row["event"] == control["event"], "Event comparison targets differ")
        timing = [json.loads(line) for line in (case_folder(args.output / event, horizon, clip, method)
                  / "stream_timing.jsonl").read_text().splitlines()]
        fallback = [i for i, t in enumerate(timing) if t["fallback"] is not None]
        recovery = next((i for i in range((fallback[-1]+1) if fallback else len(timing), len(timing))
                         if timing[i]["fallback"] is None), None)
        with np.load(row["trace"], allow_pickle=False) as trace:
            action = trace["canonical_action"]
            jump = np.max(np.abs(np.diff(action, axis=0)), axis=1) if len(action) > 1 else np.array([])
            recovery_jump = float(jump[recovery-1]) if recovery is not None and recovery > 0 else None
            phase = ({name: analyze_signal(trace["reference"][:, 7+i], trace["actual"][:, 7+i])
                      for i, name in zip(row["upper_joint_indices"], row["upper_joint_names"])}
                     if row["complete"] else None)
        rows.append(dict(event=event, method=method, horizon=horizon, clip=clip, complete=row["complete"],
            comparison_to_same_event_h10=compare(control, row), fallback_frames=len(fallback),
            first_fallback_frame=fallback[0] if fallback else None, first_recovered_frame=recovery,
            recovery_max_joint_action_jump=recovery_jump,
            action_jump_p95=float(np.quantile(jump, .95)) if len(jump) else None,
            diagnostics=row["diagnostics"], descriptive_upper_phase=phase,
            phase_is_measured_robot_delay=False, measured_end_to_end_latency_ns=None))
    save(args.output / "comparison.json", dict(status="COMPLETED", cases=rows, new_physical_cases=len(rows),
        optimizer_updates=0, synthetic_events=True, input_constraints_validated=False,
        natural_teleoperation_qualified=False, final_acceptance=False, user_accepted_checkpoint=None))


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
        return media(SimpleNamespace(output=event_folder(args)), config, runtime)
    config, runtime, plan, frozen, receipt, sources, audit, code = preflight(args)
    require({str(HERE / name) for name in NEW_FILES} <= {row["path"] for row in code}, "Missing published event code")
    nominal = nominal_evidence(args, config, audit)
    controls = [identity(path) for path in (args.config, args.runtime_config, args.pause_receipt, args.code_manifest)]
    nominal_controls = [row for row in nominal if Path(row["path"]).suffix == ".json"]
    with own_gpu() as owner:
        require(owner is not None, "Cannot identify the event controller")
        args.gpu_owner_pid, args.gpu_owner_ticks = owner["pid"], owner["start_ticks"]
        args.output.mkdir(exist_ok=False)
        save(args.output / "checkpoint_audit.json", audit)
        save(args.output / "authorization.json", dict(stage="PAUSED32K_SYNTHETIC_EVENTS", evaluation_commit=args.evaluation_commit,
            original_training_commit=COMMIT, iteration=ITERATION, evaluation_sources=code, controller_inputs=controls,
            training_source_inputs=sources, reused_nominal_evidence=nominal, gpu_lock_owner=owner,
            cases=cases(), maximum_output_bytes=3*GIB, maximum_wall_seconds=7200, phase_wall_seconds=1800,
            new_training_gpu_seconds_used=0, new_training_gpu_seconds_cap=14400, final_acceptance=False,
            synthetic_events=True, fallback="visible past pose held in ten generated slots; policy recomputes action"))
        result = dict(status="RUNNING", phases=[], controller=owner, started_utc=now())
        budget, requested, previous = Budget(config, receipt, args.output, owner), False, dict(os.environ)

        def stop(signum, frame):
            nonlocal requested
            requested = True

        handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            os.environ.update(environment(runtime))
            base = [runtime["primebot_python"], "-u", str(Path(__file__).resolve()),
                "--config", str(args.config), "--runtime-config", str(args.runtime_config), "--output", str(args.output),
                "--evaluation-commit", args.evaluation_commit, "--gpu-owner-pid", str(owner["pid"]),
                "--gpu-owner-ticks", owner["start_ticks"]]
            jobs = [("equivalence", [*base, "--child", "equivalence"])]
            for event, method, horizon, clip in cases():
                extra = ["--event-kind", event, "--tail-method", method, "--horizon", str(horizon), "--clip", clip]
                for phase in ("replay", "media"):
                    jobs.append((f"{event}_{method}_h{horizon}_{clip}_{phase}", [*base, "--child", phase, *extra]))
            jobs.append(("comparison", [*base, "--child", "comparison"]))
            for label, argv in jobs:
                require(all(identity(row["path"]) == row for row in controls+nominal_controls), "Event inputs changed")
                validate_code_manifest(args.code_manifest, args.manifest_sha256, args.evaluation_commit)
                result["phase"] = label
                save(args.output / "result.json", result)
                budget.phase_started = time.monotonic()
                value = guarded_run(argv, ROOT, budget, args.output / (label+".guard.json"),
                    args.output / (label+".log"), check_interval_s=.5, grace_s=5, stop_requested=lambda: requested)
                result["phases"].append(dict(name=label, guard=identity(args.output / (label+".guard.json"))))
                require(value["status"] == "WORKER_COMPLETED" and value["cleanup"]["worker_reaped"] is True
                        and value["cleanup"]["worker_return_code"] == 0, "Event phase failed: "+label)
            require(all(identity(row["path"]) == row for row in sources+nominal), "Frozen or reused sources changed during events")
            result.update(status="COMPLETED", source_integrity="PASSED", new_training_updates=0,
                comparison=identity(args.output / "comparison.json"), user_accepted_checkpoint=None)
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
    for name in ("pause-receipt", "audit-root", "code-manifest", "nominal-root"):
        parser.add_argument("--"+name, type=Path)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--child", choices=("equivalence", "replay", "media", "comparison"))
    parser.add_argument("--gpu-owner-pid", type=int)
    parser.add_argument("--gpu-owner-ticks")
    parser.add_argument("--event-kind", choices=EVENTS)
    parser.add_argument("--horizon", type=int, choices=(4, 5, 10))
    parser.add_argument("--tail-method", choices=("masked", "hold", "damped_cv"), default="masked")
    parser.add_argument("--clip", choices=FIXED)
    args = parser.parse_args()
    if not args.child and not all((args.pause_receipt, args.audit_root, args.code_manifest, args.nominal_root, args.manifest_sha256)):
        parser.error("Controller needs pause, audit, completed nominal root and published manifest/SHA")
    if args.child in ("replay", "media") and not all((args.event_kind, args.horizon, args.clip)):
        parser.error("Replay/media needs an event, method, horizon and fixed clip")
    execute(args)
