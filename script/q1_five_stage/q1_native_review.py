"""Evaluate the native shared Q1 policy in the identical Primebot CPU plant."""

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import sys

from contracts import valid_commit, validate_frozen
from delivery import read
from common import FIXED, ROOT, identity, now, require, save

HERE = Path(__file__).resolve().parent


def reference_window_indices(frame, frames, count=10):
    require(type(frame) is int and 0 <= frame < frames - 1 and count > 0,
            "Policy window must lead a valid physical transition")
    return [min(frame + 1 + i, frames - 1) for i in range(count)]


def native_state(robot, value, spec):
    import torch
    from morphmimic.model.state import StateParts, join_state
    position, quat, velocity, angular = robot.kinematics(value)
    indices = [robot.model.body(name).id for name in spec.tracked_bodies]
    tr = spec.translator
    physical = join_state(StateParts(tr.to_canonical(value[..., 7:29]),
        tr.to_canonical_delta(value[..., 35:57]), position[..., indices, :],
        quat[..., indices, :], velocity[..., indices, :], angular[..., indices, :]))
    return torch.cat((physical[..., :240], tr.to_canonical_delta(value[..., 57:79])), -1)


def assert_paired_reference(robot, primebot, library, spec):
    import torch
    reference = library.state(torch.arange(len(primebot)))
    require(len(reference) == len(primebot), "Robot reference lengths differ")
    native = spec.translator.to_native(reference[:, :29], torch.zeros(22))
    require(torch.allclose(native, primebot[:, 7:29], atol=2e-5, rtol=0.)
            and torch.allclose(library.root_pos_w, primebot[:, :3], atol=2e-5, rtol=0.),
            "Paired references do not describe the same Q1 poses")
    dot = (library.root_quat_w * primebot[:, 3:7]).sum(-1).abs()
    require(torch.allclose(dot, torch.ones_like(dot), atol=2e-5, rtol=0.), "Paired root orientation differs")
    derived = native_state(robot, primebot, spec)
    ref_q = reference[:, 100:156].reshape(-1, 14, 4)
    kin_q = derived[:, 100:156].reshape(-1, 14, 4)
    require(torch.allclose(reference[:, 58:100], derived[:, 58:100], atol=3e-5, rtol=0.)
            and torch.allclose((ref_q * kin_q).sum(-1).abs(), torch.ones_like(ref_q[..., 0]), atol=3e-5, rtol=0.),
            "Native reference FK differs from the common physical Q1 plant")
    return dict(frames=len(primebot), qpos_tolerance=2e-5, body_fk_tolerance=3e-5,
                velocity_note="native goal keeps its frozen library velocities; physical reset uses the common Primebot velocities")


def rollout(robot, actor, config, item, interface, seed, output):
    import numpy as np
    import torch
    from supermimic.data import Q1Motions
    from supermimic.simulator import Q1Simulator
    from evaluate import diagnostics, loads
    from metrics import tracking

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    path = Path(item["q1_primebot"]["path"])
    for key in ("q1_primebot", "q1_morph", "human"):
        require(identity(item[key]["path"]) == item[key], "Frozen paired reference changed")
    reference = Q1Motions(robot, [path], device="cpu").load(0)
    paired = actor.start(item, seed, interface)["paired_reference"]
    simulator = Q1Simulator(robot, 1, "cpu", config["action_scale"],
        use_tn=config.get("use_tn", False), physics_timestep=config.get("physics_timestep"))
    simulator.reset(torch.tensor([0]), reference[:1], randomize=False)
    state = simulator.state()
    actual, forces, actions = [state[0].numpy().copy()], [loads(simulator)], []
    failure = None
    with torch.inference_mode():
        for frame in range(len(reference) - 1):
            reply = actor.action(frame, state[0])
            native = torch.tensor(reply["native_action"], dtype=state.dtype)[None]
            action = np.asarray(reply["canonical_action"], dtype=np.float32)
            require(native.shape == (1, 22) and action.shape == (29,)
                    and torch.isfinite(native).all().item() and np.isfinite(action).all(),
                    "Invalid native policy action from its isolated runtime")
            state = simulator.step(native)
            require(torch.isfinite(state).all().item(), "Nonfinite physical common-plant state")
            actual.append(state[0].numpy().copy())
            forces.append(loads(simulator))
            actions.append(action.copy())
            if robot.terminated(state, reference[frame + 1:frame + 2]).item():
                failure = frame + 1
                break
    actual, target, forces = np.asarray(actual), reference.numpy(), np.asarray(forces)
    body, body_ref, height, slip, tilt, target_height, diagnosis = diagnostics(robot, actual, target, forces)
    valid = len(actual) - 1 if failure is None else max(0, failure - 1)
    row = dict(clip=path.stem, seed=seed, reference_sha256=item["q1_primebot"]["sha256"],
        native_reference_sha256=item["q1_morph"]["sha256"], human_sha256=item["human"]["sha256"],
        interface=interface, complete=failure is None and len(actual) == len(target), failure_frame=failure,
        valid_steps=valid, attempted_steps=len(actual) - 1, reference_steps=len(target) - 1,
        diagnostics=diagnosis, paired_reference=paired, evaluation_randomization=False,
        repeated_seed_note="nominal deterministic replays; seeds are not independent randomization trials",
        trace=str(output / "rollout.npz"), plant="identical Primebot Q1 CPU simulator and termination")
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(row["trace"], actual=actual, reference=target, body_actual=body, body_reference=body_ref,
        load=forces, sole_height=height, material_point_slip=slip, reference_sole_height=target_height,
        tilt=tilt, canonical_action=np.asarray(actions))
    if valid:
        row["valid_prefix_metrics"] = tracking(actual, target, body, body_ref, valid)
    if row["complete"]:
        row["full_clip_metrics"] = row["valid_prefix_metrics"]
    import csv
    with (output / "foot_posture.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "time_s", "left_sole_m", "right_sole_m", "left_load_n", "right_load_n",
                         "left_slip_m_s", "right_slip_m_s", "tilt_rad"))
        writer.writerows((i, i * .02, *height[i], *forces[i], *slip[i], tilt[i]) for i in range(len(actual)))
    upper = [i for i, name in enumerate(robot.joint_names) if "shoulder" in name or "elbow" in name]
    with (output / "upper_timing.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s"] + [stage + "_" + robot.joint_names[i] for stage in ("actual", "reference") for i in upper])
        writer.writerows([i, i * .02] + actual[i, 7:29][upper].tolist() + target[i, 7:29][upper].tolist()
                         for i in range(len(actual)))
    save(output / "result.json", row)
    return row


def execute(args):
    from native_review import ready_training
    require(valid_commit(args.evaluation_commit), "Require published evaluation code")
    plan, pipeline, audit, frozen = map(read, (args.plan, args.pipeline, args.audit, args.frozen))
    iteration = ready_training(plan, pipeline, identity(args.plan), audit, "q1", identity(args.checkpoint))
    validate_frozen(frozen)
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT / "data/supermimic_for_q1")
            and not args.output.exists(), "Use a new common-plant evaluation root on the sole server")
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "Common-plant evaluation must stay on CPU")
    require(Path(sys.executable).absolute() == args.plant_python,
            "Common physics must run in the original locked Primebot Python")
    require(importlib.metadata.version("mujoco") == "3.6.0"
            and importlib.metadata.version("torch").split("+")[0] == "2.7.0",
            "Common physics dependencies differ from the original Primebot evaluation")
    sys.path[:0] = [str(args.primebot_source / "src"), str(HERE.parent / "primebot_q1_review")]
    from policy_process import PolicyProcess
    from evaluate import load_actor
    from video import render_case
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", started_utc=now(), cases=[], optimizer_updates=0,
        training_replay_writes=0, backend="Locked Primebot CPU plant; isolated native v0.1 CPU policy",
        plant_runtime=dict(python=sys.executable,
                           packages={name: importlib.metadata.version(name) for name in ("torch", "mujoco")}),
        policy_checkpoint=identity(args.checkpoint), primebot_checkpoint=identity(args.primebot_checkpoint),
        checkpoint=identity(args.checkpoint), iteration=iteration, robot="q1", plant="common_primebot_cpu",
        training_plan=identity(args.plan), frozen_data=identity(args.frozen),
        evaluation_commit=args.evaluation_commit, interface=args.interface, user_accepted_checkpoint=None)
    actor = None
    try:
        command = [str(args.morph_python), "-u", str(HERE / "policy_process.py")]
        for key, value in dict(plan=args.plan, checkpoint=args.checkpoint, frozen=args.frozen,
            **{"primebot-checkpoint": args.primebot_checkpoint, "morph-source": plan["source"],
               "primebot-source": args.primebot_source}).items():
            command.extend(("--" + key, str(value)))
        actor = PolicyProcess(command, args.output / "policy_worker.log")
        metadata = actor.ready["metadata"]
        require(actor.ready["checkpoint"] == identity(args.checkpoint)
                and actor.ready["primebot_checkpoint"] == identity(args.primebot_checkpoint),
                "Policy worker loaded a different native or baseline checkpoint")
        require(metadata["iteration"] == iteration, "Loaded iteration differs from the CPU audit")
        robot, unused, config, _, _, _ = load_actor(args.primebot_checkpoint)
        del unused
        gc.collect()
        require(config["action_scale"] == .5, "Common Q1 action scale changed")
        result.update(metadata=metadata, policy_runtime=actor.ready,
            termination=robot.reward_identity(), asset_sha256=robot.asset_sha256,
            comparison_limits=["Common-plant comparison, not a claim that the native mjlab scene is identical.",
                "Motion goal uses frozen native velocities; physical initialization uses Primebot's differentiated velocities.",
                "Q1 was seen in shared training; this is not unseen-morphology zero-shot transfer."])
        items = frozen["rows"] if args.library else [r for r in frozen["rows"] if Path(r["q1_primebot"]["path"]).stem in FIXED]
        for item in items:
            for seed in ((0,) if args.library else (0, 1, 2)):
                name = Path(item["q1_primebot"]["path"]).stem
                directory = args.output / f"{name}_s{seed}"
                row = rollout(robot, actor, config, item, args.interface, seed, directory)
                row["relative"] = str(Path(item["q1_primebot"]["path"]).relative_to(config["motion_dir"]))
                if args.render and seed == 0:
                    row["video"] = render_case(Path(config["robot_xml"]), row, directory / "tracking.mp4")
                    save(directory / "result.json", row)
                result["cases"].append(row)
                save(args.output / "progress.json", result)
        require(all(identity(row[key]["path"]) == row[key] for row in items
                    for key in ("q1_primebot", "q1_morph", "human")), "Evaluation references changed during replay")
        require(not torch.cuda.is_initialized(), "CPU evaluation initialized CUDA")
        actor.close()
        result.update(status="PASSED", ended_utc=now(), complete_cases=sum(r["complete"] for r in result["cases"]),
                      cuda_initialized=False, policy_worker_reaped=True, policy_worker_return_code=actor.child.returncode)
    except BaseException as error:
        result.update(status="FAILED", ended_utc=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if actor is not None and not actor.closed:
            actor.close(graceful=False)
        save(args.output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "pipeline", "audit", "checkpoint", "primebot-checkpoint", "primebot-source",
                "plant-python", "morph-python", "frozen", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--interface", choices=("motion", "human"), required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--library", action="store_true")
    parser.add_argument("--render", action="store_true")
    execute(parser.parse_args())
