"""Evaluate the native shared Q1 policy in the identical Primebot CPU plant."""

import argparse
import gc
import json
from pathlib import Path
import random
import sys

from contracts import validate_frozen
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


def rollout(robot, actor, config, item, spec, interface, seed, output, human_root):
    import numpy as np
    import torch
    from morphmimic.constants import ANCHOR_IDX
    from morphmimic.data.motion_library import MotionLibrary
    from morphmimic.model.human import pack_human_goal
    from morphmimic.model.rewards_torch import analytic_obs, goal_features
    from morphmimic.model.state import split_state
    from morphmimic.train.obs_history import ObsHistory
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
    library = MotionLibrary([item["q1_morph"]["path"]], device="cpu", robot=spec, human_root=human_root)
    require(library.num_clips == 1 and int(library.clip_len[0]) == len(reference), "Paired clip length changed")
    paired = assert_paired_reference(robot, reference, library, spec)
    require(library.human_pos_w is not None and library.human_quat_w is not None, "Missing human reference")
    with np.load(item["human"]["path"], allow_pickle=False) as human:
        require(np.array_equal(library.human_pos_w.numpy(), human["human_pos_w"])
                and np.array_equal(library.human_quat_w.numpy(), human["human_quat_w"]),
                "MotionLibrary resolved a different human sidecar")
    simulator = Q1Simulator(robot, 1, "cpu", config["action_scale"],
        use_tn=config.get("use_tn", False), physics_timestep=config.get("physics_timestep"))
    simulator.reset(torch.tensor([0]), reference[:1], randomize=False)
    mask = actor.present
    default = spec.translator.to_canonical(torch.zeros(22))

    def observation(state):
        value = analytic_obs(state, torch.zeros(29), default, state[:, 240:], spec.imu_slot)
        return torch.cat((torch.where(mask, value[:, :29], 0.), torch.where(mask, value[:, 29:58], 0.),
                          value[:, 58:64], torch.where(mask, value[:, 64:], 0.)), -1)

    state = simulator.state()
    converted = native_state(robot, state, spec)
    history = ObsHistory(1, 10, 93, "cpu")
    history.reset(None, observation(converted))
    actual, forces, actions = [state[0].numpy().copy()], [loads(simulator)], []
    failure = None
    with torch.inference_mode():
        for frame in range(len(reference) - 1):
            offsets = torch.tensor(reference_window_indices(frame, len(reference)))
            future = library.state(offsets)[None]
            goals = dict(goal=goal_features(converted, future, anchor="reference"),
                goal_kp=pack_human_goal(library.human_pos_w[offsets][None],
                    library.human_quat_w[offsets][None], split_state(converted).body_quat[:, ANCHOR_IDX]))
            action = actor(history.flat(), goals, interface)
            native = spec.translator.to_native_delta(action)
            state = simulator.step(native)
            require(torch.isfinite(state).all().item(), "Nonfinite physical common-plant state")
            actual.append(state[0].numpy().copy())
            forces.append(loads(simulator))
            actions.append(action[0].numpy().copy())
            if robot.terminated(state, reference[frame + 1:frame + 2]).item():
                failure = frame + 1
                break
            converted = native_state(robot, state, spec)
            history.push(observation(converted))
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
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "Common-plant evaluation must stay on CPU")
    plan = read(args.plan)
    sys.path[:0] = [str(Path(plan["source"]) / "src"), str(args.primebot_source / "src"),
                   str(HERE.parent / "morphmimic_q1"), str(HERE.parent / "primebot_q1_review")]
    from bootstrap import install
    spec = install("q1")["robot"]
    from inference import load_policy
    from evaluate import load_actor
    from video import render_case
    frozen = read(args.frozen)
    validate_frozen(frozen)
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT / "data/supermimic_for_q1"),
            "Traces and checkpoints remain on the server")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", started_utc=now(), cases=[], optimizer_updates=0,
        training_replay_writes=0, backend="CPU identical Primebot plant; native v0.1 policy inputs",
        policy_checkpoint=identity(args.checkpoint), primebot_checkpoint=identity(args.primebot_checkpoint),
        interface=args.interface, user_accepted_checkpoint=None)
    try:
        actor, metadata = load_policy(args.checkpoint, spec, plan["code_commit"], expected_plan=plan)
        robot, unused, config, _, _, _ = load_actor(args.primebot_checkpoint)
        del unused
        gc.collect()
        require(config["action_scale"] == .5, "Common Q1 action scale changed")
        result.update(metadata=metadata, termination=robot.reward_identity(), asset_sha256=robot.asset_sha256,
            comparison_limits=["Common-plant comparison, not a claim that the native mjlab scene is identical.",
                "Motion goal uses frozen native velocities; physical initialization uses Primebot's differentiated velocities.",
                "Q1 was seen in shared training; this is not unseen-morphology zero-shot transfer."])
        items = frozen["rows"] if args.library else [r for r in frozen["rows"] if Path(r["q1_primebot"]["path"]).stem in FIXED]
        for item in items:
            for seed in ((0,) if args.library else (0, 1, 2)):
                name = Path(item["q1_primebot"]["path"]).stem
                directory = args.output / f"{name}_s{seed}"
                row = rollout(robot, actor, config, item, spec, args.interface, seed, directory, plan["human_root"])
                row["relative"] = str(Path(item["q1_primebot"]["path"]).relative_to(config["motion_dir"]))
                if args.render and seed == 0:
                    row["video"] = render_case(Path(config["robot_xml"]), row, directory / "tracking.mp4")
                result["cases"].append(row)
                save(args.output / "progress.json", result)
        require(not torch.cuda.is_initialized(), "CPU evaluation initialized CUDA")
        result.update(status="PASSED", ended_utc=now(), complete_cases=sum(r["complete"] for r in result["cases"]),
                      cuda_initialized=False)
    except BaseException as error:
        result.update(status="FAILED", ended_utc=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("plan", "checkpoint", "primebot-checkpoint", "primebot-source", "frozen", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--interface", choices=("motion", "human"), required=True)
    parser.add_argument("--library", action="store_true")
    parser.add_argument("--render", action="store_true")
    execute(parser.parse_args())
