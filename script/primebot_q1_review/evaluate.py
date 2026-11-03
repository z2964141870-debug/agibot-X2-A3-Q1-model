"""CPU MuJoCo evaluation of saved Primebot weights, outside training replay."""

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1"))
from common import FIXED, identity, now, require, save
from metrics import compare, progress_gate, tracking


def load_actor(checkpoint):
    import torch
    from supermimic.models import MotionPolicy
    from supermimic.robot import Q1Robot
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    config, model_id = payload["config"], payload["identity"]
    robot = Q1Robot(Path(config["robot_xml"]), modeled_action_delay=config.get("modeled_action_delay", False))
    require(robot.asset_sha256 == model_id["asset_sha256"], "Evaluation XML or meshes differ")
    robot.load_state_dict(payload["robot"])
    actor = MotionPolicy(robot, tuple(model_id["actor_hidden"]), activation=config.get("network_activation", "elu"))
    actor.load_state_dict(payload["actor"])
    actor.eval()
    iteration, transitions = payload["iteration"], payload["global_env_steps"]
    del payload
    gc.collect()
    return robot, actor, config, model_id, iteration, transitions


def loads(simulator):
    import mujoco
    import numpy as np
    model, data = simulator.cpu_models[0], simulator.cpu_data[0]
    feet = [model.body(side + "_ankle_roll_link").id for side in ("left", "right")]
    force = np.zeros(6)
    result = np.zeros(2)
    for i in range(data.ncon):
        contact = data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if model.geom_bodyid[g1] != 0 and model.geom_bodyid[g2] != 0:
            continue
        body = int(model.geom_bodyid[g2] if model.geom_bodyid[g1] == 0 else model.geom_bodyid[g1])
        if body in feet:
            mujoco.mj_contactForce(model, data, i, force)
            result[feet.index(body)] += max(0., force[0])
    return result


def diagnostics(robot, states, reference, forces):
    import mujoco
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "project_tools"))
    from audit_q1_saved_sole_geometry import foot_geometry, measure
    model = robot.model
    geometry = [foot_geometry(model, side) for side in ("left", "right")]
    data = mujoco.MjData(model)
    height, slip = measure(model, data, states[:, :29], geometry)
    target_height, _ = measure(model, data, reference[:len(states), :29], geometry)
    p, _, _, _ = robot.kinematics(torch.from_numpy(states))
    rp, _, _, _ = robot.kinematics(torch.from_numpy(reference))
    q = states[:, 3:7]
    tilt = np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1))
    target_q = reference[:len(states), 3:7]
    target_tilt = np.arccos(np.clip(1 - 2 * (target_q[:, 1] ** 2 + target_q[:, 2] ** 2), -1, 1))
    upper = [i for i, name in enumerate(robot.joint_names) if "shoulder" in name or "elbow" in name]
    events = []
    for foot in range(2):
        lifted = height[:, foot] > .01
        events += [(int(i), foot) for i in np.flatnonzero(lifted[1:] & ~lifted[:-1]) + 1]
    events.sort()
    loaded = forces[1:] >= 5.
    mean_slip = [float(slip[1:, i][loaded[:, i]].mean()) if loaded[:, i].any() else None for i in range(2)]
    summary = dict(sole_p95_m=np.quantile(height[1:], .95, axis=0).tolist(),
        reference_sole_p95_m=np.quantile(target_height[1:], .95, axis=0).tolist(),
        lift_events_above_10mm=len(events), alternating_lift_transitions=sum(a[1] != b[1] for a, b in zip(events, events[1:])),
        loaded_slip_mean_m_s=mean_slip, loaded_samples=loaded.sum(axis=0).tolist(),
        torso_tilt_rms_rad=float(np.sqrt(np.mean(tilt[1:] ** 2))),
        tilt_error_rms_rad=float(np.sqrt(np.mean((tilt[1:] - target_tilt[1:]) ** 2))),
        upper_joint_rmse_rad=float(np.sqrt(np.mean((states[1:, 7:29][:, upper] - reference[1:len(states), 7:29][:, upper]) ** 2))),
        upper_joint_peak_to_peak_rad=np.ptp(states[:, 7:29][:, upper], axis=0).tolist(),
        upper_reference_peak_to_peak_rad=np.ptp(reference[:len(states), 7:29][:, upper], axis=0).tolist(),
        root_xy_final_error_m=float(np.linalg.norm(states[-1, :2] - reference[len(states) - 1, :2])),
        load_time_note="last 2ms substep solver force; pose at control end, no continuous-support claim")
    return p.numpy(), rp.numpy(), height, slip, tilt, target_height, summary


def rollout(robot, actor, config, path, seed, output):
    import numpy as np
    import torch
    from supermimic.data import Q1Motions
    from supermimic.models import goal_features
    from supermimic.simulator import Q1Simulator
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    motion = Q1Motions(robot, [path], device="cpu").load(0)
    simulator = Q1Simulator(robot, 1, "cpu", config["action_scale"],
        use_tn=config.get("use_tn", False), physics_timestep=config.get("physics_timestep"))
    simulator.reset(torch.tensor([0]), motion[:1], randomize=False)
    state = simulator.state()
    obs = robot.observation(state, simulator.context)
    history = obs[:, None].expand(-1, 10, -1).clone()
    states = [state[0].numpy().copy()]
    forces = [loads(simulator)]
    failure = None
    with torch.inference_mode():
        for frame in range(len(motion) - 1):
            offsets = torch.arange(frame + 1, frame + 11).clamp(max=len(motion) - 1)
            references = motion[offsets][None]
            action = actor(history, goal_features(robot, references))
            state = simulator.step(action)
            require(bool(torch.isfinite(state).all()), "Nonfinite evaluated state")
            states.append(state[0].numpy().copy())
            forces.append(loads(simulator))
            if bool(robot.terminated(state, references[:, 0]).item()):
                failure = frame + 1
                break
            obs = robot.observation(state, simulator.context)
            history = torch.cat((history[:, 1:], obs[:, None]), 1)
    actual, reference, force = np.asarray(states), motion.numpy(), np.asarray(forces)
    body, body_ref, height, slip, tilt, target_height, diagnosis = diagnostics(robot, actual, reference, force)
    valid = len(actual) - 1 if failure is None else max(0, failure - 1)
    row = dict(clip=path.stem, seed=seed, reference_sha256=identity(path)["sha256"],
        complete=failure is None and len(actual) == len(reference), failure_frame=failure,
        valid_steps=valid, attempted_steps=len(actual) - 1, reference_steps=len(reference) - 1,
        diagnostics=diagnosis, evaluation_randomization=False,
        repeated_seed_note="nominal deterministic rollouts; seeds are not independent randomization trials",
        trace=str(output / "rollout.npz"))
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "rollout.npz", actual=actual, reference=reference,
        body_actual=body, body_reference=body_ref, load=force, sole_height=height,
        material_point_slip=slip, reference_sole_height=target_height, tilt=tilt)
    if valid:
        row["valid_prefix_metrics"] = tracking(actual, reference, body, body_ref, valid)
    if row["complete"]:
        row["full_clip_metrics"] = row["valid_prefix_metrics"]
    with (output / "foot_posture.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "time_s", "left_sole_m", "right_sole_m", "left_load_n", "right_load_n",
                         "left_slip_m_s", "right_slip_m_s", "tilt_rad"))
        writer.writerows((i, i * .02, *height[i], *force[i], *slip[i], tilt[i]) for i in range(len(actual)))
    upper = [i for i, name in enumerate(robot.joint_names) if "shoulder" in name or "elbow" in name]
    with (output / "upper_timing.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s"] + [stage + "_" + robot.joint_names[i] for stage in ("actual", "reference") for i in upper])
        writer.writerows([i, i * .02] + actual[i, 7:29][upper].tolist() + reference[i, 7:29][upper].tolist() for i in range(len(actual)))
    save(output / "result.json", row)
    return row


def execute(checkpoint, output, baseline=None, library=False, render=False):
    import torch
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "CPU evaluation must not initialize CUDA")
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", started_utc=now(), cases=[], optimizer_updates=0, training_replay_writes=0,
                  backend="CPU MuJoCo3.6 native Primebot simulator", weights_storage="SERVER_LOCAL_ONLY")
    try:
        robot, actor, config, model_id, iteration, transitions = load_actor(checkpoint)
        record.update(checkpoint=identity(checkpoint), iteration=iteration, training_transitions=transitions,
                      asset_sha256=robot.asset_sha256)
        paths = [Path(p) for p in model_id["motion_paths"]]
        if not library:
            paths = [p for p in paths if p.stem in FIXED]
            require(len(paths) == 3 and {p.stem for p in paths} == set(FIXED), "Fixed references missing")
        for path in paths:
            for seed in ((0,) if library else (0, 1, 2)):
                row = rollout(robot, actor, config, path, seed, output / f"{path.stem}_s{seed}")
                row["relative"] = str(path.relative_to(Path(config["motion_dir"])))
                record["cases"].append(row)
                if render and seed == 0:
                    from video import render_case
                    try:
                        row["video"] = render_case(Path(config["robot_xml"]), row, output / f"{path.stem}_s0" / "tracking.mp4")
                    except Exception as error:
                        row["video"] = dict(status="FAILED", error=f"{type(error).__name__}: {error}")
                save(output / "progress.json", record)
        if baseline is not None and not library:
            record["gate"] = progress_gate(json.loads(baseline.read_text()), record)
        record.update(status="PASSED", ended_utc=now(), complete_cases=sum(r["complete"] for r in record["cases"]),
                      cuda_initialized=torch.cuda.is_initialized())
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "result.json", record)
    print(json.dumps(dict(status=record["status"], iteration=record["iteration"], complete=record["complete_cases"],
                         cases=len(record["cases"]), gate=record.get("gate", {}).get("passed"))), flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--library", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    execute(args.checkpoint, args.output, args.baseline, args.library, args.render)
