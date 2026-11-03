"""Original native mjlab evaluation of the fresh shared G1/Q1 checkpoints."""

import argparse
from pathlib import Path
import random
import sys

from contracts import valid_commit, validate_frozen
from delivery import read
from common import FIXED, ROOT, identity, now, require, save

HERE = Path(__file__).resolve().parent


def ready_training(plan, pipeline, plan_identity, audit, robot, checkpoint):
    require(plan.get("purpose") == "train" and plan.get("robots") == ["g1", "q1"]
            and valid_commit(plan.get("code_commit")) and robot in plan["robots"],
            "Require the approved fresh shared training plan")
    require(pipeline.get("status") == "SHARED_TRAINING_COMPLETED_PENDING_EVALUATION"
            and pipeline.get("formal_training_completed") is True
            and pipeline.get("formal_plan") == plan_identity,
            "Shared formal training has not completed and passed its controller")
    iteration = audit.get("iteration")
    require(audit.get("status") == "PASSED" and audit.get("cuda_initialized") is False
            and type(iteration) is int and iteration in (0, plan["cfg"]["iterations"]),
            "Require the matching initial or final CPU checkpoint audit")
    rows = audit.get("records", [])
    require(len(rows) == 2 and {row.get("robot") for row in rows} == {"g1", "q1"}
            and all(row.get("iteration") == iteration for row in rows),
            "Both shared robot checkpoint audits are required")
    selected = next(row for row in rows if row["robot"] == robot)
    expected = Path(plan["output"]) / "run" / robot / f"ckpt_{iteration}.pt"
    require(selected.get("checkpoint") == checkpoint and Path(checkpoint["path"]) == expected,
            "Checkpoint does not match the audited native robot and iteration")
    return iteration


def episode_end(frame, frames, terminated, truncated):
    require(type(frame) is type(frames) is int and 1 <= frame < frames
            and type(terminated) is type(truncated) is bool, "Invalid native transition")
    if terminated:
        return dict(complete=False, failure_frame=frame, valid_steps=frame - 1)
    if truncated:
        require(frame == frames - 1, "Early native timeout cannot count as a complete clip")
        return dict(complete=True, failure_frame=None, valid_steps=frame)
    require(frame < frames - 1, "Native clip_end did not fire on the last target frame")
    return None


class ContactLoads:
    """Read existing MJWarp solver forces without stepping or changing the plant."""

    def __init__(self, env, spec):
        import mujoco
        import numpy as np
        import warp as wp
        from native_metrics import foot_shapes
        self.env, self.wp = env, wp
        groups = foot_shapes(env.sim.mj_model, spec)
        self.feet = [{geom for geom, *_ in shapes} for shapes in groups]
        model = env.sim.mj_model
        self.floor = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == 0
                      and int(model.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_PLANE)}
        require(len(self.floor) == 1, "Native diagnostics require one actual static plane")
        floor = next(iter(self.floor))
        require(np.allclose(model.geom_pos[floor], 0.)
                and np.allclose(model.geom_quat[floor], [1., 0., 0., 0.]), "Floor is not horizontal at z0")
        capacity = env.sim.wp_data.contact.geom.shape[0]
        self.indices = wp.array(np.arange(capacity, dtype=np.int32), dtype=wp.int32, device=env.device)
        self.forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=env.device)

    def __call__(self):
        import numpy as np
        from mujoco_warp._src.support import contact_force
        sim = self.env.sim
        count = int(sim.wp_data.nacon.numpy()[0])
        require(0 <= count <= self.indices.size, "Native contact capacity exceeded")
        if not count:
            return np.zeros(2)
        contact_force(sim.wp_model, sim.wp_data, self.indices, False, self.forces)
        forces = self.forces.numpy()[:count]
        geoms = sim.wp_data.contact.geom.numpy()[:count]
        worlds = sim.wp_data.contact.worldid.numpy()[:count]
        require(np.isfinite(forces).all() and (worlds == 0).all(), "Nonfinite or foreign-world contact force")
        result = np.zeros(2)
        for pair, force in zip(geoms, forces):
            for foot, ids in enumerate(self.feet):
                if (int(pair[0]) in ids and int(pair[1]) in self.floor
                        or int(pair[1]) in ids and int(pair[0]) in self.floor):
                    result[foot] += max(0., float(force[0]))
        return result


def rollout(env, actor, spec, item, interface, seed, output):
    import numpy as np
    import torch
    from morphmimic.constants import JOINT_NAMES, R_MAX
    from morphmimic.train.obs_history import ObsHistory
    from native_metrics import reference_poses, summarize, tracking, write_series

    require(env.num_envs == 1 and not env.cfg.auto_reset and env.step_dt == .02,
            "Native trace evaluation needs one non-resetting 50Hz plant")
    command = env.command_manager.get_term("motion")
    library = command.library
    require(command.cfg.sampling == "start" and library.has_human, "Use original eval resets and human references")
    key = "g1" if spec.name == "g1" else "q1_morph"
    name = Path(item[key]["path"]).stem
    require(library.clip_names.count(name) == 1, "Missing or ambiguous paired native clip")
    index = library.clip_names.index(name)
    start, frames = int(library.clip_start[index]), int(library.clip_len[index])
    require(frames >= 2, "Native clip has no physical transition")
    command.assign_clips(torch.tensor([0], device=env.device), torch.tensor([index], device=env.device))
    obs, _ = env.reset(env_ids=torch.tensor([0], device=env.device))
    require(int(command.global_frame[0]) == start + 1, "Native reset target must be frame1")
    reference = library.state(torch.arange(start, start + frames, device=env.device)).cpu().numpy()
    first = obs["state"][0].cpu().numpy()
    require(first.shape == (269,) and reference.shape == (frames, 269)
            and np.allclose(first[:29], reference[0, :29], atol=3e-5, rtol=0.)
            and np.allclose(first[58:100], reference[0, 58:100], atol=3e-5, rtol=0.)
            and not first[240:].any(), "Native reset did not restore frame0 and empty action memory")
    history = ObsHistory(1, 10, 93, env.device)
    history.reset(None, obs["policy"])
    goals = {key: obs[key].clone() for key in ("goal", "goal_kp")}
    sensor = ContactLoads(env, spec)
    states, poses = [first.copy()], [env.sim.data.qpos[0].cpu().numpy().copy()]
    forces, actions, rewards, end = [sensor()], [], [], None
    with torch.inference_mode():
        for frame in range(1, frames):
            require(int(command.global_frame[0]) == start + frame, "Native target frame skipped or repeated")
            action = actor(history.flat(), goals, interface)
            obs, reward, terminated, truncated, _ = env.step(action)
            require(all(torch.isfinite(value).all().item() for value in obs.values())
                    and torch.isfinite(reward).all().item(), "Nonfinite native physical episode")
            states.append(obs["state"][0].cpu().numpy().copy())
            poses.append(env.sim.data.qpos[0].cpu().numpy().copy())
            forces.append(sensor())
            actions.append(action[0].cpu().numpy().copy())
            rewards.append(float(reward[0]))
            end = episode_end(frame, frames, bool(terminated[0]), bool(truncated[0]))
            if end is not None:
                break
            history.push(obs["policy"])
            goals = {key: obs[key].clone() for key in ("goal", "goal_kp")}
    require(end is not None, "Native episode did not reach an authoritative end")
    states, poses, forces = map(np.asarray, (states, poses, forces))
    target_poses, root_address = reference_poses(env.sim.mj_model, spec, reference)
    diagnosis, series, upper = summarize(env.sim.mj_model, spec, states, reference, poses, target_poses, forces)
    row = dict(clip=name, seed=seed, robot=spec.name, interface=interface, **end,
        attempted_steps=len(states) - 1, reference_steps=frames - 1,
        reference_sha256=item[key]["sha256"], human_sha256=item["human"]["sha256"],
        trace=str(output / "rollout.npz"), evaluation_randomization=False, diagnostics=diagnosis,
        norm_return=sum(rewards) / (R_MAX * (frames - 1)), root_qpos_address=root_address,
        plant="original pinned mjlab native scene with the shared Q1 control adapter",
        repeated_seed_note="nominal deterministic evaluation; seeds are not independent randomization trials",
        metric_scope="present canonical joints and 13 tracked body positions relative to pelvis; native initial/final only")
    if end["valid_steps"]:
        row["valid_prefix_metrics"] = tracking(states, reference, actor.present.cpu().numpy(), end["valid_steps"])
    if row["complete"]:
        row["full_clip_metrics"] = row["valid_prefix_metrics"]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "rollout.npz", actual=states, reference=reference, qpos_actual=poses,
        qpos_reference=target_poses, canonical_action=np.asarray(actions), reward=np.asarray(rewards),
        joint_presence=actor.present.cpu().numpy(), **series)
    write_series(output, states, reference, series, upper, JOINT_NAMES)
    save(output / "result.json", row)
    return row


def execute(args):
    require(valid_commit(args.evaluation_commit), "Publish and independently verify the evaluation code first")
    plan, pipeline, audit, frozen = map(read, (args.plan, args.pipeline, args.audit, args.frozen))
    iteration = ready_training(plan, pipeline, identity(args.plan), audit, args.robot, identity(args.checkpoint))
    names = validate_frozen(frozen)
    require(args.output.is_absolute() and args.output.is_relative_to(ROOT / "data/supermimic_for_q1")
            and not args.output.exists(), "Use a new native evaluation root on the sole server")
    key = "g1" if args.robot == "g1" else "q1_morph"
    selected = frozen["rows"] if args.library else [
        next(row for row in frozen["rows"] if Path(row[key]["path"]).stem == name) for name in FIXED]
    sources = [item[k] for item in selected for k in (key, "human")]
    require(all(identity(item["path"]) == item for item in sources), "Native paired evaluation input changed")
    sys.path[:0] = [str(Path(plan["source"]) / "src"), str(HERE.parent / "morphmimic_q1")]
    import numpy as np
    import torch
    from bootstrap import install
    spec = install(args.robot)["robot"]
    from inference import load_policy
    actor, metadata = load_policy(args.checkpoint, spec, plan["code_commit"], "cuda:0", expected_plan=plan)
    require(metadata["iteration"] == iteration, "Loaded iteration differs from CPU audit")
    from morphmimic.eval.benchmark import make_eval_env
    import mujoco
    args.output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", started_utc=now(), cases=[], iteration=iteration,
        robot=args.robot, interface=args.interface, code_commit=args.evaluation_commit,
        plant="native_mjlab", evaluation_randomization=False,
        training_plan=identity(args.plan), checkpoint=identity(args.checkpoint), frozen_data=identity(args.frozen),
        metadata=metadata, optimizer_updates=0, training_replay_writes=0,
        training_set_only=True, unseen_morphology_zero_shot=False, user_accepted_checkpoint=None)
    env = None
    try:
        for seed in ((0,) if args.library else (0, 1, 2)):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            subset = plan["subsets"][plan["robots"].index(args.robot)]
            env = make_eval_env(subset, 1, "cuda:0", seed=seed, robot=args.robot,
                human_root=plan["human_root"], goal_anchor="reference")
            require(len(env.command_manager.get_term("motion").library.clip_names) == len(names)
                    and set(env.command_manager.get_term("motion").library.clip_names) == set(names),
                    "Native environment did not load the frozen common393")
            scene = args.output / "scene.mjb"
            if not scene.exists():
                mujoco.mj_saveModel(env.sim.mj_model, str(scene), None)
                record["scene"] = identity(scene)
            for item in selected:
                name = Path(item[key]["path"]).stem
                directory = args.output / f"{name}_s{seed}"
                row = rollout(env, actor, spec, item, args.interface, seed, directory)
                if args.render and seed == 0:
                    from native_video import render_case
                    row["video"] = render_case(scene, row, directory / "tracking.mp4")
                    save(directory / "result.json", row)
                record["cases"].append(row)
                save(args.output / "progress.json", record)
            env.close()
            env = None
        require(all(identity(item["path"]) == item for item in sources), "Native evaluation input changed during replay")
        record.update(status="PASSED", complete_cases=sum(row["complete"] for row in record["cases"]))
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
        record["ended_utc"] = now()
        save(args.output / "result.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("plan", "pipeline", "audit", "checkpoint", "frozen", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--robot", choices=("g1", "q1"), required=True)
    parser.add_argument("--interface", choices=("motion", "human"), required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--library", action="store_true")
    parser.add_argument("--render", action="store_true")
    execute(parser.parse_args())
