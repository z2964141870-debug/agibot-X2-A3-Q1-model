"""Native replay with static terrain recognition outside frozen training inputs.

The rollout matches the pinned native evaluator except for its contact reader.
The original evaluator is a frozen training input and cannot be edited in place.
"""

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "q1_five_stage"), str(HERE.parent / "primebot_q1")]

from common import require, save
from native_review import ContactLoads, episode_end


def static_floor(types, bodies, welds, positions, rotations, plane_type):
    import numpy as np
    floors = {i for i, kind in enumerate(types) if int(kind) == int(plane_type)
              and int(welds[int(bodies[i])]) == 0}
    require(len(floors) == 1, "Require one plane welded to the static world")
    floor = next(iter(floors))
    require(np.allclose(positions[floor][2], 0.)
            and np.allclose(np.asarray(rotations[floor]).reshape(3, 3)[:, 2], [0., 0., 1.]),
            "Require a horizontal upward floor at world z0")
    return floors


class StaticTerrainContactLoads(ContactLoads):
    def __init__(self, env, spec):
        import mujoco
        import numpy as np
        import warp as wp
        from native_metrics import foot_shapes
        self.env, self.wp = env, wp
        self.feet = [{geom for geom, *_ in shapes} for shapes in foot_shapes(env.sim.mj_model, spec)]
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        self.floor = static_floor(model.geom_type, model.geom_bodyid, model.body_weldid,
            data.geom_xpos, data.geom_xmat, mujoco.mjtGeom.mjGEOM_PLANE)
        capacity = env.sim.wp_data.contact.geom.shape[0]
        self.indices = wp.array(np.arange(capacity, dtype=np.int32), dtype=wp.int32, device=env.device)
        self.forces = wp.zeros(capacity, dtype=wp.spatial_vector, device=env.device)


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
    sensor = StaticTerrainContactLoads(env, spec)
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
