"""Initial/final physical Q1 evaluation on the complete frozen SOMA library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from worker import identity, prepare, save, now


@torch.inference_mode()
def rollout(env, checkpoint):
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state
    from unisupermimic.rl.contracts.supermimic import SUPERMIMIC_STATE_KEY, SUPERMIMIC_TRANSITION_KEY
    from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy

    policy = load_policy(env, checkpoint)
    mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
    _, info = env.reset_to_frames(motion.clip_start.clone())
    current = info[SUPERMIMIC_STATE_KEY]
    initial = current["s"].cpu().numpy()
    history = None
    done = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    failed = done.clone()
    returns = torch.zeros(env.num_envs, device=env.device)
    steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    records, frames, alive_rows = [], [], []
    for _ in range(int(motion.clip_len.max())):
        obs = mdp.obs(unflatten_state(current["s"]), current["a_prev"], current["xi"], current["f"])
        history = obs.unsqueeze(1).repeat(1, checkpoint["args"]["hist"], 1) if history is None else torch.cat((history[:, 1:], obs.unsqueeze(1)), 1)
        action = policy(history, mdp.goal(current["f"], checkpoint["args"]["goal_frames"]))
        state = env.step(action)
        transition = state.info[SUPERMIMIC_TRANSITION_KEY]
        alive = ~done
        records.append(transition["s1"].cpu().numpy())
        frames.append((transition["f"]+1).cpu().numpy())
        alive_rows.append(alive.cpu().numpy())
        returns += torch.where(alive, transition["r"], 0)
        steps += alive
        failed |= alive & transition["term"]
        done |= alive & transition["done"]
        current = state.info[SUPERMIMIC_STATE_KEY]
        if bool(done.all()):
            break
    if not bool(done.all()):
        raise RuntimeError("Full-library evaluation exceeded the longest reference")
    return {"initial": initial, "states": np.stack(records), "frames": np.stack(frames), "alive": np.stack(alive_rows), "steps": steps.cpu().numpy(), "failed": failed.cpu().numpy(), "returns": returns.cpu().numpy()}


def analyze(env, trace, rows):
    import mujoco
    from tools.x2_geometry import CollisionGeometry
    from unisupermimic.rl.algos.supermimic.geometry import unflatten_state

    mdp, motion = env.supermimic_mdp, env.supermimic_mdp.motion
    model = mujoco.MjModel.from_xml_path(str(env._supermimic_command.cfg.mjcf_path))
    data = mujoco.MjData(model)
    feet = [CollisionGeometry(model, (f"{side}_ankle_roll_link",)) for side in ("left", "right")]
    results = []
    for clip, record in enumerate(rows):
        count = int(trace["steps"][clip])
        actual = trace["states"][:count, clip]
        frame_ids = torch.tensor(trace["frames"][:count, clip], device=env.device)
        ref = motion.ref_state(frame_ids)
        reference_q = ref[4].cpu().numpy()
        error_q = actual[:, 13:35] - reference_q
        root_ref = ref[0].cpu().numpy()
        start = trace["initial"][clip, :3]
        full_start = motion.root_p[motion.clip_start[clip]].cpu().numpy()
        full_end = motion.root_p[motion.clip_end[clip]].cpu().numpy()
        delta = actual[-1, :2] - start[:2]
        ref_delta = full_end[:2] - full_start[:2]
        direction = ref_delta / max(np.linalg.norm(ref_delta), 1e-12)
        heights = []
        for value in actual:
            data.qpos[:] = np.r_[value[:7], value[13:35]]
            mujoco.mj_kinematics(model, data)
            heights.append([foot.minimum(data) for foot in feet])
        near = np.asarray(heights) < .015
        tensor = torch.tensor(actual, device=env.device)
        physical_bodies = mdp.kin.forward(*unflatten_state(tensor))[0].cpu().numpy()
        reference_bodies = motion.body_p[frame_ids].cpu().numpy()
        wrists = [mdp.kin.tracked_idx.index(mdp.kin.body_names.index(f"{side}_wrist_yaw_link")) for side in ("left", "right")]
        actual_wrist = physical_bodies[:, wrists]
        reference_wrist = reference_bodies[:, wrists]
        heights_wrist = actual_wrist[..., 2] - actual[:, 2, None]
        target_wrist = reference_wrist[..., 2] - root_ref[:, 2, None]
        results.append({"name": record["name"], "split": record["split"], "group": record["group"], "steps": count, "reference_steps": int(motion.clip_len[clip])-1, "failed": bool(trace["failed"][clip]), "completion_fraction": count/(int(motion.clip_len[clip])-1), "normalized_full_clip_return": float(trace["returns"][clip]/((int(motion.clip_len[clip])-1)*mdp.max_step_reward)), "net_xy_m": float(np.linalg.norm(delta)), "along_target_m": float(delta @ direction), "reference_net_xy_m": float(np.linalg.norm(ref_delta)), "path_xy_m": float(np.linalg.norm(np.diff(np.vstack((start[:2],actual[:, :2])), axis=0),axis=1).sum()), "root_xy_rmse_m": float(np.sqrt(np.mean((actual[:, :2]-root_ref[:, :2])**2))), "leg_rmse_rad": float(np.sqrt(np.mean(error_q[:, :12]**2))), "upper_limb_rmse_rad": float(np.sqrt(np.mean(error_q[:, 14:]**2))), "joint_rmse_rad": float(np.sqrt(np.mean(error_q**2))), "wrist_rmse_m": float(np.sqrt(np.mean((actual_wrist-reference_wrist)**2))), "max_wrist_height_above_root_m": heights_wrist.max(axis=0).tolist(), "reference_prefix_max_wrist_height_m": target_wrist.max(axis=0).tolist(), "wrist_peak_frame": heights_wrist.argmax(axis=0).tolist(), "reference_prefix_wrist_peak_frame": target_wrist.argmax(axis=0).tolist(), "geometric_double_near_ground_fraction": float(near.all(axis=1).mean()), "geometric_left_only_fraction": float((near[:,0]&~near[:,1]).mean()), "geometric_right_only_fraction": float((near[:,1]&~near[:,0]).mean()), "support_measurement": "Exact collision-geometry lowest point below 15mm; geometry diagnostic, not measured contact force", "actual_evaluated_prefix_seconds": count*.02})
    summary = {}
    for split in ("train", "validation"):
        selected = [r for r in results if r["split"] == split]
        summary[split] = {"clips": len(selected), "failed": sum(r["failed"] for r in selected), "mean_completion_fraction": float(np.mean([r["completion_fraction"] for r in selected])), "mean_normalized_full_clip_return": float(np.mean([r["normalized_full_clip_return"] for r in selected])), "mean_root_xy_rmse_m": float(np.mean([r["root_xy_rmse_m"] for r in selected])), "mean_upper_limb_rmse_rad": float(np.mean([r["upper_limb_rmse_rad"] for r in selected]))}
    return {"summary": summary, "per_clip": results, "limits": "Q1 simulation, own 100/15 source selection. Geometric near-ground is not a force-contact measurement. Prefix errors must be compared over common frames when termination differs."}


def render_example(runtime, trace, env, index, output):
    import imageio.v2 as imageio
    import mujoco

    motion = env.supermimic_mdp.motion
    model = mujoco.MjModel.from_xml_path(str(runtime / "q1_assets/scene.xml"))
    physical, reference = mujoco.MjData(model), mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = 2.4, 135, -12
    count = int(trace["steps"][index])
    clip_start = int(motion.clip_start[index])
    clip_length = int(motion.clip_len[index])
    source_states = [trace["initial"][index], *trace["states"][:count, index]]
    renderer = mujoco.Renderer(model, height=480, width=640)
    options = mujoco.MjvOption()
    with imageio.get_writer(output, fps=25, codec="libx264", quality=7, macro_block_size=16) as writer:
        for frame in range(0, clip_length, 2):
            actual = source_states[min(frame, len(source_states)-1)]
            physical.qpos[:] = np.r_[actual[:7], actual[13:35]]
            ref = motion.ref_state(torch.tensor([clip_start+frame],device=env.device))
            from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
            reference.qpos[:] = torch.cat((ref[0],mat_to_quat(ref[1]),ref[4]),-1)[0].cpu().numpy()
            images = []
            for data in (physical, reference):
                mujoco.mj_forward(model, data)
                camera.lookat[:] = [data.qpos[0],data.qpos[1],.5]
                renderer.update_scene(data,camera=camera,scene_option=options)
                images.append(renderer.render().copy())
            writer.append_data(np.concatenate(images,axis=1))
    renderer.close()
    return {"clip": motion.names[index], "left": "physical policy rollout (last state held after termination)", "right": "full kinematic reference", "physical_steps": count, "reference_frames": clip_length, "video": identity(output)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    library=json.loads(args.library.read_text())
    rows=library["splits"]["train"]+library["splits"]["validation"]
    combined=args.output/"evaluation_library.json"
    save(combined,{"splits":{"train":rows},"original_library":identity(args.library)})
    env=None
    results={"started_utc":now(),"library":identity(args.library),"evaluations":{}}
    try:
        options=SimpleNamespace(runtime=args.runtime,library=combined,output=args.output)
        _,env,_=prepare(options,num_envs=len(rows),eval_mode=True)
        for label in ("initial","final"):
            checkpoint_path=args.run/(label+".pt")
            checkpoint=torch.load(checkpoint_path,map_location="cuda:0",weights_only=True)
            trace=rollout(env,checkpoint)
            np.savez_compressed(args.output/(label+"_trajectory.npz"),**trace)
            result=analyze(env,trace,rows)
            walking=[(r["reference_net_xy_m"],i) for i,r in enumerate(result["per_clip"]) if r["group"]=="walking" and r["split"]=="train"]
            _,index=sorted(walking)[len(walking)//2]
            result["video"]=render_example(args.runtime,trace,env,index,args.output/(label+".mp4"))
            result["checkpoint"]=identity(checkpoint_path)
            save(args.output/(label+".json"),result)
            results["evaluations"][label]=result["summary"]
            print(json.dumps({"label":label,**result["summary"]}),flush=True)
            del checkpoint,trace
        results.update(status="INITIAL_FINAL_FULL_LIBRARY_PHYSICAL_EVALUATION_COMPLETED",completed_utc=now())
        save(args.output/"receipt.json",results)
    finally:
        if env is not None:
            env.close()
            env.backend.close()


if __name__=="__main__":
    main()
