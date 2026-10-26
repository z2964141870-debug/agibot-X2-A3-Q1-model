"""Preselect source-only motion coverage and render honest multi-clip comparisons."""

import argparse
import json
from pathlib import Path

import numpy as np


def source_metrics(row,model=None,data=None):
    from scipy.spatial.transform import Rotation
    q=np.loadtxt(row["path"],delimiter=",",ndmin=2)
    delta=np.diff(q[:,:2],axis=0)
    forward=Rotation.from_quat(q[:-1,3:7]).as_matrix()[:,:2,0]
    path=float(np.linalg.norm(delta,axis=1).sum())
    yaw=np.unwrap(np.arctan2(forward[:,1],forward[:,0]))
    result={"duration_seconds":(len(q)-1)/30,"reference_path_m":path,
            "forward_fraction":float(np.sum(delta*forward)/max(path,1e-9)),
            "lateral_fraction":float(np.sum(np.abs(delta[:,0]*forward[:,1]-delta[:,1]*forward[:,0]))/max(path,1e-9)),
            "yaw_range_degrees":float(np.ptp(yaw)*180/np.pi)}
    if model is not None and row["group"]=="gesture":
        import mujoco
        wrists=[model.body(s+"_wrist_yaw_link").id for s in ("left","right")]
        shoulders=[model.body(s+"_shoulder_roll_link").id for s in ("left","right")]
        heights=[]
        for value in q:
            data.qpos[:]=np.r_[value[:3],value[[6,3,4,5]],value[7:]]
            mujoco.mj_kinematics(model,data)
            heights.append(data.xpos[wrists,2]-data.xpos[shoulders,2])
        heights=np.asarray(heights)
        result["max_wrist_above_shoulder_m"]=float(heights.max())
        result["wrist_height_excursion_m"]=float(np.ptp(heights,axis=0).max())
    return result


def select_suite(library,runtime=None):
    rows=library["splits"]["train"]+library["splits"]["validation"]
    model=data=None
    if runtime:
        import mujoco
        model=mujoco.MjModel.from_xml_path(str(runtime/"q1_assets/scene.xml"))
        data=mujoco.MjData(model)
    metrics={r["name"]:source_metrics(r,model,data) for r in rows}
    selected=[]
    used=set()
    def add(candidates,label,count=1,priority=None):
        candidates=[r for r in candidates if r["name"] not in used]
        candidates.sort(key=lambda r:((priority(r) if priority else 0),"very_fast" in r["name"],-metrics[r["name"]]["duration_seconds"],r["name"]))
        for row in candidates[:count]:
            selected.append({"name":row["name"],"label":label,"split":row["split"],"group":row["group"],"source_metrics":metrics[row["name"]]})
            used.add(row["name"])
    # These constraints select a readable demonstration only. They never filter
    # the training library and never inspect learned-policy results.
    ordinary=[r for r in rows if r["group"]=="walking" and r["name"].startswith("walk_ff_")
              and metrics[r["name"]]["forward_fraction"]>.6 and metrics[r["name"]]["lateral_fraction"]<.4
              and metrics[r["name"]]["yaw_range_degrees"]<40]
    if not ordinary:
        raise ValueError("No ordinary forward-walk source for the requested demonstration")
    add(ordinary,"Forward walking",1)
    other_actor=[r for r in ordinary if r["actor_uid"]!=next(v["actor_uid"] for v in rows if v["name"]==selected[0]["name"])]
    add(other_actor or ordinary,"Forward walking",1)
    gestures=[r for r in rows if r["group"]=="gesture"]
    raised=([r for r in gestures if metrics[r["name"]].get("max_wrist_above_shoulder_m",-1)>.02 and metrics[r["name"]].get("wrist_height_excursion_m",0)>.05]
            if runtime else [r for r in gestures if any(s in r["name"].lower() for s in ("wave","raise","hand","cheer","stretch","scream","ear","overhead"))])
    if not raised:
        raise ValueError("No raised-hand reference in the demonstration source library")
    add(raised,"Upper-body / raised hands",1,lambda r:0 if any(v in r["name"] for v in ("triumph","wave","raise","stretch","hands_up")) else 1)
    add([r for r in gestures if metrics[r["name"]]["reference_path_m"]<.3],"Stationary gesture",1)
    for group,label in (("turning","Turning"),("jogging","Jogging"),("dancing","Dance"),("jumping","Jump")):
        add([r for r in rows if r["group"]==group],label,priority=lambda r:0 if r["name"].startswith(("jog_ff_","jump_ff_","idle_turn_","step_rotate_idle_","dance_basic_")) else 1)
    duration=sum(r["source_metrics"]["duration_seconds"] for r in selected)
    if duration<30:
        for row in sorted(rows,key=lambda r:(-metrics[r["name"]]["duration_seconds"],r["name"])):
            if row["name"] not in used:
                add([row],row["group"].title())
                duration+=metrics[row["name"]]["duration_seconds"]
                if duration>=30:
                    break
    if duration<30 or {r["group"] for r in selected}!={r["group"] for r in rows}:
        raise ValueError("Insufficient source duration or motion-type coverage")
    return {"schema":1,"selection":"Source-only, fixed before training; no policy-outcome selection. Every segment is a separate reference clip, not one continuous rollout.",
            "clips":selected,"reference_duration_seconds":duration,"playback_speed":1,"minimum_video_seconds":30}


def render_suite(runtime,trace,env,rows,suite,output,label):
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
    from worker import identity

    motion=env.supermimic_mdp.motion
    model=mujoco.MjModel.from_xml_path(str(runtime/"q1_assets/scene.xml"))
    model.vis.quality.offsamples=0
    physical,reference=mujoco.MjData(model),mujoco.MjData(model)
    camera=mujoco.MjvCamera()
    camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance,camera.azimuth,camera.elevation=2.0,135,-15
    renderer=mujoco.Renderer(model,height=480,width=640)
    options=mujoco.MjvOption()
    font_path=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font=ImageFont.truetype(str(font_path),18) if font_path.exists() else ImageFont.load_default()
    index_by_name={r["name"]:i for i,r in enumerate(rows)}
    segments=[]
    written=0
    try:
        with imageio.get_writer(output,fps=25,codec="libx264",quality=7,macro_block_size=16) as writer:
            for chosen in suite["clips"]:
                index=index_by_name[chosen["name"]]
                start,length=int(motion.clip_start[index]),int(motion.clip_len[index])
                count=int(trace["steps"][index])
                states=np.concatenate((trace["initial"][index:index+1],trace["states"][:count,index]),axis=0)
                frame_ids=torch.arange(start,start+length,2,device=env.device)
                refs=motion.ref_state(frame_ids)
                ref_q=torch.cat((refs[0],mat_to_quat(refs[1]),refs[4]),-1).cpu().numpy()
                first_video_frame=written
                for n,frame in enumerate(range(0,length,2)):
                    actual=states[min(frame,len(states)-1)]
                    physical.qpos[:]=np.r_[actual[:7],actual[13:35]]
                    reference.qpos[:]=ref_q[n]
                    images=[]
                    for data in (physical,reference):
                        mujoco.mj_kinematics(model,data)
                        mujoco.mj_comPos(model,data)
                        mujoco.mj_camlight(model,data)
                        camera.lookat[:]=[data.qpos[0],data.qpos[1],.45]
                        renderer.update_scene(data,camera=camera,scene_option=options)
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=False
                        renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False
                        images.append(renderer.render().copy())
                    canvas=Image.fromarray(np.concatenate(images,axis=1))
                    draw=ImageDraw.Draw(canvas)
                    draw.rectangle((0,0,1280,74),fill=(18,22,24))
                    draw.text((12,4),f"{label}: physical policy",font=font,fill="white")
                    draw.text((652,4),"Kinematic reference",font=font,fill="white")
                    draw.text((12,27),f"{chosen['label']} | {chosen['split']} | segment {len(segments)+1}/{len(suite['clips'])}",font=font,fill="white")
                    draw.text((12,50),chosen["name"],font=font,fill="white")
                    if frame>count:
                        draw.rectangle((0,440,640,480),fill=(115,24,24))
                        draw.text((12,449),"TERMINATED: final physical pose held",font=font,fill="white")
                    writer.append_data(np.asarray(canvas))
                    written+=1
                segments.append({**chosen,"first_video_frame":first_video_frame,"last_video_frame":written-1,
                                 "physical_steps":count,"failed":bool(trace["failed"][index]),"reference_frames_50hz":length})
    finally:
        renderer.close()
    if written/25<30:
        raise RuntimeError("Comparison movie did not reach 30 seconds")
    return {"video":identity(output),"fps":25,"frames":written,"duration_seconds":written/25,"speed":1,
            "segments":segments,"left":"Real policy physics; explicit hold after termination","right":"Kinematic Q1 reference",
            "rendering":"Saved qpos FK only, official Q1 visual meshes; MSAA/shadows/reflections disabled for headless software EGL. No policy/physics changes.",
            "continuity":"Independent clips concatenated at their original speed; no claim of one continuous 30-second rollout"}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--runtime",type=Path)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    suite=select_suite(json.loads(args.library.read_text()),args.runtime)
    args.output.write_text(json.dumps(suite,indent=2)+"\n")
    print(json.dumps(suite),flush=True)


if __name__=="__main__":
    main()
