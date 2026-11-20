"""Build the two 1x QC videos from saved CPU trials and the six-panel training plot."""
import argparse
import json
import os
os.environ.setdefault("MUJOCO_GL","egl")
from pathlib import Path
import subprocess
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from bench import ROOT,HERE,sys,BUNDLE,scene,ET
sys.path.append(str(ROOT/"runs/h003_deploy_hybrid"))
from render_hybrid_tracks_1x import _build_track_dual_vis_xml

def font(size):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",size)

def add_footprints(xml,poses,model,comparison=None):
    tree=ET.parse(xml)
    world=tree.find("worldbody")
    for geom in list(world.findall("geom")):
        if geom.get("name","").startswith("dot_"):world.remove(geom)
    for geom in world.find("body[@name='ref_pelvis']").iter("geom"):
        geom.set("rgba","0.06 0.85 0.96 0.20")
    def mark(traj,group):
        data=mujoco.MjData(model)
        last={}
        for pose in traj[::4]:
            data.qpos[:]=pose
            mujoco.mj_kinematics(model,data)
            for side in ("left","right"):
                body=model.body(side+"_ankle_roll_link").id
                sole=data.xpos[body]+data.xmat[body].reshape(3,3)@np.array([.02,0,-.022])
                if sole[2]<.015 and (side not in last or np.linalg.norm(last[side]-sole[:2])>.055):
                    ET.SubElement(world,"geom",type="cylinder",group=str(group),pos=f"{sole[0]} {sole[1]} 0.001",size=".012 .001",rgba=".9 .55 .12 .7",contype="0",conaffinity="0")
                    last[side]=sole[:2].copy()
    mark(poses,4)
    if comparison is not None: mark(comparison,5)
    tree.write(xml)

def movie(out,clips,baseline=None):
    encoder=subprocess.Popen(["ffmpeg","-nostdin","-v","error","-n","-f","rawvideo","-pixel_format","rgb24",
               "-video_size","960x674","-framerate","25","-i","-","-an","-c:v","libx264","-threads","3",
               "-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(out)],stdin=subprocess.PIPE)
    qc=[]
    stand=json.loads((BUNDLE/"config/q1_deploy_config.json").read_text())["default_q"]
    try:
        for j,(name,file) in enumerate(clips):
            saved=np.load(file)
            dynamic=saved["qpos"]
            trace=saved["trace"]
            if "reference" in saved:
                ref=saved["reference"]
            else:
                ref=np.tile(np.r_[dynamic[0,:2],.419,1,0,0,0,stand],(len(dynamic),1))
            xml=_build_track_dual_vis_xml(out.parent/"render_scene",dynamic[:,:2],ref[:,:2],f"{out.stem}_{j}")
            comparison=np.load(baseline/file.name)["qpos"] if baseline else None
            add_footprints(xml,dynamic,scene(out.parent/"render_scene/physics"),comparison)
            model=mujoco.MjModel.from_xml_path(str(xml))
            options=mujoco.MjvOption()
            sample_ticks={0,(len(dynamic)//4)*2,((len(dynamic)-1)//2)*2}
            if comparison is None:
                handoff=np.flatnonzero((trace[:,5]>0)&(trace[:,5]<1))
                sample_ticks.update((100,150, int(handoff[0]//2*2) if len(handoff) else 200))
            def frame_camera(actual,reference):
                separation=np.linalg.norm(actual[:2]-reference[:2])
                cam.lookat[:]=[*(.55*actual[:2]+.45*reference[:2]),.33]
                cam.distance=2.1+.8*separation
            d=mujoco.MjData(model)
            cam=mujoco.MjvCamera()
            cam.type=mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance=1.65
            cam.elevation=-14
            cam.azimuth=[125,72,42][j%3]
            with mujoco.Renderer(model,height=516 if baseline else 544,width=480 if baseline else 960) as renderer:
                for i in range(0,len(dynamic),2):
                    d.qpos[:29]=dynamic[i]
                    d.qpos[29:58]=ref[i]
                    mujoco.mj_forward(model,d)
                    frame_camera(dynamic[i],ref[i])
                    options.geomgroup[4:6]=[1,0]
                    renderer.update_scene(d,camera=cam,scene_option=options)
                    canvas=Image.new("RGB",(960,674),(21,26,36))
                    canvas.paste(Image.fromarray(renderer.render()),(480 if baseline else 0,66 if baseline else 38))
                    if comparison is not None:
                        d.qpos[:29]=comparison[i]
                        mujoco.mj_forward(model,d)
                        frame_camera(comparison[i],ref[i])
                        options.geomgroup[4:6]=[0,1]
                        renderer.update_scene(d,camera=cam,scene_option=options)
                        canvas.paste(Image.fromarray(renderer.render()),(0,66))
                    pen=ImageDraw.Draw(canvas)
                    if comparison is not None:
                        pen.rectangle([0,38,479,65],fill=(45,55,67))
                        pen.text((12,41),"Pure V2b",font=font(16),fill="white")
                        pen.rectangle([480,38,959,65],fill=(13,91,92))
                        pen.text((492,41),"Recovery expert + Pure V2b",font=font(16),fill="white")
                    pen.text((14,7),name,font=font(19),fill="white")
                    pen.text((942,8),f"1.0x   |   {(.02*(i+1)):.2f} s",font=font(17),fill="white",anchor="ra")
                    z,up,tau,_,_,gate,depth=trace[i,:7]
                    pen.text((15,590),f"pelvis z  {z:.3f} m     cos tilt  {up:+.2f}     g_rec  {gate:.2f}     max torque  {tau:.1f} Nm     penetration  {1000*depth:.1f} mm",font=font(16),fill="white")
                    pen.text((15,635),"g_rec",font=font(15),fill="white")
                    pts=[(80+850*k/(len(trace)-1),658-33*trace[k,5]) for k in range(0,i+1,2)]
                    if len(pts)>1:pen.line(pts,fill=(101,222,213),width=3)
                    encoder.stdin.write(canvas.tobytes())
                    if i in sample_ticks:
                        qc.append(canvas.resize((384,270)))
    finally:
        encoder.stdin.close()
        code=encoder.wait()
    if code: raise RuntimeError(f"ffmpeg exited {code}")
    sheet=Image.new("RGB",(1152,270*((len(qc)+2)//3)))
    for k,img in enumerate(qc):sheet.paste(img,(384*(k%3),270*(k//3)))
    sheet.save(out.with_suffix(".png"))

def plot(run,out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records=[json.loads(l) for l in (run/"metrics.jsonl").read_text().splitlines()]
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout="constrained")
    for ax,(key,label) in zip(axes.flat,[("reward","Mean reward"),("height","Mean pelvis height (m)"),
            ("upright_fraction","Upright fraction of sampled steps"),("exploration","Exploration standard deviation"),
            ("loss","PPO minibatch loss"),("steps","Environment steps")]):
        ax.plot([r["update"] for r in records],[r[key] for r in records],lw=1.2)
        ax.set(xlabel="PPO update",title=label)
        ax.grid(alpha=.2)
    fig.savefig(out,dpi=150)
    plt.close(fig)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("run",type=Path)
    ap.add_argument("output",type=Path)
    ap.add_argument("--e1",type=Path)
    ap.add_argument("--relay",type=Path)
    args=ap.parse_args()
    out=args.output.resolve()
    out.mkdir(exist_ok=True,parents=True)
    plot(args.run,out/"training.png")
    if args.e1:
        trials=json.loads((args.e1/"trials.json").read_text())
        clips=[]
        for group in ("prone","supine","side"):
            t=next(t for t in trials if t["pose"]==group and t["success"])
            clips.append((group+" recovery",args.e1/f"trial_{t['trial']:02}.npz"))
        movie(out/"recovery_all_poses_1x.mp4",clips,HERE/"baseline_01")
    if args.relay:
        t=next(t for t in json.loads((args.relay/"relay.json").read_text()) if t["success"])
        movie(out/"fall_recover_then_walk_1x.mp4",[("Walk, simulated perturbation, recovery and handoff",args.relay/f"relay_{t['trial']}.npz")])

