"""Three-way normal-speed comparison around the first full return to V2b."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import subprocess
from PIL import Image, ImageDraw
from delivery import (np,mujoco,font,scene,add_footprints,_build_track_dual_vis_xml)
from bench import quat_wxyz_to_mat

def release(d):
    a=int(d["fork_tick"])
    return a+int(np.flatnonzero(d["trace"][a:,5]==0)[0])

def render(root,out):
    out.mkdir(exist_ok=False,parents=True)
    prefix = out/"gait_comparison_1x"
    encoder=subprocess.Popen(["ffmpeg","-nostdin","-v","error","-n","-f","rawvideo","-pix_fmt","rgb24",
        "-s","1200x492","-r","25","-i","-","-c:v","libx264","-threads","2","-crf","19",
        "-pix_fmt","yuv420p","-movflags","+faststart",str(prefix.with_suffix(".mp4"))],stdin=subprocess.PIPE)
    names=("V2b uninterrupted","Published recovery","Faster reference restart")
    thumbs=[]
    physics=scene((out/"physics").resolve())
    try:
        for i in (0,2):
            with ExitStack() as stack:
                physical=[]
                for src in (root/"nominal01/original"/f"relay_{0 if i==0 else 1}.npz",
                            root/"relay01/original"/f"relay_{i}.npz",root/"relay01/ramp15"/f"relay_{i}.npz"):
                    with np.load(src) as d:physical.append({k:d[k] for k in d.files})
                start=release(physical[1])
                align=[int(np.flatnonzero(physical[0]["trace"][:,8]==physical[1]["trace"][start,8])[0]),
                       start,release(physical[2])]
                views=[]
                for j,(f,k) in enumerate(zip(physical,align)):
                    poses=f["qpos"][k-20:k+150]
                    ref=f["reference"][k-20:k+150]
                    xml=_build_track_dual_vis_xml((out/"visual").resolve(),poses[:,:2],ref[:,:2],f"case{i}_{j}")
                    add_footprints(xml,poses,physics)
                    model=mujoco.MjModel.from_xml_path(str(xml))
                    renderer=stack.enter_context(mujoco.Renderer(model,height=364,width=400))
                    cam=mujoco.MjvCamera()
                    R=quat_wxyz_to_mat(poses[20,3:7])
                    cam.azimuth=38+np.rad2deg(np.arctan2(R[1,0],R[0,0]))
                    cam.elevation=-12
                    cam.distance=1.96
                    views.append((renderer,mujoco.MjData(model),model,cam))
                for off in range(-20,150,2):
                    canvas=Image.new("RGB",(1200,492),(21,27,34))
                    pen=ImageDraw.Draw(canvas)
                    route="Straight" if i==0 else "Figure eight"
                    pen.text((12,7),f"{route}  |  first steps after recovery",font=font(20),fill="white")
                    pen.text((1187,8),f"1.0x    t {off*.02:+.2f} s",anchor="ra",font=font(19),fill="white")
                    for j,(f,k,(renderer,d,m,cam)) in enumerate(zip(physical,align,views)):
                        n=k+off
                        pose=f["qpos"][n]
                        d.qpos[:29]=pose
                        d.qpos[29:58]=f["reference"][n]
                        mujoco.mj_forward(m,d)
                        cam.lookat[:]=[*pose[:2],.36]
                        renderer.update_scene(d,camera=cam)
                        base=j*400
                        canvas.paste(Image.fromarray(renderer.render()),(base,70))
                        pen.rectangle((base,39,base+399,69),fill=[(57,67,79),(73,65,105),(24,92,92)][j])
                        pen.text((base+10,43),names[j],font=font(17),fill="white")
                        R=quat_wxyz_to_mat(pose[3:7])
                        heading=np.arctan2(R[1,0],R[0,0])
                        v=(pose[:2]-f["qpos"][n-1,:2])*50
                        speed=float(v@np.array([np.cos(heading),np.sin(heading)]))
                        roll=np.rad2deg(np.arctan2(R[2,1],R[2,2]))
                        pen.text((base+10,439),f"Forward {speed:+.2f} m/s    Roll {roll:+.1f} deg",font=font(15),fill="white")
                        loads=f["loads"][n]
                        pen.text((base+10,463),f"Foot load L/R   {loads[0]:.0f} / {loads[1]:.0f} N",font=font(15),fill="#b2cfdb")
                    encoder.stdin.write(canvas.tobytes())
                    if off in (0,50,140):thumbs.append(canvas.resize((660,271)))
    finally:
        encoder.stdin.close()
        rc=encoder.wait()
    if rc: raise RuntimeError(rc)
    sheet=Image.new("RGB",(1320,813))
    for j,image in enumerate(thumbs):sheet.paste(image,(660*(j//3),271*(j%3)))
    sheet.save(prefix.with_suffix(".png"))
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    render(Path.cwd(),p.parse_args().out)

