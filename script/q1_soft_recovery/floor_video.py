"""Render three complete saved development trials with the ownership timeline."""
import argparse
import json
from pathlib import Path
import subprocess

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render(run, path):
    rows=json.loads((run/"results.json").read_text())
    trials=[]
    for pose in ("prone","supine","side"):
        indices=[i for i,r in enumerate(rows) if r["pose"]==pose]
        i=next((i for i in indices if rows[i]["success"]),indices[0])
        with np.load(run/f"trial_{i:02d}.npz") as d:
            trials.append((rows[i],np.vstack((d["initial"],d["qpos"])),d["trace"].copy()))
    m=mj.MjModel.from_xml_path(str(run/"scene/fall_hard.xml"))
    for marker in ("target_pad","target_ring"):m.geom(marker).rgba[3]=0
    data=mj.MjData(m)
    camera=mj.MjvCamera();camera.distance=1.8;camera.azimuth=62;camera.elevation=-14
    font=ImageFont.truetype("DejaVuSans.ttf",16)
    width=1020;height=300
    ticks=max(len(poses) for _,poses,_ in trials)
    cmd=["ffmpeg","-n","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","1020x300","-r","25","-i","-","-c:v","libx264","-threads","2","-pix_fmt","yuv420p","-crf","20","-movflags","+faststart",str(path)]
    with mj.Renderer(m,height=214,width=340) as renderer,subprocess.Popen(cmd,stdin=subprocess.PIPE) as proc:
        for tick in range(0,ticks,2):
            image=Image.new("RGB",(width,height),"#19232c")
            pen=ImageDraw.Draw(image)
            pen.text((8,4),f"Q1 floor recovery | Simulation | 1x | t = {tick*.02:.2f} s",font=font,fill="white")
            for j,(r,poses,trace) in enumerate(trials):
                index=min(tick,len(poses)-1);data.qpos[:]=poses[index];mj.mj_forward(m,data)
                camera.lookat[:]=[data.qpos[0],data.qpos[1],.24]
                renderer.update_scene(data,camera=camera)
                image.paste(Image.fromarray(renderer.render()),(340*j,54))
                label="passed" if r["success"] else "failed"
                pen.text((340*j+8,30),f'{r["pose"]}  trial {r["trial"]}  |  {label}',font=font,fill="white")
                owner="recovery" if trace[max(0,index-1),4] else "teleoperation"
                ending="  |  end, held frame" if tick >= len(poses) else ""
                pen.text((340*j+8,273),owner+ending,font=font,fill="white")
            proc.stdin.write(np.asarray(image).tobytes())
        proc.stdin.close()
        if proc.wait():raise RuntimeError("ffmpeg failed")
    path.with_suffix(".json").write_text(json.dumps(dict(trials=[r["trial"] for r,_,_ in trials],fps=25,
        control_ms=20,ticks=ticks,frames=(ticks+1)//2,source=str(run.resolve())),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("run",type=Path);p.add_argument("output",type=Path);a=p.parse_args();render(a.run,a.output)
