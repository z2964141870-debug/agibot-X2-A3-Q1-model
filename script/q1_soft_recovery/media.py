"""Render saved states at original speed and plot actual PPO records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from common import HERE,mj


def curves(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(exist_ok=True)
    keys=[("reward","Sampled reward"),("quiet_fraction","Sampled quiet fraction"),("peak_qd","Maximum joint speed (rad/s)"),
          ("ground_mm","Maximum ground penetration (mm)"),("self_mm","Maximum arm-leg penetration (mm)"),("mean_z","Mean pelvis height (m)")]
    paths=[HERE/name/"metrics.jsonl" for name in ("teacher_stand_01","teacher_stand_02","teacher_balanced_01","teacher_balanced_02")]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout="constrained");table=[];sources={}
    for path in paths:
        if not path.exists():continue
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        x=np.array([r["steps"]/1e6 for r in rows])
        for row in rows:table.append(dict(run=path.parent.name,**row))
        for ax,(key,title) in zip(axes.flat,keys):
            y=np.array([r[key] for r in rows]);line=ax.plot(x,y,lw=.55,alpha=.6,label=path.parent.name)[0]
            if len(y)>=500:ax.plot(x[499:],np.convolve(y,np.ones(500)/500,mode="valid"),lw=2,color=line.get_color())
            ax.set_title(title);ax.set_xlabel("Million 20 ms control transitions");ax.grid(alpha=.2)
    for ax,value in ((axes[0,2],12),(axes[1,0],10),(axes[1,1],2)):ax.axhline(value,color="black",ls="--",lw=1)
    axes[0,0].legend(fontsize=8);fig.suptitle("Q1 development PPO | Different curricula | Thin: raw, thick: 500-update mean\nTraining statistics are not recovery acceptance")
    fig.savefig(out/"Q1_PPO_training.png",dpi=160);fig.savefig(out/"Q1_PPO_training.pdf")
    with (out/"Q1_PPO_training.csv").open("w") as fp:
        writer=csv.DictWriter(fp,fieldnames=sorted(set().union(*(r.keys() for r in table))));writer.writeheader();writer.writerows(table)
    (out/"curve_sources.json").write_text(json.dumps(sources,indent=2));plt.close(fig)


def video(out):
    from PIL import Image,ImageDraw,ImageFont
    out.mkdir(exist_ok=True)
    base=HERE/"stage_balanced/baseline_media";new=HERE/"stage_balanced/step480"
    b=np.load(base/"traces.npz");n=np.load(new/"traces.npz")
    np.testing.assert_array_equal(b["initial"],n["initial"])
    original=np.load(HERE/"stage_balanced/baseline/traces.npz")
    np.testing.assert_array_equal(b["trace"],original["trace"])
    records=json.loads((new/"results.json").read_text());meta=np.load(HERE/"prepared_balanced_01/starts.npz")["valid_meta"]
    high=next(i for i,r in enumerate(records) if r["final_calm_second"])
    low=int(np.argmin(abs(meta[:,1]-90)))
    panels=[(b,high,"Reference + feedback",records[high]["phase"]),(n,high,"PPO checkpoint 480",records[high]["phase"]),
            (b,low,"Reference + feedback",records[low]["phase"]),(n,low,"PPO checkpoint 480",records[low]["phase"])]
    model=mj.MjModel.from_xml_path(str(base/"scene/fall_hard.xml"));d=mj.MjData(model)
    for marker in ("target_pad","target_ring"):model.geom(marker).rgba[3]=0
    camera=mj.MjvCamera();camera.distance=1.65;camera.azimuth=63.;camera.elevation=-12.
    font=ImageFont.truetype("DejaVuSans.ttf",17);small=ImageFont.truetype("DejaVuSans.ttf",13)
    command=["ffmpeg","-n","-v","error","-f","rawvideo","-pix_fmt","rgb24","-s","896x620","-r","25","-i","-",
             "-c:v","libx264","-threads","2","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(out/"Q1_stage_balance_1x.mp4")]
    sheets=[]
    with mj.Renderer(model,height=225,width=448) as renderer,subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
        for tick in range(0,651,2):
            canvas=Image.new("RGB",(896,620),"#17222b");pen=ImageDraw.Draw(canvas)
            pen.text((12,9),"Q1 stage balance | 1x speed | Matched starts | NOT full floor recovery",font=font,fill="white")
            for panel,(data,i,label,phase) in enumerate(panels):
                x=(panel%2)*448;y=42+(panel//2)*283
                pen.text((x+10,y),f"{label} | phase {phase}/160",font=font,fill="white")
                d.qpos[:]=data["initial"][i] if tick==0 else data["state"][tick-1,i,:29]
                mj.mj_forward(model,d);camera.lookat[:]=[d.qpos[0],d.qpos[1],.23]
                renderer.update_scene(d,camera=camera);canvas.paste(Image.fromarray(renderer.render()),(x,y+25))
                met=data["trace"][max(0,tick-1),i]
                pen.text((x+10,y+253),f"t={tick*.02:.2f}s  z={met[0]:.2f}m  up={met[1]:.2f}  body-ground={int(met[3])}",font=small,fill="#dae5ed")
            proc.stdin.write(np.asarray(canvas).tobytes())
            if tick in (0,200,400,600):sheets.append(canvas.resize((448,310)))
        proc.stdin.close();code=proc.wait()
        if code:raise RuntimeError(f"ffmpeg failed {code}")
    sheet=Image.new("RGB",(896,620));
    for i,frame in enumerate(sheets):sheet.paste(frame,((i%2)*448,(i//2)*310))
    sheet.save(out/"Q1_stage_balance_contact_sheet.jpg")
    (out/"video_manifest.json").write_text(json.dumps(dict(fps=25,frames=326,speed=1.,control_s=.02,trial_indices=[high,low],
        source_traces={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (base/"traces.npz",new/"traces.npz")},
        output_sha256=hashlib.sha256((out/"Q1_stage_balance_1x.mp4").read_bytes()).hexdigest()),indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["curves","video"]);p.add_argument("out",type=Path);a=p.parse_args()
    (curves if a.action=="curves" else video)(a.out)
