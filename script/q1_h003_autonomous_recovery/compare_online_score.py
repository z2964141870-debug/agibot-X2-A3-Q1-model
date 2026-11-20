import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent/"comparison_online"
def get(seed,group,arm):
    path=root/arm/"test"/str(seed)/group
    rs=json.loads((path/"results.json").read_text())
    results=[]
    for r in rs:
        phase=r["phases"][0]
        with np.load(path/f"trial_{r['trial']:02d}.npz") as d:
            tr=d["trace"]
            pose=d["qpos"]
            t=next(iter(r["releases"]),None)
            result=dict(fell=phase["fell"],released=t is not None,steady=False,post=phase["post_1s"],
                        t=None,roll=None)
            if t is not None:
                result["t"]=(t-(r["kicks"] or [0])[0])*.02
                result["steady"]=bool(t>=25 and tr[t-25:t+1,5].all())
                if t+50<=len(tr):
                    w,x,y,z=pose[t:t+50,3:7].T
                    result["roll"]=float(np.max(np.abs(np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))))))
            results.append(result)
    return results
out={}
for group in ("static","push"):
    A=[];B=[]
    for seed in (251233,251257):
        old=get(seed,group,"legacy")
        new=get(seed,group,"exclusive")
        for i,(a,b) in enumerate(zip(old,new)):
            pa,pb=[root/arm/"test"/str(seed)/group/f"trial_{i:02d}.npz" for arm in ("legacy","exclusive")]
            with np.load(pa) as x,np.load(pb) as y:
                cap=np.flatnonzero(x["trace"][:,4]>0)
                n=cap[0] if len(cap) else len(x["trace"])
                np.testing.assert_array_equal(x["qpos"][:n],y["qpos"][:n])
        A+=old;B+=new
    sels={"legacy":A,"exclusive":B}
    summary={arm:{k:sum(r[k] for r in rows) for k in ("fell","released","steady","post")} for arm,rows in sels.items()}
    for arm,rows in sels.items():
        summary[arm]["release_time_s"]=float(np.median([r["t"] for r in rows if r["t"] is not None]))
    pairs=[(x,y) for x,y in zip(A,B) if x["released"] and y["released"]]
    delta=float(np.median([y["t"]-x["t"] for x,y in pairs]))
    out[group]=dict(n=len(A),summary=summary,paired_release_delay_s=delta,
               common_released=len(pairs),paired_steady_gain=sum(y["steady"] and not x["steady"] for x,y in zip(A,B)),
               paired_steady_loss=sum(x["steady"] and not y["steady"] for x,y in zip(A,B)),
               rows=sels)
(root/"matched_results.json").write_text(json.dumps(out,indent=2))
for k,v in out.items(): print(k,{s:x for s,x in v.items() if s!="rows"})
