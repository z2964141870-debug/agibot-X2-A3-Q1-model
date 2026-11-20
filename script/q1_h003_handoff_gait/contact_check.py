"""Sensitivity of alternating-touchdown interval CV to contact detection on paired successful trials."""
from pathlib import Path
import json
import numpy as np

def cadence(loads, first, threshold, dwell):
    state=loads[0]>=threshold
    count=np.zeros(2,int)
    events=[]
    for tick,load in enumerate(loads[:first+150]):
        wanted=load>=threshold
        count=np.where(wanted!=state,count+1,0)
        for side in np.flatnonzero(count>=dwell):
            state[side]=wanted[side]
            count[side]=0
            if state[side] and tick>=first:
                events.append((tick,side))
    dt=[(b[0]-a[0])*.02 for a,b in zip(events,events[1:]) if a[1]!=b[1]]
    if len(dt)<3: return None
    return float(np.std(dt)/np.mean(dt))

def evaluate(root, results):
    output={}
    for suite in ("relay01","blind01"):
        rows=results["walk"][suite]
        cases=[(a["trial"],a["release"],b["release"]) for a,b in zip(rows["original"],rows["ramp15"]) if a["success"] and b["success"]]
        summary=[]
        for threshold in (5,10,20):
            for dwell in (2,4):
                pairs=[]
                for i,t0,t1 in cases:
                    values=[]
                    for variant,start in (("original",t0),("ramp15",t1)):
                        with np.load(root/suite/variant/f"relay_{i}.npz") as f:
                            values.append(cadence(f["loads"],start,threshold,dwell))
                    if None not in values:pairs.append(values)
                means=np.mean(pairs,axis=0)
                summary.append(dict(threshold_N=threshold,dwell_ms=dwell*20,pairs=len(pairs),
                                    original_cv=float(means[0]),candidate_cv=float(means[1])))
        output[suite]=summary
    return output

if __name__=="__main__":
    root=Path.cwd()
    result=evaluate(root,json.loads((root/"results_full.json").read_text()))
    (root/"cadence.json").write_text(json.dumps(result,indent=2))

