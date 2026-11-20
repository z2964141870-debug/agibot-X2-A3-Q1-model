"""Gait measurements aligned to the first full handback; include unsuccessful trials."""
import argparse
import json
from pathlib import Path
import numpy as np
from bench import quat_wxyz_to_mat, score

def touchdown_sequence(loads):
    active = loads[0] >= 5.
    pending = np.zeros(2, int)
    events = []
    for k, pair in enumerate(loads):
        wanted = pair >= 5.
        pending = np.where(wanted != active, pending+1, 0)
        for side in np.flatnonzero(pending >= 2):
            active[side] = wanted[side]
            pending[side] = 0
            if active[side]:
                events.append((k, int(side)))
    return events

def gait(data, first, length):
    last = first+length
    if last>len(data["trace"]): return None
    p = data["qpos"]
    R = np.stack([quat_wxyz_to_mat(q) for q in p[first:last,3:7]])
    yaw = np.arctan2(R[:,1,0],R[:,0,0])
    roll=np.rad2deg(np.unwrap(np.arctan2(R[:,2,1],R[:,2,2])))
    v=np.diff(p[first-1:last,:2],axis=0)*50.
    lateral = -np.sin(yaw)*v[:,0]+np.cos(yaw)*v[:,1]
    forward = np.cos(yaw)*v[:,0]+np.sin(yaw)*v[:,1]
    ref=data["reference"]
    ref_v=np.diff(ref[first-1:last,:2],axis=0)*50.
    ref_fwd=np.cos(yaw)*ref_v[:,0]+np.sin(yaw)*ref_v[:,1]
    trace=data["trace"][first:last]
    events=[(k,s) for k,s in touchdown_sequence(data["loads"][:last]) if first<=k<last]
    periods=[(k1-k0)*.02 for (k0,s0),(k1,s1) in zip(events,events[1:]) if s0!=s1]
    return dict(ticks=length, lateral_rms_m_s=float(np.sqrt(np.mean(lateral**2))),
        roll_rate_rms_deg_s=float(np.sqrt(np.mean((np.diff(roll)*50)**2))),
        forward_reference_rmse=float(np.sqrt(np.mean((forward-ref_fwd)**2))),
        roll_range_deg=float(np.ptp(roll)), forward_mean_m_s=float(forward.mean()),
        displacement_m=float(np.linalg.norm(p[last-1,:2]-p[first,:2])),
        forward_speed_std=float(forward.std()), foot_width_cm=float(trace[:,7].mean()),
        min_height_m=float(trace[:,0].min()), min_cos=float(trace[:,1].min()),
        touches_L=sum(s==0 for _,s in events), touches_R=sum(s==1 for _,s in events),
        same_side_returns=sum(a[1]==b[1] for a,b in zip(events,events[1:])),
        touchdown_times_s=[round((k-first)*.02,2) for k,_ in events],
        touchdown_sides=["LR"[s] for _,s in events],
        step_time_cv=float(np.std(periods)/np.mean(periods)) if len(periods)>=3 else None)

def relay_row(data):
    trace=data["trace"]
    a=int(data["fork_tick"])
    release=np.flatnonzero((np.arange(len(trace))>=max(0,a))&(trace[:,5]==0))
    if a<0 or not len(release):return dict(success=False,release=None)
    first=int(release[0])
    after=trace[first:]
    walked=0
    for z,cosine,_,_,_,weight,*_ in after:
        walked=walked+1 if z>=.35 and cosine>.85 and weight==0 else 0
        if walked>=150:break
    # Original runner scores the last 150 steps; first-release windows always start immediately.
    last=gait(data,len(trace)-150,150)
    windows={str(n):gait(data,first,n) for n in (50,150)}
    refall=bool(np.any(after[:,5]==1))
    measured = json.loads(str(data["outcome_json"])) if "outcome_json" in data else None
    return dict(success=measured["success"] if measured else bool(walked>=150 and not refall and last["foot_width_cm"]<=22 and last["displacement_m"]>.25 and not trace[:,3].any()),
        release=first,ref_cursor=int(trace[first,8]),end_tick=len(trace),refall=refall,pd_oob=int(trace[:,3].sum()),
        first_second=windows["50"], first_three=windows["150"], seconds_2_3=gait(data,first+50,100))

def analyze(root):
    report={"e1":{}, "walk":{}}
    if (root/"e101").exists():
        for file in sorted((root/"e101/original").glob("*.npz")):
            for mode in ("original","delay10","delay20"):
                branch=root/"e101"/mode/file.name
                with np.load(branch if branch.exists() else file) as d:
                    report["e1"].setdefault(mode,[]).append(dict(trial=int(file.stem.split("_")[-1]),**score(d["trace"])))
    for case in ("relay01","held01","blind01"):
        suite=root/case
        if not suite.exists():continue
        assert (suite/"audit.json").is_file()
        rows={}
        report["walk"][case]=rows
        for folder in sorted(suite.iterdir()):
            if folder.name in ("runner","clone") or not folder.is_dir():continue
            for file in sorted((suite/"original").glob("*.npz")):
                path=folder/file.name
                with np.load(path if path.exists() else file) as data:
                    row=relay_row(data)
                i=int(file.stem.split("_")[-1])
                row["trial"]=i
                if row["release"] is not None:
                    with np.load(root/"nominal01/original"/f"relay_{0 if i%5<2 else 1}.npz") as nom:
                        idx=np.flatnonzero(nom["trace"][:,8]==row["ref_cursor"])[0]
                        row["nominal_3s"]=gait(nom,int(idx),150)
                rows.setdefault(folder.name,[]).append(row)
        if (suite/"clone").is_dir():
            for f in (suite/"clone").glob("*.npz"):
                with np.load(f) as a,np.load(suite/"original"/f.name) as b:
                    assert all(np.array_equal(a[k],b[k]) for k in a.files)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(analyze(a.root),indent=2,allow_nan=False))

