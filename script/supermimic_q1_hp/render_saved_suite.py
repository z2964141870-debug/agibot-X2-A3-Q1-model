"""Render a multi-type comparison directly from saved physical trajectories."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from worker import identity, save
from video_suite import render_suite, select_suite


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("runtime","library","evaluation","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--label",default="final",choices=("initial","final"))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    sys.path[:0]=[str(args.runtime/"src"),str(args.runtime),str(Path(__file__).resolve().parents[1]/"project_tools")]
    import torch
    from analyze_q1_hp_trial import validate_trace
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    torch.set_num_threads(8)
    library=json.loads(args.library.read_text())
    rows=library["splits"]["train"]+library["splits"]["validation"]
    suite=select_suite(library,args.runtime)
    kin=Q1Kinematics(args.runtime/"q1_assets/scene.xml","cpu")
    motion=MotionLib([r["path"] for r in rows],kin,"cpu",src_fps=30.)
    trace_path=args.evaluation/(args.label+"_trajectory.npz")
    with np.load(trace_path,allow_pickle=False) as values:
        trace={k:values[k] for k in values.files}
    validate_trace(trace,len(rows))
    env=SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion),device="cpu")
    video=render_suite(args.runtime,trace,env,rows,suite,args.output/(args.label+"_multitype.mp4"),args.label)
    receipt={"status":"SAVED_PHYSICAL_MULTITYPE_VIDEO_COMPLETED","video":video,
             "sources":[identity(trace_path),identity(args.library)],"physics_steps":0,"policy_inference_calls":0,"optimizer_updates":0}
    save(args.output/"receipt.json",receipt)
    save(args.output/"video_suite.json",suite)
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
