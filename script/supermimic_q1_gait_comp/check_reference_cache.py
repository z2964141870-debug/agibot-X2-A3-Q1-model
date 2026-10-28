"""Check real fixed Q1 reference FK on CPU, without an environment or optimizer."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--library",type=Path,required=True)
    a = parser.parse_args()
    sys.path[:0] = [str(a.runtime / "src"),str(a.runtime)]
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from unisupermimic.rl.algos.supermimic.geometry import mat_to_quat
    from rewards import FootReference
    torch.set_num_threads(4)
    names = ("walk_ff_stop_180_R_002__A095","walk_ff_start_180_R_002__A071","triumph_two_handed_001__A058")
    rows = [r for r in json.loads(a.library.read_text())["splits"]["train"] if r["name"] in names]
    if len(rows) != 3:
        raise ValueError("Fixed ordinary walks/raise missing")
    kin = Q1Kinematics(a.runtime / "q1_assets/scene.xml","cpu")
    motion = MotionLib([r["path"] for r in rows],kin,"cpu",src_fps=30.)
    reference = FootReference(SimpleNamespace(kin=kin,motion=motion))
    frames = motion.clip_start+10
    state = torch.cat((motion.root_p[frames],mat_to_quat(motion.root_R[frames]),motion.root_v[frames],
                       motion.root_w[frames],motion.q[frames],motion.qd[frames]),-1)
    features = reference.features(state,frames)
    torch.testing.assert_close(features[:,:12],torch.zeros(3,12),atol=3e-5,rtol=0.)
    if features.shape != (3,15):
        raise AssertionError("Unexpected reference foot feature dimensions")
    reward,parts = reference.reward(torch.zeros(3),state,frames,torch.zeros(3,22),torch.zeros(3,22),
                                   torch.zeros(3,14),torch.full((3,2),20.),torch.zeros(3,2))
    if not bool(torch.isfinite(reward).all()) or not bool((reward <=.05+1e-6).all()):
        raise AssertionError("Invalid real-reference reward decomposition")
    print(json.dumps({"status":"CPU_REAL_REFERENCE_CACHE_PASSED","clips":[r["name"] for r in rows],
                      "cached_frames":motion.num_frames,"feature_width":15,
                      "maximum_reference_self_error":float(features[:,:12].abs().max()),
                      "new_physical_steps":0,"optimizer_updates":0,"cuda_allocations":0}))
