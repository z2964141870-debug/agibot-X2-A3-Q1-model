"""Read saved retarget contact schedules without integration or reconversion."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from prepare_q1_gait_review import FIXED, identity, write_csv


def segments(mask):
    import numpy as np
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def point_speed(position, rotation, local, dt):
    import numpy as np
    n = len(position)
    before = np.maximum(np.arange(n)-1, 0)
    after = np.minimum(np.arange(n)+1, n-1)
    p0 = position[before] + np.einsum("nij,nj->ni", rotation[before], local)
    p1 = position[after] + np.einsum("nij,nj->ni", rotation[after], local)
    return (p1-p0) / ((after-before)[:, None]*dt)


def interior(mask):
    import numpy as np
    result = np.zeros_like(mask, dtype=bool)
    for start, end in segments(mask):
        if end-start >= 3:
            result[start+1:end-1] = True
    return result


def stable_resample(contact, count):
    import numpy as np
    at = np.arange(count)*.02*30.
    lo = np.minimum(np.floor(at+1e-8).astype(int), len(contact)-1)
    hi = np.minimum(np.ceil(at-1e-8).astype(int), len(contact)-1)
    return contact[lo] & contact[hi]


def self_test():
    import numpy as np
    p = np.c_[np.arange(5)*.1, np.zeros((5, 2))]
    r = np.tile(np.eye(3), (5, 1, 1))
    local = np.zeros((5, 3))
    np.testing.assert_allclose(point_speed(p, r, local, .1), np.tile([1.,0.,0.], (5,1)))
    angle = np.arange(5)*.1
    r[:,0,0], r[:,0,1] = np.cos(angle), -np.sin(angle)
    r[:,1,0], r[:,1,1] = np.sin(angle), np.cos(angle)
    local[:,0] = 1.
    p = -np.einsum("nij,nj->ni", r, local)
    np.testing.assert_allclose(point_speed(p,r,local,.1), 0., atol=1e-12)
    np.testing.assert_array_equal(interior(np.array([0,1,1,1,0], dtype=bool)), [0,0,1,0,0])
    np.testing.assert_array_equal(stable_resample(np.array([1,1,0], dtype=bool), 4), [1,1,0,0])
    print("PASS: translation, rotating fixed pivot, segment interior, conservative30-to50 phase alignment")


def audit(a):
    import mujoco
    import numpy as np
    import torch
    torch.set_num_threads(8)
    runtime = a.scene.parents[1]
    sys.path[:0] = [str(runtime / "src"), str(runtime), str(a.gait_code)]
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from rewards import FootReference

    if a.output.exists():
        raise FileExistsError(a.output)
    library = json.loads(a.library.read_text())
    rows = library["splits"]["train"]
    if len(rows) != 330:
        raise ValueError("Original full330 library required")
    kin = Q1Kinematics(a.scene, "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    from types import SimpleNamespace
    reference = FootReference(SimpleNamespace(motion=motion, kin=kin))
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    data = mujoco.MjData(model)
    feet = [model.body(side+"_ankle_roll_link").id for side in ("left","right")]
    result, timeseries, hashes = [], [], []
    for index, row in enumerate(rows):
        csv_path = Path(row["path"])
        if identity(csv_path)["sha256"] != row["sha256"]:
            raise ValueError("Original Q1 reference changed")
        metadata_path = csv_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text())
        raw = np.loadtxt(csv_path, delimiter=",", ndmin=2)
        contact = np.asarray(metadata["contact"], dtype=bool)
        local = np.asarray(metadata["contact_local_points"], dtype=float)
        if contact.shape != (len(raw),2) or local.shape != (len(raw),2,3):
            raise ValueError("Wrong saved support schedule shape")
        if metadata["output_sha256"] != row["sha256"]:
            raise ValueError("Retarget metadata points to different CSV")
        hashes.append(identity(metadata_path))
        pos, rot = np.empty((len(raw),2,3)), np.empty((len(raw),2,3,3))
        for frame, pose in enumerate(raw):
            data.qpos[:] = np.r_[pose[:3], pose[[6,3,4,5]], pose[7:]]
            mujoco.mj_kinematics(model, data)
            pos[frame] = data.xpos[feet]
            rot[frame] = data.xmat[feet].reshape(2,3,3)
        first, end = int(motion.clip_start[index]), int(motion.clip_end[index])
        frames = torch.arange(first, end+1)
        _, _, height, stance, swing, _ = reference.targets(frames)
        height, stance, swing = [x.cpu().numpy() for x in (height, stance, swing)]
        declared50 = stable_resample(contact, len(frames))
        for foot, side in enumerate(("left","right")):
            mask = contact[:,foot]
            inside = interior(mask)
            world = pos[:,foot]+np.einsum("nij,nj->ni", rot[:,foot], local[:,foot])
            speed = np.linalg.norm(point_speed(pos[:,foot], rot[:,foot], local[:,foot], 1/30.)[:,:2], axis=-1)
            drift = []
            for start, stop in segments(mask):
                xy = world[start:stop,:2]
                drift.extend(np.linalg.norm(xy-np.median(xy,axis=0), axis=-1).tolist())
            declared = declared50[:,foot]
            inferred = stance[:,foot]
            union = declared | inferred
            value = dict(clip=row["name"], split=row["split"], group=row["group"], side=side,
                frames30=len(raw), frames50=len(frames), declared_contact_fraction30=float(mask.mean()),
                declared_interior_frames30=int(inside.sum()),
                declared_interior_support_speed_mps=float(speed[inside].mean()) if inside.any() else None,
                declared_interior_support_speed_p95_mps=float(np.quantile(speed[inside],.95)) if inside.any() else None,
                declared_support_anchor_drift_p95_m=float(np.quantile(drift,.95)) if drift else None,
                declared_support_height_p95_m=float(np.quantile(world[mask,2],.95)) if mask.any() else None,
                declared_stable_fraction50=float(declared.mean()),
                old_heuristic_stance_fraction50=float(inferred.mean()),
                old_heuristic_swing_fraction50=float(swing[:,foot].mean()),
                old_heuristic_neither_fraction50=float((~(stance[:,foot]|swing[:,foot])).mean()),
                declared_vs_heuristic_stance_iou50=float((declared&inferred).sum()/union.sum()) if union.any() else None,
                declared_not_old_stance_fraction50=float((declared&~inferred).mean()),
                old_stance_not_declared_fraction50=float((inferred&~declared).mean()),
                ankle_relative_height_p95_m=float(np.quantile(height[:,foot],.95)))
            result.append(value)
            if row["name"] in FIXED:
                for t in range(len(frames)):
                    timeseries.append(dict(clip=row["name"], side=side, seconds=t*.02,
                        declared_stable_contact=int(declared[t]), heuristic_stance=int(inferred[t]),
                        heuristic_swing=int(swing[t,foot]), ankle_relative_height_m=float(height[t,foot])))
        if index%50 == 0:
            print(json.dumps(dict(reference_clips_processed=index+1)),flush=True)
    a.output.mkdir(parents=True, exist_ok=False)
    write_csv(a.output/"per_clip_support.csv", result)
    write_csv(a.output/"fixed_support_timeseries.csv", timeseries)
    value = dict(status="SAVED_REFERENCE_SUPPORT_AUDITED", captured_utc=datetime.now(timezone.utc).isoformat(),
        sources=[identity(p) for p in (a.library,a.scene,a.gait_code/"rewards.py",Path(__file__))],
        metadata=hashes, reference_clips=330, rows=len(result), fixed_rows=len(timeseries),
        new_integration_steps=0, policy_inference_calls=0, optimizer_updates=0,
        limits=["Declared contacts are source retarget intentions, not solved force/support labels.",
                "Original30Hz support point speed uses a fixed material point and excludes segment endpoints.",
                "50Hz declared schedule is conservative: both source neighbors must be contact.",
                "RF15 heuristic is reproduced exactly using original immutable MotionLib and FootReference.",
                "No reference changes, reconversion, rejection gate, or causal attribution of training degradation."])
    (a.output/"summary.json").write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(3,2,figsize=(12,8),sharex="row")
    for line, name in enumerate(FIXED):
        for foot, side in enumerate(("left","right")):
            data_rows=[r for r in timeseries if r["clip"]==name and r["side"]==side]
            t=[r["seconds"] for r in data_rows]
            ax=axes[line,foot]
            ax.plot(t,[r["ankle_relative_height_m"]*100 for r in data_rows],label="ankle lift cm")
            ax.plot(t,[r["declared_stable_contact"]*2 for r in data_rows],label="declared contact x2")
            ax.plot(t,[r["heuristic_stance"]*3 for r in data_rows],label="RF15 stance x3",alpha=.7)
            ax.set_title(name+" / "+side,fontsize=9)
            ax.set_xlabel("seconds");ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(a.output/"support_alignment.png",dpi=150)
    figure.savefig(a.output/"support_alignment.pdf")
    plt.close(figure)
    print(json.dumps({k:v for k,v in value.items() if k!="metadata"}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--self-test",action="store_true")
    for name in ("library","scene","gait-code","output"):
        parser.add_argument("--"+name,type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if any(getattr(args,k) is None for k in ("library","scene","gait_code","output")):
            parser.error("library, scene, gait-code and output required")
        audit(args)

