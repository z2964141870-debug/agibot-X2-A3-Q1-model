"""Use the pinned converters and filters for real Q1 CSV and same-frame humans."""

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys


def identity(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def human_job(job):
    from morphmimic.data.human_convert import convert_human_clip
    return convert_human_clip(*job)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "csv", "bvh", "metadata", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--g1-subset", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for key in ("source", "csv", "bvh", "metadata"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source / "src"))
    from bootstrap import install
    install("q1")
    import numpy as np
    import yaml
    from morphmimic.data.seed_convert import convert_all
    from morphmimic.data.manifest import filtered
    import torch

    csvs = sorted(args.csv.glob("*/*.csv"))
    bvhs = [args.bvh / p.parent.name / (p.stem + ".bvh") for p in csvs]
    inputs = [identity(p) for p in (args.metadata, *csvs, *bvhs)]
    npz = args.output / "npz50"
    df = convert_all(str(args.csv), str(npz), str(args.metadata), args.workers, robot="q1")
    if df["feas_reject"].str.startswith("error").any():
        raise RuntimeError("Retain manifest; at least one original conversion failed")
    jobs = [(str(b), str(npz / "human50" / p.parent.name / (p.stem + ".npz")))
            for p, b in zip(csvs, bvhs, strict=True)]
    with multiprocessing.Pool(args.workers) as pool:
        humans = list(pool.imap(human_job, jobs))
    for row, human in zip(df.itertuples(), humans, strict=True):
        if row.frames != human["frames"]:
            raise ValueError("Robot/human frame mismatch: " + row.path)
        with np.load(npz / row.path) as z:
            if str(z["robot"]) != "q1" or z["joint_pos"].shape != (row.frames, 22):
                raise ValueError("Not a real22-joint Q1 clip")
            if not all(np.isfinite(z[k]).all() for k in
                    ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                     "body_lin_vel_w", "body_ang_vel_w")):
                raise ValueError("Nonfinite converted Q1 clip")
    kept = filtered(df)
    paired = kept["path"].tolist()
    excluded = []
    if args.g1_subset:
        g1 = yaml.safe_load(args.g1_subset.read_text())
        g1_paths = set(g1["clips"])
        accepted = []
        for row in kept.itertuples():
            reason = None
            if row.path not in g1_paths:
                reason = "not_in_original_g1_subset"
            else:
                with np.load(Path(g1["root"]) / row.path) as z:
                    if (len(z["joint_pos"]) != row.frames or int(z["fps"]) != 50):
                        reason = "g1_frame_count_or_fps_mismatch"
                    elif len(z["joint_names"]) != 29:
                        raise ValueError("Original G1 counterpart has an unexpected asset")
            if reason:
                excluded.append(dict(path=row.path, reason=reason))
            else:
                accepted.append(row.path)
        paired = accepted
        g1["clips"] = paired
        g1["hours"] = float(kept.set_index("path").loc[paired, "seconds"].sum() / 3600)
        (args.output / "g1_paired.yaml").write_text(yaml.safe_dump(g1, sort_keys=False))
    if not paired:
        raise RuntimeError("No valid Q1 clips; do not substitute another robot")
    seconds = float(kept.set_index("path").loc[paired, "seconds"].sum())
    subset = dict(root=str(npz), hours=seconds / 3600., clips=paired)
    (args.output / "q1_paired.yaml").write_text(yaml.safe_dump(subset, sort_keys=False))
    if [identity(p) for p in (args.metadata, *csvs, *bvhs)] != inputs:
        raise RuntimeError("Conversion source changed")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU converters unexpectedly initialized CUDA")
    result = dict(status="COMPLETED", csv_count=len(csvs), converted=len(df),
        original_filter_kept=len(kept), paired_count=len(paired), paired_seconds=seconds,
        exclusions=excluded, human_sidecars=len(humans), human_robot_frames_match=True,
        source_identities=inputs, new_dynamics_steps=0, new_optimizer_updates=0,
        cuda_initialized=False, storage="SERVER_LOCAL_ONLY", user_accepted_checkpoint=None)
    (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "converted", "original_filter_kept",
        "paired_count", "paired_seconds", "human_sidecars")}), flush=True)


if __name__ == "__main__":
    main()
