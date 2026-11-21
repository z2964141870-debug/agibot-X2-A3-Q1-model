"""Expand Stage A side training pool to 400 legal settled 2 ms starts (keeps side_val.npz intact)."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

RUN_V1 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_20261121")
RISE_DIR = Path("/home/yu/projects/SuperMimicHP/runs/q1_supported_rise_20261121")
AUTON_ROOT = Path("/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery")
SCRIPT_DIR = Path(__file__).resolve().parent
for p in reversed((str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT))):
    if p not in sys.path:
        sys.path.insert(0, p)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from build_v2_pools import collect_legal_starts
from common import CONFIG, Plant
from recorded_rise import load_track


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--pools",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/pools"),
    )
    p.add_argument("--seed-start", type=int, default=71030)
    p.add_argument("--seed-end", type=int, default=71055)
    p.add_argument("--base-name", default="side_train.npz")
    p.add_argument("--out-name", default="side_train_400.npz")
    args = p.parse_args()

    pools = args.pools.resolve()
    orig = np.load(pools / args.base_name)["rows"]
    env = Plant(8, pools / "expand_side_scene", delay=11)
    model = env.model
    geom_floor = model.geom("floor").id
    arm_pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(model.pair_geom1, model.pair_geom2, strict=True)}
    files = [
        Path.home() / "projects/Q1_deploy/real_logs" / n
        for n in ("q1_official_standup_from_mcap.npz", "q1_official_standup_faceup_20261120.npz")
    ]
    refs = [load_track(f, CONFIG["joint_names"]) for f in files]

    extra_seeds = list(range(args.seed_start, args.seed_end))
    extra_by_kind, extra_side_md = collect_legal_starts(model, refs, extra_seeds, arm_pairs, geom_floor)
    env.close()

    extra = extra_by_kind[2]
    assert extra.shape == (len(extra_seeds) * 8, 79), extra.shape
    combined = np.concatenate((orig, extra), axis=0)
    assert combined.shape == (len(orig) + len(extra), 79) and np.isfinite(combined).all(), combined.shape

    out_path = pools / args.out_name
    np.savez(out_path, rows=combined)
    extra_mds = sorted(x[3] for x in extra_side_md)
    info = dict(
        out=str(out_path),
        shape=list(combined.shape),
        orig_count=len(orig),
        extra_count=len(extra),
        extra_seeds=[args.seed_start, args.seed_end - 1],
        extra_side_min_d_mm=dict(
            min=extra_mds[0],
            p5=float(np.percentile(extra_mds, 5)),
            median=float(np.median(extra_mds)),
        ),
    )
    print("EXPANDED SIDE POOL:", json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
