"""Render an intermediate RF14 checkpoint from already saved physical traces."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--video-suite", type=Path, required=True)
    parser.add_argument("--label", choices=("stage_01", "stage_02"), required=True)
    parser.add_argument("--repeat", choices=(0, 1, 2), type=int, default=0)
    args = parser.parse_args()
    evaluation = args.trial / "evaluation"
    target = evaluation / (args.label + ".mp4")
    receipt_path = evaluation / (args.label + "_media.json")
    if target.exists() or receipt_path.exists():
        parser.error("Use existing review media; do not overwrite or repeat rendering")
    status = json.loads((args.trial / "status.json").read_text())
    if status["status"] != "COMPLETED":
        parser.error("Wait until the RF14 pipeline completes")
    source = evaluation / f"{args.label}_repeat{args.repeat}.npz"
    summary_path = source.with_suffix(".json")
    summary = json.loads(summary_path.read_text())
    receipt = json.loads((evaluation / "receipt.json").read_text())
    rows = receipt["rows"]
    if len(rows) != 330 or summary["repeat"] != args.repeat:
        raise ValueError("Unexpected evaluation library or repeat identity")
    script = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime),
                   str(script / "supermimic_q1_hp"), str(script / "supermimic_q1_ppo")]
    import numpy as np
    import torch
    from resources import identity, save, stamp
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from video_suite import render_suite

    torch.set_num_threads(8)
    kin = Q1Kinematics(args.runtime / "q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    env = SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion), device=torch.device("cpu"))
    with np.load(source, allow_pickle=False) as saved:
        trace = {key: saved[key] for key in ("initial", "states", "steps", "failed")}
    if trace["steps"].shape != (330,) or trace["states"].shape[1:] != (330, 57):
        raise ValueError("Unexpected saved physical trace shape")
    media = render_suite(args.runtime, trace, env, rows,
                         json.loads(args.video_suite.read_text()), target,
                         f"RF14 {args.label} repeat{args.repeat}")
    media.update(captured_utc=stamp(), source_trace=identity(source),
                 source_summary=identity(summary_path), repeat=args.repeat,
                 cumulative_training_seconds=summary["training_seconds"],
                 new_physical_steps=0, policy_inference_calls=0, optimizer_updates=0,
                 user_checkpoint_acceptance="pending")
    save(receipt_path, media)
    print(json.dumps({"status": "SAVED_TRACE_REVIEW_VIDEO_RENDERED",
                      "label": args.label, "repeat": args.repeat,
                      "video": media["video"], "new_physical_steps": 0,
                      "policy_inference_calls": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
