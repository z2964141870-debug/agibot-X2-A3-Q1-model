"""Render fixed demonstration media from completed zero-optimizer probe traces."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def render(args):
    status = json.loads((args.trial / "status.json").read_text())
    guard = json.loads((args.trial / "guard.json").read_text())
    result = json.loads((args.trial / "results.json").read_text())
    if (status["status"] != "COMPLETED" or guard["status"] != "WORKER_COMPLETED"
            or guard["cleanup"]["worker_return_code"] != 0
            or result["optimizer_steps"] != 0 or result["scope"] != "full330"):
        raise ValueError("Require completed, successfully reaped full330 zero-optimizer probe")
    if args.candidate not in {item["candidate"] for item in result["candidates"]}:
        raise ValueError("Candidate not present in completed results")
    evaluation = args.trial / "evaluation"
    target = evaluation / (args.candidate + ".mp4")
    receipt = target.with_name(args.candidate + "_media.json")
    if target.exists() or receipt.exists():
        raise FileExistsError("Use existing media instead of overwriting or rerendering")
    source = args.trial / f"{args.candidate}_repeat{args.repeat}.npz"
    summary_path = source.with_suffix(".json")
    summary = json.loads(summary_path.read_text())
    rows = result["rows"]
    if (len(rows) != 330 or len({row["name"] for row in rows}) != 330
            or summary["candidate"] != args.candidate or summary["repeat"] != args.repeat):
        raise ValueError("Unexpected evaluation library or repeated trace identity")
    script = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime),
                   str(script / "supermimic_q1_hp"), str(script / "supermimic_q1_ppo")]
    import numpy as np
    import torch
    from resources import identity, save, stamp
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from video_suite import render_suite

    if identity(source) != summary["trace"]:
        raise ValueError("Saved trace size or SHA differs from original declaration")
    suite = json.loads(args.video_suite.read_text())
    if len(suite["clips"]) != 8 or len({row["group"] for row in suite["clips"]}) != 6:
        raise ValueError("Require the original fixed eight-segment six-type suite")
    torch.set_num_threads(8)
    kin = Q1Kinematics(args.runtime / "q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    env = SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion), device=torch.device("cpu"))
    with np.load(source, allow_pickle=False) as saved:
        trace = {key: saved[key] for key in ("initial", "states", "steps", "failed")}
    if trace["steps"].shape != (330,) or trace["states"].shape[1:] != (330, 57):
        raise ValueError("Unexpected saved physical trace shape")
    evaluation.mkdir(exist_ok=True)
    label = {"baseline": "Static 0.50", "static075": "Static 0.75",
             "activity050075": "Activity 0.50-0.75"}.get(args.candidate, args.candidate)
    media = render_suite(args.runtime, trace, env, rows, suite, target, label)
    media.update(captured_utc=stamp(), candidate=args.candidate, source_trace=identity(source),
                 source_summary=identity(summary_path), source_suite=identity(args.video_suite),
                 renderer=identity(Path(__file__)), repeat=args.repeat,
                 configured_seed=summary["seed"], new_physical_steps=0,
                 policy_inference_calls=0, optimizer_updates=0, user_accepted_checkpoint=None)
    save(receipt, media)
    print(json.dumps({"status": "SAVED_TRACE_MEDIA_RENDERED", "candidate": args.candidate,
                      "video": media["video"], "frames": media["frames"],
                      "new_physical_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("trial", "runtime", "video-suite"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--repeat", choices=(0, 1, 2), type=int, default=0)
    render(parser.parse_args())
