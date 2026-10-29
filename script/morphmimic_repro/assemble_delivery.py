#!/usr/bin/env python3
"""Assemble retained replays after a separately guarded rendering completion."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from finish_original import digest, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment
    previous_path = root / "deployment_review/progress.json"
    previous = json.loads(previous_path.read_text())
    guard = json.loads((root / "deployment_delivery/stop_replay_guard.json").read_text())
    if previous["status"] != "FAILED" or not previous.get("error", "").startswith("TimeoutExpired:"):
        raise RuntimeError("Expected the retained rendering timeout record")
    if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
        raise RuntimeError("Separate stop replay did not finish cleanly")
    if len(previous["replays"]) != 5:
        raise RuntimeError("Expected five retained completed replay artifacts")
    for replay in previous["replays"]:
        for key in ("video", "summary"):
            if digest(Path(replay[key]["path"])) != replay[key]:
                raise RuntimeError("A retained replay changed")
    output = root / "deployment_delivery"
    video = output / "it600_walk_forward_stop_005__A026.mp4"
    summary_path = video.with_suffix(".json")
    summary = json.loads(summary_path.read_text())
    replays = previous["replays"] + [{"iteration": 600, "upstream_summary": summary,
                                      "video": digest(video), "summary": digest(summary_path)}]
    expected = {(iteration, clip["name"]) for iteration in (0, 600)
                for clip in previous["plan"]["clips"]}
    if {(r["iteration"], r["upstream_summary"]["clip"]) for r in replays} != expected:
        raise RuntimeError("Replay set differs from the fixed plan")
    record = {
        "status": "DEPLOYMENT_REVIEW_COMPLETED_WITH_RENDER_TIMEOUT_RECORDED",
        "assembled_utc": datetime.now(timezone.utc).isoformat(),
        "plan": previous["plan"], "checkpoint_audit": previous["checkpoint_audit"],
        "network_max_abs_changes": previous["network_max_abs_changes"], "replays": replays,
        "prior_failure": {**digest(previous_path), "status": previous["status"],
                          "error": previous["error"], "preserved_unchanged": True},
        "rendering_completion": {
            "guard": digest(output / "stop_replay_guard.json"),
            "whole_process_limit_seconds": 600,
            "reason": "The first CPU simulation plus 1280x720 rendering reached step 106 before its 180-second process limit. That timeout is not a recorded policy failure.",
            "rerun_scope": "Only final600 stop clip from the same original initial state and unchanged policy/assets; no training resumed, no other replay repeated.",
            "training_optimizer_updates": 0,
        },
    }
    path = output / "progress.json"
    if path.exists():
        raise FileExistsError(path)
    save(path, record)
    print(json.dumps({"status": record["status"], "replays": len(replays)}))


if __name__ == "__main__":
    main()
