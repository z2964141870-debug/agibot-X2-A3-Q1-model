"""Render saved evidence on the server; validate video decoding on the Mac."""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent / "supermimic_q1_gait_goal_review"


def run(args):
    import numpy as np

    spec = importlib.util.spec_from_file_location("goal_saved_renderer", REVIEW / "deliver.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)

    def source_check(trial, media, suite, sources, rows):
        checks = []
        for label, declaration in media.items():
            with np.load(sources[label], allow_pickle=False) as trace:
                valid = all(
                    segment["physical_steps"] == int(trace["steps"][index])
                    and segment["failed"] == bool(trace["failed"][index])
                    and segment["reference_frames_50hz"] == int(trace["reference_steps"][index]) + 1
                    for segment in declaration["segments"]
                    for index in [next(i for i, row in enumerate(rows) if row["name"] == segment["name"])])
            valid &= original.identity(declaration["video"]["path"]) == declaration["video"]
            checks.append(dict(label=label, passed=bool(valid), trace=original.identity(sources[label])))
        original.save(trial / "server_media_source_check.json", dict(
            status="SAVED_MEDIA_SOURCE_CHECK_PASSED" if all(row["passed"] for row in checks) else "FAILED",
            checks=checks, decoding_status="PENDING_MAC_EXISTING_FFPROBE", new_physical_steps=0,
            policy_inference_calls=0, optimizer_updates=0))
        if not all(row["passed"] for row in checks):
            raise AssertionError("Video declarations differ from saved source trajectories")

    original.audit_media = source_check
    original.run(args)
    path = args.trial / "delivery_status.json"
    status = json.loads(path.read_text())
    status.update(scope="Server production only; Mac decoding and saved audit still required",
                  media_decoding_status="PENDING_MAC_EXISTING_FFPROBE")
    original.save(path, status)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "trial", "video-suite"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    run(parser.parse_args())
