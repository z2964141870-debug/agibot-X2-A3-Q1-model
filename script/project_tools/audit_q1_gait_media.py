"""Audit saved gait-trial MP4s locally and extract fixed demonstration frames."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from audit_q1_hp_artifacts import identity, read
from audit_q1_rf13_media import FIXED, sheet


def audit(trial, labels, output):
    if output.exists():
        raise FileExistsError(output)
    declarations = {label: read(trial / "evaluation" / (label + "_media.json"))
                    for label in labels}
    names = [row["name"] for row in declarations[labels[0]]["segments"]]
    checks, videos = [], {}
    for label, declared in declarations.items():
        path = trial / "evaluation" / (label + ".mp4")
        actual = identity(path)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
            "-show_format", "-of", "json", str(path),
        ], text=True))
        stream = probe["streams"][0]
        rate = float(Fraction(stream["avg_frame_rate"]))
        decoded = subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-xerror", "-nostats",
            "-progress", "pipe:1", "-i", str(path), "-f", "null", "-",
        ], check=True, capture_output=True, text=True)
        frames = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines()
                  if line.startswith("frame=")][-1]
        segments = declared["segments"]
        contiguous = segments[0]["first_video_frame"] == 0 and all(
            current["first_video_frame"] == previous["last_video_frame"] + 1
            for previous, current in zip(segments, segments[1:]))
        tests = {
            "size_sha_matches_server": all(
                actual[key] == declared["video"][key] for key in ("bytes", "sha256")),
            "h264_1280x480_25fps": stream["codec_name"] == "h264"
                and (stream["width"], stream["height"]) == (1280, 480)
                and rate == declared["fps"] == 25,
            "full_decode_1350_frames": frames == declared["frames"]
                == int(stream["nb_frames"]) == 1350 and not decoded.stderr.strip(),
            "duration_54s": frames / rate == declared["duration_seconds"] == 54.,
            "original_speed_contiguous_independent_segments": contiguous
                and declared["speed"] == 1
                and segments[-1]["last_video_frame"] == frames - 1
                and all(row["last_video_frame"] - row["first_video_frame"] + 1
                        == (row["reference_frames_50hz"] + 1) // 2 for row in segments),
            "fixed_eight_segments_six_categories_failure_hold":
                len(segments) == 8 and tuple(row["name"] for row in segments[:3]) == FIXED
                and [row["name"] for row in segments] == names
                and {row["group"] for row in segments} == {
                    "walking", "gesture", "turning", "jogging", "dancing", "jumping"}
                and "hold after termination" in declared["left"]
                and all(0 < row["physical_steps"] <= row["reference_frames_50hz"] - 1
                        and (row["failed"] or row["physical_steps"]
                             == row["reference_frames_50hz"] - 1) for row in segments),
        }
        checks.extend({"name": label + "_" + key, "passed": bool(value)}
                      for key, value in tests.items())
        mids = [(row["first_video_frame"] + row["last_video_frame"]) // 2 for row in segments]
        walks = [row["first_video_frame"] + round(
            (row["last_video_frame"] - row["first_video_frame"]) * fraction)
            for row in segments[:2] for fraction in (.05, .20, .35, .50, .70, .95)]
        raised = segments[2]
        raise_frames = [raised["first_video_frame"] + round(
            (raised["last_video_frame"] - raised["first_video_frame"]) * i / 7)
            for i in range(8)]
        extracted = {suffix: sheet(path, selected, path.with_name(label + "_" + suffix + ".png"))
                     for suffix, selected in (("segment_contact_sheet", mids),
                                              ("walk_sequence", walks),
                                              ("raise_sequence", raise_frames))}
        videos[label] = {
            "file": actual, "declaration": identity(path.with_name(label + "_media.json")),
            "fps": rate, "decoded_frames": frames, "seconds": frames / rate,
            "visual_review_frames": extracted,
            "failed_segments": [{"name": row["name"], "physical_steps": row["physical_steps"],
                                 "reference_steps": row["reference_frames_50hz"] - 1}
                                for row in segments if row["failed"]],
        }
    value = {
        "status": "GAIT_MEDIA_AUDIT_PASSED" if all(row["passed"] for row in checks) else "FAILED",
        "captured_utc": datetime.now(timezone.utc).isoformat(), "trial": str(trial),
        "checks": checks, "videos": videos, "source": identity(__file__),
        "new_physical_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
        "limits": "Independent source-speed segments, not a continuous54s rollout. "
                  "Full decoding and declarations verify media integrity; contact sheets "
                  "require separate visual review and do not certify natural gait.",
    }
    with output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": value["status"], "checks": len(checks),
                      "passed": sum(row["passed"] for row in checks), "output": str(output)}))
    return 0 if value["status"] == "GAIT_MEDIA_AUDIT_PASSED" else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", choices=("initial", "final", "selected"),
                        default=["initial", "final"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.labels)) != len(args.labels):
        parser.error("Video labels must be unique")
    raise SystemExit(audit(args.trial, args.labels, args.output))
