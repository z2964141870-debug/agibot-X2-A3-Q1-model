"""Audit RF13's saved media declarations and extract frames for visual review."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from audit_q1_hp_artifacts import identity, read


FIXED = (
    "walk_ff_stop_180_R_002__A095",
    "walk_ff_start_180_R_002__A071",
    "triumph_two_handed_001__A058",
)


def sheet(video, frames, output, columns=2):
    if output.exists():
        raise FileExistsError(output)
    select = "+".join("eq(n," + str(frame) + ")" for frame in frames)
    rows = (len(frames) + columns - 1) // columns
    subprocess.run([
        "ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(video), "-vf",
        "select='" + select + "',scale=640:240,tile=" + str(columns) + "x" + str(rows),
        "-frames:v", "1", "-fps_mode", "vfr", str(output),
    ], check=True, capture_output=True, text=True)
    return {"file": identity(output), "video_frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declarations = {
        label: read(args.trial / "evaluation" / (label + "_media.json"))
        for label in ("initial", "final")
    }
    checks, videos = [], {}
    for label, declared in declarations.items():
        path = args.trial / "evaluation" / (label + ".mp4")
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
            for previous, current in zip(segments, segments[1:])
        )
        tests = {
            "size_sha_matches_server": all(
                actual[key] == declared["video"][key] for key in ("bytes", "sha256")),
            "h264_1280x480_25fps": stream["codec_name"] == "h264"
                and (stream["width"], stream["height"]) == (1280, 480)
                and rate == declared["fps"] == 25,
            "full_decode_1350_frames": frames == declared["frames"]
                == int(stream["nb_frames"]) == 1350 and not decoded.stderr.strip(),
            "duration_54s": frames / rate == declared["duration_seconds"] == 54.0,
            "original_speed_contiguous_independent_segments": contiguous
                and declared["speed"] == 1
                and segments[-1]["last_video_frame"] == frames - 1
                and all(row["last_video_frame"] - row["first_video_frame"] + 1
                        == (row["reference_frames_50hz"] + 1) // 2 for row in segments),
            "fixed_eight_segments_six_categories_and_failure_hold":
                len(segments) == 8
                and tuple(row["name"] for row in segments[:3]) == FIXED
                and [row["name"] for row in segments]
                    == [row["name"] for row in declarations["initial"]["segments"]]
                and {row["group"] for row in segments} == {
                    "walking", "gesture", "turning", "jogging", "dancing", "jumping"}
                and "hold after termination" in declared["left"]
                and all(0 < row["physical_steps"] < row["reference_frames_50hz"]
                        and row["failed"] == (
                            row["physical_steps"] < row["reference_frames_50hz"] - 1)
                        for row in segments),
        }
        checks.extend({"name": label + "_" + key, "passed": bool(value)}
                      for key, value in tests.items())
        mids = [(row["first_video_frame"] + row["last_video_frame"]) // 2
                for row in segments]
        walk_frames = [
            row["first_video_frame"] + round(
                (row["last_video_frame"] - row["first_video_frame"]) * fraction)
            for row in segments[:2] for fraction in (.05, .20, .35, .50, .70, .95)
        ]
        raise_row = segments[2]
        raise_frames = [
            raise_row["first_video_frame"] + round(
                (raise_row["last_video_frame"] - raise_row["first_video_frame"]) * i / 7)
            for i in range(8)
        ]
        extracted = {}
        for suffix, selected in (
            ("segment_contact_sheet", mids),
            ("walk_sequence", walk_frames),
            ("raise_sequence", raise_frames),
        ):
            extracted[suffix] = sheet(
                path, selected, path.with_name(label + "_" + suffix + ".png"))
        videos[label] = {
            "file": actual, "declaration": identity(
                path.with_name(label + "_media.json")),
            "codec": stream["codec_name"], "width": stream["width"],
            "height": stream["height"], "fps": rate, "decoded_frames": frames,
            "seconds": frames / rate, "visual_review_frames": extracted,
            "failed_segments": [
                {"name": row["name"], "physical_steps": row["physical_steps"],
                 "reference_steps": row["reference_frames_50hz"] - 1}
                for row in segments if row["failed"]
            ],
        }
    result = {
        "status": "RF13_MEDIA_AUDIT_PASSED" if all(x["passed"] for x in checks)
                  else "RF13_MEDIA_AUDIT_FAILED",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "videos": videos, "source": identity(__file__),
        "physics_integration_steps": 0, "policy_inference_calls": 0,
        "optimizer_updates": 0,
        "limits": "Independent source-speed segments, not one continuous rollout. "
                  "Frame sheets require separate visual review; these twelve checks "
                  "establish media integrity and declared protocol, not motion quality.",
    }
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({
        "status": result["status"], "passed": sum(x["passed"] for x in checks),
        "checks": len(checks), "output": str(args.output),
        "failures": [x for x in checks if not x["passed"]],
    }))
    return 0 if all(x["passed"] for x in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
