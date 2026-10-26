"""Validate local multi-motion MP4 delivery and extract segment contact sheets."""

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from audit_q1_hp_artifacts import identity, read


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    checks, videos = [], {}
    for label in ("initial", "final"):
        path = args.trial / "evaluation" / (label + ".mp4")
        declared = read(path.with_suffix(".json"))["video"]
        actual_identity = identity(path)
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
            "-show_format", "-of", "json", str(path)], text=True))
        stream = probe["streams"][0]
        rate = float(Fraction(stream["avg_frame_rate"]))
        decoded = subprocess.run([
            "ffmpeg", "-v", "error", "-xerror", "-nostats", "-progress", "pipe:1",
            "-i", str(path), "-f", "null", "-"], text=True, capture_output=True, check=True)
        frames = [int(line.split("=", 1)[1]) for line in decoded.stdout.splitlines()
                  if line.startswith("frame=")][-1]
        segments = declared["segments"]
        contiguous = segments[0]["first_video_frame"] == 0 and all(
            segments[index]["first_video_frame"] == segments[index - 1]["last_video_frame"] + 1
            for index in range(1, len(segments)))
        ordinary = [row for row in segments if row["label"] == "Forward walking"]
        tests = {
            "size_sha_matches_server": all(actual_identity[key] == declared["video"][key]
                                           for key in ("bytes", "sha256")),
            "h264_1280x480_25fps": stream["codec_name"] == "h264"
                and (stream["width"], stream["height"]) == (1280, 480) and rate == 25.,
            "full_decode_all_declared_frames": frames == declared["frames"]
                == int(stream["nb_frames"]) and not decoded.stderr.strip(),
            "duration_at_least_30s": frames / rate == declared["duration_seconds"] >= 30.,
            "original_speed_contiguous_independent_segments": contiguous
                and declared["speed"] == 1 and segments[-1]["last_video_frame"] == frames - 1
                and all(row["last_video_frame"] - row["first_video_frame"] + 1
                        == (row["reference_frames_50hz"] + 1) // 2 for row in segments),
            "six_categories_walk_and_raised_hands": len(segments) >= 8 and len(ordinary) >= 2
                and any(row["label"] == "Upper-body / raised hands" for row in segments)
                and {row["group"] for row in segments} == {
                    "walking", "gesture", "turning", "jogging", "dancing", "jumping"},
        }
        checks.extend({"name": label + "_" + key, "passed": bool(value)}
                      for key, value in tests.items())
        sheet = path.with_name(label + "_segment_contact_sheet.png")
        if sheet.exists():
            raise FileExistsError(sheet)
        mids = [(row["first_video_frame"] + row["last_video_frame"]) // 2 for row in segments]
        select = "+".join("eq(n," + str(frame) + ")" for frame in mids)
        subprocess.run([
            "ffmpeg", "-v", "error", "-n", "-i", str(path), "-vf",
            "select='" + select + "',scale=640:240,tile=2x4", "-frames:v", "1",
            "-fps_mode", "vfr", str(sheet)], check=True, capture_output=True, text=True)
        videos[label] = {"file": actual_identity, "codec": stream["codec_name"],
                         "width": stream["width"], "height": stream["height"],
                         "fps": rate, "decoded_frames": frames, "seconds": frames / rate,
                         "contact_sheet": identity(sheet), "sample_frames": mids}
    result = {"status": "HP_Q1_MEDIA_AUDIT_PASSED" if all(row["passed"] for row in checks)
              else "HP_Q1_MEDIA_AUDIT_FAILED", "created_utc": datetime.now(timezone.utc).isoformat(),
              "checks": checks, "videos": videos, "source": identity(__file__),
              "physics_integration_steps": 0, "policy_inference_calls": 0, "optimizer_updates": 0,
              "limits": "Eight independent source-speed segments, not one continuous rollout. "
              "Contact sheets are for separate visual review; this receipt checks media integrity "
              "and frozen segment declarations, not motion quality."}
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps({"status": result["status"], "checks": checks, "videos": videos}))
    return 0 if all(row["passed"] for row in checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
