"""Decode the four published-policy videos without running a simulation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw


def inspect(root, ffmpeg, ffprobe):
    progress = json.loads((root / "policy_suite_progress.json").read_text())
    if progress["status"] != "COMPLETED" or len(progress["replays"]) != 4:
        raise RuntimeError("Require the completed, fixed four-case suite")
    checks = []
    for replay in progress["replays"]:
        summary = replay["upstream_summary"]
        path = root / Path(replay["video"]).name
        expected = summary["frames"]
        metadata = json.loads(subprocess.check_output([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
            "-show_format", "-of", "json", str(path)], text=True))
        stream = metadata["streams"][0]
        width, height = stream["width"], stream["height"]
        numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
        fps = numerator / denominator
        if (width, height) != (1280, 720) or fps != 50:
            raise RuntimeError(f"Unexpected video geometry/rate: {path.name}")
        selections = {
            "overview": np.linspace(0, expected - 1, 9).round().astype(int).tolist(),
            "gait": [round(t * fps) for t in np.linspace(0.6, 2.2, 9)],
        }
        sheets = {key: Image.new("RGB", (1152, 738), "white")
                  for key in selections}
        process = subprocess.Popen([
            ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo",
            "-pix_fmt", "rgb24", "pipe:1"], stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        count, std_min, changed = 0, float("inf"), 0
        previous = None
        try:
            while True:
                raw = process.stdout.read(width * height * 3)
                if not raw:
                    break
                if len(raw) != width * height * 3:
                    raise RuntimeError(f"Truncated decoded frame: {path.name}")
                frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                sample = frame[::8, ::8].astype(np.int16)
                std_min = min(std_min, float(sample.std()))
                if previous is not None and np.any(sample != previous):
                    changed += 1
                previous = sample
                for key, indices in selections.items():
                    for slot, wanted in enumerate(indices):
                        if count != wanted:
                            continue
                        x, y = (slot % 3) * 384, (slot // 3) * 246
                        sheets[key].paste(Image.fromarray(frame).resize((384, 216)),
                                          (x, y + 30))
                        ImageDraw.Draw(sheets[key]).text(
                            (x + 6, y + 8),
                            f"{replay['robot']} | frame {count} | {count / fps:.2f}s",
                            fill="black")
                count += 1
            error = process.stderr.read().decode(errors="replace")
            code = process.wait(timeout=30)
            if code or error or count != expected or std_min < 5 or changed == 0:
                raise RuntimeError(f"Media check failed: {path.name}, {count}, {error}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
            process.stderr.close()
        contacts = []
        for key, sheet in sheets.items():
            target = path.with_suffix(f".{key}.png")
            sheet.save(target)
            contacts.append(target.name)
        checks.append(dict(video=path.name, bytes=path.stat().st_size,
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           decoded_frames=count, expected_frames=expected,
                           width=width, height=height, fps=fps,
                           duration_seconds=count / fps,
                           sampled_pixel_std_min=std_min,
                           adjacent_sampled_frames_changed=changed,
                           contact_sheets=contacts, selected_frames=selections,
                           upstream_result=summary["result"],
                           padded_or_replaced=False))
    result = dict(status="DECODED_MEDIA_CHECKS_PASSED", count=len(checks),
                  checks=checks, new_physics_steps=0, optimizer_updates=0,
                  visual_review="Pending separate image inspection")
    (root / "media_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/homebrew/bin/ffmpeg")
    parser.add_argument("--ffprobe", default="/opt/homebrew/bin/ffprobe")
    args = parser.parse_args()
    inspect(args.review, args.ffmpeg, args.ffprobe)
