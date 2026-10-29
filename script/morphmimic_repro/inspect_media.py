#!/usr/bin/env python3
"""Decode original replay videos and create labeled review contact sheets."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw

    progress = json.loads((args.review / "progress.json").read_text())
    if progress["status"] != "DEPLOYMENT_REVIEW_COMPLETED":
        raise RuntimeError("Deployment must be finished before media inspection")
    checks = []
    for replay in progress["replays"]:
        path = Path(replay["video"]["path"])
        summary = replay["upstream_summary"]
        expected = (summary["frames"] if summary["result"] == "completed"
                    else int(summary["result"].rsplit(" ", 1)[1]) + 1)
        indices = np.linspace(0, expected - 1, 5).round().astype(int).tolist()
        reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        metadata = next(reader)
        width, height = metadata["size"]
        sheet = Image.new("RGB", (384 * 5, 254), "white")
        draw = ImageDraw.Draw(sheet)
        frames = 0
        std_min = float("inf")
        try:
            for index, raw in enumerate(reader):
                array = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
                std_min = min(std_min, float(array.std()))
                for column, wanted in enumerate(indices):
                    if index == wanted:
                        thumb = Image.fromarray(array).resize((384, 216))
                        sheet.paste(thumb, (384 * column, 38))
                        draw.text((384 * column + 8, 7),
                                  f"it{replay['iteration']} | frame {index} | {index / 50:.2f}s",
                                  fill="black")
                frames += 1
        finally:
            reader.close()
        if frames != expected or abs(metadata["fps"] - 50.0) > 0.01 or std_min < 5:
            raise RuntimeError(f"Invalid media {path.name}: {frames}/{expected}, {metadata}")
        contact = path.with_suffix(".contact.png")
        sheet.save(contact)
        checks.append({"video": path.name, "decoded_frames": frames,
                       "expected_frames": expected, "fps": metadata["fps"],
                       "size": [width, height], "encoded_seconds": frames / metadata["fps"],
                       "last_state_time_seconds": (frames - 1) / 50,
                       "minimum_frame_pixel_std": std_min, "contact_sheet": contact.name,
                       "selected_frames": indices, "upstream_result": summary["result"],
                       "padded_or_replaced": False})
    result = {"status": "DECODED_MEDIA_CHECKS_PASSED", "count": len(checks),
              "checks": checks, "visual_review": "Pending human/agent image inspection; decoding is not visual acceptance."}
    (args.review / "media_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
