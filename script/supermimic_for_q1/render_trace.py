#!/usr/bin/env python3
"""Render saved Q1 evaluation and reference poses without new physics steps."""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from supermimic.q1_assets import Q1Assets, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=12.)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assets = Q1Assets(args.root)
    evaluation = json.loads(args.evaluation.read_text())
    row, = [r for r in evaluation["rows"] if r["clip"] == args.clip]
    for key in ("trace", "reference"):
        if identity(row[key]["path"])["sha256"] != row[key]["sha256"]:
            raise ValueError("Saved evaluation artifact changed")
    with np.load(row["trace"]["path"], allow_pickle=False) as trace:
        actual = trace["state"]
    with np.load(row["reference"]["path"], allow_pickle=False) as reference:
        target = reference["qpos"]
    if actual.shape != (row["executed_steps"], 57) or len(target) != row["expected_steps"]:
        raise ValueError("Wrong Q1 trace shape or evaluation denominator")
    indices = np.arange(0, min(len(actual), int(args.max_seconds * 50)), 2)
    if len(indices) == 0:
        raise ValueError("Empty trace")
    model, data = assets.model, mujoco.MjData(assets.model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 135., -15.
    width, height, header, fps = 640, 480, 68, 25
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    video = args.output / "comparison.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{2 * width}x{height + header}", "-r", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(video)]
    standards = []
    saved_images = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with (args.output / "ffmpeg.log").open("x") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for output_frame, index in enumerate(indices):
                    camera.lookat[:] = (target[index, :3] + actual[index, :3]) * .5
                    camera.lookat[2] = .4
                    separation = float(np.linalg.norm(target[index, :2] - actual[index, :2]))
                    camera.distance = max(1.8, 1.2 + 1.5 * separation)
                    panels = []
                    for pose in (target[index], actual[index, :29]):
                        data.qpos[:] = pose
                        mujoco.mj_kinematics(model, data)
                        renderer.update_scene(data, camera=camera)
                        panels.append(renderer.render().copy())
                    standards.extend(float(panel.std()) for panel in panels)
                    frame = Image.new("RGB", (2 * width, height + header), "#f4f6f7")
                    for offset, panel in enumerate(panels):
                        frame.paste(Image.fromarray(panel), (offset * width, header))
                    draw = ImageDraw.Draw(frame)
                    draw.text((16, 8), "Q1 reference", fill="#182028", font=font)
                    draw.text((width + 16, 8), "Q1 policy: saved physical evaluation", fill="#182028", font=font)
                    draw.text((16, 36), f"{args.clip} | t={index / 50:.2f}s", fill="#39444d", font=font)
                    draw.text((width + 16, 36), f"Early termination: {row['failed']} | completed {len(actual)}/{len(target)} steps",
                              fill="#912b3d" if row["failed"] else "#176444", font=font)
                    process.stdin.write(frame.tobytes())
                    if output_frame in {0, len(indices) // 2, len(indices) - 1}:
                        path = args.output / f"frame_{output_frame:04d}.png"
                        frame.save(path)
                        saved_images.append(identity(path))
                process.stdin.close()
                process.wait(timeout=60)
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            if process.returncode != 0:
                raise RuntimeError(f"Encoding failed: {process.returncode}")
    if min(standards) < 1.:
        raise ValueError("Blank render")
    report = {"status": "SAVED_TRACE_RENDER_COMPLETED", "clip": args.clip,
              "evaluation": identity(args.evaluation), "assets": assets.manifest(),
              "trace": row["trace"], "reference": row["reference"], "rendered_frames": len(indices),
              "fps": fps, "pixel_std_min": min(standards), "artifacts": [identity(video), *saved_images],
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "source": identity(__file__), "storage_status": "LOCAL_ONLY",
              "alignment": "Post-control evaluated state and that control step's reference; same camera"}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
