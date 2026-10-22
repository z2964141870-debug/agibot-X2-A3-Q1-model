#!/usr/bin/env python3
"""Render saved A3 evaluation states beside their reference, without physics steps."""

import argparse
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from complete_run import health, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    if args.fps < 1 or args.fps > 50 or args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("Use 1..50 FPS and a positive optional preview duration")
    args.output.mkdir(parents=True, exist_ok=False)
    initial_health = health()
    evaluation = json.loads(args.evaluation.read_text())
    admission = json.loads(args.admission.read_text())
    if evaluation["status"] != "DETERMINISTIC_PHYSICS_EVALUATION_COMPLETED":
        raise ValueError("A completed physical evaluation is required")
    if identity(evaluation["checkpoint"]["path"]) != evaluation["checkpoint"]:
        raise ValueError("Evaluation checkpoint identity changed")
    rows = [row for row in evaluation["rows"] if row["clip"] == args.clip]
    if len(rows) != 1:
        raise ValueError("Expected exactly one matching evaluated clip")
    row = rows[0]
    for key in ("reference", "trace"):
        if identity(row[key]["path"]) != row[key]:
            raise ValueError(f"Evaluation {key} changed")
    source_model = admission["assets"]["mjcf"]
    if identity(source_model["path"]) != source_model:
        raise ValueError("A3 geometry source changed")
    with np.load(row["trace"]["path"], allow_pickle=False) as trace:
        actual = trace["state"].copy()
    with np.load(row["reference"]["path"], allow_pickle=False) as reference:
        target = reference["qpos"].copy()
        reference_fps = float(reference["fps"])
    if reference_fps != 50 or actual.shape != (row["executed_steps"], 83):
        raise ValueError("Expected the evaluated A3 50-Hz state layout")
    if len(target) != row["expected_steps"] or not np.isfinite(actual).all() or not np.isfinite(target).all():
        raise ValueError("Reference count or finite-state contract failed")
    available_seconds = len(actual) / reference_fps
    seconds = min(available_seconds, args.max_seconds) if args.max_seconds else available_seconds
    indices = np.minimum((np.arange(int(np.ceil(seconds * args.fps))) * reference_fps / args.fps).astype(int), len(actual) - 1)
    model = mujoco.MjModel.from_xml_path(source_model["path"])
    if (model.nq, model.nv) != (42, 41):
        raise ValueError("A3 model dimensions changed")
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 135., -15.
    width, height, header = 640, 480, 64
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    frame_path, video_path = args.output / "last_frame.png", args.output / "trace_comparison.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{2 * width}x{height + header}", "-r", str(args.fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(video_path)]
    last_frame, sampled_health, pixel_std = None, [], []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        with (args.output / "ffmpeg.log").open("x") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            try:
                for output_index, index in enumerate(indices):
                    if output_index % 50 == 0:
                        sampled_health.append(health())
                    camera.lookat[:] = .5 * (target[index, :3] + actual[index, :3])
                    camera.lookat[2] = .85
                    separation = np.linalg.norm(target[index, :2] - actual[index, :2])
                    camera.distance = max(3.2, 2.5 + separation * 1.5)
                    panels = []
                    for pose in (target[index], actual[index, :42]):
                        data.qpos[:] = pose
                        data.qvel[:] = 0.
                        mujoco.mj_kinematics(model, data)
                        renderer.update_scene(data, camera=camera)
                        panels.append(renderer.render().copy())
                    pixel_std.append([float(panel.std()) for panel in panels])
                    frame = Image.new("RGB", (2 * width, height + header), "#f4f6f7")
                    for panel_index, panel in enumerate(panels):
                        frame.paste(Image.fromarray(panel), (panel_index * width, header))
                    draw = ImageDraw.Draw(frame)
                    draw.text((16, 9), "Reference target", font=font, fill="#13191e")
                    draw.text((width + 16, 9), "Evaluated policy state", font=font, fill="#13191e")
                    label = f"{args.clip} | t={index / reference_fps:.2f}s | full clip={len(target) / reference_fps:.2f}s"
                    draw.text((16, 36), label, font=font, fill="#39444d")
                    outcome = f"Evaluation terminated early: {row['failed']} | executed={available_seconds:.2f}s"
                    draw.text((width + 16, 36), outcome, font=font, fill="#8a2436" if row["failed"] else "#176444")
                    process.stdin.write(frame.tobytes())
                    last_frame = frame
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
                raise RuntimeError(f"Video encoding failed with exit {process.returncode}")
    if not pixel_std or min(min(row) for row in pixel_std) < 1.:
        raise ValueError("Rendered robot panels appear blank")
    last_frame.save(frame_path)
    result = {"status": "SAVED_TRACE_RENDER_COMPLETED", "entry": identity(__file__),
              "evaluation": identity(args.evaluation), "admission": identity(args.admission),
              "clip": args.clip, "trace": row["trace"], "reference": row["reference"], "mjcf": source_model,
              "rendered_frames": len(indices), "fps": args.fps, "rendered_seconds": len(indices) / args.fps,
              "executed_seconds": available_seconds, "full_clip_seconds": len(target) / reference_fps,
              "preview_truncated": seconds < available_seconds, "evaluation_failed": row["failed"],
              "render_panel_pixel_std_min": float(np.min(pixel_std)),
              "alignment": "Post-transition policy state and that control step's target, matching evaluation rewards. Both panels share the same camera.",
              "new_physics_steps": 0, "new_policy_inference": 0, "new_optimizer_updates": 0,
              "initial_health": initial_health, "sampled_health": sampled_health, "final_health": health(),
              "artifacts": [identity(path) for path in (frame_path, video_path)], "storage_status": "LOCAL_ONLY",
              "not_new_policy_performance_evidence": True}
    with (args.output / "receipt.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
