"""Original-speed actual/reference video from saved physical trajectories."""

from pathlib import Path
import csv
from fractions import Fraction
import os
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1"))
from common import identity, require


def decoded_metadata(text, frames):
    headers = {}
    lines = []
    for line in text.splitlines():
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            headers[key.strip()] = value.strip()
        elif line and not line.startswith("#"):
            lines.append(line)
    rows = list(csv.reader(lines, skipinitialspace=True))
    width, height = map(int, headers["dimensions 0"].split("x"))
    timebase = Fraction(headers["tb 0"])
    require((width, height, timebase) == (1280, 392, Fraction(1, 50)),
            "Decoded video dimensions or timebase differ")
    require(len(rows) == frames and all(len(row) == 6 and int(row[0]) == 0
            and int(row[1]) == i and int(row[2]) == i and int(row[3]) == 1
            and int(row[4]) == width * height * 3 and len(row[5].strip()) == 64
            for i, row in enumerate(rows)), "Decoded frames or timestamps differ")
    return dict(width=width, height=height, r_frame_rate="50/1", nb_frames=frames,
                duration=frames / 50, verification="complete FFmpeg RGB framehash decode")


def render_case(xml, case, output):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    ffmpeg = os.environ.get("Q1_REVIEW_FFMPEG") or shutil.which("ffmpeg")
    require(ffmpeg and Path(ffmpeg).is_file(), "An explicitly configured existing FFmpeg is required")
    with np.load(case["trace"], allow_pickle=False) as saved:
        actual, reference = saved["actual"].copy(), saved["reference"].copy()
    require(len(actual) <= len(reference), "Video trajectory exceeds reference")
    if len(actual) < len(reference):
        actual = np.concatenate((actual, np.repeat(actual[-1:], len(reference) - len(actual), axis=0)))
    model = mujoco.MjModel.from_xml_path(str(xml))
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(360, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.8, 125, -15
    command = [ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x392", "-r", "50", "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=360, width=640) as renderer:
            for frame in range(len(reference)):
                views = []
                for pose in (actual[frame], reference[frame]):
                    camera.lookat[:] = pose[:3]
                    camera.lookat[2] = max(.25, float(camera.lookat[2]))
                    data.qpos[:] = pose[:29]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    views.append(renderer.render().copy())
                image = Image.new("RGB", (1280, 392))
                image.paste(Image.fromarray(views[0]), (0, 32))
                image.paste(Image.fromarray(views[1]), (640, 32))
                draw = ImageDraw.Draw(image)
                failed = case["failure_frame"] is not None and frame >= case["failure_frame"]
                draw.text((8, 8), f"Q1 POLICY  {frame / 50:.2f}s  {'FAILED / HOLD' if failed else 'TRACKING'}", fill="red" if failed else "white")
                draw.text((648, 8), "REPAIRED SOMA REFERENCE  1x", fill="white")
                if frame in (0, len(reference) // 2, len(reference) - 1):
                    image.save(output.with_name(f"frame_{frame:05d}.png"))
                process.stdin.write(image.tobytes())
    except BaseException:
        process.stdin.close()
        process.terminate()
        process.wait()
        raise
    process.stdin.close()
    require(process.wait() == 0, "Video encoder failed")
    hashes = subprocess.check_output([ffmpeg, "-nostdin", "-v", "error", "-i", str(output),
        "-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "framehash", "-hash", "sha256", "pipe:1"], text=True)
    probe = decoded_metadata(hashes, len(reference))
    return dict(status="PASSED", **identity(output), probe=probe, original_speed=True,
                actual_frames_before_hold=case["attempted_steps"] + 1, failure_preserved=True)
