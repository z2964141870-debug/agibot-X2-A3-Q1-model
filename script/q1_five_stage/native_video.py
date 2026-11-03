"""Original-speed native scene media, including the unedited failure hold."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

from contracts import require
from common import identity


def render_case(scene, case, output):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1_review"))
    from video import decoded_metadata
    ffmpeg = os.environ.get("Q1_REVIEW_FFMPEG") or shutil.which("ffmpeg")
    require(ffmpeg and Path(ffmpeg).is_file() and not output.exists(), "Use the existing encoder and new media path")
    with np.load(case["trace"], allow_pickle=False) as saved:
        actual, reference = saved["qpos_actual"].copy(), saved["qpos_reference"].copy()
    require(1 < len(actual) <= len(reference), "Invalid native failure trajectory")
    if len(actual) < len(reference):
        actual = np.concatenate((actual, np.repeat(actual[-1:], len(reference) - len(actual), axis=0)))
    model = mujoco.MjModel.from_binary_path(str(scene))
    require(actual.shape[1] == reference.shape[1] == model.nq, "Native media scene differs from qpos")
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(360, model.vis.global_.offheight)
    data, camera = mujoco.MjData(model), mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = (2.8 if case["robot"] == "g1" else 1.8), 125, -15
    command = [ffmpeg, "-nostdin", "-v", "error", "-n", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "1280x392", "-r", "50", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "23", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(model, height=360, width=640) as renderer:
            for frame in range(len(reference)):
                views = []
                for pose in (actual[frame], reference[frame]):
                    start = case["root_qpos_address"]
                    camera.lookat[:] = pose[start:start + 3]
                    camera.lookat[2] = max(.25, float(camera.lookat[2]))
                    data.qpos[:] = pose
                    mujoco.mj_kinematics(model, data)
                    mujoco.mj_comPos(model, data)
                    renderer.update_scene(data, camera=camera)
                    views.append(renderer.render().copy())
                image = Image.new("RGB", (1280, 392))
                image.paste(Image.fromarray(views[0]), (0, 32))
                image.paste(Image.fromarray(views[1]), (640, 32))
                failed = case["failure_frame"] is not None and frame >= case["failure_frame"]
                draw = ImageDraw.Draw(image)
                draw.text((8, 8), f"{case['robot'].upper()} {case['interface'].upper()} {frame / 50:.2f}s "
                    + ("FAILED / HOLD" if failed else "TRACKING"), fill="red" if failed else "white")
                draw.text((648, 8), "SAME SOMA REFERENCE  1x", fill="white")
                if frame in (0, len(reference) // 2, len(reference) - 1):
                    image.save(output.with_name(f"frame_{frame:05d}.png"))
                process.stdin.write(image.tobytes())
    except BaseException:
        process.stdin.close()
        process.terminate()
        process.wait()
        raise
    process.stdin.close()
    require(process.wait() == 0, "Native media encoder failed")
    hashes = subprocess.check_output([ffmpeg, "-nostdin", "-v", "error", "-i", str(output),
        "-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "framehash", "-hash", "sha256", "pipe:1"], text=True)
    return dict(status="PASSED", **identity(output), probe=decoded_metadata(hashes, len(reference)),
                original_speed=True, actual_frames_before_hold=case["attempted_steps"] + 1,
                failure_preserved=True, scene=identity(scene))
