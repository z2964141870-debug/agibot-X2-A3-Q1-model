"""Original-speed actual/reference video from saved physical trajectories."""

from pathlib import Path
import json
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1"))
from common import identity, require


def render_case(xml, case, output):
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    require(ffmpeg and ffprobe, "Existing ffmpeg and ffprobe are required")
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
    probe = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(output)], text=True))["streams"][0]
    require((probe["width"], probe["height"], probe["r_frame_rate"], int(probe["nb_frames"])) ==
            (1280, 392, "50/1", len(reference)), "Video time/geometry differs from saved trajectory")
    subprocess.run([ffmpeg, "-v", "error", "-i", str(output), "-f", "null", "-"], check=True)
    return dict(status="PASSED", **identity(output), probe=probe, original_speed=True,
                actual_frames_before_hold=case["attempted_steps"] + 1, failure_preserved=True)
