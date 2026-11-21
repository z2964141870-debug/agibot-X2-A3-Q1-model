"""Render the first prone case from four matched CPU physics runs at wall-clock speed."""
import argparse
from pathlib import Path
import subprocess

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANELS = [("native", "Formal / 5 ms", "direct target"),
          ("direct2", "Formal / 2 ms", "direct target"),
          ("formal500", "Formal / 2 ms", "500 Hz ramp + 11 ms state delay"),
          ("slow_v3", "Slow v3 / 2 ms", "500 Hz ramp + 11 ms state delay")]


def render(root, output):
    model = mujoco.MjModel.from_xml_path(str(root / "native/static/scene/fall_hard.xml"))
    for marker in ("target_pad", "target_ring"):
        model.geom(marker).rgba[3] = 0
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.7, 60., -11.
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    small = ImageFont.truetype("DejaVuSans.ttf", 14)
    states = []
    for name, _, _ in PANELS:
        with np.load(root / name / "static/trial_00.npz") as sample:
            states.append(np.vstack([sample["initial"], sample["qpos"]]))
    for seq in states[1:]:
        np.testing.assert_array_equal(seq[0], states[0][0])
    cmd = ["ffmpeg", "-n", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "896x584",
           "-r", "25", "-i", "-", "-c:v", "libx264", "-threads", "2", "-crf", "19",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as movie, mujoco.Renderer(model, height=222, width=448) as renderer:
        try:
            for tick in range(0, max(map(len, states)), 2):
                frame = Image.new("RGB", (896, 584), "#18212a")
                pen = ImageDraw.Draw(frame)
                pen.text((10, 7), f"Q1 simulation    |    Prone #00    |    1x    |    {tick * .02:.2f} s", font=font)
                for i, (_, name, detail) in enumerate(PANELS):
                    x, y = (i % 2) * 448, 38 + (i // 2) * 272
                    seq = states[i]
                    data.qpos[:] = seq[min(tick, len(seq)-1)]
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = [*data.qpos[:2], .24]
                    renderer.update_scene(data, camera=camera)
                    frame.paste(Image.fromarray(renderer.render()), (x, y+39))
                    pen.text((x+9, y), name, font=font)
                    label = detail if tick < len(seq) else "End of trial / frame held"
                    pen.text((x+9, y+20), label, font=small, fill="#c3dbdf")
                movie.stdin.write(frame.tobytes())
        finally:
            movie.stdin.close()
        if movie.wait():
            raise RuntimeError("video encoder failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    render(args.root.resolve(), args.output)
