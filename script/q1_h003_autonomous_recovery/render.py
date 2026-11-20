"""Show three independent falls and autonomous standing without an operator."""

from contextlib import ExitStack
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from evaluate import D, mujoco
from delivery import font


def main():
    src = D / "runs/test/251233/offline"
    with ExitStack() as stack:
        panels = []
        for i in range(3):
            tr = stack.enter_context(np.load(src / f"trial_{i:02d}.npz"))
            model = mujoco.MjModel.from_xml_path(str(src / "scene/fall_hard.xml"))
            for marker in ("target_pad", "target_ring"):
                model.geom(marker).rgba[3] = 0
            renderer = stack.enter_context(mujoco.Renderer(model, height=276, width=320))
            cam = mujoco.MjvCamera()
            cam.distance = 2.15
            cam.azimuth = 55.0
            cam.elevation = -15.0
            panels.append((tr, model, mujoco.MjData(model), renderer, cam))
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "960x372",
            "-r",
            "25",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(D / "recovery_offline_1x.mp4"),
        ]
        encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        thumbs = []
        try:
            for t in range(0, len(panels[0][0]["qpos"]), 2):
                frame = Image.new("RGB", (960, 372), (24, 32, 40))
                pen = ImageDraw.Draw(frame)
                pen.text(
                    (12, 8),
                    f"Q1 autonomous recovery  |  Operator disconnected  |  {t * 0.02:05.2f} s",
                    font=font(20),
                    fill="white",
                )
                pen.text((945, 8), "1.0x", font=font(18), anchor="ra", fill="white")
                for i, (d, m, state, ren, cam) in enumerate(panels):
                    state.qpos[:] = d["qpos"][t]
                    mujoco.mj_forward(m, state)
                    cam.lookat[:] = [*state.qpos[:2], 0.31]
                    ren.update_scene(state, camera=cam)
                    frame.paste(Image.fromarray(ren.render()), (i * 320, 41))
                    pen.text(
                        (i * 320 + 9, 322),
                        ("Prone", "Supine", "Side")[i] + "  |  Recovery owns control",
                        font=font(15),
                        fill="white",
                    )
                    pen.text(
                        (i * 320 + 9, 345),
                        "Stand confirmed" if d["trace"][t, 8] else "Stabilizing / rising",
                        font=font(15),
                        fill="#9bd8c4",
                    )
                encoder.stdin.write(frame.tobytes())
                if t in (0, 250, 700):
                    thumbs.append(frame.resize((660, 256)))
        finally:
            encoder.stdin.close()
            if encoder.wait():
                raise RuntimeError("encoder failed")
        sheet = Image.new("RGB", (660, 256 * len(thumbs)))
        for i, im in enumerate(thumbs):
            sheet.paste(im, (0, 256 * i))
        sheet.save(D / "preview.png")


if __name__ == "__main__":
    main()
