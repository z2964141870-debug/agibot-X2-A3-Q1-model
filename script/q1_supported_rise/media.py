"""Create a real-time video and a six-panel figure from saved experiments."""

import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
PANELS = [
    ("matched/77331/formal", "Frozen recovery", "Ground contact only"),
    ("matched/77331/fast_catch", "Earlier autonomous stand", "Ground contact only"),
    (
        "limbs/77331/fast_catch",
        "Same earlier stand",
        "96 arm-leg collision pairs enabled",
    ),
    (
        "limbs/77331/paced",
        "Measured joint reference",
        "96 arm-leg collision pairs enabled",
    ),
]


def video(phase=False, approach=False):
    trial = 3 if phase else 8
    panels = PANELS
    title = "Supine #08   |   Same sampled fall"
    output = "Q1_supine_1x.mp4"
    if phase:
        panels = [
            (
                "phase_repeat/base",
                "Original balance transition",
                "Prone record at 6.80 s",
            ),
            ("phase_repeat/low", "Crouch balance transition", "Prone record at 6.80 s"),
            (
                "phase_repeat/base",
                "Original balance transition",
                "Supine record at 6.80 s",
            ),
            (
                "phase_repeat/low",
                "Crouch balance transition",
                "Supine record at 6.80 s",
            ),
        ]
        title = "Balance from recorded crouches   |   Seed 92041"
        output = "Q1_crouch_1x.mp4"
    if approach:
        trial = 0
        title = "Prone #00   |   Arm-leg collision enabled"
        output = "Q1_prone_supported_1x.mp4"
        panels = [
            (
                "limbs/77331/fast_catch",
                "Fast rise with earlier stand",
                "96 arm-leg pairs enabled",
            ),
            (
                "limbs/77331/paced",
                "Measured joint reference",
                "96 arm-leg pairs enabled",
            ),
            (
                "toward_eval/77331",
                "Optimized ground support",
                "96 arm-leg pairs enabled",
            ),
            (
                "transfer_eval/77331",
                "Optimized support + transfer",
                "96 arm-leg pairs enabled",
            ),
        ]
    paths = [ROOT / p if phase else ROOT / p / "static" for p, _, _ in panels]
    frames, owners, labels = [], [], []
    for i, path in enumerate(paths):
        if phase:
            trial = 3 if i < 2 else 15
        with np.load(path / f"trial_{trial:02d}.npz") as f:
            frames.append(np.vstack((f["initial"], f["qpos"])))
            owners.append(f["trace"][:, 4])
        if phase:
            labels.append(
                json.loads((path / "standing_phase.json").read_text())[trial][
                    "stood_and_handed_back"
                ]
            )
        else:
            labels.append(
                json.loads((path / "results.json").read_text())[trial]["success"]
            )
    for a, b in ((0, 1), (2, 3)):
        np.testing.assert_array_equal(frames[a][0], frames[b][0])
    if approach:
        for seq in frames[1:]:
            np.testing.assert_array_equal(frames[0][0], seq[0])
    m = mj.MjModel.from_xml_path(str(paths[0] / "scene/fall_hard.xml"))
    for marker in ("target_pad", "target_ring"):
        m.geom(marker).rgba[3] = 0
    d = mj.MjData(m)
    camera = mj.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.95, 63.0, -11.0
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    small = ImageFont.truetype("DejaVuSans.ttf", 14)
    cmd = [
        "ffmpeg",
        "-n",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "896x604",
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
        str(ROOT / output),
    ]
    with (
        mj.Renderer(m, height=222, width=448) as renderer,
        subprocess.Popen(cmd, stdin=subprocess.PIPE) as movie,
    ):
        try:
            for tick in range(0, max(map(len, frames)), 2):
                image = Image.new("RGB", (896, 604), "#15202b")
                pen = ImageDraw.Draw(image)
                pen.text(
                    (10, 7),
                    f"Q1 simulation   |   {title}   |   1x   |   {tick * 0.02:.2f} s",
                    font=font,
                )
                for i, (_, label, conditions) in enumerate(panels):
                    x, y = (i % 2) * 448, 40 + (i // 2) * 280
                    seq = frames[i]
                    d.qpos[:] = seq[min(tick, len(seq) - 1)]
                    mj.mj_forward(m, d)
                    camera.lookat[:] = [*d.qpos[:2], 0.34]
                    renderer.update_scene(d, camera=camera)
                    image.paste(Image.fromarray(renderer.render()), (x, y + 47))
                    pen.text((x + 8, y), label, font=font)
                    owner = owners[i][min(max(tick - 1, 0), len(owners[i]) - 1)]
                    state = "teleop" if not owner else "recovery"
                    if tick >= len(seq):
                        state = "end: " + ("passed" if labels[i] else "failed")
                    pen.text(
                        (x + 8, y + 25),
                        f"{conditions}  |  {state}",
                        font=small,
                        fill="#b8dbe2",
                    )
                movie.stdin.write(image.tobytes())
        finally:
            movie.stdin.close()
        if movie.wait():
            raise RuntimeError("ffmpeg failed")


def figure(latest=False):
    curves = [
        ("fast", "Fast rise"),
        ("landing", "Support objective"),
        ("clear", "With arm-leg contacts"),
    ]
    if latest:
        curves = [
            ("clear", "Arm-leg contacts"),
            ("toward", "Ground to crouch"),
            ("transfer", "Crouch to stand"),
        ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), layout="constrained")
    for col, (run, title) in enumerate(curves):
        data = [
            json.loads(t) for t in (ROOT / (run + ".jsonl")).read_text().splitlines()
        ]
        a = axes[0, col]
        for k, label in (
            ("train", "Train best"),
            ("validation", "Selection"),
            ("best", "Best retained"),
        ):
            a.plot([row[k] for row in data], label=label, lw=1.5)
        a.set(title=title, xlabel="CEM generation", ylabel="Search objective")
        a.legend(fontsize=8)
        a = axes[1, col]
        a.plot([row["speed"] for row in data], label="Speed (rad/s)", color="#ab520c")
        a.plot([row["depth_mm"] for row in data], label="Overlap (mm)", color="#13685f")
        a.set(xlabel="CEM generation", ylabel="Selection-set peak")
        a.legend(fontsize=8)
    fig.suptitle("Q1 trajectory search    |    distinct objectives across columns")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.savefig(
        ROOT / ("Q1_support_training.png" if latest else "Q1_training.png"), dpi=145
    )
    plt.close(fig)


if __name__ == "__main__":
    if sys.argv[1] in ("video", "phase", "approach"):
        video(sys.argv[1] == "phase", sys.argv[1] == "approach")
    else:
        figure(sys.argv[1] == "support")
