"""Plot complete fits and render one paired gain and loss on the second holdout."""

import json
import sys
import subprocess
from pathlib import Path
from contextlib import ExitStack

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from delivery import mujoco, font


def read(path):
    return json.loads((ROOT / path).read_text())


def plot():
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained")
    labels = ["low", "nominal", "high", "payload", "delay", "combined"]
    for row, (fit, test, arms, title) in enumerate(
        [
            (
                "selection.json",
                "results.json",
                ["published", "selected", "bias_only"],
                "Early adaptation",
            ),
            (
                "selection_late.json",
                "results_late.json",
                ["published", "late_selected", "late_bias"],
                "Late adaptation",
            ),
        ]
    ):
        f = read(fit)
        res = read(test)
        ax = axes[row, 0]
        names = [k for k in f["scores"] if k != "published"]
        values = [f["scores"][k]["utility"] for k in names]
        ax.scatter(range(1, len(values) + 1), values, color="#377d87", s=21)
        ax.axhline(f["scores"]["published"]["utility"], color="#565656", ls="--", label="Published")
        j = names.index(f["selected"])
        ax.scatter(j + 1, values[j], color="#c46a35", marker="*", s=115, label="Selected", zorder=3)
        ax.set(xlabel="Candidate evaluated", ylabel="Training score", title=f"{title} | fit")
        ax.legend(fontsize=8)
        for col, (key, denom) in enumerate([("static", 24), ("relay", 5)], start=1):
            ax = axes[row, col]
            for j, (arm, label, color) in enumerate(
                zip(arms, ["Published", "Feedback", "Bias only"], ["#576778", "#29836e", "#d49551"])
            ):
                ax.bar(
                    np.arange(6) + (j - 1) * 0.25,
                    [res["conditions"][s][arm][key] for s in labels],
                    0.24,
                    label=label,
                    color=color,
                )
            ax.set_xticks(range(6), labels, rotation=23, ha="right")
            ax.set(
                ylim=(0, denom + 1), ylabel=f"Passed / {denom}", title=f"{title} | {key} holdout"
            )
            ax.legend(fontsize=7, ncol=3)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(ROOT / "training_eval.png", dpi=145)
    fig.savefig(ROOT / "training_eval.pdf")
    plt.close(fig)


def video():
    res = read("results_late.json")["conditions"]
    examples = []
    for outcome in ("gains", "losses"):
        for surface in ("low", "high", "nominal", "payload", "delay", "combined"):
            ids = res[surface]["late_selected"]["static_pairs"][outcome]
            if ids:
                examples.append((surface, ids[0], outcome))
                break
    if len(examples) < 2:
        examples.append(("low", 0, "fixed example"))
    encoder = subprocess.Popen(
        [
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
            "960x512",
            "-r",
            "25",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(ROOT / "late_comparison_1x.mp4"),
        ],
        stdin=subprocess.PIPE,
    )
    thumbs = []
    try:
        for surface, idx, label in examples:
            with ExitStack() as stack:
                arms = []
                for arm in ("published", "late_selected"):
                    folder = ROOT / "runs/test_late" / arm / surface
                    d = stack.enter_context(np.load(folder / "static" / f"trial_{idx:02}.npz"))
                    m = mujoco.MjModel.from_xml_path(
                        str(folder / "static_scene/scene/fall_hard.xml")
                    )
                    ren = stack.enter_context(mujoco.Renderer(m, height=390, width=480))
                    cam = mujoco.MjvCamera()
                    cam.distance = 1.8
                    cam.elevation = -16
                    cam.azimuth = 55.0
                    arms.append((d, m, mujoco.MjData(m), ren, cam))
                for i in range(0, 300, 2):
                    canvas = Image.new("RGB", (960, 512), (22, 31, 40))
                    pen = ImageDraw.Draw(canvas)
                    pen.text(
                        (12, 7),
                        f"{surface}  |  static trial {idx}  |  {label}",
                        font=font(20),
                        fill="white",
                    )
                    pen.text(
                        (945, 8),
                        f"1.0x   {(i + 1) * 0.02:.2f} s",
                        anchor="ra",
                        font=font(18),
                        fill="white",
                    )
                    for j, (d, m, data, ren, cam) in enumerate(arms):
                        left = j * 480
                        pose = d["qpos"][i]
                        data.qpos[:] = pose
                        mujoco.mj_forward(m, data)
                        cam.lookat[:] = [*pose[:2], 0.25]
                        ren.update_scene(data, camera=cam)
                        canvas.paste(Image.fromarray(ren.render()), (left, 75))
                        pen.rectangle((left, 40, left + 479, 74), fill=("#495969", "#296d63")[j])
                        pen.text(
                            (left + 12, 46),
                            ("Published", "Late feedback")[j],
                            font=font(18),
                            fill="white",
                        )
                        z, up, *_ = d["trace"][i]
                        pen.text(
                            (left + 12, 475),
                            f"Pelvis {z:.2f} m     Upright cosine {up:+.2f}",
                            font=font(17),
                            fill="white",
                        )
                    encoder.stdin.write(canvas.tobytes())
                    if i in (0, 150, 298):
                        thumbs.append(canvas.resize((480, 256)))
    finally:
        encoder.stdin.close()
        if encoder.wait():
            raise RuntimeError("video encoding failed")
    sheet = Image.new("RGB", (480 * len(examples), 768))
    for i, im in enumerate(thumbs):
        sheet.paste(im, (480 * (i // 3), 256 * (i % 3)))
    sheet.save(ROOT / "comparison_qc.png")
    (ROOT / "video_cases.json").write_text(json.dumps(examples))


if __name__ == "__main__":
    plot()
    video()
