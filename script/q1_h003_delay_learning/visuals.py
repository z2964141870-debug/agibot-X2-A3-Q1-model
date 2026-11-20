"""Six-panel training and closed-loop results; matched 1x simulation video."""

from contextlib import ExitStack
from pathlib import Path
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from protocol import ROOT, EVALUATION
from run import read

sys.path.insert(0, str(ROOT.parent))
from delivery import mujoco, font


def curves(result):
    fit = read(ROOT / "trained/fit.json")
    motion = read(ROOT / "motion/fit.json")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.2), layout="constrained")
    ax = axes[0, 0]
    h = fit["history"]
    for field, label in (("valid_static_rmse", "Static"), ("valid_relay_rmse", "Relay")):
        ax.plot([x["epoch"] for x in h], [x[field] for x in h], label=label)
    ax.axvline(fit["epoch"], color="#777", ls="--", lw=1)
    ax.legend()
    ax.set(title="Next target on validation data", xlabel="Training epoch", ylabel="RMSE (rad)")
    ax = axes[0, 1]
    h = motion["history"]
    for key, label in (("pop_mean", "Population mean"), ("validation", "Separate starts")):
        ax.plot([x["generation"] for x in h], [x[key] for x in h], label=label)
    ax.axhline(motion["base_on_validation"], color="#777", ls="--")
    ax.legend(fontsize=8)
    ax.set(
        title="Delayed rise motion search", xlabel="Population generation", ylabel="Screening score"
    )
    arms = result["arms"]

    def bars(ax, labels, vals, denom, title):
        colors = ["#55616d", "#c49a4e", "#268593", "#7b5698"]
        for j, (name, color) in enumerate(
            zip(("Published", "Linear", "Predictor", "Refit motion"), colors)
        ):
            ax.bar(
                np.arange(len(labels)) + (j - 1.5) * 0.2,
                [v[j] for v in vals],
                0.195,
                label=name,
                color=color,
            )
        ax.set_xticks(range(len(labels)), labels, rotation=12)
        ax.set(title=title, ylabel=f"Passed / {denom}", ylim=(0, denom * 1.08))

    for idx, ax, title in (
        (0, axes[1, 0], "Static recovery, two unseen seeds"),
        (1, axes[1, 1], "Walking relay, two unseen seeds"),
    ):
        bars(
            ax,
            ["0 ms", "20 ms", "Low friction", "Payload", "Combined"],
            [[result["primary"][s][a][idx] for a in arms] for s in EVALUATION],
            (48, 20)[idx],
            title,
        )
    for kind, ax, denom in (("static", axes[0, 2], 24), ("relay", axes[1, 2], 10)):
        data = result["per_split"]["test2"]
        bars(
            ax,
            ["10 ms", "20 ms", "30 ms"],
            [[data[s][a][kind]["success"] for a in arms] for s in ("half", "lag", "long")],
            denom,
            f"Fixed 20 ms settings | {kind}",
        )
    axes[1, 0].legend(fontsize=7, ncol=2)
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(ROOT / "training_eval.png", dpi=145)
    fig.savefig(ROOT / "training_eval.pdf")
    plt.close(fig)


def movie(res):
    motion = res["arms"][-1]
    rows = read(ROOT / "runs/test0/lag" / motion / "relay/relay.json")
    clips = []
    for success in (True, False):
        eligible = [r for r in rows if r["fallen"] and r["success"] == success]
        if eligible:
            clips.append(eligible[0]["trial"])
    if not clips:
        return
    mp4 = ROOT / "delay_relay_1x.mp4"
    thumbs = []
    enc = subprocess.Popen(
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
            "960x508",
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
            str(mp4),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for i in clips:
            with ExitStack() as stack:
                sides = []
                for arm in ("published", motion):
                    path = ROOT / "runs/test0/lag" / arm / "relay"
                    d = stack.enter_context(np.load(path / f"relay_{i}.npz"))
                    m = mujoco.MjModel.from_xml_path(str(path / "xml/fall_hard.xml"))
                    for marker in ("target_pad", "target_ring"):
                        m.geom(marker).rgba[3] = 0.0
                    ren = stack.enter_context(mujoco.Renderer(m, height=366, width=480))
                    camera = mujoco.MjvCamera()
                    camera.distance = 1.85
                    camera.azimuth = 55.0
                    camera.elevation = -16.0
                    sides.append(
                        (
                            d,
                            m,
                            mujoco.MjData(m),
                            ren,
                            camera,
                            read(path / "relay.json")[i]["success"],
                        )
                    )
                kick = int(sides[0][0]["kicks"][0])
                start = kick - 16
                end = max(len(s[0]["trace"]) for s in sides)
                ticks = list(range(start, end, 2))
                for j, t in enumerate(ticks):
                    frame = Image.new("RGB", (960, 508), (23, 31, 39))
                    pen = ImageDraw.Draw(frame)
                    pen.text(
                        (12, 8),
                        f"20 ms | walking perturbation {i} | after push {(t - kick) * 0.02:+.2f} s",
                        font=font(20),
                        fill="white",
                    )
                    pen.text((945, 8), "1.0x", font=font(19), anchor="ra", fill="white")
                    for side, (d, m, data, ren, cam, passed) in enumerate(sides):
                        left = 480 * side
                        pen.rectangle((left, 40, left + 479, 73), fill=("#475969", "#6c4c82")[side])
                        pen.text(
                            (left + 12, 45),
                            ("Published", "Delay-trained rise motion")[side],
                            font=font(18),
                            fill="white",
                        )
                        if t >= len(d["trace"]):
                            pen.text(
                                (left + 120, 228),
                                "Trial ended: " + ("pass" if passed else "fail"),
                                font=font(22),
                                fill="white",
                            )
                            continue
                        pose = d["qpos"][t]
                        data.qpos[:] = pose
                        mujoco.mj_forward(m, data)
                        cam.lookat[:] = [*pose[:2], 0.30]
                        ren.update_scene(data, camera=cam)
                        frame.paste(Image.fromarray(ren.render()), (left, 74))
                        z, up, _, _, _, g, *_ = d["trace"][t]
                        pen.text(
                            (left + 12, 454),
                            f"Height {z:.2f} m   Upright {up:+.2f}   Recovery {g:.2f}",
                            font=font(16),
                            fill="white",
                        )
                    enc.stdin.write(frame.tobytes())
                    if j in (0, len(ticks) // 2, len(ticks) - 1):
                        thumbs.append(frame.resize((480, 254)))
    finally:
        enc.stdin.close()
        if enc.wait():
            raise RuntimeError("encoder failed")
    sheet = Image.new("RGB", (480 * len(clips), 762))
    for i, img in enumerate(thumbs):
        sheet.paste(img, (480 * (i // 3), 254 * (i % 3)))
    sheet.save(ROOT / "video_qc.png")
    import json

    (ROOT / "video_cases.json").write_text(json.dumps(clips))


if __name__ == "__main__":
    r = read(ROOT / "results.json")
    curves(r)
    movie(r)
