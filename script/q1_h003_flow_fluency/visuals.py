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

from design import D as ROOT, destination
from assess import read

sys.path.insert(0, str(ROOT.parent))
from delivery import mujoco, font


def curves(result):
    fit = read(ROOT / "fit.json")
    selection = read(ROOT / "selection.json")
    a, b = selection["pace"], selection["gain"]
    arms = ["published", a, b, a + "+" + b]
    names = ["Published", "Pace", "Age", "Combined"]
    colors = ["#576872", "#bb7136", "#258578", "#6c5292"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6), layout="constrained")
    h = np.array(fit["history"])
    ax = axes[0, 0]
    ax.plot(h[:, 0], np.sqrt(h[:, 1]), label="Fit (weighted)", ls="--", color="#888")
    ax.plot(h[:, 0], h[:, 2], label="Static validation")
    ax.plot(h[:, 0], h[:, 3], label="Walking validation")
    ax.axvline(fit["epoch"], ls=":", color="black")
    ax.set(title="Command prediction during training", xlabel="Training epoch", ylabel="RMSE (rad)")
    ax.legend(fontsize=8)
    dev = selection["pace_dev"]
    ax = axes[0, 1]
    ax.bar(list(dev), [dev[k]["counts"]["pass5"] for k in dev], color="#bb7136")
    ax.tick_params(axis="x", labelrotation=18)
    ax.set(
        title="Pace selection on development cases", ylabel="5 s recoveries / 10", ylim=(0, 10.6)
    )
    ax = axes[0, 2]
    for i, arm in enumerate(arms):
        d = result["daily"][arm]
        ax.bar(
            np.arange(3) + (i - 1.5) * 0.19,
            [d["static"] / 48, d["counts"]["pass5"] / 40, d["repeat"]["pass5"] / 10],
            0.18,
            color=colors[i],
            label=names[i],
        )
    ax.set_xticks(range(3), ["Static (48)", "Single push (40)", "Two pushes (10)"])
    ax.set(title="Daily holdout", ylabel="Passed fraction", ylim=(0, 1.14))
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1, 0]
    keys = ["support_roll", "three_roll", "late_roll"]
    for i, arm in enumerate(arms):
        d = result["daily"][arm]["paired"]
        ax.bar(
            np.arange(3) + (i - 1.5) * 0.19,
            [100 * (d[k]["ratio"] - 1) for k in keys],
            0.18,
            color=colors[i],
        )
    ax.axhline(0, color="#424850", lw=0.8)
    ax.set_xticks(range(3), ["First support", "First 3 s", "3–5 s"])
    ax.set(title="Paired trunk roll peaks", ylabel="Median change vs published (%)")
    for col, kind in [(1, "static"), (2, "relay")]:
        ax = axes[1, col]
        scenes = list(result["stress"])
        for arm, c, name in zip(arms, colors, names):
            vals = [result["stress"][s][arm] for s in scenes]
            ax.plot(
                range(len(scenes)),
                [v["static"] if kind == "static" else v["counts"]["pass5"] for v in vals],
                marker="o",
                label=name,
                color=c,
            )
        ax.set_xticks(
            range(len(scenes)),
            ["10 ms", "20 ms", "30 ms", "Variable", "Slick", "Load", "Both"],
            rotation=25,
        )
        ax.set(
            title="Stress holdout | " + kind,
            ylabel="Passed / " + ("24" if kind == "static" else "10"),
        )
    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.18)
    fig.savefig(ROOT / "training_eval.png", dpi=145)
    fig.savefig(ROOT / "training_eval.pdf")
    plt.close(fig)


def movie(res):
    sel = read(ROOT / "selection.json")
    a, b = sel["pace"], sel["gain"]
    clips = []
    for kind in ("roll", "gain", "failure"):
        scenes = (
            ["daily"]
            if kind == "roll"
            else ["lag", "variable", "long", "half", "slick", "payload", "combined"]
        )
        for scene in scenes:
            arm = a if scene == "daily" else b
            base = read(destination("hold0", scene, "published", "relay") / "relay.json")
            cand = read(destination("hold0", scene, arm, "relay") / "relay.json")
            if kind == "roll":
                eligible = [
                    r["trial"]
                    for r in cand
                    if r["pass5"]
                    and base[r["trial"]]["pass5"]
                    and "support_roll" in r
                    and "support_roll" in base[r["trial"]]
                ]
                eligible.sort(key=lambda i: cand[i]["support_roll"] - base[i]["support_roll"])
            elif kind == "gain":
                eligible = [
                    r["trial"] for r in cand if r["pass5"] and not base[r["trial"]]["pass5"]
                ]
            else:
                eligible = [r["trial"] for r in cand if r["down"] and not r["pass5"]]
            if eligible:
                clips.append((scene, arm, eligible[0]))
                break
    mp4 = ROOT / "flow_1x.mp4"
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
        for scene, candidate, i in clips:
            with ExitStack() as stack:
                sides = []
                for arm in ("published", candidate):
                    path = destination("hold0", scene, arm, "relay")
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
                            read(path / "relay.json")[i]["pass5"],
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
                        f"{scene} | walking perturbation {i} | after push {(t - kick) * 0.02:+.2f} s",
                        font=font(20),
                        fill="white",
                    )
                    pen.text((945, 8), "1.0x", font=font(19), anchor="ra", fill="white")
                    for side, (d, m, data, ren, cam, passed) in enumerate(sides):
                        left = 480 * side
                        pen.rectangle((left, 40, left + 479, 73), fill=("#475969", "#6c4c82")[side])
                        pen.text(
                            (left + 12, 45),
                            (
                                "Published",
                                "Pace ramp" if scene == "daily" else "Age-aware prediction",
                            )[side],
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
