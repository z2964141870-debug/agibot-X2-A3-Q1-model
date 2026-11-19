"""Render H002-SONIC v3 Publication 6-Panel Chart, 6-Frame 3D Filmstrip PNG, and 2x2 MuJoCo Comparison Video."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/yu/projects/SuperMimicHP")
Q1_XML = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
BASE = ROOT / "runs/h002_sonic_v3"


def _render_frame(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer, qpos: np.ndarray) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(qpos[0]), float(qpos[1]), 0.42]
    cam.distance = 1.68
    cam.azimuth = 136.0
    cam.elevation = -16.0
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def _annotate(img: np.ndarray, title: str, subtitle: str = "", bar_color: tuple[int, int, int] = (24, 52, 70)) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    W, H = im.size
    draw.rectangle([0, 0, W, 36], fill=bar_color)
    draw.text((10, 8), title, fill=(255, 255, 255))
    if subtitle:
        draw.rectangle([0, H - 24, W, H], fill=(18, 22, 28))
        draw.text((10, H - 18), subtitle, fill=(225, 238, 248))
    return np.asarray(im)


def build_filmstrip_and_video() -> None:
    model = mujoco.MjModel.from_xml_path(str(Q1_XML))
    data = mujoco.MjData(model)
    m = json.loads((BASE / "v3_sonic_metrics.json").read_text(encoding="utf-8"))
    ms_traj = np.load(BASE / "sonic_v3_modeswitch_30s_qpos.npz")["qpos"]
    h2h = np.load(BASE / "v2_vs_v3_push_qpos.npz")
    v2_q, v3_q = h2h["v2_qpos"], h2h["v3_qpos"]
    h2h_m = m["head_to_head_push70_delay40ms"]
    v2_ok, v2_st, v2_bg = h2h_m["V2_RobotPose_ONNX"]["completed"], h2h_m["V2_RobotPose_ONNX"]["steps_run"], h2h_m["V2_RobotPose_ONNX"]["budget"]
    v3_ok, v3_st, v3_bg = h2h_m["V3_RobotPose_ONNX"]["completed"], h2h_m["V3_RobotPose_ONNX"]["steps_run"], h2h_m["V3_RobotPose_ONNX"]["budget"]

    bvh_npz = np.load(BASE / "v3_sample_bvh_qpos.npz")
    bvh_q = bvh_npz["qpos"]
    bvh_name = str(bvh_npz["name"])

    r_strip = mujoco.Renderer(model, height=360, width=480)
    film_specs = [
        ("1. V3 RobotPose +0.65m/s Shove Recovery", "30s Demo t=3.4s (+20ms Bus Delay)", ms_traj, 170, (38, 70, 83)),
        ("2. V3 Zero-IK Raw Human BVH", f"{bvh_name[:28]} t=3.0s", bvh_q, min(150, len(bvh_q) - 1), (109, 40, 217)),
        ("3. V3 VR 3-Pt -0.60m/s Lateral Shove", "30s Demo t=15.4s (Sparse Head+Wrists)", ms_traj, 770, (21, 101, 192)),
        ("4. V3 Gamepad PlannerCmd Turn+Squat", "30s Demo t=21.6s (100D Joystick Cmd)", ms_traj, 1080, (42, 157, 143)),
        ("5. 400ms Dropout -> Watchdog Takeover", "30s Demo t=26.9s (Seamless Crossfade)", ms_traj, 1345, (180, 83, 9)),
        (f"6. H2H 0.70m/s Push+40ms ({h2h_m['clip_alias'][:18]})", f"v3: {v3_st}/{v3_bg} | v2: {v2_st}/{v2_bg}", v3_q, min(180, len(v3_q) - 1), (15, 118, 110)),
    ]
    panels = []
    for title, sub, arr, idx, col in film_specs:
        i = min(idx, len(arr) - 1)
        panels.append(_annotate(_render_frame(model, data, r_strip, arr[i]), title, sub, bar_color=col))
    strip = np.concatenate((np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)), axis=0)
    strip_path = BASE / "q1_v3_push_multimodal_filmstrip.png"
    Image.fromarray(strip).save(strip_path, optimize=True)
    r_strip.close()
    print(f"SAVED_FILMSTRIP -> {strip_path} ({strip_path.stat().st_size} B)", flush=True)

    r_vid = mujoco.Renderer(model, height=320, width=424)
    vid_dir = Path("/tmp/v3_vid_frames")
    shutil.rmtree(vid_dir, ignore_errors=True)
    vid_dir.mkdir(parents=True)

    n_frames = 210
    for k in range(n_frames):
        i_v2 = min(int(k * len(v2_q) / n_frames), len(v2_q) - 1) if not v2_ok else min(k, len(v2_q) - 1)
        if v2_ok:
            i_v2 = min(int(k * (len(v2_q) - 1) / (n_frames - 1)), len(v2_q) - 1)
        else:
            i_v2 = min(k, len(v2_q) - 1)
        i_v3 = min(int(k * (len(v3_q) - 1) / (n_frames - 1)), len(v3_q) - 1)
        i_bvh = min(int(k * (len(bvh_q) - 1) / (n_frames - 1)), len(bvh_q) - 1)
        i_ms = min(int(k * (len(ms_traj) - 1) / (n_frames - 1)), len(ms_traj) - 1)

        s_v2 = "PUSHED @1.2s | " + ("PASS" if v2_ok else ("FELL!" if i_v2 >= len(v2_q) - 1 else f"t={i_v2*0.02:.2f}s"))
        s_v3 = f"PUSHED @1.2s (0.70m/s+40ms) | t={i_v3*0.02:.2f}s" + (" PASS" if v3_ok else f" {v3_st}/{v3_bg}")
        p_tl = _annotate(_render_frame(model, data, r_vid, v2_q[i_v2]), "v2 SONIC (200-clip): 0.70m/s Push+40ms", s_v2, (180, 45, 45) if not v2_ok else (70, 90, 110))
        p_tr = _annotate(_render_frame(model, data, r_vid, v3_q[i_v3]), "v3 SONIC (1,727-clip Hardened): Same Push", s_v3, (15, 118, 110))
        p_bl = _annotate(_render_frame(model, data, r_vid, bvh_q[i_bvh]), f"v3 Zero-IK Raw BVH ({bvh_name[:18]})", f"t={i_bvh*0.02:.2f}s | HumanSMPL 430D", (88, 28, 135))

        sec_ms = i_ms * 0.02
        if sec_ms < 6.0:
            ph = "RobotPose + 0.65m/s Shove"
        elif sec_ms < 12.0:
            ph = "Zero-IK HumanSMPL (BVH)"
        elif sec_ms < 18.0:
            ph = "VR 3-Pt - 0.60m/s Shove"
        elif sec_ms < 24.0:
            ph = "PlannerCmd Turn+Squat"
        elif sec_ms < 26.5:
            ph = "RobotPose Pre-Dropout"
        elif sec_ms < 26.9:
            ph = "400ms Dropout -> Watchdog"
        else:
            ph = "PlannerCmd Safe Recover"
        p_br = _annotate(_render_frame(model, data, r_vid, ms_traj[i_ms]), "v3 30s 4-Modality+2xPush+Watchdog", f"t={sec_ms:.1f}s | {ph}", (29, 53, 87))
        grid = np.concatenate((np.concatenate((p_tl, p_tr), axis=1), np.concatenate((p_bl, p_br), axis=1)), axis=0)
        Image.fromarray(grid).save(vid_dir / f"frame_{k:04d}.png")
    r_vid.close()

    mp4_path = BASE / "q1_v3_push_delay_demo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "25", "-i", str(vid_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(mp4_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(vid_dir, ignore_errors=True)
    print(f"SAVED_MP4 -> {mp4_path} ({mp4_path.stat().st_size} B)", flush=True)


def build_6panel_chart() -> None:
    m = json.loads((BASE / "v3_sonic_metrics.json").read_text(encoding="utf-8"))
    onnx_v = json.loads((BASE / "onnx_v3_verify.json").read_text(encoding="utf-8"))
    ms = json.loads((BASE / "sonic_v3_modeswitch_30s.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 8.6, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.8), dpi=160)

    ho_nom = m["held_out_200_nominal"]
    ho_str = m["held_out_200_hardened_stress"]

    # Panel A: Multi-Dataset Scaling (12 -> 200 -> 1,727 Train Clips)
    ax = axes[0, 0]
    scales = ["v1 H002\n(12 clips)", "v2 SONIC\n(200 clips)", "v3 SONIC\n(1,727 clips)"]
    x = np.arange(len(scales))
    w = 0.25
    nom_200 = [
        ho_nom["V1_H002_12clip"]["all"]["survival_rate"] * 100,
        ho_nom["V2_RobotPose_200clip"]["all"]["survival_rate"] * 100,
        ho_nom["V3_RobotPose"]["all"]["survival_rate"] * 100,
    ]
    str_200 = [
        ho_str["V1_H002_12clip"]["all"]["survival_rate"] * 100,
        ho_str["V2_RobotPose_200clip"]["all"]["survival_rate"] * 100,
        ho_str["V3_RobotPose"]["all"]["survival_rate"] * 100,
    ]
    bvh_v2_s = m["zero_ik_raw_bvh_24"]["V2_HumanSMPL_ONNX"]["survival_rate"] * 100
    bvh_v3_s = m["zero_ik_raw_bvh_24"]["V3_HumanSMPL_ONNX"]["survival_rate"] * 100
    bvh_24 = [0.0, bvh_v2_s, bvh_v3_s]

    r1 = ax.bar(x - w, nom_200, w, label="200-HO Nominal", color="#1d3557")
    r2 = ax.bar(x, str_200, w, label="200-HO Hardened Push+Delay", color="#e76f51")
    r3 = ax.bar(x + w, bvh_24, w, label="24 Raw BVH Zero-IK", color="#2a9d8f")
    for bars, skip0 in ((r1, False), (r2, False), (r3, True)):
        for idx, bar in enumerate(bars):
            v = bar.get_height()
            lbl = "N/A" if (skip0 and idx == 0) else f"{v:.1f}%"
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.4, lbl, ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x); ax.set_xticklabels(scales, fontsize=8.0)
    ax.set_ylim(0, 116); ax.set_ylabel("Survival Rate (%)")
    ax.set_title("A. Dataset Scaling: 12 -> 200 -> 1,727 Train Clips", fontweight="bold")
    ax.legend(fontsize=7.0, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: 200-Clip Held-Out All Modalities (Nominal vs Hardened Stress)
    ax = axes[0, 1]
    specs_b = [
        ("Zeroed_0ms", "Zeroed"),
        ("V1_H002_12clip", "v1\nRobot"),
        ("V2_RobotPose_200clip", "v2\nRobot"),
        ("V3_RobotPose", "v3\nRobot"),
        ("V3_HumanSMPL", "v3\nSMPL"),
        ("V3_VR3Pt", "v3\nVR3Pt"),
        ("V3_PlannerCmd", "v3\nPlan"),
        ("Oracle_90ms", "Oracle\n90ms"),
    ]
    xb = np.arange(len(specs_b))
    wb = 0.36
    vb_nom = [ho_nom[k]["all"]["survival_rate"] * 100 for k, _ in specs_b]
    vb_str = [ho_str[k]["all"]["survival_rate"] * 100 if k in ho_str else np.nan for k, _ in specs_b]
    p1 = ax.bar(xb - wb / 2, vb_nom, wb, label="200-HO Nominal (%)", color="#264653")
    p2 = ax.bar(xb + wb / 2, np.nan_to_num(vb_str, nan=0.0), wb, label="200-HO 0.5x Fric+Push+20ms", color="#e63946")
    for bar, v in zip(p1, vb_nom):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6)
    for bar, v in zip(p2, vb_str):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.6, color="#9b2226")
    ax.set_xticks(xb); ax.set_xticklabels([l for _, l in specs_b], fontsize=7.2)
    ax.set_ylim(0, 116); ax.set_ylabel("Survival Rate (%)")
    ax.set_title("B. 200 Unseen Held-Out: All 4 SONIC v3 Modalities", fontweight="bold")
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel C: Push-Impulse Sweep (0 .. 0.95 m/s)
    ax = axes[0, 2]
    ps = m["push_impulse_sweep_50"]
    mags = [0.0, 0.35, 0.55, 0.75, 0.95]
    styles = [
        ("V3_RobotPose", "#0f766e", "-", "o", "v3 RobotPose (Hardened)"),
        ("V3_HumanSMPL", "#7c3aed", "-", "s", "v3 HumanSMPL (Zero-IK)"),
        ("V2_RobotPose", "#264653", "--", "^", "v2 RobotPose (200-clip)"),
        ("V1_H002_12clip", "#f4a261", "-.", "d", "v1 H002 (12-clip)"),
        ("Zeroed_0ms", "#9ca3af", ":", "x", "Zeroed Baseline"),
    ]
    for k, col, ls, mk, lbl in styles:
        ys = [ps[k][f"{dv:.2f}"]["survival_rate"] * 100 for dv in mags]
        ax.plot(mags, ys, color=col, linestyle=ls, marker=mk, linewidth=2.0, markersize=5.5, label=lbl)
    ax.set_xticks(mags); ax.set_ylim(0, 108)
    ax.set_xlabel("Bidirectional Lateral Shove Magnitude (m/s)"); ax.set_ylabel("50-Clip Survival (%)")
    ax.set_title("C. Push-Impulse Survival Curve (0.0 -> 0.95 m/s)", fontweight="bold")
    ax.legend(fontsize=7.0, loc="lower left", framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel D: Actuator-Delay Sweep (0 .. 60 ms + 0.35 m/s push)
    ax = axes[1, 0]
    ds = m["actuator_delay_sweep_50"]
    d_ms = [0, 20, 40, 60]
    for k, col, ls, mk, lbl in styles:
        ys = [ds[k][f"{d}ms"]["survival_rate"] * 100 for d in d_ms]
        ax.plot(d_ms, ys, color=col, linestyle=ls, marker=mk, linewidth=2.0, markersize=5.5, label=lbl)
    ax.set_xticks(d_ms); ax.set_ylim(0, 108)
    ax.set_xlabel("Actuator Bus Delay (ms) + 0.35m/s Shove"); ax.set_ylabel("50-Clip Survival (%)")
    ax.set_title("D. Actuator-Delay Robustness Curve (0 -> 60 ms)", fontweight="bold")
    ax.legend(fontsize=7.0, loc="lower left", framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel E: Per-Dataset Breakdown + ONNX Latency Inset
    ax = axes[1, 1]
    srcs = ["lafan1", "soma", "phuma", "synthetic"]
    slabs = ["LAFAN1\n(24)", "SOMA\n(51)", "PHUMA\n(114)", "Synth\n(11)"]
    xs = np.arange(len(srcs))
    ws = 0.25
    v2s = [ho_nom["V2_RobotPose_200clip"]["by_source"][s]["survival_rate"] * 100 for s in srcs]
    v3_nom_s = [ho_nom["V3_RobotPose"]["by_source"][s]["survival_rate"] * 100 for s in srcs]
    v3_str_s = [ho_str["V3_RobotPose"]["by_source"][s]["survival_rate"] * 100 for s in srcs]
    e1 = ax.bar(xs - ws, v2s, ws, label="v2 RobotPose Nom", color="#64748b")
    e2 = ax.bar(xs, v3_nom_s, ws, label="v3 RobotPose Nom", color="#0f766e")
    e3 = ax.bar(xs + ws, v3_str_s, ws, label="v3 Hardened Stress", color="#e76f51")
    for bars in (e1, e2, e3):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.4)
    p50_rob = onnx_v["models"]["robot_pose"]["cpu_cgfa_tok_head_p50_ms"]
    p99_rob = onnx_v["models"]["robot_pose"]["cpu_cgfa_tok_head_p99_ms"]
    p50_smpl = onnx_v["models"]["human_smpl"]["cpu_cgfa_tok_head_p50_ms"]
    ax.text(
        0.02, 0.04,
        f"1-Thread CPU ORT (50Hz=20ms):\nRobotPose p50={p50_rob:.2f}ms (p99={p99_rob:.2f}ms)\nHumanSMPL p50={p50_smpl:.2f}ms",
        transform=ax.transAxes, fontsize=6.8, bbox=dict(boxstyle="round,pad=0.32", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
    )
    ax.set_xticks(xs); ax.set_xticklabels(slabs, fontsize=7.8)
    ax.set_ylim(0, 132); ax.set_ylabel("Survival Rate (%)")
    ax.set_title("E. Per-Dataset Generalization (v2 vs v3) & CPU Latency", fontweight="bold")
    ax.legend(fontsize=6.8, loc="upper center", ncol=3, framealpha=0.94)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel F: 30s Continuous 4-Modality Mode-Switch + Shoves + 20ms Delay + Watchdog
    ax = axes[1, 2]
    tr = ms["trace_sampled"]
    ts = [r["t_s"] for r in tr]
    pzs = [r["pelvis_z_m"] for r in tr]
    spds = [r["speed_mps"] for r in tr]
    ax_v = ax.twinx()
    ax.plot(ts, pzs, color="#1d3557", linewidth=1.8, label="Pelvis Z (m)")
    ax_v.plot(ts, spds, color="#e76f51", alpha=0.72, linewidth=1.1, label="XY Speed (m/s)")
    ax.axvspan(26.5, 26.9, color="#fbbf24", alpha=0.32, label="400ms Dropout")
    for tx, txt in [(3.0, "Push+0.65"), (15.0, "Push-0.60")]:
        ax.axvline(tx, color="#dc2626", linestyle="--", linewidth=1.1)
        ax.text(tx + 0.3, 0.495, txt, color="#b91c1c", fontsize=6.6, fontweight="bold")
    for xv, txt in [(0.2, "A:Rob"), (6.2, "B:SMPL"), (12.2, "C:VR"), (18.2, "D:Plan"), (24.2, "E:Rob"), (27.1, "G:Rec")]:
        ax.axvline(xv - 0.2, color="#94a3b8", linestyle=":", linewidth=0.8)
        ax.text(xv, 0.595, txt, fontsize=6.4, fontweight="bold", color="#0f172a", va="top")
    ax.set_xlim(0, 30.0); ax.set_ylim(0.18, 0.61); ax_v.set_ylim(0.0, 2.3)
    ax.set_xlabel("Continuous Episode Time (s) | 20ms Bus Delay Throughout"); ax.set_ylabel("Pelvis Z (m)", color="#1d3557")
    ax_v.set_ylabel("Root Speed (m/s)", color="#c0392b")
    s30 = "30/30s OK" if ms["completed_30s"] else f"{ms['steps_survived']}/{ms['budget']}"
    ax.set_title(f"F. 30s 4-Modality + 2xPush + Watchdog ({s30}, OOB={ms['pd_oob_count']})", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax_v.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="upper right", bbox_to_anchor=(0.99, 0.88), framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("H002-SONIC v3: 1,727-Clip Multi-Dataset Scaling, 4,096-Env Push/Delay RL & Full 200-Clip Sim Verification", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    out_png = BASE / "h002_v3_scaling_push_delay.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"SAVED_CHART -> {out_png} ({out_png.stat().st_size} B)", flush=True)


if __name__ == "__main__":
    build_filmstrip_and_video()
    build_6panel_chart()
