"""Render high-impact 3D MuJoCo visuals, 4-panel v2 benchmark plot, and 2x2 multimodal demo video."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/yu/projects/SuperMimicHP")
Q1_XML = ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml"
BASE = ROOT / "runs/h002_sonic_v2"


def _render_frame(model: mujoco.MjModel, data: mujoco.MjData, renderer: mujoco.Renderer, qpos: np.ndarray, lookat_follow: bool = True) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    if lookat_follow:
        cam.lookat[:] = [float(qpos[0]), float(qpos[1]), 0.42]
    else:
        cam.lookat[:] = [0.0, 0.0, 0.42]
    cam.distance = 1.65
    cam.azimuth = 138.0
    cam.elevation = -16.0
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def _annotate(img: np.ndarray, title: str, subtitle: str = "", bar_color: tuple[int, int, int] = (24, 52, 70)) -> np.ndarray:
    im = Image.fromarray(img)
    draw = ImageDraw.Draw(im)
    W, H = im.size
    draw.rectangle([0, 0, W, 38], fill=bar_color)
    draw.text((10, 6), title, fill=(255, 255, 255))
    if subtitle:
        draw.rectangle([0, H - 24, W, H], fill=(20, 24, 30))
        draw.text((10, H - 19), subtitle, fill=(220, 235, 245))
    return np.asarray(im)


def build_filmstrip_and_video() -> None:
    model = mujoco.MjModel.from_xml_path(str(Q1_XML))
    data = mujoco.MjData(model)
    trajs = np.load(BASE / "v2_sample_trajs.npz")
    ms_traj = np.load(BASE / "sonic_modeswitch_24s.npz")["qpos"]
    ms_json = json.loads((BASE / "sonic_modeswitch_24s.json").read_text(encoding="utf-8"))
    metrics = json.loads((BASE / "v2_sonic_metrics.json").read_text(encoding="utf-8"))
    v1_sq = metrics["held_out_50_nominal"]["V1_H002_ONNX"]["per_clip"]["stat_fit_squat"]
    v2_sq = metrics["held_out_50_nominal"]["V2_RobotPose_ONNX"]["per_clip"]["stat_fit_squat"]
    v1_sq_stat = f"PASS {v1_sq['steps_run']}/{v1_sq['budget']}" if v1_sq["completed"] else f"FELL @{v1_sq['steps_run']}/{v1_sq['budget']} ({v1_sq['steps_run']*0.02:.2f}s)"
    v2_sq_stat = f"PASS {v2_sq['steps_run']}/{v2_sq['budget']}" if v2_sq["completed"] else f"FELL @{v2_sq['steps_run']}/{v2_sq['budget']}"

    # 1. High-Res Filmstrip (6 key poses across Held-Out & 24s Mode-Switch): 480x360 panels -> 3x2 grid
    r_strip = mujoco.Renderer(model, height=360, width=480)
    film_specs = [
        ("1. Held-Out Deep Squat (stat_fit_squat)", f"v2: {v2_sq_stat} | v1: {v1_sq_stat}", trajs["V2_RobotPose_ONNX__stat_fit_squat__pol"], 100, (38, 70, 83)),
        ("2. Held-Out 180° Pivot Turn (CGFA)", "walk2_subject1 t=5.0s | Zeroed fell", trajs["V2_RobotPose_ONNX__walk2_subject1__pol"], 250, (42, 157, 143)),
        ("3. Held-Out SOMA Manipulation", "eating_apple t=3.2s | Anchor Locked", trajs["V2_RobotPose_ONNX__soma_stat_3_eating_apple_R_002__A276__pol"], 160, (29, 53, 87)),
        ("4. Mode-Switch: Gamepad Walk->Squat", "24s Demo t=6.8s (PlannerCmd 100D)", ms_traj, 340, (231, 111, 81)),
        ("5. Mode-Switch: Zero-IK Raw BVH SMPL", "24s Demo t=11.6s (HumanSMPL 430D)", ms_traj, 580, (142, 68, 173)),
        ("6. Mode-Switch: VR 3-Pt + Watchdog", "24s Demo t=17.0s (VR3Pt 360D)", ms_traj, 850, (21, 101, 192)),
    ]
    panels = []
    for title, sub, arr, idx, col in film_specs:
        i = min(idx, len(arr) - 1)
        frame = _render_frame(model, data, r_strip, arr[i], lookat_follow=True)
        panels.append(_annotate(frame, title, sub, bar_color=col))
    row1 = np.concatenate(panels[:3], axis=1)
    row2 = np.concatenate(panels[3:], axis=1)
    strip = np.concatenate((row1, row2), axis=0)
    strip_path = BASE / "q1_multimodal_filmstrip.png"
    Image.fromarray(strip).save(strip_path, optimize=True)
    r_strip.close()
    print(f"SAVED_FILMSTRIP -> {strip_path} ({strip_path.stat().st_size} B)", flush=True)

    # 2. 2x2 Comparison Video (180 frames @ 25fps = 7.2s smooth showcase)
    r_vid = mujoco.Renderer(model, height=320, width=424)
    sq_v1 = trajs["V1_H002_ONNX__stat_fit_squat__pol"]
    sq_v2 = trajs["V2_RobotPose_ONNX__stat_fit_squat__pol"]
    w2_v2 = trajs["V2_RobotPose_ONNX__walk2_subject1__pol"]

    vid_dir = Path("/tmp/v2_vid_frames")
    if vid_dir.exists():
        import shutil; shutil.rmtree(vid_dir)
    vid_dir.mkdir(parents=True)

    for k in range(180):
        i_sq1 = min(k, len(sq_v1) - 1)
        i_sq2 = min(k, len(sq_v2) - 1)
        i_w2 = min(k * 2, len(w2_v2) - 1)
        i_ms = min(k * 6, len(ms_traj) - 1)

        tl_tag = "DONE OK" if v1_sq["completed"] else f"FELL @{v1_sq['steps_run']}"
        tr_tag = "STABLE FULL CLIP" if v2_sq["completed"] else f"FELL @{v2_sq['steps_run']}"
        p_tl = _annotate(_render_frame(model, data, r_vid, sq_v1[i_sq1]), "v1 H002 Baseline: stat_fit_squat", f"t={i_sq1*0.02:.2f}s | {tl_tag if i_sq1 >= len(sq_v1)-1 else 'tracking'}", (180, 50, 50) if not v1_sq["completed"] else (60, 100, 120))
        p_tr = _annotate(_render_frame(model, data, r_vid, sq_v2[i_sq2]), "v2 SONIC+WM_Cal: stat_fit_squat", f"t={i_sq2*0.02:.2f}s | {tr_tag}", (42, 157, 143))
        p_bl = _annotate(_render_frame(model, data, r_vid, w2_v2[i_w2]), "v2 Held-Out: walk2 180° Pivot", f"t={i_w2*0.02:.1f}s CGFA Turn-Gated", (38, 70, 83))
        # Determine phase label for ms
        sec_ms = i_ms * 0.02
        if sec_ms < 4.0:
            ph = "StandIdle (PlannerCmd)"
        elif sec_ms < 9.0:
            ph = "Walk+Squat (PlannerCmd)"
        elif sec_ms < 14.0:
            ph = "Zero-IK HumanSMPL (BVH)"
        elif sec_ms < 19.0:
            ph = "VR 3-Pt (PICO Head+Wrists)"
        elif sec_ms < 21.0:
            ph = "400ms Dropout -> Watchdog"
        else:
            ph = "StandRecover (PlannerCmd)"
        p_br = _annotate(_render_frame(model, data, r_vid, ms_traj[i_ms]), "v2 Live 4-Modality Mode-Switch (24s)", f"t={sec_ms:.1f}s | {ph}", (29, 53, 87))

        grid = np.concatenate((np.concatenate((p_tl, p_tr), axis=1), np.concatenate((p_bl, p_br), axis=1)), axis=0)
        Image.fromarray(grid).save(vid_dir / f"frame_{k:04d}.png")
    r_vid.close()

    mp4_path = BASE / "q1_sonic_multimodal_demo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "25", "-i", str(vid_dir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", str(mp4_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    import shutil; shutil.rmtree(vid_dir, ignore_errors=True)
    print(f"SAVED_MP4 -> {mp4_path} ({mp4_path.stat().st_size} B)", flush=True)


def build_4panel_chart() -> None:
    m = json.loads((BASE / "v2_sonic_metrics.json").read_text(encoding="utf-8"))
    ms = json.loads((BASE / "sonic_modeswitch_24s.json").read_text(encoding="utf-8"))

    plt.rcParams.update({"font.size": 8.8, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4), dpi=160)

    # Panel A: 50 Unseen Held-Out Clips Across Modalities & Baselines (Nominal vs 20-Clip Stress)
    ax = axes[0, 0]
    ho_nom = m["held_out_50_nominal"]
    ho_str = m["held_out_20_stress"]
    specs_a = [
        ("Zeroed_unadapted_0ms", "Zeroed\nBase"),
        ("V1_H002_ONNX", "v1 H002\n(12-clip)"),
        ("Ablation_WM_UnfrozenDec", "WM-Unfroz\nDec (Abl)"),
        ("V2_RobotPose_ONNX", "v2 Robot\nPose"),
        ("V2_HumanSMPL_ONNX", "v2 Human\nSMPL"),
        ("V2_VR3Pt_ONNX", "v2 VR\n3-Point"),
        ("V2_PlannerCmd_ONNX", "v2 Planner\n(Loco)*"),
        ("Oracle_scheme_a_0ms", "Oracle\nCeiling"),
    ]
    specs_a = [s for s in specs_a if s[0] in ho_nom]
    x = np.arange(len(specs_a))
    w = 0.36
    nom_vals = [
        ho_nom[k]["loco_squat_subset"]["survival_rate"] * 100.0 if k == "V2_PlannerCmd_ONNX" else ho_nom[k]["all_50"]["survival_rate"] * 100.0
        for k, _ in specs_a
    ]
    str_vals = [ho_str[k]["survival_rate"] * 100.0 if k in ho_str else np.nan for k, _ in specs_a]
    b1 = ax.bar(x - w/2, nom_vals, w, label="Unseen 50 Nominal (%)", color="#264653")
    b2 = ax.bar(x + w/2, np.nan_to_num(str_vals, nan=0.0), w, label="Unseen 20 Stress (0.55x Fric+Jitter+IMU)", color="#e76f51")
    for bar, v in zip(b1, nom_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5)
    for bar, v in zip(b2, str_vals):
        if np.isfinite(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.5, color="#a03018")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in specs_a], fontsize=7.0)
    ax.set_ylim(0, 118); ax.set_ylabel("Survival Rate (%)")
    ax.set_title("A. 50 Unseen Held-Out Survival (v1->v2 Scaling & 4 SONIC Modalities)", fontweight="bold")
    ax.legend(fontsize=7.2, loc="lower right", framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel B: v2 Multi-Modal Quality (NormReturn & Body RMSE & RootXY Drift on Held-Out 50 + Zero-IK BVH)
    ax = axes[0, 1]
    mods = ["Zeroed_unadapted_0ms", "V1_H002_ONNX", "V2_RobotPose_ONNX", "V2_HumanSMPL_ONNX", "V2_VR3Pt_ONNX"]
    mlab = ["Zeroed", "v1 H002", "v2 RobotPose", "v2 HumanSMPL", "v2 VR 3-Pt"]
    xm = np.arange(len(mods))
    ax2 = ax.twinx()
    ret_v = [ho_nom[k]["all_50"]["norm_return"] for k in mods]
    rxy_v = [ho_nom[k]["all_50"]["root_xy_rmse_cm"] for k in mods]
    b_r = ax.bar(xm - 0.18, ret_v, 0.36, label="NormReturn (Higher Better)", color="#2a9d8f")
    b_d = ax2.bar(xm + 0.18, rxy_v, 0.36, label="Root XY World Drift cm (Lower Better)", color="#f4a261")
    for bar in b_r:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in b_d:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xm); ax.set_xticklabels(mlab, fontsize=7.8)
    ax.set_ylim(0, 1.08); ax2.set_ylim(0, max(rxy_v) * 1.28 + 5)
    ax.set_ylabel("Normalized Return", color="#2a9d8f"); ax2.set_ylabel("Root XY Drift (cm)", color="#c06014")
    zero_ik_surv = m["zero_ik_raw_bvh_8"]["survival_rate"] * 100.0
    ax.set_title(f"B. Unseen 50 Return & Drift | Zero-IK Raw BVH (8 clips): {zero_ik_surv:.0f}% Surv", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel C: Deep Squat Held-Out Comparison (v1 vs v2) + v1-16 Re-eval
    ax = axes[1, 0]
    trajs = np.load(BASE / "v2_sample_trajs.npz")
    pz_v1 = trajs["V1_H002_ONNX__stat_fit_squat__pol"][:, 2]
    pz_v2 = trajs["V2_RobotPose_ONNX__stat_fit_squat__pol"][:, 2]
    pz_ref = trajs["V2_RobotPose_ONNX__stat_fit_squat__ref"][:, 2]
    v1_sq = ho_nom["V1_H002_ONNX"]["per_clip"]["stat_fit_squat"]
    v2_sq = ho_nom["V2_RobotPose_ONNX"]["per_clip"]["stat_fit_squat"]
    v1_lbl = f"v1 H002 ({'PASS' if v1_sq['completed'] else 'Fell @ '}{v1_sq['steps_run']}/{v1_sq['budget']})"
    v2_lbl = f"v2 RobotPose ({'PASS ' if v2_sq['completed'] else 'Fell @ '}{v2_sq['steps_run']}/{v2_sq['budget']})"
    ax.plot(np.arange(len(pz_ref)) * 0.02, pz_ref, "--", color="#444", linewidth=1.6, label="Reference Pelvis Z")
    ax.plot(np.arange(len(pz_v2)) * 0.02, pz_v2, color="#2a9d8f", linewidth=2.0, label=v2_lbl)
    ax.plot(np.arange(len(pz_v1)) * 0.02, pz_v1, color="#e63946", linewidth=1.8, label=v1_lbl)
    if not v1_sq["completed"]:
        ax.scatter([(len(pz_v1) - 1) * 0.02], [pz_v1[-1]], color="#e63946", s=55, zorder=5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Pelvis Height Z (m)")
    v1_16_s = m["v1_16_match"]["V2_RobotPose_ONNX"]["survival_rate"] * 100.0
    ax.set_title(f"C. Held-Out Deep Squat Trajectory | v1-16 Benchmark: 87.5% -> {v1_16_s:.1f}%", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel D: 24s Continuous 4-Modality Mode-Switch & 400ms Watchdog Telemetry
    ax = axes[1, 1]
    tr = ms["trace_sampled"]
    ts = [r["t_s"] for r in tr]
    pzs = [r["pelvis_z_m"] for r in tr]
    slips = [r["slip_mm_s"] for r in tr]
    ax_s = ax.twinx()
    ax.plot(ts, pzs, color="#264653", linewidth=1.8, label="Pelvis Z (m)")
    ax_s.plot(ts, slips, color="#e76f51", alpha=0.65, linewidth=1.1, label="Sole Slip (mm/s)")
    ylim_lo = min(min(pzs) - 0.04, 0.22)
    ylim_hi = max(max(pzs) + 0.07, 0.52)
    lbl_y = ylim_hi - 0.014
    for xv, txt in [(0.25, "A:Stand"), (4.25, "B:Walk+Squat"), (9.25, "C:ZeroIK\nSMPL"), (14.25, "D:VR3Pt"), (19.15, "E:Drop\n400ms"), (21.20, "F:Recover")]:
        ax.axvline(xv if xv > 0.5 else 0.0, color="#888", linestyle=":", linewidth=0.9)
        ax.text(xv, lbl_y, txt, fontsize=6.4, fontweight="bold", color="#1d3557", va="top")
    ax.set_xlim(0, 24.0); ax.set_ylim(ylim_lo, ylim_hi); ax_s.set_ylim(0, max(max(slips), 250) * 1.15)
    ax.set_xlabel("Continuous Episode Time (s)"); ax.set_ylabel("Pelvis Height (m)", color="#264653")
    ax_s.set_ylabel("Stance Slip (mm/s)", color="#c0392b")
    surv_ms = "100%" if ms["completed_24s"] else f"{ms['steps_survived']}/{ms['budget']}"
    ax.set_title(f"D. 24s Continuous 4-Modality Switch + Watchdog (PD OOB={ms['pd_oob_count']}, Surv={surv_ms})", fontweight="bold")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax_s.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7.0, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle("H002-SONIC v2: 200-Clip Scaled Training, 4-Modality Universal Tokens & 50 Held-Out Sim Verification", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    out_png = BASE / "h002_v2_sonic_benchmark.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"SAVED_CHART -> {out_png} ({out_png.stat().st_size} B)", flush=True)


if __name__ == "__main__":
    build_filmstrip_and_video()
    build_4panel_chart()
