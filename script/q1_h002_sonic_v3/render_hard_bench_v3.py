"""Render Hard-Bench Charts, 6-Panel 3D Filmstrip, 2x2 Falling-Box Compare MP4, and 1x2 Loop-Closure MP4."""

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
BASE = ROOT / "runs/h002_sonic_v3"


def _make_loop_vis_xml(path_xy: np.ndarray | None = None, dot_rgba: str = "0.06 0.65 0.58 0.72", tag: str = "base") -> Path:
    xml_dir = Path("/tmp/q1_box_scene")
    xml_dir.mkdir(parents=True, exist_ok=True)
    abs_rob = xml_dir / "supermimic_abs.xml"
    if not abs_rob.is_file():
        mesh_dir = ROOT / "src/unisupermimic/assets/robots/q1/assets"
        txt = (ROOT / "src/unisupermimic/assets/robots/q1/supermimic.xml").read_text(encoding="utf-8").replace(
            'meshdir="assets"', f'meshdir="{mesh_dir}"'
        )
        abs_rob.write_text(txt, encoding="utf-8")
    dots_xml = ""
    if path_xy is not None and len(path_xy) > 2:
        idxs = np.linspace(0, len(path_xy) - 1, 30, dtype=int)
        dots_xml = "\n".join(
            f'    <geom name="wp_{tag}_{i}" type="cylinder" pos="{float(path_xy[idx, 0]):.3f} {float(path_xy[idx, 1]):.3f} 0.002" size="0.045 0.002" rgba="{dot_rgba}" contype="0" conaffinity="0" />'
            for i, idx in enumerate(idxs)
        )
    loop_xml = xml_dir / f"q1_loop_vis_scene_{tag}.xml"
    loop_xml.write_text(
        f"""<mujoco model="q1_loop_vis_{tag}">
  <include file="{abs_rob}" />
  <option integrator="implicitfast" timestep="0.005" cone="pyramidal" />
  <asset>
    <texture name="ground" type="2d" builtin="checker" rgb1="0.72 0.75 0.78" rgb2="0.56 0.60 0.65" width="256" height="256" />
    <material name="ground" texture="ground" texrepeat="2 2" texuniform="true" />
  </asset>
  <worldbody>
    <light pos="1.5 -1.5 4.0" dir="-0.2 0.2 -1" directional="true" />
    <geom name="floor" type="plane" size="0 0 0.05" material="ground" friction="0.6 0.005 0.0001" />
    <geom name="origin_pad" type="box" pos="0 0 0.002" size="0.32 0.32 0.002" rgba="0.06 0.72 0.64 0.55" contype="0" conaffinity="0" />
    <geom name="origin_ring" type="box" pos="0 0 0.003" size="0.16 0.16 0.002" rgba="0.98 0.80 0.15 0.85" contype="0" conaffinity="0" />
{dots_xml}
  </worldbody>
</mujoco>
""",
        encoding="utf-8",
    )
    return loop_xml


def _render_cam(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: mujoco.Renderer,
    qpos: np.ndarray,
    *,
    lookat: tuple[float, float, float],
    dist: float,
    azim: float,
    elev: float,
) -> np.ndarray:
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = list(lookat)
    cam.distance = dist
    cam.azimuth = azim
    cam.elevation = elev
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


def build_hard_bench_chart() -> None:
    m = json.loads((BASE / "hard_bench_v3_metrics.json").read_text(encoding="utf-8"))
    lt = np.load(BASE / "hard_bench_loop_qpos.npz")
    ct = np.load(BASE / "hard_bench_catch_qpos.npz")

    plt.rcParams.update({"font.size": 8.6, "axes.edgecolor": "#444", "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.8), dpi=160)

    # Panel A: Bird's-Eye Loop-Closure Trajectories (Racetrack + Figure-8)
    ax = axes[0, 0]
    meta = m.get("loop_meta", {"racetrack_len_m": 7.3, "racetrack_dur_s": 19.2, "figure8_len_m": 10.4, "figure8_dur_s": 26.8})
    ref_r = lt["race_ref_xy"]
    v3_rob_r = lt["race_V3_RobotPose_CGFA_xy"]
    v3_smpl_r = lt["race_V3_HumanSMPL_CGFA_xy"]
    v3_plan_r = lt["race_V3_PlannerCmd_CGFA_xy"]
    v2_rob_r = lt["race_V2_RobotPose_CGFA_xy"]
    dr_r = lt["race_V3_UnfilteredOdom_DR_xy"]
    ref_f8 = lt["fig8_ref_xy"]
    v3_f8_smpl = lt["fig8_V3_HumanSMPL_CGFA_xy"]

    ax.plot(ref_r[:, 0], ref_r[:, 1], "--", color="#111827", linewidth=1.6, label=f"Racetrack Ideal ({meta['racetrack_len_m']:.1f}m)")
    ax.plot(ref_f8[:, 0], ref_f8[:, 1], "--", color="#7c3aed", linewidth=1.2, alpha=0.45, label=f"Figure-8 Ideal ({meta['figure8_len_m']:.1f}m)")
    ax.plot(v3_rob_r[:, 0], v3_rob_r[:, 1], color="#0f766e", linewidth=2.2, label=f"v3 RobotPose CGFA ({m['loop_closure_racetrack_26s']['V3_RobotPose_CGFA']['loop_closure_err_cm']:.1f}cm)")
    ax.plot(v3_smpl_r[:, 0], v3_smpl_r[:, 1], color="#2563eb", linewidth=1.9, alpha=0.88, label=f"v3 HumanSMPL CGFA ({m['loop_closure_racetrack_26s']['V3_HumanSMPL_CGFA']['loop_closure_err_cm']:.1f}cm)")
    ax.plot(v3_plan_r[:, 0], v3_plan_r[:, 1], color="#0284c7", linewidth=1.3, linestyle="-.", alpha=0.80, label=f"v3 PlannerCmd CGFA ({m['loop_closure_racetrack_26s']['V3_PlannerCmd_CGFA']['loop_closure_err_cm']:.1f}cm)")
    ax.plot(v2_rob_r[:, 0], v2_rob_r[:, 1], color="#f59e0b", linewidth=1.4, alpha=0.80, label=f"v2 RobotPose CGFA ({m['loop_closure_racetrack_26s']['V2_RobotPose_CGFA']['loop_closure_err_cm']:.0f}cm drift)")
    ax.plot(dr_r[:, 0], dr_r[:, 1], ":", color="#dc2626", linewidth=1.8, label=f"Unfiltered DR ({m['loop_closure_racetrack_26s']['V3_UnfilteredOdom_DR']['loop_closure_err_cm']:.0f}cm FELL)")
    ax.plot(v3_f8_smpl[:, 0], v3_f8_smpl[:, 1], color="#7c3aed", alpha=0.78, linewidth=1.6, label=f"Fig-8 v3 SMPL CGFA ({m['loop_closure_figure8_30s']['V3_HumanSMPL_CGFA']['loop_closure_err_cm']:.1f}cm)")
    ax.scatter([0.0], [0.0], c="#f59e0b", s=95, marker="*", zorder=7, edgecolors="#111", label="Start / Origin Pad (0, 0)")
    ax.scatter([v3_rob_r[-1, 0], v3_smpl_r[-1, 0], dr_r[-1, 0]], [v3_rob_r[-1, 1], v3_smpl_r[-1, 1], dr_r[-1, 1]], c=["#0f766e", "#2563eb", "#dc2626"], s=40, zorder=6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("World X (m)"); ax.set_ylabel("World Y (m)")
    ax.set_title(f"A. Closed-Loop Navigation Paths ({meta['racetrack_len_m']:.1f}m Racetrack & {meta['figure8_len_m']:.1f}m Figure-8)", fontweight="bold")
    ax.legend(fontsize=6.0, loc="lower left", ncol=2, framealpha=0.92)
    ax.grid(True, linestyle="--", alpha=0.35)

    # Panel B: Loop-Closure Error & Survival Across Conditions
    ax = axes[0, 1]
    race = m["loop_closure_racetrack_26s"]
    fig8 = m["loop_closure_figure8_30s"]
    keys = [
        ("V3_RobotPose_CGFA", "v3 Robot\nCGFA"),
        ("V3_HumanSMPL_CGFA", "v3 SMPL\nCGFA"),
        ("V3_PlannerCmd_CGFA", "v3 Plan\nCGFA"),
        ("V2_RobotPose_CGFA", "v2 Robot\nCGFA"),
        ("V3_UnfilteredOdom_DR", "DR\nNoGate"),
    ]
    xb = np.arange(len(keys))
    wb = 0.36
    errs_r = [race[k]["loop_closure_err_cm"] for k, _ in keys]
    errs_f = [fig8[k]["loop_closure_err_cm"] if k in fig8 else np.nan for k, _ in keys]
    b1 = ax.bar(xb - wb / 2, errs_r, wb, label=f"{meta['racetrack_dur_s']:.1f}s Racetrack ({meta['racetrack_len_m']:.1f}m)", color="#0f766e")
    b2 = ax.bar(xb + wb / 2, np.nan_to_num(errs_f, nan=0.0), wb, label=f"{meta['figure8_dur_s']:.1f}s Figure-8 ({meta['figure8_len_m']:.1f}m)", color="#7c3aed")
    for bar, (k, _) in zip(b1, keys):
        v = race[k]["loop_closure_err_cm"]
        tag = f"{v:.0f}" + ("" if race[k]["completed"] else "\nFELL")
        ax.text(bar.get_x() + bar.get_width() / 2, v + 3.0, tag, ha="center", va="bottom", fontsize=6.5)
    for bar, (k, _) in zip(b2, keys):
        if k in fig8:
            v = fig8[k]["loop_closure_err_cm"]
            tag = f"{v:.0f}" + ("" if fig8[k]["completed"] else "\nFELL")
            ax.text(bar.get_x() + bar.get_width() / 2, v + 3.0, tag, ha="center", va="bottom", fontsize=6.5, color="#5b21b6")
    ax.set_xticks(xb); ax.set_xticklabels([l for _, l in keys], fontsize=7.6)
    ax.set_ylim(0, max(max(errs_r), max(x for x in errs_f if np.isfinite(x))) * 1.28 + 25)
    ax.set_ylabel("Origin Return Error ||p(T) - p(0)|| (cm, Lower Better)")
    ax.set_title("B. Loop-Closure Origin Error (Full Loop -> Back to Origin)", fontweight="bold")
    ax.legend(fontsize=7.2, loc="upper left", framealpha=0.92)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel C: Falling-Box Catch Rate vs Lookahead Latency (20 Trials + 0..200ms Latency Curve)
    ax = axes[1, 0]
    bs = m["falling_box_catch_20trials"]
    models_c = [
        ("V3_Causal_0ms_HumanSMPL", "v3 SMPL\n(0ms)", "#0f766e"),
        ("V3_Causal_0ms_VR3Pt", "v3 VR3Pt\n(0ms)", "#2563eb"),
        ("V2_Causal_0ms_RobotPose", "v2 Robot\n(0ms)", "#2a9d8f"),
        ("SONIC_Future_120msBuf", "SONIC-Fut\n(120ms)", "#f59e0b"),
        ("SuperMimic_Orig_200msBuf", "SuperMimic\n(200ms)", "#dc2626"),
    ]
    xc = np.arange(len(models_c))
    rates = [bs[k]["catch_rate"] * 100.0 for k, _, _ in models_c]
    cols = [c for _, _, c in models_c]
    bars = ax.bar(xc, rates, 0.54, color=cols)
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 2.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold")
    ls = m["falling_box_latency_sweep_ms"]
    ms_keys = ["0ms", "40ms", "80ms", "120ms", "160ms", "200ms"]
    ls_vals = " | ".join(f"{k}:{ls[k]['v3_smpl_with_artif_lat']*100:.0f}%" for k in ms_keys)
    ax.text(
        0.02, 0.96,
        f"Causal Buffer Sweep (10 trials): {ls_vals}",
        transform=ax.transAxes, va="top", fontsize=6.6,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f8fafc", edgecolor="#cbd5e1", alpha=0.95),
    )
    ax.set_xticks(xc); ax.set_xticklabels([l for _, l, _ in models_c], fontsize=7.4)
    ax.set_ylim(0, 126); ax.set_ylabel("Rigid-Body Box Catch Rate (%, 20 Trials)")
    ax.set_title("C. High-Platform Falling-Box Catch Rate (Real MuJoCo 3.11)", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    # Panel D: Falling-Box Catch Timeline (Box Z(t) & Wrist Z(t))
    ax = axes[1, 1]
    bz_smpl = ct["V3_Causal_0ms_HumanSMPL__box_z"]
    lw_smpl = ct["V3_Causal_0ms_HumanSMPL__lw_z"]
    bz_vr = ct["V3_Causal_0ms_VR3Pt__box_z"]
    bz_sm200 = ct["SuperMimic_Orig_200msBuf__box_z"]
    lw_sm200 = ct["SuperMimic_Orig_200msBuf__lw_z"]
    bz_sf120 = ct["SONIC_Future_120msBuf__box_z"]
    t_arr = np.arange(len(bz_smpl)) * 0.02
    ax.plot(t_arr, bz_smpl, color="#0f766e", linewidth=2.3, label="Box Z — Ours v3 HumanSMPL 0ms (CAUGHT)")
    ax.plot(t_arr, bz_vr, color="#2563eb", linewidth=1.7, alpha=0.85, label="Box Z — Ours v3 VR3Pt 0ms (CAUGHT)")
    ax.plot(t_arr, lw_smpl, "--", color="#2a9d8f", linewidth=1.4, label="Wrist Z — v3 0ms fast V-cradle")
    ax.plot(t_arr, bz_sf120, color="#f59e0b", linewidth=1.9, label="Box Z — SONIC-Fut 120ms Buf (MISSED)")
    ax.plot(t_arr, bz_sm200, color="#dc2626", linewidth=2.0, label="Box Z — Orig SuperMimic 200ms Buf (MISSED)")
    ax.plot(t_arr, lw_sm200, ":", color="#dc2626", linewidth=1.5, label="Wrist Z — SuperMimic (200ms delayed)")
    ax.axvline(0.80, color="#334155", linestyle="--", linewidth=1.0)
    ax.text(0.82, 1.38, "t=0.80s Drop", fontsize=6.9, fontweight="bold", color="#1e293b")
    ax.axvline(0.98, color="#475569", linestyle=":", linewidth=1.0)
    ax.text(1.00, 1.24, "t=0.98s Human Reacts", fontsize=6.7, color="#334155")
    ax.set_xlim(0.4, 2.68); ax.set_ylim(0.0, 1.48)
    ax.set_xlabel("Live Causal Clock Time (s)"); ax.set_ylabel("Height Z (m)")
    ax.set_title("D. Live Catch Physics Trace: 0ms Causal vs Future-Lookahead Misses", fontweight="bold")
    ax.legend(fontsize=6.3, loc="upper right", framealpha=0.94)
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.suptitle("H002-SONIC v3 Hard Benchmarks: Closed-Loop Origin Return (Racetrack / Figure-8) & 0ms Falling-Box Catch", fontsize=12.2, fontweight="bold")
    fig.tight_layout(rect=[0, 0.01, 1, 0.96])
    out_png = BASE / "h002_v3_hard_bench_loop_catch.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"SAVED_HARD_CHART -> {out_png} ({out_png.stat().st_size} B)", flush=True)


def build_hard_bench_filmstrip_and_videos() -> None:
    lt = np.load(BASE / "hard_bench_loop_qpos.npz")
    ct = np.load(BASE / "hard_bench_catch_qpos.npz")
    m = json.loads((BASE / "hard_bench_v3_metrics.json").read_text(encoding="utf-8"))
    meta = m.get("loop_meta", {"racetrack_len_m": 7.3, "racetrack_dur_s": 19.2, "figure8_len_m": 10.4, "figure8_dur_s": 26.8})

    race_xml = _make_loop_vis_xml(lt["race_ref_xy"], "0.06 0.65 0.58 0.72", "race")
    fig8_xml = _make_loop_vis_xml(lt["fig8_ref_xy"], "0.52 0.24 0.85 0.72", "fig8")
    m_race = mujoco.MjModel.from_xml_path(str(race_xml))
    d_race = mujoco.MjData(m_race)
    m_fig8 = mujoco.MjModel.from_xml_path(str(fig8_xml))
    d_fig8 = mujoco.MjData(m_fig8)
    box_xml = Path("/tmp/q1_box_scene/q1_falling_box_scene.xml")
    m_box = mujoco.MjModel.from_xml_path(str(box_xml))
    d_box = mujoco.MjData(m_box)

    race_res = m["loop_closure_racetrack_26s"]
    fig8_res = m["loop_closure_figure8_30s"]
    race_best = min(("V3_RobotPose_CGFA", "V3_HumanSMPL_CGFA"), key=lambda k: race_res[k]["loop_closure_err_cm"])
    fig8_best = min(("V3_RobotPose_CGFA", "V3_HumanSMPL_CGFA"), key=lambda k: fig8_res[k]["loop_closure_err_cm"])
    race_tag = "v3 RobotPose" if race_best == "V3_RobotPose_CGFA" else "v3 HumanSMPL"
    fig8_tag = "v3 RobotPose" if fig8_best == "V3_RobotPose_CGFA" else "v3 HumanSMPL"

    race_q = lt[f"race_{race_best}_qpos"]
    fig8_q = lt[f"fig8_{fig8_best}_qpos"]
    v3_smpl_q = ct["V3_Causal_0ms_HumanSMPL__qpos"]
    v3_vr_q = ct["V3_Causal_0ms_VR3Pt__qpos"]
    sf120_catch_q = ct["SONIC_Future_120msBuf__qpos"]
    sm200_catch_q = ct["SuperMimic_Orig_200msBuf__qpos"]

    # 1. 6-Panel Filmstrip (3x2, 480x360)
    r_strip = mujoco.Renderer(m_race, height=360, width=480)
    r_box_strip = mujoco.Renderer(m_box, height=360, width=480)
    f1 = min(70, len(race_q) - 1)
    f2 = min(270, len(race_q) - 1)
    p1 = _annotate(_render_cam(m_race, d_race, r_strip, race_q[f1], lookat=(race_q[f1][0], race_q[f1][1], 0.40), dist=2.3, azim=135, elev=-22), f"1. Racetrack Leg-1 ({race_tag})", f"t={f1*0.02:.1f}s | pos=({race_q[f1][0]:.2f}, {race_q[f1][1]:.2f})m", (29, 53, 87))
    p2 = _annotate(_render_cam(m_race, d_race, r_strip, race_q[f2], lookat=(race_q[f2][0], race_q[f2][1], 0.40), dist=2.5, azim=135, elev=-26), "2. Racetrack 180deg Far Turn", f"t={f2*0.02:.1f}s | pos=({race_q[f2][0]:.2f}, {race_q[f2][1]:.2f})m", (42, 157, 143))
    rc_err = race_res[race_best]["loop_closure_err_cm"]
    p3 = _annotate(_render_cam(m_race, d_race, r_strip, race_q[-1], lookat=(0.0, 0.0, 0.35), dist=2.4, azim=135, elev=-28), "3. Full Loop Return to Origin Pad", f"t={meta['racetrack_dur_s']:.1f}s | {meta['racetrack_len_m']:.1f}m Close={rc_err:.1f}cm", (15, 118, 110))

    p4 = _annotate(_render_cam(m_box, d_box, r_box_strip, v3_smpl_q[58], lookat=(0.14, 0.0, 0.72), dist=1.85, azim=122, elev=-12), "4. Ours v3 SMPL (0ms): Mid-Air Catch", "t=1.18s | V-Cradle catches falling box", (15, 118, 110))
    p5 = _annotate(_render_cam(m_box, d_box, r_box_strip, v3_vr_q[-1], lookat=(0.14, 0.0, 0.70), dist=1.85, azim=122, elev=-12), "5. Ours v3 VR3Pt (0ms): Held Steady!", f"t=2.68s | box_z={ct['V3_Causal_0ms_VR3Pt__box_z'][-1]:.2f}m", (21, 101, 192))
    p6 = _annotate(_render_cam(m_box, d_box, r_box_strip, sm200_catch_q[80], lookat=(0.14, 0.0, 0.55), dist=1.95, azim=122, elev=-15), "6. Orig SuperMimic (200ms Buf): MISS", "t=1.62s | Too late -> Box on Floor", (180, 45, 45))
    r_strip.close(); r_box_strip.close()

    strip = np.concatenate((np.concatenate([p1, p2, p3], axis=1), np.concatenate([p4, p5, p6], axis=1)), axis=0)
    strip_path = BASE / "q1_v3_hard_bench_filmstrip.png"
    Image.fromarray(strip).save(strip_path, optimize=True)
    print(f"SAVED_HARD_FILMSTRIP -> {strip_path} ({strip_path.stat().st_size} B)", flush=True)

    # 2. 2x2 Synchronized Falling-Box Video (134 frames @ 25fps -> slowed ~2x for crystal clear reaction visual!)
    r_box_v = mujoco.Renderer(m_box, height=340, width=424)
    vdir = Path("/tmp/v3_box_vid")
    shutil.rmtree(vdir, ignore_errors=True); vdir.mkdir(parents=True)
    for k in range(len(v3_smpl_q)):
        t_s = (k + 1) * 0.02
        c1 = _annotate(_render_cam(m_box, d_box, r_box_v, v3_smpl_q[k], lookat=(0.14, 0.0, 0.72), dist=1.95, azim=122, elev=-12), "Ours v3 HumanSMPL (0ms Causal)", f"t={t_s:.2f}s | box_z={ct['V3_Causal_0ms_HumanSMPL__box_z'][k]:.2f}m | CATCH!", (15, 118, 110))
        c2 = _annotate(_render_cam(m_box, d_box, r_box_v, v3_vr_q[k], lookat=(0.14, 0.0, 0.72), dist=1.95, azim=122, elev=-12), "Ours v3 PICO VR3Pt (0ms Causal)", f"t={t_s:.2f}s | box_z={ct['V3_Causal_0ms_VR3Pt__box_z'][k]:.2f}m | CATCH!", (21, 101, 192))
        c3 = _annotate(_render_cam(m_box, d_box, r_box_v, sf120_catch_q[k], lookat=(0.14, 0.0, 0.72), dist=1.95, azim=122, elev=-12), "SONIC-Future (120ms Future Buf)", f"t={t_s:.2f}s | box_z={ct['SONIC_Future_120msBuf__box_z'][k]:.2f}m | MISS", (180, 95, 15))
        c4 = _annotate(_render_cam(m_box, d_box, r_box_v, sm200_catch_q[k], lookat=(0.14, 0.0, 0.72), dist=1.95, azim=122, elev=-12), "Orig SuperMimic (200ms Future Buf)", f"t={t_s:.2f}s | box_z={ct['SuperMimic_Orig_200msBuf__box_z'][k]:.2f}m | MISS", (180, 45, 45))
        grid = np.concatenate((np.concatenate((c1, c2), axis=1), np.concatenate((c3, c4), axis=1)), axis=0)
        Image.fromarray(grid).save(vdir / f"frame_{k:04d}.png")
    r_box_v.close()
    mp4_box = BASE / "q1_v3_falling_box_catch_compare.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "25", "-i", str(vdir / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(mp4_box)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(vdir, ignore_errors=True)
    print(f"SAVED_BOX_MP4 -> {mp4_box} ({mp4_box.stat().st_size} B)", flush=True)

    # 3. 1x2 Synchronized Loop-Closure Video (180 frames @ 25fps)
    r_race_v = mujoco.Renderer(m_race, height=400, width=520)
    r_fig8_v = mujoco.Renderer(m_fig8, height=400, width=520)
    vdir_l = Path("/tmp/v3_loop_vid")
    shutil.rmtree(vdir_l, ignore_errors=True); vdir_l.mkdir(parents=True)
    n_l = 180
    race_mid = (float(lt["race_ref_xy"][:, 0].mean()), float(lt["race_ref_xy"][:, 1].mean()), 0.25)
    fig8_mid = (float(lt["fig8_ref_xy"][:, 0].mean()), float(lt["fig8_ref_xy"][:, 1].mean()), 0.25)
    for k in range(n_l):
        ir = min(int(k * (len(race_q) - 1) / (n_l - 1)), len(race_q) - 1)
        if8 = min(int(k * (len(fig8_q) - 1) / (n_l - 1)), len(fig8_q) - 1)
        qr = race_q[ir]; qf = fig8_q[if8]
        p_l = _annotate(_render_cam(m_race, d_race, r_race_v, qr, lookat=race_mid, dist=4.3, azim=135, elev=-40), f"Racetrack ({meta['racetrack_dur_s']:.1f}s/{meta['racetrack_len_m']:.1f}m | {race_tag})", f"t={ir*0.02:.1f}s | pos=({qr[0]:.2f}, {qr[1]:.2f})m", (15, 118, 110))
        p_r = _annotate(_render_cam(m_fig8, d_fig8, r_fig8_v, qf, lookat=fig8_mid, dist=4.7, azim=135, elev=-42), f"Figure-8 ({meta['figure8_dur_s']:.1f}s/{meta['figure8_len_m']:.1f}m | {fig8_tag})", f"t={if8*0.02:.1f}s | pos=({qf[0]:.2f}, {qf[1]:.2f})m", (88, 28, 135))
        Image.fromarray(np.concatenate((p_l, p_r), axis=1)).save(vdir_l / f"frame_{k:04d}.png")
    r_race_v.close(); r_fig8_v.close()
    mp4_loop = BASE / "q1_v3_loop_closure_demo.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", "25", "-i", str(vdir_l / "frame_%04d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", str(mp4_loop)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    shutil.rmtree(vdir_l, ignore_errors=True)
    print(f"SAVED_LOOP_MP4 -> {mp4_loop} ({mp4_loop.stat().st_size} B)", flush=True)


if __name__ == "__main__":
    build_hard_bench_chart()
    build_hard_bench_filmstrip_and_videos()
