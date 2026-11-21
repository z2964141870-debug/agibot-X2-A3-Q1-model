"""One-shot reserved unseen final acceptance runner for Q1 Soft Recovery V2."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_V1 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_20261121")
RISE_DIR = Path("/home/yu/projects/SuperMimicHP/runs/q1_supported_rise_20261121")
AUTON_ROOT = Path("/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery")
for p in reversed((str(SCRIPT_DIR), str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT))):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_v2
assert hasattr(eval_v2.evaluate, "main"), f"Wrong evaluate module loaded: {eval_v2.evaluate.__file__}"
from command_support import sample_record
from common import BUNDLE, CONFIG, mj
from unified_v3 import UnifiedRecoveryV3, verify_onnx_and_benchmark


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_push_grid_video(push_dir: Path, seed: int, out_mp4: Path):
    """Render 4-panel 1.0x (25 fps) video for push trials, prioritizing any failure/worst peaks."""
    from PIL import Image, ImageDraw, ImageFont

    res = json.loads((push_dir / "results.json").read_text())
    fell = [r for r in res if r["phases"][0]["fell"]]
    fails = sorted([r for r in fell if not r["success"]], key=lambda r: -r.get("peak_qd", 0.0))
    passes = sorted(
        [r for r in fell if r["success"]],
        key=lambda r: -max(r.get("peak_qd", 0.0) / 12.0, r.get("ground_mm", 0.0) / 10.0, r.get("self_mm", 0.0) / 2.0),
    )
    ordered = fails + passes + [r for r in res if not r["phases"][0]["fell"]]
    picks = []
    seen = set()
    for r in ordered:
        if r["trial"] not in seen:
            seen.add(r["trial"])
            tag = "FAIL" if not r["success"] else "OK"
            picks.append((r, f"PUSH ({tag})"))
        if len(picks) == 4:
            break

    panels = []
    max_ticks = 0
    for r, label in picks:
        data = np.load(push_dir / f"trial_{r['trial']:02d}.npz")
        max_ticks = max(max_ticks, len(data["qpos"]))
        panels.append((r, label, data))

    model = mj.MjModel.from_xml_path(str(push_dir / "scene/fall_hard.xml"))
    for marker in ("target_pad", "target_ring"):
        try:
            model.geom(marker).rgba[3] = 0
        except Exception:
            pass
    d = mj.MjData(model)
    camera = mj.MjvCamera()
    camera.distance = 1.65
    camera.azimuth = 63.0
    camera.elevation = -14.0
    font = ImageFont.truetype("DejaVuSans.ttf", 17)
    small = ImageFont.truetype("DejaVuSans.ttf", 13)

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "896x620",
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
        str(out_mp4),
    ]
    n_frames = 0
    sheets = []
    with mj.Renderer(model, height=225, width=448) as renderer, subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for tick in range(0, max_ticks + 1, 2):
            canvas = Image.new("RGB", (896, 620), "#16202a")
            pen = ImageDraw.Draw(canvas)
            pen.text((12, 9), f"Q1 Recovery V2 Push | seed {seed} | 1.0x speed (25 fps, 2 ms physics)", font=font, fill="white")
            for p_idx, (r, label, data) in enumerate(panels):
                x = (p_idx % 2) * 448
                y = 42 + (p_idx // 2) * 283
                qpos = data["qpos"]
                trace = data["trace"]
                peaks = data["peaks"]
                t_clamped = min(max(0, tick - 1), len(qpos) - 1)
                ended = tick > len(qpos)
                d.qpos[:] = data["initial"] if tick == 0 else qpos[t_clamped]
                mj.mj_forward(model, d)
                camera.lookat[:] = [d.qpos[0], d.qpos[1], 0.25]
                renderer.update_scene(d, camera=camera)
                canvas.paste(Image.fromarray(renderer.render()), (x, y + 25))
                own = int(trace[t_clamped, 4])
                pk_qd = float(peaks[: t_clamped + 1, 0].max())
                pk_gnd = float(peaks[: t_clamped + 1, 6].max() * 1000.0)
                pk_self = float(peaks[: t_clamped + 1, 7].max() * 1000.0)
                tag = "[STOP]" if ended else ("RECOV" if own else "TELEOP")
                pen.text((x + 10, y), f"T{r['trial']:02d} {label} | {tag}", font=font, fill="white")
                pen.text(
                    (x + 10, y + 253),
                    f"t={t_clamped*.02:4.2f}s qd={pk_qd:4.1f} gnd={pk_gnd:4.1f}mm self={pk_self:4.1f}mm",
                    font=small,
                    fill="#d7e4ef",
                )
            proc.stdin.write(np.asarray(canvas).tobytes())
            n_frames += 1
            if len(sheets) < 4 and tick in (0, 140, 280, min(460, max_ticks - (max_ticks % 2))):
                sheets.append(canvas.resize((448, 310)))
        proc.stdin.close()
        code = proc.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed with {code}")
    if sheets:
        sheet = Image.new("RGB", (896, 620), "#16202a")
        for idx, im in enumerate(sheets[:4]):
            sheet.paste(im, ((idx % 2) * 448, (idx // 2) * 310))
        sheet.save(out_mp4.with_suffix(".jpg"))
    return dict(
        video=str(out_mp4),
        contact_sheet=str(out_mp4.with_suffix(".jpg")),
        fps=25,
        frames=n_frames,
        duration_s=round(n_frames / 25.0, 2),
        speed=1.0,
        trials=[r["trial"] for r, _ in picks],
        sha256=sha256_file(out_mp4),
    )


def summarize_unseen_static(static_dirs: list[tuple[int, Path]], plan: dict) -> dict:
    per_seed = []
    pose_agg = {
        pose: dict(succ=0, soft=0, n=0, worst_qd=0.0, worst_gnd=0.0, worst_self=0.0, max_confirm_s=0.0)
        for pose in ("prone", "supine", "side")
    }
    failures_or_violations = []
    total_rejected = 0
    reject_reasons = {}
    all_rows = []
    confirmed_times_all = []
    confirmed_times_succ = []

    qd_max_lim = float(plan["recovery_peaks"]["joint_speed_maximum_rad_s"])
    gnd_max_lim = float(plan["recovery_peaks"]["ground_penetration_maximum_mm"])
    self_max_lim = float(plan["recovery_peaks"]["arm_leg_penetration_maximum_mm"])
    deadline_s = float(plan["static"]["standing_confirmation_deadline_s_from_takeover"])

    for seed, d in static_dirs:
        res = json.loads((d / "results.json").read_text())
        rej = json.loads((d / "rejected_initializations.json").read_text())
        inits = json.loads((d / "initialization.json").read_text())
        total_rejected += len(rej)
        for item in rej:
            rs = item.get("reason", "unknown")
            reject_reasons[rs] = reject_reasons.get(rs, 0) + 1

        by_pose = {}
        for pose in ("prone", "supine", "side"):
            sub = [r for r in res if r["pose"] == pose]
            succ = [r for r in sub if r["success"]]
            soft = [
                r
                for r in succ
                if r.get("peak_qd", 99) <= qd_max_lim
                and r.get("ground_mm", 99) <= gnd_max_lim
                and r.get("self_mm", 99) <= self_max_lim
            ]
            conf = [
                r["first_confirmation_s"] - r["takeovers"][0] * 0.02
                for r in sub
                if r.get("first_confirmation_s") is not None and r.get("takeovers")
            ]
            conf_succ = [
                r["first_confirmation_s"] - r["takeovers"][0] * 0.02
                for r in succ
                if r.get("first_confirmation_s") is not None and r.get("takeovers")
            ]
            by_pose[pose] = dict(
                succ=len(succ),
                soft=len(soft),
                n=len(sub),
                worst_qd=max(r.get("peak_qd", 0.0) for r in sub),
                worst_gnd=max(r.get("ground_mm", 0.0) for r in sub),
                worst_self=max(r.get("self_mm", 0.0) for r in sub),
                max_confirm_s=max(conf, default=None),
                max_confirm_s_successful=max(conf_succ, default=None),
            )
            pa = pose_agg[pose]
            pa["succ"] += len(succ)
            pa["soft"] += len(soft)
            pa["n"] += len(sub)
            pa["worst_qd"] = max(pa["worst_qd"], by_pose[pose]["worst_qd"])
            pa["worst_gnd"] = max(pa["worst_gnd"], by_pose[pose]["worst_gnd"])
            pa["worst_self"] = max(pa["worst_self"], by_pose[pose]["worst_self"])
            if conf:
                pa["max_confirm_s"] = max(pa["max_confirm_s"], max(conf))

        seed_conf = [
            r["first_confirmation_s"] - r["takeovers"][0] * 0.02
            for r in res
            if r.get("first_confirmation_s") is not None and r.get("takeovers")
        ]
        seed_conf_succ = [
            r["first_confirmation_s"] - r["takeovers"][0] * 0.02
            for r in res
            if r["success"] and r.get("first_confirmation_s") is not None and r.get("takeovers")
        ]
        confirmed_times_all.extend(seed_conf)
        confirmed_times_succ.extend(seed_conf_succ)
        per_seed.append(
            dict(
                seed=seed,
                success=sum(r["success"] for r in res),
                soft_success=sum(
                    1
                    for r in res
                    if r["success"]
                    and r.get("peak_qd", 99) <= qd_max_lim
                    and r.get("ground_mm", 99) <= gnd_max_lim
                    and r.get("self_mm", 99) <= self_max_lim
                ),
                n=len(res),
                rejected_init_attempts=len(rej),
                max_init_attempts=max(x["attempts"] for x in inits),
                wrong_release=sum(r["wrong_release"] for r in res),
                max_confirm_s=max(seed_conf, default=None),
                max_confirm_s_successful=max(seed_conf_succ, default=None),
                owned_peak_qd=max(r.get("peak_qd", 0.0) for r in res),
                owned_ground_mm=max(r.get("ground_mm", 0.0) for r in res),
                owned_self_mm=max(r.get("self_mm", 0.0) for r in res),
                trial_peak_qd=max(r["trial_peak_qd"] for r in res),
                trial_ground_mm=max(r["trial_ground_mm"] for r in res),
                trial_self_mm=max(r["trial_self_mm"] for r in res),
                by_pose=by_pose,
            )
        )
        for r in res:
            all_rows.append((seed, r))
            cs = (
                r["first_confirmation_s"] - r["takeovers"][0] * 0.02
                if r.get("first_confirmation_s") is not None and r.get("takeovers")
                else None
            )
            soft_ok = (
                r["success"]
                and r.get("peak_qd", 99) <= qd_max_lim
                and r.get("ground_mm", 99) <= gnd_max_lim
                and r.get("self_mm", 99) <= self_max_lim
            )
            if not soft_ok:
                failures_or_violations.append(
                    dict(
                        seed=seed,
                        trial=r["trial"],
                        pose=r["pose"],
                        success=r["success"],
                        confirm_s=cs,
                        takeovers=r["takeovers"],
                        releases=r["releases"],
                        peak_qd=r.get("peak_qd", 0.0),
                        peak_qd_joint=r.get("peak_qd_joint"),
                        ground_mm=r.get("ground_mm", 0.0),
                        self_mm=r.get("self_mm", 0.0),
                        all_self_mm=r.get("all_self_mm", 0.0),
                        trial_peak_qd=r["trial_peak_qd"],
                        trial_ground_mm=r["trial_ground_mm"],
                        trial_self_mm=r["trial_self_mm"],
                    )
                )

    total_n = len(all_rows)
    total_succ = sum(1 for _, r in all_rows if r["success"])
    total_soft = sum(
        1
        for _, r in all_rows
        if r["success"]
        and r.get("peak_qd", 99) <= qd_max_lim
        and r.get("ground_mm", 99) <= gnd_max_lim
        and r.get("self_mm", 99) <= self_max_lim
    )
    init_gnd_max = max(r["initial_ground_mm"] for _, r in all_rows)
    init_self_max = max(r["initial_self_mm"] for _, r in all_rows)
    init_all_self_max = max(r["initial_all_self_mm"] for _, r in all_rows)

    return dict(
        total=total_n,
        timely_success=total_succ,
        timely_rate=total_succ / max(1, total_n),
        simultaneous_soft_success=total_soft,
        soft_rate=total_soft / max(1, total_n),
        within_qd_12=sum(1 for _, r in all_rows if r.get("peak_qd", 99) <= qd_max_lim),
        within_ground_10=sum(1 for _, r in all_rows if r.get("ground_mm", 99) <= gnd_max_lim),
        within_self_2=sum(1 for _, r in all_rows if r.get("self_mm", 99) <= self_max_lim),
        wrong_release=sum(r["wrong_release"] for _, r in all_rows),
        max_confirm_s=max(confirmed_times_all, default=None),
        max_confirm_s_all_confirmed=max(confirmed_times_all, default=None),
        max_confirm_s_successful=max(confirmed_times_succ, default=None),
        all_timely_within_deadline=all(
            (r["first_confirmation_s"] - r["takeovers"][0] * 0.02) <= deadline_s + 1e-8
            for _, r in all_rows
            if r["success"]
        ),
        legal_settled_starts=sum(
            1
            for _, r in all_rows
            if r["initial_ground_mm"] <= plan["static"]["settled_initial_ground_maximum_mm"]
            and r["initial_self_mm"] <= plan["static"]["settled_initial_arm_leg_maximum_mm"]
        ),
        initial_ground_mm_max=init_gnd_max,
        initial_arm_leg_self_mm_max=init_self_max,
        initial_all_self_mm_max=init_all_self_max,
        rejected_initializations_total=total_rejected,
        rejected_reasons=reject_reasons,
        pd_oob=sum(r["pd_oob"] for _, r in all_rows),
        teleop_calls_while_owned=sum(r["teleop_calls_while_owned"] for _, r in all_rows),
        max_command_step=max(r["max_command_step"] for _, r in all_rows),
        owned_peaks_including_failed=dict(
            peak_qd=max(r.get("peak_qd", 0.0) for _, r in all_rows),
            ground_mm=max(r.get("ground_mm", 0.0) for _, r in all_rows),
            self_mm=max(r.get("self_mm", 0.0) for _, r in all_rows),
        ),
        full_trial_peaks_including_teleop=dict(
            trial_peak_qd=max(r["trial_peak_qd"] for _, r in all_rows),
            trial_ground_mm=max(r["trial_ground_mm"] for _, r in all_rows),
            trial_self_mm=max(r["trial_self_mm"] for _, r in all_rows),
        ),
        by_pose=pose_agg,
        per_seed=per_seed,
        failures_or_violations=failures_or_violations,
    )


def summarize_unseen_push(push_dirs: list[tuple[int, Path]], plan: dict) -> dict:
    per_seed = []
    all_trials = []
    qd_max_lim = float(plan["recovery_peaks"]["joint_speed_maximum_rad_s"])
    gnd_max_lim = float(plan["recovery_peaks"]["ground_penetration_maximum_mm"])
    self_max_lim = float(plan["recovery_peaks"]["arm_leg_penetration_maximum_mm"])

    for seed, d in push_dirs:
        res = json.loads((d / "results.json").read_text())
        seed_trials = []
        for r in res:
            tr = np.load(d / f"trial_{r['trial']:02d}.npz")
            peaks = tr["peaks"]
            cap = r["takeovers"][0] if r["takeovers"] else -1
            if cap >= 0:
                s1 = slice(cap, min(len(peaks), cap + 40))
                s2 = slice(min(len(peaks), cap + 40), len(peaks))
                p1 = [float(peaks[s1, 0].max()), float(peaks[s1, 6].max() * 1000), float(peaks[s1, 7].max() * 1000)]
                p2 = (
                    [float(peaks[s2, 0].max()), float(peaks[s2, 6].max() * 1000), float(peaks[s2, 7].max() * 1000)]
                    if s2.start < s2.stop
                    else [0.0, 0.0, 0.0]
                )
            else:
                p1 = [0.0, 0.0, 0.0]
                p2 = [0.0, 0.0, 0.0]
            cs = (
                r["first_confirmation_s"] - r["takeovers"][0] * 0.02
                if r.get("first_confirmation_s") is not None and r.get("takeovers")
                else None
            )
            row = dict(
                seed=seed,
                trial=r["trial"],
                fell=bool(r["phases"][0]["fell"]),
                success=bool(r["success"]),
                pre_impact_fall=bool(r["pre_impact_fall"]),
                wrong_release=int(r["wrong_release"]),
                takeovers=r["takeovers"],
                releases=r["releases"],
                confirm_s=cs,
                owned_peak_qd=r.get("peak_qd", 0.0),
                owned_peak_qd_joint=r.get("peak_qd_joint"),
                owned_ground_mm=r.get("ground_mm", 0.0),
                owned_self_mm=r.get("self_mm", 0.0),
                trial_peak_qd=r["trial_peak_qd"],
                trial_ground_mm=r["trial_ground_mm"],
                trial_self_mm=r["trial_self_mm"],
                cap_plus40=p1,
                after_40=p2,
                pd_oob=r["pd_oob"],
                teleop_calls_while_owned=r["teleop_calls_while_owned"],
                max_command_step=r["max_command_step"],
            )
            seed_trials.append(row)
            all_trials.append(row)

        fell_s = [x for x in seed_trials if x["fell"]]
        owned_s = [x for x in seed_trials if x["takeovers"]]
        per_seed.append(
            dict(
                seed=seed,
                n=len(seed_trials),
                actual_falls=len(fell_s),
                recovered=sum(x["success"] for x in fell_s),
                pre_impact_falls=sum(x["pre_impact_fall"] for x in seed_trials),
                wrong_release=sum(x["wrong_release"] for x in seed_trials),
                owned_peaks_including_failed=dict(
                    peak_qd=max((x["owned_peak_qd"] for x in owned_s), default=0.0),
                    ground_mm=max((x["owned_ground_mm"] for x in owned_s), default=0.0),
                    self_mm=max((x["owned_self_mm"] for x in owned_s), default=0.0),
                ),
                after_40_peaks=dict(
                    peak_qd=max((x["after_40"][0] for x in owned_s), default=0.0),
                    ground_mm=max((x["after_40"][1] for x in owned_s), default=0.0),
                    self_mm=max((x["after_40"][2] for x in owned_s), default=0.0),
                ),
                full_trial_peaks=dict(
                    trial_peak_qd=max(x["trial_peak_qd"] for x in seed_trials),
                    trial_ground_mm=max(x["trial_ground_mm"] for x in seed_trials),
                    trial_self_mm=max(x["trial_self_mm"] for x in seed_trials),
                ),
            )
        )

    fell_all = [x for x in all_trials if x["fell"]]
    owned_all = [x for x in all_trials if x["takeovers"]]
    rec_cnt = sum(x["success"] for x in fell_all)
    n_fell = len(fell_all)
    return dict(
        total_trials=len(all_trials),
        actual_falls=n_fell,
        recovered_over_actual_falls=rec_cnt,
        recovery_rate_over_actual_falls=rec_cnt / max(1, n_fell),
        pre_impact_falls=sum(x["pre_impact_fall"] for x in all_trials),
        wrong_release=sum(x["wrong_release"] for x in all_trials),
        pd_oob=sum(x["pd_oob"] for x in all_trials),
        teleop_calls_while_owned=sum(x["teleop_calls_while_owned"] for x in all_trials),
        max_command_step=max(x["max_command_step"] for x in all_trials),
        owned_count=len(owned_all),
        owned_within_qd_12=sum(1 for x in owned_all if x["owned_peak_qd"] <= qd_max_lim),
        owned_within_ground_10=sum(1 for x in owned_all if x["owned_ground_mm"] <= gnd_max_lim),
        owned_within_self_2=sum(1 for x in owned_all if x["owned_self_mm"] <= self_max_lim),
        after_40_within_soft=sum(
            1
            for x in owned_all
            if x["after_40"][0] <= qd_max_lim and x["after_40"][1] <= gnd_max_lim and x["after_40"][2] <= self_max_lim
        ),
        owned_peaks_including_failed=dict(
            peak_qd=max((x["owned_peak_qd"] for x in owned_all), default=0.0),
            ground_mm=max((x["owned_ground_mm"] for x in owned_all), default=0.0),
            self_mm=max((x["owned_self_mm"] for x in owned_all), default=0.0),
        ),
        after_40_peaks=dict(
            peak_qd=max((x["after_40"][0] for x in owned_all), default=0.0),
            ground_mm=max((x["after_40"][1] for x in owned_all), default=0.0),
            self_mm=max((x["after_40"][2] for x in owned_all), default=0.0),
        ),
        full_trial_peaks_all30=dict(
            trial_peak_qd=max(x["trial_peak_qd"] for x in all_trials),
            trial_ground_mm=max(x["trial_ground_mm"] for x in all_trials),
            trial_self_mm=max(x["trial_self_mm"] for x in all_trials),
        ),
        full_trial_peaks_fell_subset=dict(
            trial_peak_qd=max((x["trial_peak_qd"] for x in fell_all), default=0.0),
            trial_ground_mm=max((x["trial_ground_mm"] for x in fell_all), default=0.0),
            trial_self_mm=max((x["trial_self_mm"] for x in fell_all), default=0.0),
        ),
        per_seed=per_seed,
        trials=all_trials,
    )


def summarize_regressions(off_dir: Path, flat_dir: Path, dbl_dir: Path, own_res: dict) -> dict:
    off_r = json.loads((off_dir / "results.json").read_text())
    flt_r = json.loads((flat_dir / "results.json").read_text())
    dbl_r = json.loads((dbl_dir / "results.json").read_text())
    return dict(
        offline_84101=dict(
            trials=len(off_r),
            releases=sum(len(x["releases"]) for x in off_r),
            teleop_calls_while_owned=sum(x["teleop_calls_while_owned"] for x in off_r),
            terminal_standing=sum(x["terminal_standing"] for x in off_r),
            owned_peaks=dict(
                peak_qd=max(x.get("peak_qd", 0.0) for x in off_r),
                ground_mm=max(x.get("ground_mm", 0.0) for x in off_r),
                self_mm=max(x.get("self_mm", 0.0) for x in off_r),
            ),
        ),
        flat_84101=dict(
            trials=len(flt_r),
            takeovers=sum(len(x["takeovers"]) for x in flt_r),
            upright_fraction_min=min(x["upright_fraction"] for x in flt_r),
            terminal_standing=sum(x["terminal_standing"] for x in flt_r),
            trial_peaks=dict(
                trial_peak_qd=max(x["trial_peak_qd"] for x in flt_r),
                trial_ground_mm=max(x["trial_ground_mm"] for x in flt_r),
                trial_self_mm=max(x["trial_self_mm"] for x in flt_r),
            ),
        ),
        double_84111=dict(
            trials=len(dbl_r),
            both_fell=sum(1 for x in dbl_r if len(x["phases"]) == 2 and all(p["fell"] for p in x["phases"])),
            both_phases_recover=sum(
                1 for x in dbl_r if len(x["phases"]) == 2 and all(p["post_1s"] for p in x["phases"])
            ),
            wrong_release=sum(x["wrong_release"] for x in dbl_r),
            releases_confirmed_all=all(
                len(x["releases"]) == len(x["release_checks"]) and all(c["confirmed"] for c in x["release_checks"])
                for x in dbl_r
            ),
            owned_peaks=dict(
                peak_qd=max(x.get("peak_qd", 0.0) for x in dbl_r),
                ground_mm=max(x.get("ground_mm", 0.0) for x in dbl_r),
                self_mm=max(x.get("self_mm", 0.0) for x in dbl_r),
            ),
            trial_peaks=dict(
                trial_peak_qd=max(x["trial_peak_qd"] for x in dbl_r),
                trial_ground_mm=max(x["trial_ground_mm"] for x in dbl_r),
                trial_self_mm=max(x["trial_self_mm"] for x in dbl_r),
            ),
        ),
        ownership_stale_deterministic=own_res,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, default=SCRIPT_DIR / "acceptance_plan.json")
    p.add_argument(
        "--expert",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/unified_v3_08/expert.onnx"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/final_acceptance_20261121"),
    )
    p.add_argument("--roll", type=Path, default=RUN_V1 / "joined_roll_dev/roll.npz")
    p.add_argument(
        "--side-ckpt",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/side_v2_07/best_side_v3.pt"),
    )
    p.add_argument(
        "--supine-ckpt",
        type=Path,
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/supine_v2_04/best.pt"),
    )
    p.add_argument("--blend-ticks", type=int, default=None)
    args = p.parse_args()

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.plan.read_text())
    assert int(plan["physics_ms"]) == 2
    assert int(plan["control_ms"]) == 20
    assert int(plan["observation_delay_ms_assumed"]) == 11
    assert int(plan["explicit_arm_leg_pairs"]) == 96

    expert = args.expert.resolve()
    blend_ticks = eval_v2.DEFAULT_BLEND_TICKS if args.blend_ticks is None else args.blend_ticks

    print(f"=== Verify ONNX Equivalence & Latency: {expert} ===", flush=True)
    roll_fit = np.load(args.roll)
    side_ckpt = torch.load(args.side_ckpt, map_location="cpu", weights_only=False)
    supine_ckpt = torch.load(args.supine_ckpt, map_location="cpu", weights_only=False)
    faceup_cmd = sample_record("q1_official_standup_faceup_20261120")["command_position"]
    v3_model = UnifiedRecoveryV3(roll_fit, side_ckpt, supine_ckpt["actor"], faceup_cmd).cpu().eval()
    onnx_stats = verify_onnx_and_benchmark(v3_model, expert)

    static_seeds = [int(s) for s in plan["static"]["reserved_seeds"]]
    push_seeds = [int(s) for s in plan["push"]["reserved_seeds"]]

    static_dirs = []
    for seed in static_seeds:
        d = out / f"static_{seed}"
        print(f"=== Unseen Static Seed {seed} ===", flush=True)
        eval_v2.run_group(expert, "static", seed, d, blend_ticks)
        static_dirs.append((seed, d))

    push_dirs = []
    for seed in push_seeds:
        d = out / f"push_{seed}"
        print(f"=== Unseen Push Seed {seed} ===", flush=True)
        eval_v2.run_group(expert, "push", seed, d, blend_ticks)
        push_dirs.append((seed, d))

    print("=== Unseen Regressions: offline_84101, flat_84101, double_84111, check_soft_ownership ===", flush=True)
    off_dir = out / "offline_84101"
    eval_v2.run_group(expert, "offline", 84101, off_dir, blend_ticks)
    flat_dir = out / "flat_84101"
    eval_v2.run_group(expert, "flat", 84101, flat_dir, blend_ticks)
    dbl_dir = out / "double_84111"
    eval_v2.run_group(expert, "double", 84111, dbl_dir, blend_ticks)
    own_res = eval_v2.check_soft_ownership(expert, blend_ticks=blend_ticks)

    static_sum = summarize_unseen_static(static_dirs, plan)
    push_sum = summarize_unseen_push(push_dirs, plan)
    reg_sum = summarize_regressions(off_dir, flat_dir, dbl_dir, own_res)

    print("=== Rendering Original-Speed (1.0x, 25 fps) Unseen Videos & Contact Sheets ===", flush=True)
    videos = []
    for seed, d in static_dirs:
        v = eval_v2.render_grid_video(d, seed, out / f"Q1_recovery_v2_unseen_{seed}_1x.mp4")
        v["group"] = "static"
        v["seed"] = seed
        v["contact_sheet"] = str(Path(v["video"]).with_suffix(".jpg"))
        videos.append(v)
    pv = render_push_grid_video(push_dirs[0][1], push_seeds[0], out / f"Q1_recovery_v2_unseen_push_{push_seeds[0]}_1x.mp4")
    pv["group"] = "push"
    pv["seed"] = push_seeds[0]
    videos.append(pv)

    # Strict acceptance gates from acceptance_plan.json
    st_plan = plan["static"]
    pu_plan = plan["push"]
    pk_plan = plan["recovery_peaks"]
    all_owned_qd = max(static_sum["owned_peaks_including_failed"]["peak_qd"], push_sum["owned_peaks_including_failed"]["peak_qd"])
    all_owned_gnd = max(static_sum["owned_peaks_including_failed"]["ground_mm"], push_sum["owned_peaks_including_failed"]["ground_mm"])
    all_owned_self = max(static_sum["owned_peaks_including_failed"]["self_mm"], push_sum["owned_peaks_including_failed"]["self_mm"])

    gates = {
        "static_total_ge_114": static_sum["timely_success"] >= int(st_plan["minimum_success"]),
        "static_prone_ge_36": static_sum["by_pose"]["prone"]["succ"] >= int(st_plan["minimum_success_per_orientation"]),
        "static_supine_ge_36": static_sum["by_pose"]["supine"]["succ"] >= int(st_plan["minimum_success_per_orientation"]),
        "static_side_ge_36": static_sum["by_pose"]["side"]["succ"] >= int(st_plan["minimum_success_per_orientation"]),
        "static_confirm_le_15s": static_sum["all_timely_within_deadline"] and (static_sum["max_confirm_s_successful"] or 99.0) <= float(st_plan["standing_confirmation_deadline_s_from_takeover"]) + 1e-8,
        "static_wrong_handbacks_0": static_sum["wrong_release"] <= int(st_plan["incorrect_handbacks_maximum"]),
        "static_legal_settled_starts_120": static_sum["legal_settled_starts"] == static_sum["total"],
        "push_actual_falls_ge_20": push_sum["actual_falls"] >= int(pu_plan["minimum_actual_falls"]),
        "push_recovery_rate_ge_90pct": push_sum["recovery_rate_over_actual_falls"] >= float(pu_plan["recovery_rate_over_actual_falls_minimum"]) - 1e-9,
        "push_pre_impact_falls_0": push_sum["pre_impact_falls"] == 0,
        "push_wrong_handbacks_0": push_sum["wrong_release"] == 0,
        "owned_peak_qd_le_12_incl_failed": all_owned_qd <= float(pk_plan["joint_speed_maximum_rad_s"]) + 1e-9,
        "owned_ground_mm_le_10_incl_failed": all_owned_gnd <= float(pk_plan["ground_penetration_maximum_mm"]) + 1e-9,
        "owned_arm_leg_mm_le_2_incl_failed": all_owned_self <= float(pk_plan["arm_leg_penetration_maximum_mm"]) + 1e-9,
        "regression_offline_clean": (
            reg_sum["offline_84101"]["releases"] == 0
            and reg_sum["offline_84101"]["teleop_calls_while_owned"] == 0
            and reg_sum["offline_84101"]["terminal_standing"] == reg_sum["offline_84101"]["trials"]
        ),
        "regression_flat_no_takeover": reg_sum["flat_84101"]["takeovers"] == 0,
        "regression_ownership_stale_deterministic": all(bool(v) for v in own_res.values()),
        "regression_double_clean": (
            reg_sum["double_84111"]["wrong_release"] == 0
            and reg_sum["double_84111"]["releases_confirmed_all"]
            and reg_sum["double_84111"]["both_phases_recover"] == reg_sum["double_84111"]["both_fell"]
        ),
    }

    report = dict(
        status="strict_pass" if all(gates.values()) else "partial_fail_see_gates",
        overall_strict_acceptance_pass=all(gates.values()),
        acceptance_plan_sha256=sha256_file(args.plan),
        expert=str(expert),
        expert_sha256=sha256_file(expert),
        gate_config=dict(
            calm_upright_gate=eval_v2.GATE_CONFIG,
            blend_ticks=blend_ticks,
        ),
        onnx_verification=onnx_stats,
        gates=gates,
        static_unseen=static_sum,
        push_unseen=push_sum,
        combined_owned_peaks_including_failed=dict(
            peak_qd=all_owned_qd,
            ground_mm=all_owned_gnd,
            self_mm=all_owned_self,
        ),
        regressions=reg_sum,
        videos=videos,
    )

    run_base = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121")
    frozen_assets = [
        BUNDLE / "config/q1_deploy_config.json",
        BUNDLE / "robot_model/supermimic.xml",
        RUN_V1 / "eval_core.py",
        RUN_V1 / "evaluate.py",
        AUTON_ROOT / "q1_autonomous_recovery.py",
        RISE_DIR / "limbs.py",
        RUN_V1 / "common.py",
        SCRIPT_DIR / "eval_v2.py",
        SCRIPT_DIR / "unified_v3.py",
        Path(__file__).resolve(),
        args.plan.resolve(),
        args.roll.resolve(),
        run_base / "side_v2_06/best_landing.pt",
        args.side_ckpt.resolve(),
        args.supine_ckpt.resolve(),
        expert,
        run_base / "side_v2_06/train_curve_a1.png",
        run_base / "side_v2_06/train_curve_a1.csv",
        run_base / "side_v2_07/train_curve_a2.png",
        run_base / "side_v2_07/train_curve_a2.csv",
        run_base / "supine_v2_03/train_curve.png",
        run_base / "supine_v2_03/train_curve.csv",
        run_base / "supine_v2_04/train_curve.png",
        run_base / "supine_v2_04/train_curve.csv",
    ]
    snapshot = dict(
        runtime=dict(
            physics_ms=2,
            control_ms=20,
            sender_hz=500,
            observation_delay_ms_assumed=11,
            explicit_arm_leg_pairs=96,
            mujoco=mj.__version__,
            bundle=str(BUNDLE),
            deploy_codec_fingerprint=CONFIG["codec_fingerprint"],
            torque_safe_pd=CONFIG["torque_safe_pd"],
            sender_target_box_rad=plan["sender"]["target_box_from_default_rad"],
            sender_increment_per_2ms_rad=plan["sender"]["increment_per_physics_step_rad"],
        ),
        model=dict(
            expert=str(expert),
            expert_sha256=sha256_file(expert),
            onnx_verification=onnx_stats,
            gate_config=eval_v2.GATE_CONFIG,
            blend_ticks=blend_ticks,
            stage_configs={
                "side_v2_06": json.loads((run_base / "side_v2_06/config.json").read_text()),
                "side_v2_07": json.loads((run_base / "side_v2_07/config.json").read_text()),
                "supine_v2_03": json.loads((run_base / "supine_v2_03/config.json").read_text()),
                "supine_v2_04": json.loads((run_base / "supine_v2_04/config.json").read_text()),
            },
        ),
        files={str(f): dict(bytes=f.stat().st_size, sha256=sha256_file(f)) for f in frozen_assets},
    )

    (out / "frozen_config_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    (out / "final_acceptance_report.json").write_text(json.dumps(report, indent=2))
    print("=== UNSEEN FINAL ACCEPTANCE COMPLETE ===", flush=True)
    print(json.dumps(dict(overall=report["overall_strict_acceptance_pass"], gates=gates), indent=2), flush=True)


if __name__ == "__main__":
    main()
