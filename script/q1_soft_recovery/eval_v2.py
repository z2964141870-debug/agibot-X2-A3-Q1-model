"""Evaluate UnifiedRecoveryV2 with CalmUprightRecovery handoff & render original-speed videos."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np

RUN_V1 = Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_20261121")
RISE_DIR = Path("/home/yu/projects/SuperMimicHP/runs/q1_supported_rise_20261121")
AUTON_ROOT = Path("/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery")
for p in reversed((str(RUN_V1), str(RISE_DIR), str(AUTON_ROOT))):
    if p not in sys.path:
        sys.path.insert(0, p)

import evaluate
import eval_core
from common import BUNDLE, mj
from recovery import Gate

BASE_RECOVERY_CLS = eval_core.Q1AutonomousRecovery

DEFAULT_BLEND_TICKS = 16
GATE_CONFIG = dict(
    h_min=0.355,
    up_z_min=0.965,
    rms_leg_max=0.35,
    w_norm_max=0.70,
    qd_rms_max=0.95,
    calm_ticks=4,
)


class CalmUprightGate(Gate):
    """Switch from recovery expert to private V2b stand only once upright and calm (or standard stand dwell)."""

    def reset(self):
        super().reset()
        self.calm_ticks = 0

    def advance(self, x, width):
        if self.mode == "recover":
            qd = x[22:44] * 10.0
            w = x[44:47] * 4.0
            upright_calm = (
                bool(x[-1] > 0.5)
                and float(x[-2]) > GATE_CONFIG["h_min"]
                and -float(x[49]) > GATE_CONFIG["up_z_min"]
                and float(np.sqrt(np.mean(x[:12] ** 2))) < GATE_CONFIG["rms_leg_max"]
                and float(np.linalg.norm(w)) < GATE_CONFIG["w_norm_max"]
                and float(np.sqrt(np.mean(qd ** 2))) < GATE_CONFIG["qd_rms_max"]
            )
            self.calm_ticks = self.calm_ticks + 1 if upright_calm else 0
            if self.calm_ticks >= GATE_CONFIG["calm_ticks"]:
                self.mode = "track"
                self.tilt_ticks = self.stand_ticks = self.calm_ticks = self.blend_ticks = 0
                return 0.0, True
        else:
            self.calm_ticks = 0
        return super().advance(x, width)


def make_calm_controller(
    blend_ticks: int = DEFAULT_BLEND_TICKS,
    leg_speed_cap: float = 8.5,
    leg_brake_gain: float = 0.5,
):
    class SoftRecoveryOwner(BASE_RECOVERY_CLS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rise.expert.gate = CalmUprightGate()
            self.rise.expert.gate.reset()

        def reset(self):
            super().reset()
            self.origin_target = None
            self.blend_index = blend_ticks
            self.protecting = False
            self.imminent_ticks = 0
            self.protect_pose = None

        def step(self, R, w, q, qd, previous, base, anchor, *, sample_time_s, teleop_stamp_s):
            c = self.tracker
            if (
                not math.isfinite(sample_time_s)
                or (self.last is not None and sample_time_s <= self.last)
                or not all(np.isfinite(x).all() for x in (R, w, q, qd, previous))
            ):
                self.since = None
                self.standing_confirmed = False
                self.fall_count = 0
                self.imminent_ticks = 0
                raise ValueError("expected finite proprioception and increasing sample time")
            fresh = (
                teleop_stamp_s is not None
                and math.isfinite(teleop_stamp_s)
                and 0.0 <= sample_time_s - teleop_stamp_s <= 0.1 + 1e-9
            )
            if self.owner == "teleop" and fresh and not self.protecting:
                up = R[2]
                vertical_rate = float(np.cross(up, w)[2])
                imminent = (
                    up[2] < 0.88 and vertical_rate < -2.0 and up[2] + 0.18 * vertical_rate < 0.50
                ) or (up[2] < 0.71 and vertical_rate < -0.6)
                severe = (up[2] < 0.82 and vertical_rate < -2.1) or (up[2] < 0.70)
                self.imminent_ticks = self.imminent_ticks + 1 if imminent else 0
                if self.imminent_ticks >= 2 or severe:
                    self.owner = "recovery"
                    self.phase = "protect"
                    self.entered = sample_time_s
                    self.standing_confirmed = False
                    self.since = self.hold_R = None
                    self.fall_count = 0
                    self.protecting = True
                    self.imminent_ticks = 0
                    self.rise.reset()
                    self.rise.expert.gate.mode = "recover"
                    self.protect_pose = None
            elif not fresh or self.owner != "teleop":
                self.imminent_ticks = 0

            if self.protecting:
                age = sample_time_s - self.entered
                exit_protect = (
                    age >= 0.20
                    and abs(R[2, 2]) < 0.42
                    and float(np.linalg.norm(w)) < 1.3
                    and float(np.max(np.abs(qd))) < 4.5
                ) or age >= 0.36
                if not exit_protect:
                    self.last = sample_time_s
                    self.rise.expert.state.fill(0.0)
                    target = q.copy()
                    target[15] = q[15] + 0.35 * (max(q[15], 0.65) - q[15])
                    target[19] = q[19] + 0.35 * (min(q[19], -0.65) - q[19])
                    target[17] = q[17] + 0.25 * (min(q[17], 0.30) - q[17])
                    target[21] = q[21] + 0.25 * (min(q[21], 0.30) - q[21])
                    target = np.clip(target, c.default_q - 2.0, c.default_q + 2.0)
                    tau = np.clip(c.kp * (target - q) - c.kd * qd, -c.eff, c.eff)
                    return target, tau, 1.0
                self.protecting = False
                self.phase = "rise"

            before = self.rise.expert.gate.mode
            target, tau, owned = super().step(
                R,
                w,
                q,
                qd,
                previous,
                base,
                anchor,
                sample_time_s=sample_time_s,
                teleop_stamp_s=teleop_stamp_s,
            )
            after = self.rise.expert.gate.mode
            if owned == 0.0:
                self.protecting = False
                self.imminent_ticks = 0
                self.protect_pose = None
                self.origin_target = None
                self.blend_index = blend_ticks
                return target, tau, owned
            if before == "recover" and after in ("track", "handoff") and owned:
                self.origin_target = c.default_q + c.action_scale * previous
                self.blend_index = 0
            if after == "recover":
                self.blend_index = blend_ticks
                return target, tau, owned

            out = target.copy()
            if self.blend_index < blend_ticks and self.origin_target is not None:
                self.blend_index += 1
                weight = 0.5 - 0.5 * math.cos(math.pi * self.blend_index / blend_ticks)
                out = self.origin_target + weight * (out - self.origin_target)

            # Proprioceptive leg-joint speed guard during private V2b settle to avoid foot-stomp spikes.
            tau_cmd = np.clip(c.kp * (out - q) - c.kd * qd, -c.eff, c.eff)
            leg_idx = slice(0, 12)
            excess = np.maximum(np.abs(qd[leg_idx]) - leg_speed_cap, 0.0)
            if excess.any():
                tau_cmd[leg_idx] = np.clip(
                    tau_cmd[leg_idx] - np.sign(qd[leg_idx]) * excess * (leg_brake_gain * c.kd[leg_idx]),
                    -c.eff[leg_idx],
                    c.eff[leg_idx],
                )
            guarded_target = np.clip(q + (tau_cmd + c.kd * qd) / c.kp, c.default_q - 2.0, c.default_q + 2.0)
            tau_cmd = np.clip(c.kp * (guarded_target - q) - c.kd * qd, -c.eff, c.eff)
            return guarded_target, tau_cmd, owned

    return SoftRecoveryOwner


def run_group(expert: Path, group: str, seed: int, out: Path, blend_ticks: int):
    eval_core.Q1AutonomousRecovery = make_calm_controller(blend_ticks=blend_ticks)
    evaluate.main(
        argparse.Namespace(
            group=group,
            legacy=False,
            catch=False,
            out=out,
            seed=seed,
            expert=expert,
            delay=11,
        )
    )


def summarize_static(static_dirs: list[tuple[int, Path]]):
    rows = []
    for seed, d in static_dirs:
        res = json.loads((d / "results.json").read_text())
        by_pose = {}
        for pose in ("prone", "supine", "side"):
            sub = [r for r in res if r["pose"] == pose]
            succ = [r for r in sub if r["success"]]
            soft = [
                r
                for r in succ
                if r.get("peak_qd", 99) <= 12.0 and r.get("ground_mm", 99) <= 10.0 and r.get("self_mm", 99) <= 2.0
            ]
            by_pose[pose] = dict(
                succ=len(succ),
                soft=len(soft),
                n=len(sub),
                worst_qd=max(r.get("peak_qd", 0.0) for r in sub),
                worst_gnd=max(r.get("ground_mm", 0.0) for r in sub),
                worst_self=max(r.get("self_mm", 0.0) for r in sub),
            )
        all_succ = sum(r["success"] for r in res)
        all_soft = sum(
            1
            for r in res
            if r["success"] and r.get("peak_qd", 99) <= 12.0 and r.get("ground_mm", 99) <= 10.0 and r.get("self_mm", 99) <= 2.0
        )
        confirm_s = [
            r["first_confirmation_s"] - r["takeovers"][0] * 0.02
            for r in res
            if r["success"] and r.get("first_confirmation_s") is not None and r.get("takeovers")
        ]
        rows.append(
            dict(
                seed=seed,
                success=all_succ,
                soft_success=all_soft,
                n=len(res),
                wrong_release=sum(r["wrong_release"] for r in res),
                max_confirm_s=max(confirm_s, default=None),
                peak_qd=max(r.get("peak_qd", 0.0) for r in res),
                ground_mm=max(r.get("ground_mm", 0.0) for r in res),
                self_mm=max(r.get("self_mm", 0.0) for r in res),
                by_pose=by_pose,
            )
        )
    return rows


def render_grid_video(static_dir: Path, seed: int, out_mp4: Path):
    from PIL import Image, ImageDraw, ImageFont

    res = json.loads((static_dir / "results.json").read_text())
    picks = []
    for pose in ("prone", "supine", "side"):
        sub = [r for r in res if r["pose"] == pose]
        succ = [r for r in sub if r["success"]]
        picks.append((succ[0] if succ else sub[0], f"{pose.upper()} ({'OK' if (succ or sub)[0]['success'] else 'FAIL'})"))

    # 4th panel: distinct worst failure if any, else distinct worst softness peak trial
    picked_ids = {p[0]["trial"] for p in picks}
    rem = [r for r in res if r["trial"] not in picked_ids] or res
    fails = [r for r in rem if not r["success"]]
    if fails:
        worst = fails[0]
        w_label = f"WORST FAIL T{worst['trial']:02d} ({worst['pose']})"
    else:
        worst = max(rem, key=lambda r: max(r.get("peak_qd", 0) / 12, r.get("ground_mm", 0) / 10, r.get("self_mm", 0) / 2))
        w_label = f"WORST PEAK T{worst['trial']:02d} ({worst['pose']})"
    picks.append((worst, w_label))

    panels = []
    max_ticks = 0
    for r, label in picks:
        t = r["trial"]
        data = np.load(static_dir / f"trial_{t:02d}.npz")
        max_ticks = max(max_ticks, len(data["qpos"]))
        panels.append((r, label, data))

    model = mj.MjModel.from_xml_path(str(static_dir / "scene/fall_hard.xml"))
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
            pen.text((12, 9), f"Q1 Recovery V2 | seed {seed} | 1.0x speed (25 fps, 2 ms physics)", font=font, fill="white")
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
            if len(sheets) < 4 and tick in (0, 120, 260, min(420, max_ticks - (max_ticks % 2))):
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
    return dict(video=str(out_mp4), fps=25, frames=n_frames, speed=1.0, sha256=hashlib.sha256(out_mp4.read_bytes()).hexdigest())


def check_soft_ownership(expert: Path, blend_ticks: int = DEFAULT_BLEND_TICKS) -> dict:
    """Deterministic ownership, finite/monotonic timestamp guard, and stale-input regression check."""
    tracker = eval_core.bench.Q1V2bOnnxController(BUNDLE, ("robot_pose",))
    owner_cls = make_calm_controller(blend_ticks=blend_ticks)
    branch = owner_cls(tracker, expert, BUNDLE / "robot_model/supermimic.xml")
    q = tracker.default_q.copy()
    zero = np.zeros_like(q)
    w = np.zeros(3, dtype=np.float32)
    upright = np.eye(3, dtype=np.float32)
    down = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    calls = []

    def live():
        calls.append("teleop")
        return q.copy(), zero.copy(), {}

    def anchor():
        calls.append("anchor")

    def tick(t, stamp, R=upright, rate=w, joints=q):
        return branch.step(
            R, rate, joints, zero, zero, live, anchor, sample_time_s=t, teleop_stamp_s=stamp
        )[2]

    assert tick(0.0, 0.0) == 0.0 and branch.owner == "teleop"
    assert calls == ["teleop"]
    calls.clear()

    assert tick(0.02, 0.02, down) == 1.0
    assert branch.owner == "recovery" and branch.phase == "protect" and branch.protecting
    assert not calls

    try:
        tick(0.02, 0.02, down)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate timestamp accepted during protect")
    nan_q = q.copy()
    nan_q[0] = float("nan")
    try:
        tick(0.03, 0.03, upright, w, nan_q)
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite proprioception accepted during protect")

    for k in range(2, 19):
        assert tick(k * 0.02, k * 0.02, down) == 1.0
    assert not calls
    original_step = branch.rise.step
    branch.rise.step = lambda *args: (q.copy(), zero.copy(), 0.0)

    for k in range(19, 60):
        assert tick(k * 0.02, None) == 1.0
    assert branch.standing_confirmed and not calls and branch.owner == "recovery"

    assert tick(1.22, 0.92) == 1.0
    assert tick(1.24, 1.26) == 1.0
    assert tick(1.26, 0.01) == 1.0
    assert not calls
    assert tick(1.28, 1.28, rate=np.ones(3, np.float32)) == 1.0
    assert not branch.standing_confirmed and not calls

    for k in range(65, 81):
        assert tick(k * 0.02, k * 0.02) == 1.0
    assert tick(2.30, 2.30) == 1.0 and not calls
    for k in range(116, 140):
        assert tick(k * 0.02, k * 0.02) == 1.0
    assert not calls
    assert tick(2.80, 2.80) == 0.0
    assert calls == ["anchor", "teleop"] and branch.owner == "teleop"

    calls.clear()
    assert tick(2.82, 2.82, down) == 1.0 and branch.owner == "recovery" and not calls

    branch.reset()
    assert branch.owner == "teleop" and not branch.standing_confirmed
    assert tick(0.0, None, down) == 1.0
    branch.rise.step = original_step
    for k in range(1, 1501):
        assert tick(k * 0.02, k * 0.02, down) == 1.0
    assert not calls and branch.owner == "recovery"

    return {
        "automatic_takeover": True,
        "protect_duplicate_timestamp_rejected": True,
        "protect_nonfinite_rejected": True,
        "offline_hold": True,
        "stale_future_pre_takeover_rejected": True,
        "instability_restarts_dwell": True,
        "sample_gap_restarts_dwell": True,
        "anchor_before_teleop": True,
        "second_fall": True,
        "timestamp_reset": True,
        "persistent_down_no_timeout_s": 30,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "expert",
        type=Path,
        nargs="?",
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/unified_v3_08/expert.onnx"),
    )
    p.add_argument(
        "out",
        type=Path,
        nargs="?",
        default=Path("/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/dev_eval_07"),
    )
    p.add_argument("--blend-ticks", type=int, default=DEFAULT_BLEND_TICKS)
    p.add_argument("--groups", default="static,push,offline,flat,double")
    p.add_argument("--no-videos", action="store_true")
    p.add_argument("--ownership-only", action="store_true")
    args = p.parse_args()
    expert = args.expert.resolve()
    own_res = check_soft_ownership(expert, blend_ticks=args.blend_ticks)
    print("=== OWNERSHIP CHECK ===", json.dumps(own_res), flush=True)
    if args.ownership_only:
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    wanted = set(g.strip() for g in args.groups.split(",") if g.strip())

    static_dirs = []
    if "static" in wanted:
        for seed in (92017, 92019, 92117):
            d = out / f"static_{seed}"
            print(f"=== Running static seed {seed} ===", flush=True)
            run_group(expert, "static", seed, d, args.blend_ticks)
            static_dirs.append((seed, d))

    push_dir = None
    if "push" in wanted:
        print("=== Running push seed 92023 ===", flush=True)
        push_dir = out / "push_92023"
        run_group(expert, "push", 92023, push_dir, args.blend_ticks)

    reg_dirs = {}
    for grp, seed in (("offline", 92017), ("flat", 92017), ("double", 92023)):
        if grp in wanted:
            print(f"=== Running regression {grp} seed {seed} ===", flush=True)
            rd = out / f"{grp}_{seed}"
            run_group(expert, grp, seed, rd, args.blend_ticks)
            reg_dirs[grp] = rd

    static_summary = summarize_static(static_dirs) if static_dirs else []

    push_detail = []
    push_res = []
    if push_dir is not None:
        push_res = json.loads((push_dir / "results.json").read_text())
        for r in push_res:
            t = r["trial"]
            tr = np.load(push_dir / f"trial_{t:02d}.npz")
            peaks = tr["peaks"]
            cap = r["takeovers"][0] if r["takeovers"] else -1
            if cap >= 0:
                s1 = slice(cap, min(len(peaks), cap + 40))
                s2 = slice(min(len(peaks), cap + 40), len(peaks))
                p1 = (float(peaks[s1, 0].max()), float(peaks[s1, 6].max() * 1000), float(peaks[s1, 7].max() * 1000))
                p2 = (
                    (float(peaks[s2, 0].max()), float(peaks[s2, 6].max() * 1000), float(peaks[s2, 7].max() * 1000))
                    if s2.start < s2.stop
                    else (0.0, 0.0, 0.0)
                )
            else:
                p1 = p2 = (0.0, 0.0, 0.0)
            push_detail.append(
                dict(
                    trial=t,
                    fell=r["phases"][0]["fell"],
                    success=r["success"],
                    pre_impact_fall=r["pre_impact_fall"],
                    peak_qd=r.get("peak_qd", 0.0),
                    ground_mm=r.get("ground_mm", 0.0),
                    self_mm=r.get("self_mm", 0.0),
                    cap_plus40=p1,
                    after_40=p2,
                )
            )

    reg_summary = {}
    for grp, rd in reg_dirs.items():
        rr = json.loads((rd / "results.json").read_text())
        if grp == "offline":
            reg_summary[grp] = dict(
                trials=len(rr),
                releases=sum(len(x["releases"]) for x in rr),
                teleop_calls=sum(x["teleop_calls_while_owned"] for x in rr),
                terminal_standing=sum(x["terminal_standing"] for x in rr),
            )
        elif grp == "flat":
            reg_summary[grp] = dict(
                trials=len(rr),
                takeovers=sum(len(x["takeovers"]) for x in rr),
                terminal_standing=sum(x["terminal_standing"] for x in rr),
            )
        else:
            reg_summary[grp] = dict(
                trials=len(rr),
                both_fell=sum(1 for x in rr if len(x["phases"]) == 2 and all(p["fell"] for p in x["phases"])),
                both_phases_recover=sum(
                    1
                    for x in rr
                    if len(x["phases"]) == 2 and all(p["post_1s"] for p in x["phases"])
                ),
                peak_qd=max(x.get("peak_qd", 0.0) for x in rr),
                ground_mm=max(x.get("ground_mm", 0.0) for x in rr),
                self_mm=max(x.get("self_mm", 0.0) for x in rr),
            )

    videos = []
    if not args.no_videos:
        for seed, d in static_dirs:
            v = render_grid_video(d, seed, out / f"Q1_recovery_v2_{seed}_1x.mp4")
            videos.append(v)

    push_summary = (
        dict(
            fell=sum(x["fell"] for x in push_detail),
            success=sum(x["success"] for x in push_detail),
            n=len(push_detail),
            pre_impact_falls=sum(x["pre_impact_fall"] for x in push_detail),
            wrong_release=sum(r["wrong_release"] for r in push_res),
            peak_qd=max(x["peak_qd"] for x in push_detail),
            ground_mm=max(x["ground_mm"] for x in push_detail),
            self_mm=max(x["self_mm"] for x in push_detail),
            trials=push_detail,
        )
        if push_detail
        else {}
    )
    report = dict(
        expert=str(expert),
        expert_sha256=hashlib.sha256(expert.read_bytes()).hexdigest(),
        gate_config=GATE_CONFIG,
        blend_ticks=args.blend_ticks,
        ownership_check=own_res,
        static_summary=static_summary,
        push_summary=push_summary,
        regressions=reg_summary,
        videos=videos,
    )
    (out / "dev_report.json").write_text(json.dumps(report, indent=2))
    print("=== DEV BENCHMARK REPORT ===", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
