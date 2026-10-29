"""Saved full-library tables, truthful six-panel plots and source-speed media."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "project_tools"),
               str(HERE.parent / "supermimic_q1_hp"), str(HERE.parent / "supermimic_q1_ppo")]
from resources import identity, save, stamp
from evidence import review_tables, write_csv


def dashboard(trial, line):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plot_training_dashboard import episode_points, rolling_ratio

    source = trial / "run/training_evidence.jsonl"
    rows = [json.loads(row) for row in source.read_text().splitlines()]
    if not rows or any(b["step"] <= a["step"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Nonempty increasing completed-iteration records required")
    panels = [(episode_points(rows, key), title) for key, title in (
        ("normalized_return_sum", "Completed Episode Normalized Return"),
        ("hard_terminations", "Hard Terminations / Completed Episodes"),
        ("length_steps_sum", "Completed Episode Length (steps)"))]
    panels.append(([(row["step"], row["reward_sum"], row["transitions"] * row["reward_maximum"])
                    for row in rows], "Step Reward / Maximum"))
    for key, title in (("wm_err8", "World Model Step-8 Error"),
                       ("imag_survival_H", "Imagined Endpoint Survival")):
        panels.append(([(row["step"], row[key], 1.) for row in rows if row[key] is not None], title))
    target = trial / "dashboard"
    target.mkdir(exist_ok=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    exported = []
    for axis, (points, title) in zip(axes.flat, panels):
        axis.set_title(title, fontsize=11)
        if points:
            raw = [num / den if den else float("nan") for _, num, den in points]
            mean = rolling_ratio(points, 500)
            axis.plot([p[0] for p in points], raw, alpha=.3, linewidth=.6, color="#23765e")
            axis.plot([p[0] for p in points], [v if v is not None else float("nan") for v in mean],
                      linewidth=1.8, color="#23765e", label="500-iteration pooled mean")
            axis.legend(fontsize=8)
            exported.extend(dict(metric=title, iteration=step, numerator=num, denominator=den,
                                 raw=num / den if den else None, mean500=average)
                            for (step, num, den), average in zip(points, mean))
        else:
            axis.text(.5, .5, "N/A - physical PPO" if line == "ppo" else "Not recorded",
                      ha="center", va="center", transform=axis.transAxes)
        axis.set_xlabel("Completed training iteration")
        axis.grid(alpha=.2)
    batch = "WM/actor1024, H32" if line == "base" else "PPO minibatch24576, 24 steps/env"
    fig.suptitle(f"GAIT_GOAL01 {line} | 4096 environments | {batch} | "
                 f"{rows[-1]['training_seconds']:.1f} actual training seconds", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(target / ("training_dashboard." + suffix), dpi=160)
    plt.close(fig)
    write_csv(target / "training_dashboard.csv", exported)
    save(target / "receipt.json", dict(status="TRAINING_DASHBOARD_COMPLETED", source=identity(source),
        rolling_window_iterations=500, episode_weight="Actual completed episodes",
        step_weight="Actual physical transitions", line=line,
        not_applicable=["wm_err8", "imag_survival_H"] if line == "ppo" else [],
        artifacts=[identity(path) for path in sorted(target.iterdir())]))


def audit_media(trial, media, suite, sources, rows):
    import numpy as np
    from audit_q1_rf13_media import sheet

    checks, videos = [], {}
    for label, declaration in media.items():
        if "alias_of" in declaration:
            continue
        video = Path(declaration["video"]["path"])
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(video)
        ], text=True))["streams"][0]
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-nostats",
            "-progress", "pipe:1", "-i", str(video), "-f", "null", "-"],
            check=True, capture_output=True, text=True)
        frames = [int(row.split("=", 1)[1]) for row in decoded.stdout.splitlines() if row.startswith("frame=")][-1]
        segments = declaration["segments"]
        with np.load(sources[label], allow_pickle=False) as trace:
            matched = all(segment["physical_steps"] == int(trace["steps"][index])
                          and segment["failed"] == bool(trace["failed"][index])
                          and segment["reference_frames_50hz"] == int(trace["reference_steps"][index]) + 1
                          for segment in segments
                          for index in [next(i for i, row in enumerate(rows) if row["name"] == segment["name"])])
        tests = dict(size_sha_matches=identity(video) == declaration["video"],
            h264_1280x480_25fps=probe["codec_name"] == "h264" and
                (probe["width"], probe["height"]) == (1280, 480) and Fraction(probe["avg_frame_rate"]) == 25,
            full_decode_1350_frames=frames == int(probe["nb_frames"]) == declaration["frames"] == 1350
                and not decoded.stderr.strip(),
            original_speed_54_seconds=declaration["speed"] == 1 and declaration["fps"] == 25
                and declaration["duration_seconds"] == frames / 25 == 54.,
            fixed_segments_and_failure_evidence=matched and len(segments) == 8
                and [row["name"] for row in segments] == [row["name"] for row in suite["clips"]]
                and "hold after termination" in declaration["left"],
            contiguous_segments=segments[0]["first_video_frame"] == 0
                and segments[-1]["last_video_frame"] == frames - 1
                and all(b["first_video_frame"] == a["last_video_frame"] + 1 for a, b in zip(segments, segments[1:]))
                and all(row["last_video_frame"] - row["first_video_frame"] + 1
                        == (row["reference_frames_50hz"] + 1) // 2 for row in segments))
        checks.extend(dict(name=label + "_" + key, passed=bool(value)) for key, value in tests.items())
        mids = [(row["first_video_frame"] + row["last_video_frame"]) // 2 for row in segments]
        walk = [row["first_video_frame"] + round((row["last_video_frame"] - row["first_video_frame"]) * fraction)
                for row in segments[:2] for fraction in (.05, .2, .35, .5, .7, .95)]
        raised = segments[2]
        raise_frames = [raised["first_video_frame"] + round((raised["last_video_frame"] - raised["first_video_frame"]) * i / 7)
                        for i in range(8)]
        sheets = {name: sheet(video, selected, video.with_name(label + "_" + name + ".png"))
                  for name, selected in (("segment_sheet", mids), ("walk_sequence", walk), ("raise_sequence", raise_frames))}
        videos[label] = dict(video=identity(video), contact_sheets=sheets)
    passed = bool(checks) and all(row["passed"] for row in checks)
    save(trial / "media_audit.json", dict(status="MEDIA_AUDIT_PASSED" if passed else "FAILED",
        checks=checks, videos=videos, aliases={key: value["alias_of"] for key, value in media.items() if "alias_of" in value},
        visual_review_completed=False, new_physical_steps=0, optimizer_updates=0))
    if not passed:
        raise AssertionError("Saved media integrity/protocol failed")


def run(args):
    import numpy as np
    import torch

    status = args.trial / "delivery_status.json"
    def progress(phase):
        save(status, dict(status="RUNNING", phase=phase, updated_utc=stamp()))
    progress("saved_tables_and_dashboard")
    tables = review_tables(args.trial)
    dashboard(args.trial, args.line)
    sys.path[:0] = [str(args.runtime / "src"), str(args.runtime)]
    from unisupermimic.tasks.motion_tracking.q1.supermimic.kinematics import Q1Kinematics
    from unisupermimic.tasks.motion_tracking.q1.supermimic.motion import MotionLib
    from video_suite import render_suite
    torch.set_num_threads(8)
    receipt = json.loads((args.trial / "evaluation/receipt.json").read_text())
    rows = receipt["rows"]
    suite = json.loads(args.video_suite.read_text())
    if [row["name"] for row in suite["clips"][:3]] != [
            "walk_ff_stop_180_R_002__A095", "walk_ff_start_180_R_002__A071", "triumph_two_handed_001__A058"]:
        raise ValueError("Frozen ordinary-walk/raise suite required")
    kin = Q1Kinematics(args.runtime / "q1_assets/scene.xml", "cpu")
    motion = MotionLib([row["path"] for row in rows], kin, "cpu", src_fps=30.)
    env = SimpleNamespace(supermimic_mdp=SimpleNamespace(motion=motion), device=torch.device("cpu"))
    media, canonical, sources = {}, {}, {}
    for label in ("initial", "final", "recommended"):
        source = Path(receipt[label + "_traces"][0])
        sources[label] = source
        digest = identity(source)["sha256"]
        if digest in canonical:
            original = canonical[digest]
            media[label] = dict(media[original], alias_of=original, requested_role=label)
        else:
            progress(label + "_saved_trajectory_video")
            with np.load(source, allow_pickle=False) as saved:
                trace = {key: saved[key] for key in ("initial", "states", "steps", "failed")}
            media[label] = render_suite(args.runtime, trace, env, rows, suite,
                args.trial / f"evaluation/{label}.mp4", f"GAIT_GOAL01 {args.line} {label}, upper0.50")
            media[label]["trace"] = identity(source)
            canonical[digest] = label
        save(args.trial / f"evaluation/{label}_media.json", media[label])
    progress("saved_media_audit")
    audit_media(args.trial, media, suite, sources, rows)
    save(status, dict(status="COMPLETED", completed_utc=stamp(), tables=tables, media=media,
        new_physical_steps=0, policy_inference_calls=0, optimizer_updates=0, visual_review_completed=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "trial", "video-suite"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    run(parser.parse_args())
