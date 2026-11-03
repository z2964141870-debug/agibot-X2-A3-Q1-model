"""Complete shared evaluation inventory, conclusions, and independent Mac acceptance."""

import argparse
from pathlib import Path
import subprocess
import sys

from comparison import case_index
from controller import updates
from contracts import expected_transitions
from delivery import artifact, check_visuals, load_manifest, read
from common import FIXED, ROOT, TARGET_TRANSITIONS, identity, now, require, save


def required_paths(names):
    from shared_review import comparisons, jobs
    require(len(names) == 393 and len(set(names)) == 393 and set(FIXED) <= set(names), "Wrong frozen delivery clips")
    paths = {"evaluation.json", "guard_config.json", "conclusions.json", "README_RESULTS.md",
             "training_pipeline.json", "training_plan.json", "checkpoint_audit_initial.json",
             "checkpoint_audit_final.json", "primebot_checkpoint_audit.json", "frozen_data.json"}
    videos = set()
    for job in jobs():
        phase = job["name"]
        paths.update(phase + "/" + name for name in ("status.json", "guard.json", "transition_receipt.json", "review/result.json"))
        if job["plant"] == "common":
            paths.add(phase + "/review/policy_worker.log")
        selected, seeds = (names, (0,)) if job["mode"] == "library" else (FIXED, (0, 1, 2))
        for name in selected:
            for seed in seeds:
                prefix = f"{phase}/review/{name}_s{seed}/"
                paths.update(prefix + member for member in ("result.json", "foot_posture.csv", "upper_timing.csv"))
                if job["mode"] == "fixed" and seed == 0:
                    videos.add(prefix + "tracking.mp4")
    for job in comparisons():
        paths.update(job["name"] + "/" + member for member in
                     ("status.json", "guard.json", "transition_receipt.json", "comparison.json"))
    for robot in ("g1", "q1"):
        paths.update(f"curves/{robot}/" + member for member in
                     ("status.json", "guard.json", "transition_receipt.json", "plots/metrics.csv",
                      "plots/training.png", "plots/training.pdf", "plots/source_receipt.json"))
    return paths | videos, videos


def diagnostic_table(comparisons):
    lines = ["| Comparison | Side | Complete Clips | No Alternating Events | Median Sole p95 L/R (mm) | Median Loaded Slip L/R (mm/s) | Median Upper RMSE (rad) |",
             "| --- | --- | ---: | ---: | --- | --- | ---: |"]
    for name, comparison in comparisons.items():
        for side, motion in comparison["complete_motion"].items():
            diagnosis = motion["diagnostics"]
            stats = diagnosis["statistics"]

            def median(key, scale=1.):
                value = stats.get(key, {}).get("median")
                return "Unavailable" if value is None else f"{value * scale:.6g}"

            sole = " / ".join(median("sole_p95_" + foot + "_m", 1000) for foot in ("left", "right"))
            slip = " / ".join(median("loaded_slip_" + foot + "_m_s", 1000) for foot in ("left", "right"))
            count = diagnosis["counts"].get("clips_with_no_alternating_events", "Unavailable")
            lines.append(f"| {name} | {side} | {diagnosis['complete_clips']} / {diagnosis['total_clips']} "
                         f"| {count} | {sole} | {slip} | {median('upper_joint_rmse_rad')} |")
    return lines


def write_conclusions(root, plan, config, exposure):
    from shared_review import comparisons
    primebot_root = Path(config["primebot_review"])
    primebot_audit = read(primebot_root / "checkpoint_audit.json")
    comparisons_value = {job["name"].split("/")[-1]: read(root / job["name"] / "comparison.json")
                         for job in comparisons()}
    result = dict(status="EVALUATED_PENDING_MAC_ACCEPTANCE", created_utc=now(),
        primebot=dict(initial=primebot_audit["initial"], final=primebot_audit["final"],
                      fixed_final_complete=read(primebot_root / "final_fixed/result.json")["complete_cases"],
                      library_count=len(read(primebot_root / "final_library/result.json")["cases"])),
        shared=dict(configuration=plan["cfg"], training_commit=plan["code_commit"],
                    per_robot_transitions=expected_transitions(plan["cfg"]),
                    per_robot_optimizer_updates=updates(plan["cfg"], plan["cfg"]["iterations"])),
        comparisons=comparisons_value, equal_exposure=dict(exposure,
            target_q1_transitions=TARGET_TRANSITIONS, additional_training_authorized=False),
        conclusions=dict(primebot_run_through_gate_passed=True, natural_gait_qualified=False,
            g1_shared_initial_final_is_not_g1_single_robot_control=True,
            training_set_results_only=True, unseen_morphology_zero_shot=False,
            complete_recipe_and_equal_exposure_must_remain_separate=True),
        limits=["Q1 V3 differs from the upstream external V2 example; both routes share the repaired393 references.",
               "Common CPU Q1 physics uses the locked original Primebot Python/MuJoCo runtime; only native policy inputs/actions cross a CPU worker pipe.",
               "Native metrics use present joints and13 tracked bodies; the policy worker uses its own frozen Morph dependencies.",
               "Native mjlab and common CPU plant results have different control/termination implementations and cannot be pooled.",
               "Nominal seeds0/1/2 do not establish performance under independent randomized disturbances.",
               "Foot contact load is sampled at control endpoints and does not establish continuous support.",
               "G1 initial/final results alone cannot isolate the effect of sharing versus a matched G1-only training control.",
               "Early-stopped Primebot lacks equal Q1 exposure; no extra training is authorized to fill it."],
        goal_completed=False, user_accepted_checkpoint=None)
    save(root / "conclusions.json", result)
    lines = ["# Primebot And Shared G1/Q1 Results", "",
        "Server evaluation is complete. Mac hash/decode/visual acceptance remains required. The five-stage goal is not settled.", "",
        "Primebot passed the approved run-through gate. Survival and tracking errors do not establish natural gait quality.", "",
        "| Comparison | Left/Right Complete | Joint Common Prefix (rad) | Relative Body Common Prefix (m) |",
        "| --- | --- | --- | --- |"]
    for name, row in comparisons_value.items():
        counts = row["complete_motion"]
        scores = row["common_prefix_metrics"]
        metrics = [f"{scores[key]['left']:.6g} / {scores[key]['right']:.6g}" if key in scores else "Unavailable"
                   for key in ("joint_rmse_rad", "body_relative_rmse_m")]
        lines.append(f"| {name} | {counts['left']['complete_cases']} / {counts['right']['complete_cases']} "
                     f"of {counts['right']['total_cases']} | {metrics[0]} | {metrics[1]} |")
    lines.extend(["", "Complete-motion diagnostics are aggregated with equal clip weights. Failed clips are excluded "
                  "and reported as missing full-motion evidence. Threshold crossings are not verified steps.", "",
                  *diagnostic_table(comparisons_value)])
    lines.extend(["", f"Primebot stopped at {primebot_audit['final']['iteration']} iterations and "
                  f"{primebot_audit['final']['global_env_steps']} Q1 transitions. Shared training used "
                  f"{plan['cfg']['iterations']} iterations and {expected_transitions(plan['cfg'])} transitions per robot.",
                  "", "Equal-exposure comparison: " + exposure["status"] + ". No additional training was started.",
                  "", "Per-case foot clearance, alternating lift events, loaded slip, tilt and upper-body timing are in the JSON/CSV records. "
                  "Fixed videos retain early failures and labeled holds.", "",
                  *["- " + item for item in result["limits"]], ""])
    (root / "README_RESULTS.md").write_text("\n".join(lines))


def prepare(root, config, names, output):
    from shared_review import comparisons, jobs, phase_passed
    require(not output.exists() and root.is_absolute() and root.is_relative_to(ROOT / "data/supermimic_for_q1"),
            "Build a fresh delivery manifest on the sole server")
    evaluation = read(root / "evaluation.json")
    expected_phases = {job["name"] for job in jobs() + comparisons()} | {"curves/g1", "curves/q1"}
    phases = evaluation.get("phases", [])
    require(evaluation.get("status") == "SHARED_EVALUATION_COMPLETED_PENDING_MAC_DELIVERY"
            and len(phases) == len(expected_phases) and {row.get("name") for row in phases} == expected_phases,
            "Full serial evaluation and comparisons did not finish")
    for row in phases:
        phase_passed(read(root / row["name"] / "transition_receipt.json"))
        require(identity(row["result"]["path"]) == row["result"], "Phase result changed after evaluation")
    paths, expected_videos = required_paths(names)
    media = []
    for job in jobs():
        result = read(root / job["name"] / "review/result.json")
        selected, seeds = (names, (0,)) if job["mode"] == "library" else (FIXED, (0, 1, 2))
        cases = case_index(result, selected, seeds, job["interface"])
        for row in cases.values():
            relative = f"{job['name']}/review/{row['clip']}_s{row['seed']}/"
            require(read(root / relative / "result.json") == row, "Per-case result differs from its evaluation")
            if job["mode"] != "fixed" or row["seed"] != 0:
                continue
            video = row.get("video", {})
            require(video.get("status") == "PASSED" and video.get("original_speed") is True
                    and video.get("failure_preserved") is True
                    and video.get("actual_frames_before_hold") == row["attempted_steps"] + 1,
                    "Missing original-speed video or preserved failure")
            frames = row["reference_steps"] + 1
            probe = video["probe"]
            require(probe["nb_frames"] == frames and probe["r_frame_rate"] == "50/1"
                    and (probe["width"], probe["height"]) == (1280, 392), "Wrong fixed video timing or framing")
            video_path = relative + "tracking.mp4"
            item = artifact(root, video_path)
            require((item["bytes"], item["sha256"]) == (video["bytes"], video["sha256"]), "Rendered video changed")
            contacts = [relative + f"frame_{frame:05d}.png" for frame in (0, frames // 2, frames - 1)]
            paths.update(contacts)
            media.append(dict(relative=video_path, probe=probe, contact_frames=contacts))
    require(len(media) == len(expected_videos) == 36 and {row["relative"] for row in media} == expected_videos,
            "Need every36 distinct fixed initial/final native/common video")
    value = dict(status="READY_FOR_MAC_VERIFICATION", created_utc=now(), review_root=str(root),
        manifest_server_path=str(output), clip_names=names, frozen_data=identity(config["frozen_data"]),
        artifacts=[artifact(root, name) for name in sorted(paths)], media=media,
        checkpoint_npz_scene_storage="SERVER_LOCAL_ONLY", goal_completed=False, user_accepted_checkpoint=None)
    save(output, value)
    return value


def verify(manifest_path, local_root, visual_path, output, ffmpeg):
    require(not output.exists(), "Preserve an existing Mac acceptance receipt")
    manifest, visual = load_manifest(manifest_path), read(visual_path)
    required, expected_videos = required_paths(manifest["clip_names"])
    rows = manifest.get("artifacts", [])
    relatives = [row["relative"] for row in rows]
    media = manifest.get("media", [])
    require(manifest.get("status") == "READY_FOR_MAC_VERIFICATION" and len(set(relatives)) == len(relatives)
            and required <= set(relatives) and len(media) == 36
            and {row["relative"] for row in media} == expected_videos
            and {name for name in relatives if name.endswith(".mp4")} == expected_videos
            and not any(Path(name).suffix.lower() in (".pt", ".npz", ".mjb") for name in relatives),
            "Incomplete, duplicate, or oversized shared delivery inventory")
    for row in media:
        probe = row["probe"]
        contacts = row.get("contact_frames", [])
        expected_contacts = [str(Path(row["relative"]).parent / f"frame_{i:05d}.png")
                             for i in (0, probe["nb_frames"] // 2, probe["nb_frames"] - 1)]
        require(contacts == expected_contacts and set(contacts) <= set(relatives), "Missing exact video contact frames")
    record = dict(status="VERIFYING", started_utc=now(), delivery_manifest=manifest["identity"],
        artifacts=[], media_checks=[], visual_receipt=identity(visual_path), new_physical_steps=0,
        new_optimizer_updates=0, goal_completed=False, user_accepted_checkpoint=None)
    try:
        for row in rows:
            require(artifact(local_root, row["relative"]) == row, "Mac artifact identity differs: " + row["relative"])
            record["artifacts"].append(row)
        check_visuals(manifest, visual)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "primebot_q1_review"))
        from video import decoded_metadata
        for row in media:
            path = local_root / row["relative"]
            hashes = subprocess.check_output([str(ffmpeg), "-nostdin", "-v", "error", "-i", str(path),
                "-map", "0:v:0", "-pix_fmt", "rgb24", "-f", "framehash", "-hash", "sha256", "pipe:1"],
                text=True, timeout=900)
            probe = decoded_metadata(hashes, row["probe"]["nb_frames"])
            require(probe == row["probe"], "Mac full decode differs from server timing")
            record["media_checks"].append(dict(relative=row["relative"], sha256=identity(path)["sha256"],
                                                decoded_frames=probe["nb_frames"], visual_reviewed=True))
        record.update(status="VERIFIED", ended_utc=now())
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("manifest", "local-root", "visual-receipt", "output", "ffmpeg"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.manifest, args.local_root, args.visual_receipt, args.output, args.ffmpeg)["status"])
