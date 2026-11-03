"""Fail-closed contracts for the approved Primebot-to-MorphMimic transition."""

import copy
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "primebot_q1"))
from common import FIXED, TARGET_TRANSITIONS, evaluation_gate, morph_settings, require


COUNTS = dict(zip(("wm_opt", "policy_opt", "critic_opt", "kp_opt", "critic_kp_opt"),
                  (80, 8, 160, 8, 160)))
MODULES = {"wm", "policy", "critic", "critic_kp"}
REFERENCE_KEYS = ("q1_primebot", "q1_morph", "g1", "human")


def valid_commit(value):
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def validate_frozen(frozen, expected_count=393):
    rows = frozen.get("rows", [])
    require(len(rows) == expected_count, "Frozen paired clip count changed")
    names = []
    for row in rows:
        stems = []
        for key in REFERENCE_KEYS:
            item = row[key]
            require(set(item) == {"path", "bytes", "sha256"} and item["bytes"] > 0
                    and len(item["sha256"]) == 64, "Incomplete frozen reference identity")
            stems.append(Path(item["path"]).stem)
        require(len(set(stems)) == 1, "Paired robot/human clips differ")
        names.append(stems[0])
    require(len(set(names)) == len(names) and set(FIXED) <= set(names),
            "Missing fixed clips or duplicate paired names")
    return names


def required_delivery_paths(expected_names):
    paths = {"result.json", "checkpoint_audit.json", "paired_fixed_metrics.json",
             "exposure_comparison_checkpoint.json"}
    for stage in ("initial", "final"):
        for mode in ("fixed", "library"):
            prefix = stage + "_" + mode
            paths.add(prefix + "/result.json")
            names = FIXED if mode == "fixed" else expected_names
            seeds = (0, 1, 2) if mode == "fixed" else (0,)
            for name in names:
                for seed in seeds:
                    case = f"{prefix}/{name}_s{seed}/"
                    paths.update(case + filename for filename in
                                 ("result.json", "foot_posture.csv", "upper_timing.csv"))
                    if mode == "fixed" and seed == 0:
                        paths.add(case + "tracking.mp4")
    return paths


def primebot_gate(server, pipeline, pairs, fixed, libraries, expected_names, audit, mac, manifest):
    """Recompute the full gate; a provisional observer can never launch Morph."""
    reasons = []
    if pipeline.get("status") != "SERVER_EVALUATION_COMPLETED_PENDING_MAC_DELIVERY":
        reasons.append("primebot_pipeline_not_finished")
    if server.get("status") != "COMPLETED" or server.get("local_mac_delivery_required_before_morph") is not True:
        reasons.append("final_server_delivery_incomplete")
    if server.get("library_count") != len(expected_names):
        reasons.append("wrong_final_library_count")
    expected_fixed = {(clip, seed) for clip in FIXED for seed in (0, 1, 2)}
    video_ids = set()
    media_ok = True
    for stage in ("initial", "final"):
        value = fixed.get(stage, {})
        keys = [(r.get("clip"), r.get("seed")) for r in value.get("cases", [])]
        if value.get("status") != "PASSED" or len(keys) != 9 or set(keys) != expected_fixed:
            reasons.append("incomplete_" + stage + "_fixed_evaluation")
        for row in value.get("cases", []):
            if row.get("seed") == 0:
                video = row.get("video", {})
                media_ok = media_ok and video.get("status") == "PASSED"
                video_ids.add((stage, row.get("clip")))
    media_ok = media_ok and video_ids == {(s, c) for s in ("initial", "final") for c in FIXED}
    library_ok = True
    for stage in ("initial", "final"):
        value = libraries.get(stage, {})
        cases = value.get("cases", [])
        names = [row.get("clip") for row in cases]
        library_ok = library_ok and (value.get("status") == "PASSED"
            and len(names) == len(expected_names) and set(names) == set(expected_names)
            and all(row.get("seed") == 0 for row in cases))
    audit_ok = (audit.get("status") == "PASSED"
        and audit.get("initial", {}).get("iteration") == 0
        and audit.get("final", {}).get("iteration") == server.get("final_iteration"))
    base = evaluation_gate(pairs, library_ok, media_ok, audit_ok)
    reasons.extend(base["gate_reason"] if not base["passed"] else [])
    if server.get("passed") is not True:
        reasons.append("server_gate_did_not_pass")
    expected = manifest.get("artifacts", [])
    keys = [row.get("relative") for row in expected]
    if (manifest.get("status") != "READY_FOR_MAC_VERIFICATION" or not expected
            or len(set(keys)) != len(keys) or mac.get("status") != "VERIFIED"
            or mac.get("delivery_manifest") != manifest.get("identity")
            or mac.get("artifacts") != expected):
        reasons.append("mac_delivery_missing_or_mismatched")
    if not required_delivery_paths(expected_names) <= set(keys):
        reasons.append("incomplete_mac_numerical_artifacts")
    videos = {row["relative"] for row in expected if row["relative"].endswith(".mp4")}
    checks = mac.get("media_checks", [])
    if (len(videos) != 6 or len(checks) != len(videos)
            or {row.get("relative") for row in checks} != videos
            or any(row.get("status") != "PASSED" or row.get("visual_reviewed") is not True
                   or type(row.get("decoded_frames")) is not int or row["decoded_frames"] <= 0
                   for row in checks)):
        reasons.append("mac_original_speed_media_not_verified")
    reasons = list(dict.fromkeys(reasons))
    return dict(passed=not reasons, gate_reason=reasons or ["full_primebot_gate_and_mac_delivery_passed"],
        paired_metrics=base["paired_metrics"], authorizes_morph_training=not reasons,
        natural_gait_thresholds_are_diagnostic=True, user_accepted_checkpoint=None)


def build_plans(template, output, subsets, human_root, commit, environments):
    require(valid_commit(commit), "Require an independently verified full commit")
    require(environments in (1024, 512) and len(subsets) == 2, "Unapproved shared capacity tier")
    require(template["robots"] == ["g1", "q1"], "Require the G1/Q1 shared native template")
    profile = copy.deepcopy(template)
    profile.update(purpose="profile", output=str(Path(output) / "profile"), code_commit=commit,
        subsets=list(map(str, subsets)), human_root=str(human_root), receipt_every=1,
        initial_admission_gib=24, reuse_profile_weights_for_formal_training=False)
    for key in tuple(profile):
        if key.startswith("formal_target_") or key == "scope":
            profile.pop(key)
    settings = morph_settings(environments)
    profile["cfg"].update(settings, iterations=8, wm_only_iters=0, ckpt_every=8, log_every=1)
    formal = copy.deepcopy(profile)
    formal.update(purpose="train", output=str(Path(output) / "train"), receipt_every=50)
    formal["cfg"].update(settings, log_every=50)
    for plan in (profile, formal):
        cfg = plan["cfg"]
        require(not any(k in cfg for k in ("resume", "checkpoint", "resume_from")), "Fresh initialization required")
        require(cfg["wm_hidden"] == [2048, 2048, 2048]
            and cfg["policy_hidden"] == [2048, 1024, 512, 256]
            and cfg["critic_hidden"] == [256, 128]
            and cfg["horizon"] == 32 and cfg["window_k"] == 8 and cfg["wm_batches"] == 10
            and cfg["kp_interface"] is True and cfg["kp_student"] == "return"
            and cfg["latent_align_weight"] == .05 and cfg["utd"] == .4,
            "Preserve the approved complete networks, five optimizers and paired alignment")
        plan["target_transitions_per_robot"] = TARGET_TRANSITIONS
    require(expected_transitions(formal["cfg"]) == TARGET_TRANSITIONS, "Shared exposure target changed")
    return profile, formal


def expected_transitions(cfg):
    # Native Trainer accumulates 1/utd whole environment steps per complete iteration.
    steps = math.floor(cfg["iterations"] / cfg["utd"] + 1e-8)
    return (cfg["warmup_env_steps"] + steps) * cfg["num_envs"]


def check_profile(workers, profile, expected_names):
    require(len(workers) == 2, "Both native ranks must complete")
    for name, worker in zip(("g1", "q1"), workers):
        cfg = worker.get("resolved_cfg", {})
        require(worker.get("status") == "COMPLETED" and worker.get("robot") == name
            and worker.get("code_commit") == profile["code_commit"]
            and worker.get("formal_training") is False and worker.get("iterations_completed") == 8
            and worker.get("reference_clips") == len(expected_names)
            and worker.get("collected_environment_transitions") == expected_transitions(profile["cfg"])
            and worker.get("optimizer_updates") == COUNTS
            and worker.get("optimizer_adam_steps") == {k: [v] for k, v in COUNTS.items()}
            and worker.get("native_rank_checks_required") is True
            and worker.get("native_rank_checks_passed") is True
            and len(worker.get("peer_audit", [])) == 2, "Incomplete native shared profile")
        require(all(cfg.get(k) == v for k, v in profile["cfg"].items()), "Profile configuration changed")
        clips = worker.get("common_clip_names", [])
        require(len(clips) == len(expected_names) and set(clips) == set(expected_names), "Wrong shared clips")
        require(worker.get("nonfinite_updates") and not any(worker["nonfinite_updates"].values()),
                "Nonfinite updates invalidate the profile")
        timing = worker.get("iteration_seconds", [])
        require(len(timing) == 8 and all(math.isfinite(t) and t > 0 for t in timing), "Missing eight full timings")
        require(worker.get("latent_draws") and worker.get("last_latent_draw")
            and set(worker.get("shared_parameter_sha", {})) == MODULES, "Missing actual paired draws or shared hashes")
    for key in ("common_clip_names", "shared_parameter_sha", "latent_draws", "last_latent_draw"):
        require(workers[0][key] == workers[1][key], "Native ranks diverged: " + key)
    require(workers[0]["initial_buffers"]["joint_presence"] != workers[1]["initial_buffers"]["joint_presence"],
            "Robot-specific buffers were overwritten")


def allow_capacity_fallback(phase, logs):
    if phase.get("status") != "FAILED" or phase.get("inputs_unchanged") is not True:
        return False
    guard = phase.get("guard", {})
    cleanup = guard.get("cleanup", {})
    if (guard.get("status") != "WORKER_FAILED" or cleanup.get("worker_reaped") is not True
            or type(cleanup.get("worker_return_code")) is not int
            or cleanup["worker_return_code"] <= 0):
        return False
    if phase.get("cgroup_oom_kill_delta") != 0 or phase.get("gpu_recovery_action") != "None":
        return False
    text = logs.lower()
    faults = ("unspecified launch failure", "illegal memory access", "device-side assert",
              "requires reset", "memory reserve", "wall limit", "frozen phase sources changed",
              "disk reserve", "temperature reserve", "worker reserve reached", "under_oom")
    return "cuda out of memory" in text and not any(token in text for token in faults)


def disk_budget(checkpoints, cfg):
    require(set(checkpoints) == {"g1", "q1"} and all(n > 0 for n in checkpoints.values()),
            "Measure both real native checkpoints before formal training")
    count = math.ceil(cfg["iterations"] / cfg["ckpt_every"]) + 1
    # Replay is not persisted by the native v0.1 Trainer. Account for both latest
    # copies and concurrent checkpoint/latest temporary files, plus final evidence.
    persisted = sum(checkpoints.values()) * (count + 3)
    return dict(measured_checkpoint_bytes=checkpoints, numbered_checkpoints_per_robot=count,
        replay_persisted=False, maximum_experiment_bytes=math.ceil((persisted + 8 * 1024**3) * 1.25),
        required_free_reserve_bytes=100 * 1024**3)
