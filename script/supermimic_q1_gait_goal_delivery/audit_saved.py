"""Audit saved training and mirrored Mac media without replay or optimizer work."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REVIEW = HERE.parent / "supermimic_q1_gait_goal_review"
sys.path[:0] = [str(REVIEW), str(HERE.parent / "supermimic_q1_ppo")]
from resources import identity, save, stamp
from evidence import candidate_rank


def content_match(actual, expected):
    return all(actual[key] == expected[key] for key in ("bytes", "sha256"))


def delivery_scope(guard, controller, error_log, recover):
    if guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_return_code"] == 0:
        return "SERVER_RENDER_COMPLETED"
    if (recover and controller["status"] == "FAILED" and controller["phase"] == "delivery"
            and guard["status"] == "WORKER_FAILED" and guard["cleanup"]["worker_return_code"] == 1
            and "FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'" in error_log):
        return "ORIGINAL_MISSING_FFPROBE_FAILURE_PRESERVED_MAC_COMPLETES_SAVED_MEDIA"
    raise ValueError("Unrecognized delivery failure; no automatic recovery")


def audit(args):
    import numpy as np
    import torch

    spec = importlib.util.spec_from_file_location("goal_frozen_persistence", REVIEW / "audit_saved.py")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    checks = []

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    trial, run = args.trial, args.trial / "run"
    controller = json.loads((trial / "status.json").read_text())
    result = json.loads((run / "receipt.json").read_text())
    clock = json.loads((run / "training_clock.json").read_text())
    plan = json.loads((trial / "plan.json").read_text())
    check("matching_trial_line", result["line"] == controller["line"] == plan["line"] == args.line)
    check("actual_training_budget", result["status"] == "TRAINING_COMPLETED"
        and 0 <= result["training_seconds"] < 1200 and clock["active_started_monotonic"] is None
        and clock["inherited_training_seconds"] == 0 and clock["absolute_maximum_seconds"] == 1200
        and abs(result["training_seconds"] - clock["accumulated_training_seconds"]) < 1e-6)
    changed = [row["path"] for row in plan["source"] if identity(row["path"]) != row]
    check("all_frozen_sources_unchanged", not changed, count=len(plan["source"]), changed=changed)
    for name in ("import_guard.json", "train_guard.json", "delivery_guard.json"):
        guard = json.loads((trial / name).read_text())
        if name == "delivery_guard.json":
            log = Path("/opt/data/private/ZHY/logs") / (trial.name + "_delivery.log")
            scope = delivery_scope(guard, controller, log.read_text() if log.exists() else "",
                                   args.recover_missing_ffprobe)
            check(name, guard["cleanup"]["worker_reaped"], scope=scope)
        else:
            check(name, guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_reaped"]
                  and guard["cleanup"]["worker_return_code"] == 0)
    selection = json.loads((run / "checkpoint_selection.json").read_text())
    check("reviewed_selection_and_null_adoption", selection["numeric_recommendation"] ==
          min(selection["candidates"], key=candidate_rank) and selection["user_accepted_checkpoint"] is None
          and selection["goal_complete"] is False)
    for candidate in selection["candidates"]:
        check(candidate["label"] + "_checkpoint", identity(candidate["checkpoint"]["path"]) == candidate["checkpoint"])
    stages = json.loads((run / "stage_decisions.json").read_text())
    final = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    stage = torch.load(stages[-1]["checkpoint"]["path"], map_location="cpu", weights_only=True)
    keys = ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v") if args.line == "base" else ("actor", "critic", "optimizer")
    check("final_persistence_without_post_gate_updates", all(original.equal(final[k], stage[k]) for k in keys))
    del final, stage
    receipt = json.loads((trial / "evaluation/receipt.json").read_text())
    check("original300_30_library", len(receipt["rows"]) == 330
        and sum(row["split"] == "train" for row in receipt["rows"]) == 300
        and len({row["name"] for row in receipt["rows"]}) == 330)
    traces = {row["path"]: row for candidate in selection["candidates"] for row in candidate["traces"]}
    for path, expected in traces.items():
        valid = identity(path) == expected
        with np.load(path, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in
                     ("steps", "reference_steps", "states", "feet", "failed", "alive", "frames")}
            valid &= saved["steps"].shape == (330,) and saved["feet"].shape[1:] == (330, 2, 3)
            valid &= bool(np.isfinite(saved["states"]).all() and np.isfinite(saved["feet"]).all())
            summary = json.loads(Path(path).with_suffix(".json").read_text())
            valid &= len(summary["per_clip"]) == 330
            for index, row in enumerate(summary["per_clip"]):
                count, length = int(saved["steps"][index]), int(saved["reference_steps"][index])
                valid &= 0 < count <= length and row["steps"] == count
                valid &= row["failed"] == bool(saved["failed"][index])
                valid &= bool(saved["alive"][:count, index].all() and not saved["alive"][count:, index].any())
                first = int(saved["frames"][0, index])
                valid &= bool(np.array_equal(saved["frames"][:count, index], first + np.arange(count)))
        check(Path(path).stem + "_full330_exact_prefix", valid)
    labels = {candidate["label"] for candidate in selection["candidates"]}
    import csv
    for name, expected in (("per_clip", 990 * len(labels)), ("fixed", 9 * len(labels))):
        with (trial / "review" / (name + ".csv")).open(newline="") as stream:
            check(name + "_table_rows", sum(1 for _ in csv.DictReader(stream)) == expected, expected=expected)
    check("actual_episode_six_panel", json.loads((trial / "dashboard/receipt.json").read_text())["status"]
          == "TRAINING_DASHBOARD_COMPLETED")
    media = json.loads(args.media_audit.read_text())
    check("mac_full_decode_and_protocol", media["status"] == "RF13_MEDIA_AUDIT_PASSED"
          and len(media["checks"]) == 12 and all(row["passed"] for row in media["checks"]))
    for label, evidence in media["videos"].items():
        declaration_path = trial / f"evaluation/{label}_media.json"
        declaration = json.loads(declaration_path.read_text())
        check(label + "_mac_server_declaration_identity", content_match(identity(declaration_path), evidence["declaration"]))
        check(label + "_mac_server_video_identity", content_match(
            identity(declaration["video"]["path"]), evidence["file"]))
        source = receipt[("initial" if label == "initial" else "final") + "_traces"][0]
        with np.load(source, allow_pickle=False) as trace:
            valid = all(segment["physical_steps"] == int(trace["steps"][index])
                and segment["failed"] == bool(trace["failed"][index])
                and segment["reference_frames_50hz"] == int(trace["reference_steps"][index]) + 1
                for segment in declaration["segments"]
                for index in [next(i for i, row in enumerate(receipt["rows"]) if row["name"] == segment["name"])])
        check(label + "_declared_source_failure_prefix", valid)
        for sheet in evidence["visual_review_frames"].values():
            local = trial / "evaluation" / Path(sheet["file"]["path"]).name
            check(local.name + "_mirrored_identity", content_match(identity(local), sheet["file"]))
    recommended = json.loads((trial / "evaluation/recommended_media.json").read_text())
    check("recommended_media_identity", identity(recommended["video"]["path"]) == recommended["video"])
    passed = all(row["passed"] for row in checks)
    audit_path = trial / "postdelivery_saved_audit.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)
    save(audit_path, dict(status="SAVED_EVIDENCE_AUDIT_PASSED" if passed else "FAILED", created_utc=stamp(),
        checks=checks, original_pipeline_status=controller["status"], media_audit=identity(args.media_audit),
        new_physical_steps=0, policy_inference_calls=0, optimizer_updates=0,
        goal_complete=False, user_accepted_checkpoint=None, visual_review_completed=False))
    if passed:
        completed = dict(status="COMPLETED", completed_utc=stamp(), original_pipeline_status=controller["status"],
            saved_audit=identity(audit_path), original_failure_preserved=controller["status"] == "FAILED",
            goal_complete=False, user_accepted_checkpoint=None)
        save(trial / "postdelivery_status.json", completed)
        if controller["status"] == "AWAITING_MAC_MEDIA_AUDIT":
            controller.update(status="COMPLETED", phase="pipeline_completed", completed_utc=stamp(),
                postdelivery_saved_audit=identity(audit_path), goal_complete=False, user_accepted_checkpoint=None)
            save(trial / "status.json", controller)
    print(json.dumps(dict(passed=passed, checks=len(checks), failed=[row for row in checks if not row["passed"]])))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    parser.add_argument("--media-audit", type=Path, required=True)
    parser.add_argument("--recover-missing-ffprobe", action="store_true")
    raise SystemExit(audit(parser.parse_args()))
