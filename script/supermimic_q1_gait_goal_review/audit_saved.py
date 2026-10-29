"""Audit GAIT_GOAL01 persistence and delivery without new physics or updates."""

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_ppo")]
from resources import identity, save, stamp
from evidence import candidate_rank


def equal(left, right):
    import torch
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def audit(args):
    import numpy as np
    import torch

    checks = []
    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))
    run = args.trial / "run"
    result = json.loads((run / "receipt.json").read_text())
    clock = json.loads((run / "training_clock.json").read_text())
    plan = json.loads((args.trial / "plan.json").read_text())
    check("independent_actual_training_budget", result["status"] == "TRAINING_COMPLETED"
        and 0 <= result["training_seconds"] < 1200 and clock["active_started_monotonic"] is None
        and clock["inherited_training_seconds"] == 0 and clock["absolute_maximum_seconds"] == 1200
        and abs(result["training_seconds"] - clock["accumulated_training_seconds"]) < 1e-6)
    changed = [row["path"] for row in plan["source"] if identity(row["path"]) != row]
    check("original_and_review_sources_unchanged", not changed, count=len(plan["source"]), changed=changed)
    for name in ("import_guard.json", "train_guard.json", "delivery_guard.json"):
        guard = json.loads((args.trial / name).read_text())
        check(name, guard["status"] == "WORKER_COMPLETED" and guard["cleanup"]["worker_reaped"]
            and guard["cleanup"]["worker_return_code"] == 0)
    selection = json.loads((run / "checkpoint_selection.json").read_text())
    check("reviewed_selection_and_null_adoption", selection["numeric_recommendation"] ==
        min(selection["candidates"], key=candidate_rank) and selection["user_accepted_checkpoint"] is None
        and selection["goal_complete"] is False)
    for candidate in selection["candidates"]:
        check(candidate["label"] + "_checkpoint_identity", identity(candidate["checkpoint"]["path"]) == candidate["checkpoint"])
    stages = json.loads((run / "stage_decisions.json").read_text())
    final = torch.load(run / "final.pt", map_location="cpu", weights_only=True)
    stage = torch.load(stages[-1]["checkpoint"]["path"], map_location="cpu", weights_only=True)
    keys = ("wm", "pi", "critic", "opt_wm", "opt_pi", "opt_v") if args.line == "base" else ("actor", "critic", "optimizer")
    check("final_persistence_without_post_gate_updates", all(equal(final[key], stage[key]) for key in keys))
    del final, stage
    receipt = json.loads((args.trial / "evaluation/receipt.json").read_text())
    check("original300_30_fixed_library", len(receipt["rows"]) == 330
        and sum(row["split"] == "train" for row in receipt["rows"]) == 300
        and len({row["name"] for row in receipt["rows"]}) == 330)
    traces = {row["path"]: row for candidate in selection["candidates"] for row in candidate["traces"]}
    for path, expected in traces.items():
        valid = identity(path) == expected
        with np.load(path, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in ("steps", "reference_steps", "states", "feet", "failed", "alive", "frames")}
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
        check(Path(path).stem + "_exact_saved_prefix_and_full330", valid)
    delivery = json.loads((args.trial / "delivery_status.json").read_text())
    check("tables_and_media_delivered", delivery["status"] == "COMPLETED"
        and delivery["tables"]["per_clip_rows"] == 990 * len(delivery["tables"]["labels"])
        and delivery["tables"]["fixed_rows"] == 9 * len(delivery["tables"]["labels"])
        and delivery["new_physical_steps"] == delivery["optimizer_updates"] == 0)
    check("saved_media_integrity", json.loads((args.trial / "media_audit.json").read_text())["status"] == "MEDIA_AUDIT_PASSED")
    check("six_panel_actual_episode_dashboard", json.loads((args.trial / "dashboard/receipt.json").read_text())["status"] == "TRAINING_DASHBOARD_COMPLETED")
    passed = all(row["passed"] for row in checks)
    save(args.trial / "saved_audit.json", dict(status="SAVED_EVIDENCE_AUDIT_PASSED" if passed else "FAILED",
        created_utc=stamp(), checks=checks, new_physical_steps=0, optimizer_updates=0,
        goal_complete=False, physical_success_claimed=False, user_accepted_checkpoint=None))
    print(json.dumps(dict(passed=passed, checks=len(checks), failed=[row for row in checks if not row["passed"]])))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--line", choices=("base", "ppo"), required=True)
    raise SystemExit(audit(parser.parse_args()))
