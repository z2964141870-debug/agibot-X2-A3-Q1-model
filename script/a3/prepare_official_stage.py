"""Reuse hash-registered diagnostics in the independent A3 workspace."""

import json
from pathlib import Path
import shutil

from script.a3.fullchain_support import DATA, ROOT, digest, mark, write_json


def main():
    history = ROOT / "data/references/a3_fullchain_20261010_history"
    old_root = history.resolve().parents[2]
    inventory = json.loads((ROOT / "data/manifests/a3_fullchain_20261010.json").read_text())
    prefix = "data/experiments/a3_fullchain_20261010/"
    names = {"input_audit.json", "retarget_summary.json", "a3_retarget_config.json",
             "retarget_qpos0_diagnostic.json", "buffer_summary.json"}
    reused = []
    for row in inventory["artifacts"]:
        if not row["path"].startswith(prefix):
            continue
        relative = row["path"][len(prefix):]
        if relative not in names and not relative.startswith(("mocap/", "motionlib/")):
            continue
        source, target = old_root / row["path"], DATA / relative
        if source.stat().st_size != row["bytes"] or digest(source) != row["sha256"]:
            raise ValueError(f"History artifact changed: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if digest(target) != row["sha256"]:
            raise ValueError(f"Copied artifact changed: {target}")
        reused.append({**row, "source": str(source), "target": str(target)})
    old_split = json.loads((history / "split.json").read_text())
    new_split = json.loads((DATA / "split.json").read_text())
    for role in ("train", "heldout"):
        if [(r["name"], r["sha256"]) for r in old_split[role]] != [
                (r["name"], r["sha256"]) for r in new_split[role]]:
            raise ValueError("Fixed whole-motion split changed")
    receipt = DATA / "reused_history.json"
    write_json(receipt, {"artifacts": reused, "split_assignments_unchanged": True,
                         "old_ledgers_copied": False, "backup_status": "LOCAL_ONLY"})
    old_tasks = json.loads((history / "tasks.json").read_text())["tasks"]
    for task in ("input_audit", "retarget", "retarget_visual", "reference_buffer",
                 "mocap_quality", "mocap_finetune"):
        previous = old_tasks[task]
        mark(task, previous["status"], reused_evidence=str(receipt),
             historical_evidence=previous, scope="HASH_VERIFIED_HISTORY_REUSE")
    mark("isaac_baseline", "deferred", reason="missing_smpl_sim_metrics",
         user_action="Provide compatible smpl_sim.smpllib.smpl_eval; independent validate required")
    mark("rknn", "deferred", reason="rknn_toolkit_unavailable",
         user_action="Provide compatible RKNN toolkit; board execution remains untested")
    smoke = DATA / "official_smoke/explicit_summary.json"
    if json.loads(smoke.read_text())["motions"][0]["windows"]["full_rollout"]["policy_steps"] != 10:
        raise ValueError("Official smoke incomplete")
    mark("mujoco_wrapper_check", "passed", evidence=str(smoke), scope="OFFICIAL_10_STEP_SMOKE")


if __name__ == "__main__":
    main()
