"""Generate and reread a SHA inventory of closed-loop preparation artifacts."""

import json
from pathlib import Path

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import OFFICIAL
from script.a3.fullchain_support import DATA, ROOT


def main():
    directories = [ROOT / "data/experiments" / name for name in (
        "a3_joint_ablation_20261010_E16", "a3_joint_dynamics_20261010_E17",
        "a3_causal_reference_20261010_E18", "a3_controller_reference_20261010_E19",
        "a3_controller_adapter_20261010_E20", "a3_controller_adapter_20261010_E21",
        "a3_longtrain_readiness_20261010")]
    paths = set()
    paths.update(p for p in OFFICIAL.parent.glob("*") if p.is_file())
    paths.add(DATA / "split.json")
    paths.update(p for p in (DATA / "motionlib").rglob("*.pkl") if p.is_file())
    for experiment in ("E12", "E14"):
        predictor = ROOT / f"data/experiments/a3_future_reference_20261010_{experiment}/ridge_joint_future.npz"
        if predictor.is_file():
            paths.add(predictor)
    for directory in directories:
        if directory.exists():
            paths.update(p for p in directory.rglob("*") if p.is_file() and not p.is_symlink())
    for directory in (ROOT / "logs").glob("a3_*_20261010_E*"):
        if any(directory.name.endswith(f"E{number}") for number in range(16, 22)):
            paths.update(p for p in directory.rglob("*") if p.is_file() and not p.is_symlink())
    paths.update(p for p in (ROOT / "script/a3").glob("*") if p.is_file())
    paths.update(p for p in (ROOT / "reports").glob("README_A3_*_20261010.md") if p.is_file())
    rows = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(paths)]
    result = dict(scope="E16_E21_AND_LONGTRAIN_PREPARATION", artifacts=rows, files=len(rows),
        total_bytes=sum(row["bytes"] for row in rows), backup_status="LOCAL_ONLY",
        note="Inventory integrity does not prove experiment completion or effect acceptance")
    path = ROOT / "data/manifests/a3_closed_loop_artifacts_20261010.json"
    atomic_json(path, result)
    loaded = json.loads(path.read_text())
    for row in loaded["artifacts"]:
        artifact = ROOT / row["path"]
        if artifact.stat().st_size != row["bytes"] or sha256(artifact) != row["sha256"]:
            raise ValueError(f"Artifact changed during inventory recheck: {artifact}")
    atomic_json(ROOT / "data/manifests/a3_closed_loop_inventory_verified_20261010.json",
        dict(status="passed", inventory_sha256=sha256(path), files=len(rows), independent_second_read=True,
             method="REOPEN_EVERY_FILE_AND_REHASH", backup_status="LOCAL_ONLY"))
    print(json.dumps({key: result[key] for key in ("scope", "files", "total_bytes", "backup_status")}))


if __name__ == "__main__":
    main()
