"""Compact evidence inventory for independent A3 direction experiments."""

import json
from pathlib import Path

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import ROOT


def main():
    e09 = ROOT / "data/experiments/a3_likelihood_20261010_E09"
    offline = json.loads((e09 / "likelihood_diagnostic.json").read_text())
    if offline["parameters_before_sha256"] != offline["parameters_after_sha256"]:
        raise ValueError("Offline replay changed parameters")
    if sha256(e09 / "action_means.pt") != offline["means_sha256"]:
        raise ValueError("Offline replay capture changed")
    e10 = ROOT / "data/experiments/a3_gradient_20261010_E10"
    stopped = json.loads((e10 / "state.json").read_text())
    if stopped["status"] != "blocked" or len(stopped["attempts"]) != 2:
        raise ValueError("Expected the preserved two no-progress interruptions")
    names = ["reports/README_A3_DIRECTION_20261010.md",
             "data/manifests/a3_gradient_20261010.json", "data/manifests/a3_likelihood_20261010.json"]
    for pattern in ("*gradient_probe.py", "*likelihood_probe.py", "direction_receipt.py", "*future_reference_probe.py"):
        names.extend(str(path.relative_to(ROOT)) for path in (ROOT / "script/a3").glob(pattern))
    for directory in (e09, e10, ROOT / "logs/a3_likelihood_20261010_E09", ROOT / "logs/a3_gradient_20261010_E10"):
        names.extend(str(path.relative_to(ROOT)) for path in directory.rglob("*") if path.is_file())
    future_path = ROOT / "data/manifests/a3_future_reference_20261010.json"
    future = json.loads(future_path.read_text()) if future_path.exists() else None
    if future:
        names.append(str(future_path.relative_to(ROOT)))
        names.extend(str(path.relative_to(ROOT)) for path in
                     (ROOT / "data/experiments/a3_future_reference_20261010_E11").rglob("*") if path.is_file())
    for experiment in ("E12", "E13"):
        manifest_path = ROOT / "data/manifests" / f"a3_future_reference_20261010_{experiment}.json"
        if manifest_path.exists():
            names.append(str(manifest_path.relative_to(ROOT)))
            names.extend(str(path.relative_to(ROOT)) for path in
                         (ROOT / "data/experiments" / f"a3_future_reference_20261010_{experiment}").rglob("*") if path.is_file())
    names.extend(str(path.relative_to(ROOT)) for path in (ROOT / "script/a3").glob("future_reference_*.py"))
    names.extend(str(path.relative_to(ROOT)) for directory in (ROOT / "logs").glob("a3_future_reference_20261010_*")
                 for path in directory.rglob("*") if path.is_file())
    result = dict(status="E09_DIAGNOSTIC_E10_DEFERRED", e09_optimizer_steps=offline["optimizer_steps"],
        e09_sampling_reproduced=False, e10_status=stopped["status"], e10_reason=stopped["reason"],
        e10_attempts=stopped["attempts"], initial_ratio_root_cause="UNKNOWN",
        future_joint_predictor_status=future["status"] if future else "pending",
        future_joint_predictor_selected=future["selected_joint_predictor"] if future else None,
        sources_reviewed=["https://arxiv.org/html/2608.19182v1", "https://arxiv.org/html/2409.16578v2",
                          "https://arxiv.org/html/2610.09055v1", "https://arxiv.org/abs/2508.02373",
                          "https://sonic-agibot-x2.github.io/sonic-transfer/"],
        current_boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        artifacts=[dict(path=name, bytes=(ROOT/name).stat().st_size, sha256=sha256(ROOT/name))
                   for name in sorted(set(names))], policy_improvement_verified=False, backup_status="LOCAL_ONLY")
    atomic_json(ROOT / "data/manifests/a3_direction_20261010.json", result)
    print(json.dumps(dict(artifact_count=len(result["artifacts"]), e10="DEFERRED", e11=result["future_joint_predictor_status"])))


if __name__ == "__main__":
    main()
