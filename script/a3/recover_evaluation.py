"""Quarantine invalid restart artifacts; preserve valid completed motions."""

import argparse
import json
from pathlib import Path
import subprocess

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.evaluate_mujoco import rollout_windows
from script.a3.fullchain_support import ROOT


def recover(output):
    output = output.resolve()
    if not output.is_relative_to(ROOT / "data"):
        raise ValueError("Evaluation must belong to this workspace")
    busy = subprocess.run(["pgrep", "-f", r"script.a3.(evaluate_mujoco|sim_entry)"], capture_output=True)
    if busy.returncode == 0:
        raise RuntimeError("Cannot recover artifacts while an evaluator is active")
    path = output / "explicit_manifest.json"
    manifest = json.loads(path.read_text())
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    report = {"boot_id": boot, "kept": [], "rerun": []}
    for name, row in manifest["runs"].items():
        try:
            if row["status"] != "passed":
                raise ValueError("Attempt did not durably complete")
            metrics = output / f"{name}.metrics.json"
            if sha256(metrics) != row["metrics_sha256"]:
                raise ValueError("Metrics checksum changed after interruption")
            summary = json.loads(metrics.read_text())["motions"][0]
            trace = json.loads((output / f"{name}.timeseries.json").read_text())["motions"][0]
            rollout_windows(summary, trace)
            report["kept"].append(name)
            continue
        except (OSError, ValueError, KeyError, IndexError) as error:
            reason = str(error)
        target = output / "incomplete_artifacts" / boot / name
        target.mkdir(parents=True, exist_ok=True)
        for artifact in output.glob(f"{name}.*"):
            if artifact.is_file():
                artifact.rename(target / artifact.name)
        atomic_json(target / "previous_attempt.json", row)
        row.update(status="interrupted_artifacts_unverified", recovery_reason=reason,
                   preserved_artifacts=str(target))
        report["rerun"].append({"motion": name, "reason": reason})
    atomic_json(path, manifest)
    atomic_json(output / f"recovery_{boot}.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    recover(parser.parse_args().output)
