"""Validate a three-clip repair, then freeze an unchanged-screen full library."""

import argparse
import json
from pathlib import Path
import subprocess

from common import HERE, ROOT, require, save, now
from guard import run
from prepare import read_yaml


def execute(config, config_path, output):
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="RUNNING", phase="repair_fixed", started_utc=now(),
                  new_policy_optimizer_updates=0, formal_training_started=False)
    try:
        for scope in ("fixed", "all"):
            phase = output / ("repair_" + scope)
            record["phase"] = phase.name
            save(output / "pipeline.json", record)
            run(config, phase, [config["morph_python"], "-u", str(HERE / "repair_reference.py"),
                "--config", str(config_path), "--output", str(phase / "data"),
                "--scope", scope, "--workers", "3"], 7200, 16 * 1024**3, [config_path])
            repair = json.loads((phase / "data/result.json").read_text())
            if scope == "fixed":
                require(repair["projected"] == 3, "Fixed motion repair failed a geometry/deviation audit")
            q1 = json.loads((phase / "data/q1_paired.yaml").read_text())
            g1 = read_yaml(config["g1_subset"], config["morph_python"])
            g1["clips"] = q1["clips"]
            save(phase / "data/g1_paired.yaml", g1)
            revised = dict(config, q1_subset=str(phase / "data/q1_paired.yaml"),
                g1_subset=str(phase / "data/g1_paired.yaml"), expected_candidates=len(q1["clips"]))
            revised_path = output / ("screen_" + scope + "_config.json")
            save(revised_path, revised)
            screen = output / ("screen_" + scope)
            record["phase"] = screen.name
            save(output / "pipeline.json", record)
            run(revised, screen, [config["primebot_python"], "-u", str(HERE / "prepare.py"),
                "--config", str(revised_path), "--output", str(screen / "dataset")],
                7200, 16 * 1024**3, [config_path, revised_path])
            admitted = json.loads((screen / "dataset/result.json").read_text())
            if scope == "fixed":
                require(admitted["accepted"] == 3, "The repaired fixed motions did not pass the original screen")
            record[scope] = dict(repaired=repair["projected"], accepted=admitted["accepted"],
                dataset=str(screen / "dataset"))
        record.update(status="DATA_ADMITTED_PENDING_NATIVE_AND_CAPACITY", ended_utc=now())
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "pipeline.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    execute(json.loads(args.config.read_text()), args.config.resolve(), args.output.resolve())
