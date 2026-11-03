"""Actual native control check and the approved single-use capacity ladder."""

import argparse
import json
from pathlib import Path
import subprocess

from common import CAPACITY, HERE, ROOT, capacity_failure, identity, now, require, save


def phase(config, config_path, output, name, command, seconds=7200, maximum_bytes=32 * 1024**3, inputs=()):
    directory = output / name
    cmd = [config["primebot_python"], "-u", str(HERE / "guard.py"), "--config", str(config_path),
        "--output", str(directory), "--seconds", str(seconds), "--maximum-bytes", str(maximum_bytes)]
    for path in inputs:
        cmd += ["--input", str(path)]
    with (output / (name + "_controller.log")).open("x") as log:
        result = subprocess.run([*cmd, "--", *command], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
    return result.returncode == 0


def profile_plan(config, dataset, output, environments, wm, policy):
    return dict(purpose="profile", recipe=str(Path(config["adapted_source"]) / "configs/q1-fresh-single.example.toml"),
        overrides=dict(robot_xml=config["xml"], motion_dir=str(dataset / "accepted"), output_dir=str(output / "run"),
            num_envs=environments, model_batch_size=wm, policy_batch_size=policy, iterations=8,
            profile_iterations=8, seed=0))


def execute(config, config_path, dataset, output, commit):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="RUNNING", started_utc=now(), code_commit=commit, stages=[],
                  formal_training_started=False, user_accepted_checkpoint=None)
    try:
        admitted = json.loads((dataset / "result.json").read_text())
        require(admitted["status"] == "COMPLETED" and admitted["accepted"] > 0
                and admitted["original_screening_thresholds"], "No admissible original-screen dataset")
        result["dataset"] = identity(dataset / "result.json")
        frozen_path = dataset / "admission_frozen.json"
        frozen = json.loads(frozen_path.read_text())
        require(frozen["status"] == "FROZEN" and frozen["admitted"] == admitted["accepted"],
                "Common paired data has not been frozen")
        for row in frozen["rows"]:
            for key in ("q1_primebot", "q1_morph", "g1", "human"):
                require(identity(row[key]["path"]) == row[key], "Frozen paired source changed")
        result["frozen_data"] = identity(frozen_path)
        require(phase(config, config_path, output, "native", [config["primebot_python"], "-u", str(HERE / "native.py"),
            "--config", str(config_path), "--dataset", str(dataset), "--output", str(output / "native/result.json")],
            inputs=[dataset / "result.json"]), "Actual native V3 control verification failed")
        native = json.loads((output / "native/result.json").read_text())
        require(native["status"] == "PASSED" and native["actual_substep_control"]["calls"] == 240
                and native["new_optimizer_updates"] == 0, "Incomplete native control audit")
        result["stages"].append(dict(stage="native", status="PASSED"))
        for environments, wm, policy in CAPACITY:
            name = "profile_e" + str(environments)
            result["phase"] = name
            save(output / "pipeline.json", result)
            plan = profile_plan(config, dataset, output / name, environments, wm, policy)
            path = output / (name + "_plan.json")
            save(path, plan)
            passed = phase(config, config_path, output, name,
                [config["primebot_python"], "-u", str(HERE / "train.py"), "--plan", str(path)], inputs=[path, dataset / "result.json"])
            if not passed:
                guard = json.loads((output / name / "guard.json").read_text())
                log = (output / name / "stdout.log").read_text()
                require(guard["status"] == "WORKER_FAILED" and capacity_failure(log),
                        "Failure is not an explicit worker CUDA capacity OOM; no fallback")
                result["stages"].append(dict(stage=name, status="CUDA_CAPACITY_OOM"))
                continue
            audit = output / (name + "_audit.json")
            require(phase(config, config_path, output, name + "_audit", [config["primebot_python"], "-u",
                str(HERE / "audit.py"), "--run", str(output / name / "run"), "--profile", "--output", str(audit)]),
                "Profile checkpoint audit failed")
            result.update(status="VALIDATED_READY_FOR_FRESH_FORMAL", selected_plan=plan,
                audit=identity(audit), profile=identity(path), environments=environments,
                model_batch_size=wm, policy_batch_size=policy, ended_utc=now())
            return result
        raise RuntimeError("All three approved capacity tiers failed")
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "pipeline.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "dataset", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    require(len(args.code_commit) == 40, "Use the published full commit")
    execute(json.loads(args.config.read_text()), args.config.resolve(), args.dataset.resolve(), args.output.resolve(), args.code_commit)
