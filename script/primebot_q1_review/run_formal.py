"""Fresh monitored Primebot training, exact save audit and automatic evaluation."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "primebot_q1"))
from common import ROOT, identity, now, require, save
from formal import disk_budget
from validate import phase


def run(config, config_path, validation, output, commit):
    output.mkdir(parents=True, exist_ok=False)
    result = dict(status="INITIALIZING", started_utc=now(), code_commit=commit,
        fresh_initialization=True, profile_weights_reused=False, automatic_retry=False,
        maximum_iterations=250000, user_accepted_checkpoint=None)
    try:
        validated = json.loads((validation / "pipeline.json").read_text())
        require(validated["status"] == "VALIDATED_READY_FOR_FRESH_FORMAL", "Native/capacity validation incomplete")
        audit_path = Path(validated["audit"]["path"])
        require(identity(audit_path) == validated["audit"], "Validation save audit changed")
        audited = json.loads(audit_path.read_text())
        require(audited["status"] == "PASSED", "Profile audit did not pass")
        frozen_path = Path(validated["frozen_data"]["path"])
        require(identity(frozen_path) == validated["frozen_data"], "Frozen dataset manifest changed")
        frozen = json.loads(frozen_path.read_text())
        for row in frozen["rows"]:
            for key in ("q1_primebot", "q1_morph", "g1", "human"):
                require(identity(row[key]["path"]) == row[key], "Frozen common reference changed")
        monitor_adaptation = json.loads((Path(config["adapted_source"]) / "live_review_adaptation.json").read_text())
        source_inputs = json.loads((validation / f'profile_e{validated["environments"]}/status.json').read_text())["inputs"]
        before_source = monitor_adaptation["before"]
        require(any(p["path"].endswith("/src/supermimic/train.py") and p["sha256"] == before_source["sha256"]
                    for p in source_inputs), "Monitored trainer is not derived from the validated source")
        graphics = config.get("graphics_dir")
        if graphics:
            os.environ.update(Q1_REVIEW_GRAPHICS=graphics, LD_LIBRARY_PATH=str(Path(graphics) / "lib"),
                __EGL_VENDOR_LIBRARY_FILENAMES=str(HERE.parent / "morphmimic_cross_robot/nvidia_egl_595_20261102.json"),
                PYOPENGL_PLATFORM="egl", EGL_PLATFORM="surfaceless")
        budget = disk_budget(audited)
        # Iteration1000 is an additional observation-only learner snapshot.
        budget["checkpoint_count_bound"] += 1
        budget["maximum_experiment_bytes"] += int(1.25 * (budget["measured_learner_bytes"] + budget["measured_rank_metadata_bound_bytes"]))
        require(shutil.disk_usage(ROOT).free >= budget["maximum_experiment_bytes"] + 100 * 1024**3,
                "Whole monitored recipe disk budget does not fit")
        plan = copy.deepcopy(validated["selected_plan"])
        plan.update(purpose="train", recipe=str(Path(config["adapted_source"]) / "configs/q1-fresh-single.example.toml"))
        plan["overrides"].update(iterations=250000, profile_iterations=0, output_dir=str(output / "train/run"))
        plan_path = output / "train_plan.json"
        save(plan_path, plan)
        seconds = min(30 * 86400, max(86400, audited["steady_iteration_seconds"] * 250000 * 2.5 + 14400))
        result.update(status="TRAINING", validation=identity(validation / "pipeline.json"), plan=identity(plan_path),
            disk_budget=budget, iteration_only_hours_estimate=audited["steady_iteration_seconds"] * 250000 / 3600,
            estimate_excludes_compilation_evaluation_and_io=True, monitor_first_iteration=1000, monitor_every=5000,
            stop_rule="three consecutive provisional fixed9/9 and both tracking improvements")
        save(output / "pipeline.json", result)
        inputs = [plan_path, audit_path, frozen_path, *HERE.glob("*.py"),
                  HERE.parent / "project_tools/audit_q1_saved_sole_geometry.py",
                  HERE.parent / "project_tools/prepare_q1_gait_review.py"]
        if graphics:
            inputs.append(HERE.parent / "morphmimic_cross_robot/nvidia_egl_595_20261102.json")
        require(phase(config, config_path, output, "train", [config["primebot_python"], "-u", str(HERE / "train_monitored.py"),
            "--plan", str(plan_path)], seconds=seconds, maximum_bytes=budget["maximum_experiment_bytes"], inputs=inputs),
            "Formal training stopped abnormally; preserve the run without restarting")
        run_path = output / "train/run"
        marker = run_path / "early_stop.json"
        iteration = json.loads(marker.read_text())["iteration"] if marker.exists() else 250000
        result.update(status="TRAINING_COMPLETED_PENDING_EVALUATION", completed_iterations=iteration,
            truncated_recipe=iteration < 250000, training_outcome="sustained_tracking_early_stop" if marker.exists() else "full_recipe_completed")
        save(output / "pipeline.json", result)
        require(phase(config, config_path, output, "delivery", [config["primebot_python"], "-u", str(HERE / "deliver.py"),
            "--run", str(run_path), "--iteration", str(iteration), "--output", str(output / "delivery/review")],
            seconds=24 * 3600, maximum_bytes=32 * 1024**3, inputs=inputs), "Final evaluation/delivery failed")
        delivered = json.loads((output / "delivery/review/result.json").read_text())
        result.update(status="SERVER_EVALUATION_COMPLETED_PENDING_MAC_DELIVERY", server_gate=delivered,
            conditional_morph_training_started=False, ended_utc=now(),
            next_action="verify local original-speed videos and numerical artifacts, then apply full gate")
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "pipeline.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "validation", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    require(len(args.code_commit) == 40, "Use an independently verified full commit")
    run(json.loads(args.config.read_text()), args.config.resolve(), args.validation.resolve(), args.output.resolve(), args.code_commit)
