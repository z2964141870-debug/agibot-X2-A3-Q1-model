"""Start one fresh full Primebot recipe only after complete capacity/save audit."""

import argparse
import copy
import json
from pathlib import Path
import shutil

from common import HERE, ROOT, identity, now, require, save
from validate import phase


def disk_budget(profile_audit, iterations=250000):
    final = profile_audit["final"]
    learner = final["checkpoint"]["bytes"]
    replay = final["full_capacity_replay_bytes"]
    rank_metadata = max(1024**2, final["rank_shard"]["bytes"] - final["initialized_replay_bytes"])
    # Periodic, zero, first-compiled, iteration100, exposure and final snapshots;
    # atomic final replay publication also needs room for a complete temporary.
    snapshots = iterations // 5000 + 5
    payload = snapshots * (learner + rank_metadata) + 2 * replay + 8 * 1024**3
    return dict(checkpoint_count_bound=snapshots, measured_learner_bytes=learner,
        measured_rank_metadata_bound_bytes=rank_metadata, full_replay_bytes=replay,
        maximum_experiment_bytes=int(payload * 1.25), minimum_free_reserve_bytes=100 * 1024**3)


def execute(config, config_path, validation, output, commit):
    output.mkdir(parents=True, exist_ok=False)
    record = dict(status="INITIALIZING", started_utc=now(), code_commit=commit,
                  fresh_initialization=True, reuse_profile_weights=False, automatic_retry=False,
                  user_accepted_checkpoint=None)
    try:
        check = json.loads((validation / "pipeline.json").read_text())
        require(check["status"] == "VALIDATED_READY_FOR_FRESH_FORMAL", "Validation is incomplete")
        audit_path = Path(check["audit"]["path"])
        require(identity(audit_path) == check["audit"], "Profile checkpoint audit changed")
        audit = json.loads(audit_path.read_text())
        require(audit["status"] == "PASSED", "Eight-update CPU audit did not pass")
        budget = disk_budget(audit)
        require(shutil.disk_usage(ROOT).free >= budget["maximum_experiment_bytes"] + budget["minimum_free_reserve_bytes"],
                "Whole recipe checkpoint/replay disk admission failed")
        plan = copy.deepcopy(check["selected_plan"])
        plan.update(purpose="train")
        plan["overrides"].update(iterations=250000, profile_iterations=0, output_dir=str(output / "train/run"))
        path = output / "train_plan.json"
        save(path, plan)
        seconds = min(30 * 86400, max(86400, audit["steady_iteration_seconds"] * 250000 * 2.5 + 7200))
        record.update(status="TRAINING", validation=identity(validation / "pipeline.json"),
            plan=identity(path), disk_budget=budget,
            iteration_only_hours_estimate=audit["steady_iteration_seconds"] * 250000 / 3600,
            external_wall_seconds=seconds, estimate_excludes_compilation_and_io=True)
        save(output / "pipeline.json", record)
        require(phase(config, config_path, output, "train", [config["primebot_python"], "-u",
            str(HERE / "train.py"), "--plan", str(path)], seconds=seconds,
            maximum_bytes=budget["maximum_experiment_bytes"], inputs=[path, audit_path]),
            "Formal training stopped abnormally; preserve the run and do not restart")
        audit_output = output / "final_checkpoint_audit.json"
        require(phase(config, config_path, output, "final_audit", [config["primebot_python"], "-u",
            str(HERE / "audit.py"), "--run", str(output / "train/run"), "--iteration", "250000",
            "--output", str(audit_output)]), "Full-iteration final checkpoint audit failed")
        record.update(status="TRAINING_COMPLETED_PENDING_EVALUATION", final_audit=identity(audit_output), ended_utc=now())
    except BaseException as error:
        record.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=now())
        raise
    finally:
        save(output / "pipeline.json", record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "validation", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    execute(json.loads(args.config.read_text()), args.config.resolve(), args.validation.resolve(),
            args.output.resolve(), args.code_commit)
