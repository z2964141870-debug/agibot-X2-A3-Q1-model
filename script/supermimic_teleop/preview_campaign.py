"""Single-use sequential preview campaign that survives SSH disconnection.

Each study keeps the existing runtime, H10 equality and original-training gates.
Controllers run in this process so their owned worker cleanup completes before
another study starts. No training, lock acquisition or failure retries occur.
"""

import argparse
import copy
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import preview_review as review
from common import ROOT, identity, now, require, save
from delivery import read
from interim_review import GIB, ReviewBudget, TRAINING
from runtime_guard import allocated_tree

DEFAULT_METHODS = ("hold", "damped_cv", "cv")
MAXIMUM_WALL_SECONDS = 7200


def studies(output, baseline_output, methods):
    methods = tuple(methods)
    require(methods and len(set(methods)) == len(methods) and set(methods) <= set(DEFAULT_METHODS),
            "Select each independent forecast method at most once")
    baseline = output / "masked" if baseline_output is None else baseline_output
    rows = ([] if baseline_output is not None else
            [dict(tail_method="masked", output=baseline, baseline_output=None)])
    return rows + [dict(tail_method=method, output=output / ("forecast_" + method),
                        baseline_output=baseline) for method in methods]


class CampaignBudget(ReviewBudget):
    def __init__(self, config, output, campaign_root, deadline):
        super().__init__(config, output)
        self.campaign_root, self.deadline = campaign_root, deadline

    def check(self):
        require(time.monotonic() < self.deadline, "Preview campaign reached its two-hour bound")
        require(allocated_tree(self.campaign_root)["unique_allocated_bytes"] < 3 * GIB,
                "Preview campaign reached its combined output bound")
        return super().check()


def execute(args):
    args.child, args.iteration = None, 16000
    schedule = studies(args.output, args.baseline_output, args.tail_methods)
    sources = review.validate_code_manifest(args.code_manifest, args.manifest_sha256,
                                            args.evaluation_commit)
    _, _, _, _, training_sources = review.preflight(args)
    frozen_paths = {row["path"] for row in read(TRAINING / "e1024/train/status.json")["inputs"]}
    require(str(Path(__file__).resolve()) not in frozen_paths,
            "Independent campaign must not be a frozen training input")
    control_inputs = [identity(path) for path in
                      (args.config, args.runtime_config, args.reuse_audit, args.code_manifest)]
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    deadline, requested = started + MAXIMUM_WALL_SECONDS, False
    result = dict(status="RUNNING", stage="LOW_LATENCY_PREVIEW_CAMPAIGN", started_utc=now(),
        evaluation_commit=args.evaluation_commit, studies=[],
        schedule=[dict(tail_method=row["tail_method"], output=str(row["output"]),
                       baseline_output=None if row["baseline_output"] is None else
                       str(row["baseline_output"])) for row in schedule],
        original_training_commit=review.COMMIT, maximum_wall_seconds=MAXIMUM_WALL_SECONDS,
        maximum_combined_output_bytes=3 * GIB, maximum_study_wall_seconds=1800,
        training_signals_sent=0, acquire_training_gpu_lock=False,
        final_acceptance=False, measured_end_to_end_latency_ms=None,
        user_accepted_checkpoint=None)
    save(args.output / "authorization.json", dict(result,
         evaluation_sources=sources, training_source_inputs=training_sources,
         controller_inputs=control_inputs, retry_failed_study=False))

    def stop(signum, frame):
        nonlocal requested
        requested = True

    previous = {signum: signal.signal(signum, stop) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        for row in schedule:
            require(not requested, "Preview campaign stop requested")
            require(time.monotonic() < deadline, "Preview campaign reached its two-hour bound")
            require(all(identity(value["path"]) == value for value in control_inputs),
                    "Campaign configuration or published manifest changed")
            child = copy.copy(args)
            child.tail_method, child.output = row["tail_method"], row["output"]
            child.baseline_output = row["baseline_output"]
            save(args.output / "result.json", dict(result, current_study=row["tail_method"]))
            value = review.execute(child, budget_factory=lambda config, output:
                CampaignBudget(config, output, args.output, deadline))
            require(value.get("status") == "COMPLETED" and value.get("source_integrity") == "PASSED",
                    "Preview study incomplete; do not dispatch another method")
            result["studies"].append(dict(tail_method=row["tail_method"],
                result=identity(child.output / "result.json"), comparison=value["comparison"],
                new_cases=value["total_cases"], videos=value["media_videos"]))
            save(args.output / "result.json", result)
        require(all(identity(value["path"]) == value
                    for value in sources + training_sources + control_inputs),
                "Campaign inputs changed")
        result.update(status="COMPLETED", ended_utc=now(), source_integrity="PASSED",
            total_new_cases=sum(value["new_cases"] for value in result["studies"]),
            total_videos=sum(value["videos"] for value in result["studies"]),
            elapsed_seconds=time.monotonic() - started)
        return result
    except BaseException as error:
        result.update(status="FAILED", ended_utc=now(), error=f"{type(error).__name__}: {error}",
                      elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        try:
            save(args.output / "result.json", result)
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("config", "runtime-config", "output", "reuse-audit", "code-manifest"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--evaluation-commit", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--baseline-output", type=Path)
    parser.add_argument("--tail-methods", nargs="+", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    execute(parser.parse_args())
