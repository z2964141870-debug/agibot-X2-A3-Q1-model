"""Compare frozen RF11 with two arm-target corrections, without optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo")]
from worker import identity, now, prepare, process_identity, save
from run_trial import Q1Budget
from runtime_guard import guarded_run
from q1_upper_reference import ReferenceCenteredUpper


def policy_digest(policy):
    value = hashlib.sha256()
    for key, tensor in policy.state_dict().items():
        value.update(key.encode("ascii"))
        value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def worker(args):
    import numpy as np
    import torch
    from evaluation import FIXED_DEMOS, clip_metrics, compare, rollout, summarize

    torch.set_num_threads(8)
    source = json.loads(args.library.read_text())
    rows = [dict(row, split=split) for split, values in source["splits"].items()
            for row in values if split == "validation" or row["name"] in FIXED_DEMOS]
    if len([row for row in rows if row["split"] == "validation"]) != 30:
        raise ValueError("Retain all original 30 validation clips")
    combined = args.output / "evaluation_library.json"
    save(combined, {"splits": {"train": rows}})
    (args.output / "run").mkdir(exist_ok=False)
    status_path = args.output / "run/status.json"
    status = dict(status="INITIALIZING", started_utc=now(), optimizer_steps=0,
                  process=process_identity(), selected_clips=len(rows),
                  library=identity(args.library), checkpoint=identity(args.checkpoint),
                  candidates={"RF11": None, "reference": 0., "residual015": .15},
                  protocol="Fixed original validation30 + fixed walks/raised hands; same environment batch for each candidate; no noise, no optimization.")
    save(status_path, status)
    env = None
    try:
        _, env, _ = prepare(SimpleNamespace(runtime=args.runtime, library=combined,
                                           output=args.output),
                             num_envs=len(rows), eval_mode=True)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        policy = load_policy(env, torch.load(args.checkpoint, map_location=env.device,
                                             weights_only=True))
        policy.eval()
        before = policy_digest(policy)
        results = {name: [] for name in ("RF11", "reference", "residual015")}
        for repeat in range(3):
            baseline = None
            for name, radius in status["candidates"].items():
                def progress(frame):
                    status.update(status="RUNNING", repeat=repeat, candidate=name,
                                  frame=frame, updated_utc=now())
                    save(status_path, status)
                selected = policy if radius is None else ReferenceCenteredUpper(policy, env, radius)
                trace = rollout(env, selected, progress=progress)
                path = args.output / f"{name}_repeat{repeat}.npz"
                np.savez_compressed(path, **trace)
                summary = summarize(trace, rows)
                summary["fixed_demos"] = [dict(name=row["name"], **clip_metrics(trace, i))
                                           for i, row in enumerate(rows) if row["name"] in FIXED_DEMOS]
                if name == "RF11":
                    baseline = trace
                else:
                    summary["RF11_common_prefix_comparison"] = compare(trace, baseline, rows)
                save(path.with_suffix(".json"), summary)
                results[name].append(summary)
                print(json.dumps({"candidate": name, "repeat": repeat,
                                  "validation": summary["summary"]["validation"],
                                  "comparison": summary.get("RF11_common_prefix_comparison")}), flush=True)
            del baseline
        if policy_digest(policy) != before:
            raise AssertionError("Frozen policy or normalizer changed")
        save(args.output / "results.json", dict(results=results, rows=rows,
             policy_sha256=before, optimizer_steps=0,
             limitation="33-environment diagnostic, not a new 4096 training or full330 benchmark. No validation tuning, no reward/PD/reference changes. Three predetermined repeats/conditions; physical effects include closed-loop coupling."))
        status.update(status="COMPLETED", completed_utc=now(), frozen_policy_unchanged=True,
                      result=identity(args.output / "results.json"))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=now(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        save(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("runtime", "library", "checkpoint", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    status_path = args.output / "status.json"
    result = dict(status="RUNNING", started_utc=now(), optimizer_steps=0,
                  controller=process_identity(), source=identity(Path(__file__)),
                  candidate_source=identity(HERE / "q1_upper_reference.py"))
    save(status_path, result)

    class DiagnosticBudget(Q1Budget):
        def check(self):
            resources = super().check()
            if time.monotonic() - self.started > 1500:
                raise RuntimeError("Bounded nontraining diagnostic exceeded 25 minutes")
            return resources

    budget = DiagnosticBudget(SimpleNamespace(output=args.output, runtime=args.runtime,
                                               mode="diagnosis", expected_oom_kill=0))
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
               "-u", str(Path(__file__)), *sys.argv[1:], "--worker"]
    try:
        guard = guarded_run(command, Path("/opt/data/private/ZHY"), budget,
                            args.output / "guard.json",
                            Path("/opt/data/private/ZHY/logs") / (args.output.name + ".log"),
                            grace_s=45, maximum_idle_s=900,
                            progress_path=args.output / "run/status.json")
        if guard["status"] != "WORKER_COMPLETED":
            raise RuntimeError("Diagnostic guard: " + guard["status"])
        observed = json.loads((args.output / "run/status.json").read_text())
        if observed["status"] != "COMPLETED":
            raise RuntimeError("Diagnostic worker did not complete")
        result.update(status="COMPLETED", worker_result=observed["result"])
    except BaseException as error:
        result.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        result["ended_utc"] = now()
        save(status_path, result)


if __name__ == "__main__":
    main()
