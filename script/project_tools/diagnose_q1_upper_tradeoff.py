"""Five predetermined closed-loop arm blends; frozen RF11, zero optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo")]
from worker import identity, now, prepare, process_identity, save
from run_trial import Q1Budget
from runtime_guard import guarded_run
from diagnose_q1_upper_reference import policy_digest
from q1_upper_blend import BlendedUpper

ALPHAS = {"alpha000": 0., "alpha025": .25, "alpha050": .5,
          "alpha075": .75, "alpha100": 1.}
METRICS = ("root_xy_rmse", "upper_rmse", "leg_rmse", "wrist_relative_rmse",
           "target_oscillation", "loaded_slip")


def check_sources(items):
    for item in items:
        if identity(item["path"]) != item:
            raise ValueError("Frozen source changed: " + item["path"])
    return {"count": len(items), "bytes": sum(item["bytes"] for item in items),
            "all_size_sha256_match": True}


def worker(args):
    import numpy as np
    import torch
    from evaluation import FIXED_DEMOS, clip_metrics, compare, rollout, summarize

    torch.set_num_threads(8)
    source = json.loads(args.library.read_text())
    rows = [dict(row, split=split) for split, values in source["splits"].items()
            for row in values if split == "validation" or row["name"] in FIXED_DEMOS]
    if len(rows) != 33 or sum(r["split"] == "validation" for r in rows) != 30:
        raise ValueError("Retain original validation30 plus the fixed three demos")
    if set(FIXED_DEMOS) != {r["name"] for r in rows if r["split"] != "validation"}:
        raise ValueError("Fixed demo identity changed")
    library = args.output / "evaluation_library.json"
    save(library, {"splits": {"train": rows}})
    (args.output / "run").mkdir(exist_ok=False)
    status_path = args.output / "run/status.json"
    status = dict(status="INITIALIZING", started_utc=now(), optimizer_steps=0,
                  process=process_identity(), selected_clips=33)
    save(status_path, status)
    env = None
    try:
        _, env, _ = prepare(SimpleNamespace(runtime=args.runtime, library=library,
                                           output=args.output), num_envs=33, eval_mode=True)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        policy = load_policy(env, torch.load(args.checkpoint, map_location=env.device,
                                             weights_only=True))
        policy.eval()
        before = policy_digest(policy)
        results = {name: [] for name in ALPHAS}
        common_indices = [i for i, row in enumerate(rows) if row["split"] == "validation"]
        for repeat in range(3):
            traces = {}
            for name, alpha in ALPHAS.items():
                def progress(frame):
                    status.update(status="RUNNING", repeat=repeat, candidate=name,
                                  frame=frame, updated_utc=now())
                    save(status_path, status)
                selected = BlendedUpper(policy, env, alpha)
                trace = rollout(env, selected, progress=progress)
                np.savez_compressed(args.output / f"{name}_repeat{repeat}.npz", **trace)
                traces[name] = trace
                summary = summarize(trace, rows)
                summary["alpha"] = alpha
                summary["fixed_demos"] = [dict(name=row["name"], **clip_metrics(trace, i))
                    for i, row in enumerate(rows) if row["name"] in FIXED_DEMOS]
                summary["RF11_common_prefix_comparison"] = compare(trace, traces["alpha000"], rows)
                results[name].append(summary)
                save(args.output / f"{name}_repeat{repeat}.json", summary)
                print(json.dumps(dict(candidate=name, repeat=repeat,
                     validation=summary["summary"]["validation"],
                     comparison=summary["RF11_common_prefix_comparison"])), flush=True)
            lengths = np.minimum.reduce([t["steps"] for t in traces.values()])
            for name, trace in traces.items():
                per_clip = [dict(name=rows[i]["name"], **clip_metrics(trace, i, lengths[i]))
                            for i in common_indices]
                measured = {key: float(np.mean([r[key] for r in per_clip]))
                    if all(r[key] is not None for r in per_clip) else None for key in METRICS}
                results[name][-1]["all_alpha_common_prefix"] = dict(measurements=measured,
                    mean_seconds=float(lengths[common_indices].mean() * .02), per_clip=per_clip)
                save(args.output / f"{name}_repeat{repeat}.json", results[name][-1])
        if policy_digest(policy) != before:
            raise AssertionError("Frozen policy or normalizer changed")
        result_path = args.output / "results.json"
        save(result_path, dict(results=results, rows=rows, optimizer_steps=0,
             policy_sha256=before, alphas=ALPHAS,
             protocol="33 real closed-loop environments; no observation noise; fixed ascending alpha order in each of three repeats; original full30 validation plus fixed three demos; common-prefix and complete-episode outcomes retained.",
             limitations="Existing validation set diagnostic, not a held-out test or training. Same reset-to-reference protocol, not identical hidden solver states. No optimality, significance, monotonicity or safety guarantee. All-alpha shared prefixes exclude later behavior: report completion/failure alongside errors."))
        status.update(status="COMPLETED", completed_utc=now(),
                      frozen_policy_unchanged=True, result=identity(result_path))
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
    for name in ("runtime", "library", "checkpoint", "frozen-plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    sources = json.loads(args.frozen_plan.read_text())["source"]
    own = [identity(HERE / name) for name in
           ("diagnose_q1_upper_tradeoff.py", "q1_upper_blend.py",
            "diagnose_q1_upper_reference.py", "q1_upper_reference.py")]
    plan = dict(started_utc=now(), alphas=ALPHAS, repeats=3, optimizer_steps=0,
        protocol="Fixed validation30 + three demos; lower14 always use real closed-loop history. Blended physical upper PD targets: (1-alpha)*policy + alpha*reference[f+1], original action clamp/scale/base retained.",
        source=own, checkpoint=identity(args.checkpoint), library=identity(args.library),
        frozen_plan=identity(args.frozen_plan), old_sources_before=check_sources(sources),
        wall_cap_seconds=1500, maximum_idle_seconds=900, grace_seconds=45,
        lower_control="Frozen RF11 responds to real observations and previous executed actions, not replay or masked sensors.")
    save(args.output / "plan.json", plan)
    status = dict(status="RUNNING", started_utc=now(), optimizer_steps=0,
                  controller=process_identity())
    save(args.output / "status.json", status)

    class DiagnosticBudget(Q1Budget):
        def check(self):
            resources = super().check()
            if time.monotonic() - self.started > 1500:
                raise RuntimeError("Nontraining trade-off diagnostic exceeded25 minutes")
            return resources

    budget = DiagnosticBudget(SimpleNamespace(output=args.output, runtime=args.runtime,
                                               mode="diagnosis", expected_oom_kill=0))
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
               "-u", str(Path(__file__)), *sys.argv[1:], "--worker"]
    try:
        guard = guarded_run(command, Path("/opt/data/private/ZHY"), budget,
            args.output / "guard.json", Path("/opt/data/private/ZHY/logs") / (args.output.name + ".log"),
            grace_s=45, maximum_idle_s=900, progress_path=args.output / "run/status.json")
        if guard["status"] != "WORKER_COMPLETED" or not guard["cleanup"]["worker_reaped"] or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("Diagnostic guard did not reap a successful worker")
        observed = json.loads((args.output / "run/status.json").read_text())
        if observed["status"] != "COMPLETED":
            raise RuntimeError("Diagnostic worker did not complete")
        check_sources(own + [plan["checkpoint"], plan["library"]])
        status.update(status="COMPLETED", result=observed["result"],
                      old_sources_after=check_sources(sources))
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["ended_utc"] = now()
        save(args.output / "status.json", status)


if __name__ == "__main__":
    main()
