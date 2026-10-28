"""Bounded zero-update base/arm-blend comparison before overnight training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path("/opt/data/private/ZHY")
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_hp"),
               str(HERE.parent / "supermimic_q1_ppo"), str(HERE.parent / "project_tools")]
from worker import prepare, process_identity
from resources import save, identity, stamp
from run_trial import Q1Budget
from runtime_guard import guarded_run


def worker(a):
    import numpy as np
    import torch
    from physical import rollout, FootReference, FIXED_DEMOS, summarize, gait_summary
    from diagnose_q1_upper_reference import policy_digest
    from metrics import checkpoint_rank

    torch.set_num_threads(8)
    source = json.loads(a.library.read_text())
    rows = [dict(row, split=split) for split in ("train", "validation")
            for row in source["splits"][split]
            if split == "validation" or row["name"] in FIXED_DEMOS]
    if len(rows) != 33 or sum(r["split"] == "validation" for r in rows) != 30:
        raise ValueError("Keep validation30 and exactly three fixed demonstrations")
    library = a.output / "evaluation_library.json"
    save(library, {"splits": {"train": rows}})
    run = a.output / "run"
    run.mkdir(exist_ok=False)
    status = dict(status="INITIALIZING", process=process_identity(), started_utc=stamp(), optimizer_steps=0)
    save(run / "status.json", status)
    env = None
    try:
        _, env, _ = prepare(SimpleNamespace(runtime=a.runtime, library=library, output=run),
                            num_envs=33, eval_mode=True)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        foot = FootReference(env.supermimic_mdp)
        thresholds = json.loads(a.calibration.read_text())["thresholds"]
        results = []
        for base in json.loads(a.candidates.read_text())["bases"]:
            path = Path(base["checkpoint"])
            teacher = load_policy(env, torch.load(path, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
            digest = policy_digest(teacher)
            for alpha in (0., .5, .75):
                label = base["name"] + "_alpha" + str(round(alpha*100)).zfill(3)
                repeats = []
                for repeat in range(3):
                    def progress(frame):
                        status.update(status="RUNNING", candidate=label, repeat=repeat, frame=frame, updated_utc=stamp())
                        save(run / "status.json", status)
                    trace = rollout(env, teacher, None, thresholds, foot,
                                    "teacher" if alpha == 0 else "static075", progress, upper_alpha=alpha)
                    target = a.output / (label + f"_repeat{repeat}.npz")
                    np.savez_compressed(target, **trace)
                    summary = summarize(trace, rows)
                    summary.update(gait=gait_summary(trace, rows), repeat=repeat, upper_alpha=alpha,
                                   checkpoint=identity(path), trace=identity(target))
                    save(target.with_suffix(".json"), summary)
                    repeats.append(summary)
                rank = checkpoint_rank(repeats)
                results.append(dict(label=label, checkpoint=identity(path), upper_alpha=alpha,
                                    rank=rank, summaries=repeats))
                print(json.dumps(dict(candidate=label, rank=rank,
                    validation_failures=[r["summary"]["validation"]["failures"] for r in repeats],
                    fixed=[r["gait"] for r in repeats])), flush=True)
            if policy_digest(teacher) != digest:
                raise AssertionError("Probe changed frozen teacher/normalization")
        save(a.output / "results.json", dict(status="COMPLETED", optimizer_steps=0,
            rows=rows, candidates=results, rank_protocol="Fixed/validation failures, forward progress, then gait diagnostics; agent reviews upper amplitude/timing before training choice",
            limitations="Validation used for candidate exploration, not an independent test; three repeats are not training seeds; physical resets do not guarantee identical hidden solver state",
            user_accepted_checkpoint=None))
        status.update(status="COMPLETED", completed_utc=stamp(), candidates=len(results), frozen_teachers_unchanged=True)
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        save(run / "status.json", status)


def main(a):
    if a.worker:
        worker(a)
        return
    a.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(a.frozen_plan.read_text())["source"]
    sources = frozen + [identity(p) for p in (Path(__file__), HERE / "physical.py", a.candidates, a.library, a.calibration)]
    for item in sources:
        if identity(item["path"]) != item:
            raise RuntimeError("Prior frozen source changed: " + item["path"])
    save(a.output / "plan.json", dict(source=sources, started_utc=stamp(), process=process_identity(),
         optimizer_steps=0, wall_cap_seconds=1500, candidates=identity(a.candidates)))
    class Budget(Q1Budget):
        def check(self):
            value = super().check()
            if time.monotonic()-self.started > 1500:
                raise RuntimeError("Zero-update diagnostic exceeded25 minutes")
            return value
    budget = Budget(SimpleNamespace(output=a.output, runtime=a.runtime, mode="diagnosis", expected_oom_kill=0))
    command = [sys.executable, "-u", str(HERE.parent / "supermimic_q1_hp/activate.py"),
               "-u", str(Path(__file__)), *sys.argv[1:], "--worker"]
    status = dict(status="RUNNING", process=process_identity(), started_utc=stamp(), optimizer_steps=0)
    save(a.output / "status.json", status)
    try:
        guard = guarded_run(command, ROOT, budget, a.output / "guard.json",
            ROOT / "logs" / (a.output.name+".log"), progress_path=a.output / "run/status.json",
            maximum_idle_s=900, grace_s=45)
        if guard["status"] != "WORKER_COMPLETED" or guard["cleanup"]["worker_return_code"] != 0:
            raise RuntimeError("Probe failed; no automatic restart")
        for item in sources:
            if identity(item["path"]) != item:
                raise RuntimeError("Probe source changed: " + item["path"])
        status.update(status="COMPLETED", ended_utc=stamp(), frozen_source_count=len(sources))
    except BaseException as error:
        status.update(status="FAILED", ended_utc=stamp(), error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(a.output / "status.json", status)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "calibration", "candidates", "frozen-plan", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--worker", action="store_true")
    main(p.parse_args())
