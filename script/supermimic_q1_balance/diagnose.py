"""Three-repeat, zero-optimizer fallback check on the fixed 30+3 clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "supermimic_q1_hp"), str(HERE.parent / "project_tools")]
from worker import prepare, process_identity
from balance_evaluation import rollout, summarize_balance, compare, FIXED_DEMOS
from control import ResidualHead
from resources import save, stamp
from diagnose_q1_upper_reference import policy_digest


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    status = {"status": "INITIALIZING", "started_utc": stamp(), "process": process_identity(), "optimizer_steps": 0}
    save(a.output / "status.json", status)
    env = None
    try:
        torch.set_num_threads(8)
        source = json.loads(a.library.read_text())
        rows = [dict(row, split=split) for split, values in source["splits"].items()
                for row in values if split == "validation" or row["name"] in FIXED_DEMOS]
        if len(rows) != 33:
            raise ValueError("Original 30 validation and three fixed demos required")
        lib = a.output / "library.json"
        save(lib, {"splits": {"train": rows}})
        _, env, _ = prepare(SimpleNamespace(runtime=a.runtime, library=lib, output=a.output), num_envs=33, eval_mode=True)
        from unisupermimic.tasks.motion_tracking.x2.supermimic.evaluate import load_policy
        teacher = load_policy(env, torch.load(a.checkpoint, map_location=env.device, weights_only=True)).requires_grad_(False).eval()
        head = ResidualHead().to(env.device)
        digest = policy_digest(teacher)
        thresholds = json.loads(a.calibration.read_text())["thresholds"]
        result = {"rows": rows, "optimizer_steps": 0, "repeats": []}
        for repeat in range(3):
            traces, summaries = {}, {}
            for mode in ("teacher", "static075", "gated"):
                def progress(frame):
                    status.update(status="RUNNING", repeat=repeat, mode=mode, frame=frame, updated_utc=stamp())
                    save(a.output / "status.json", status)
                trace = rollout(env, teacher, head, thresholds, mode, progress)
                np.savez_compressed(a.output / f"{mode}_repeat{repeat}.npz", **trace)
                traces[mode] = trace
                summary = summarize_balance(trace, rows)
                summary["RF11_comparison"] = compare(trace, traces["teacher"], rows)
                if mode == "gated":
                    summary["static075_comparison"] = compare(trace, traces["static075"], rows)
                summaries[mode] = summary
                save(a.output / f"{mode}_repeat{repeat}.json", summary)
                print(json.dumps({"repeat": repeat, "mode": mode, "summary": summary["summary"],
                                  "intervention": summary["intervention"], "ratios": summary["RF11_comparison"]["ratios"]}), flush=True)
            result["repeats"].append(summaries)
        if policy_digest(teacher) != digest:
            raise AssertionError("Frozen teacher changed")
        save(a.output / "comparison.json", result)
        status.update(status="DIAGNOSTIC_COMPLETED", completed_utc=stamp(), teacher_unchanged=True)
        save(a.output / "receipt.json", status)
    except BaseException as error:
        status.update(status="FAILED", error=f"{type(error).__name__}: {error}", ended_utc=stamp())
        raise
    finally:
        if env is not None:
            env.close()
            env.backend.close()
        save(a.output / "status.json", status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "library", "checkpoint", "calibration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
