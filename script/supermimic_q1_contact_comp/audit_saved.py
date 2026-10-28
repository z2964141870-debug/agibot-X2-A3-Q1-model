"""Audit already saved RF14/RF16 evidence without simulation or optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent / "supermimic_q1_ppo"))
from resources import identity, save, stamp
from contact_limits import OPTIMIZATION_DEADLINE


def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys() == b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return len(a) == len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a == b


def audit(a):
    checks = []
    def check(name,passed,**details):
        checks.append(dict(name=name,passed=bool(passed),**details))
    run = a.trial / "run"
    result = json.loads((run / "receipt.json").read_text())
    clock = json.loads((run / "training_clock.json").read_text())
    plan = json.loads((a.trial / "plan.json").read_text())
    check("training_terminal_and_budget",result["status"] == "TRAINING_COMPLETED"
          and result["training_seconds"] <=1300 and clock["active_started_monotonic"] is None
          and abs(result["training_seconds"]-clock["accumulated_training_seconds"]) < 1e-6)
    changed = [item["path"] for item in plan["source"] if identity(item["path"]) != item]
    check("frozen_sources_unchanged",not changed,count=len(plan["source"]),changed=changed)
    for name in ("train_guard.json","delivery_guard.json"):
        guard = json.loads((a.trial / name).read_text())
        cleanup = guard.get("cleanup",{})
        check(name,guard["status"] == "WORKER_COMPLETED" and cleanup.get("worker_reaped") and cleanup.get("worker_return_code") ==0)
    stages = json.loads((run / "stage_decisions.json").read_text())
    check("stage_evaluations_saved",bool(stages))
    final = torch.load(run / "final.pt",map_location="cpu",weights_only=True)
    stage = torch.load(stages[-1]["checkpoint"]["path"],map_location="cpu",weights_only=True)
    keys = ("wm","pi","critic","opt_wm","opt_pi","opt_v") if a.mode == "base" else ("actor","critic","optimizer")
    check("evaluation_and_persistence_did_not_optimize",all(equal(final[k],stage[k]) for k in keys))
    if a.mode == "base":
        check("inherited_budget_accounted",abs(clock["inherited_training_seconds"]-3500.2659895760007) < 1e-6
              and final["iteration"] >=3271 and final["sim_steps"] >=33755136)
        selection = json.loads((run / "checkpoint_selection.json").read_text())
        check("selection_uses_predeclared_rank",selection["selected"] ==min(selection["candidates"],key=lambda v:v["rank"]))
        check("selected_checkpoint_identity",identity(selection["selected"]["checkpoint"]["path"]) == selection["selected"]["checkpoint"])
    else:
        check("source_phase_cache_frozen", bool(final.get("phase_contract")) and final["phase_contract"] == json.loads((run / "reference_phase_contract.json").read_text()))
        check("only_gait_head_and_critic_new",final["new_world_model_updates"] ==0 and final["input_features"] ==1420
              and final["latent_std"] ==.1 and final["residual_radius_rad"] ==.15)
        check("overnight_budget_and_continuation_evidence",clock["current_cap_seconds"] ==1300
              and clock["optimization_wall_deadline_unix"] ==OPTIMIZATION_DEADLINE
              and all(row["overnight_continuation"]["continue_authorized"]
                      for row in stages[:-1] if row["training_seconds"] >=3500))
    receipt = json.loads((a.trial / "evaluation/receipt.json").read_text())
    for label in ("initial_traces","final_traces"):
        paths = receipt[label]
        valid = len(paths) ==3
        for path in paths:
            with np.load(path,allow_pickle=False) as saved:
                valid = valid and saved["steps"].shape ==(330,) and saved["feet"].shape[1:] ==(330,2,3)
                valid = valid and bool(np.isfinite(saved["states"]).all()) and bool(np.isfinite(saved["feet"]).all())
                valid = valid and bool((saved["steps"] <=saved["reference_steps"]).all())
        check(label+"_physical_evidence",valid)
    check("pipeline_completed",json.loads((a.trial / "status.json").read_text())["status"] =="COMPLETED")
    value = {"status":"SAVED_EVIDENCE_AUDIT_PASSED" if all(r["passed"] for r in checks) else "FAILED",
             "captured_utc":stamp(),"trial":str(a.trial),"mode":a.mode,"checks":checks,
             "optimizer_updates":0,"new_physical_steps":0,"training_code_commit":plan["code_commit"],
             "physical_success_claimed":False}
    save(a.output,value)
    print(json.dumps(value))
    return 0 if value["status"] =="SAVED_EVIDENCE_AUDIT_PASSED" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial",type=Path,required=True)
    parser.add_argument("--mode",choices=("base","gait"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use existing audit; do not repeat or overwrite")
    raise SystemExit(audit(args))

