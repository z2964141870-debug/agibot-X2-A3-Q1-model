"""Snapshot terminal experiments without interpreting partial training as acceptance."""
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import torch
from common import HERE,mj,BUNDLE


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize():
    report=dict(status="development_in_progress",final_acceptance="not_run",runtime=dict(python=platform.python_version(),torch=torch.__version__,mujoco=mj.__version__),
                training={},validation={},models={},artifacts={})
    for name in ("teacher_stand_01","teacher_stand_02","teacher_balanced_01","teacher_balanced_02"):
        p=HERE/name;rows=[json.loads(s) for s in (p/"metrics.jsonl").read_text().splitlines()]
        latest=torch.load(p/"latest.pt",map_location="cpu",weights_only=False)
        report["training"][name]=dict(last_record=rows[-1],saved_update=latest["update"],record_count=len(rows),
            metrics_sha256=sha(p/"metrics.jsonl"),configuration=latest["args"],
            terminal_state="stopped_for_validation_review" if name=="teacher_balanced_01" else "completed_requested_updates")
        for path in sorted(p.glob("*.pt"))+sorted(p.glob("*.onnx")):
            report["models"][str(path.relative_to(HERE))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    for family in ("stage_dev","stage_balanced"):
        for path in sorted((HERE/family).glob("*/results.json")):
            rows=json.loads(path.read_text())
            final=[r for r in rows if r.get("final_calm_second",False)]
            report["validation"][str(path.parent.relative_to(HERE))]=dict(n=len(rows),standing_half_second=sum(r["standing"] for r in rows),
                final_calm_second=sum(r.get("final_calm_second",False) for r in rows),
                final_phases=[r["phase"] for r in final],
                successful_subset_peaks={key:max([r[key] for r in final],default=None) for key in ("peak_qd","ground_mm","self_mm")},
                all_trial_peaks={key:max(r[key] for r in rows) for key in ("peak_qd","ground_mm","self_mm")},sha256=sha(path))
    files=[HERE/"plant.so",HERE/"prepared_v3/contract.json",HERE/"acceptance_plan.json",BUNDLE/"config/q1_deploy_config.json",BUNDLE/"robot_model/supermimic.xml"]
    files+=list((HERE/"media_01").glob("*"))+list((HERE/"media_02").glob("*"))
    for path in files:
        if path.is_file():report["artifacts"][str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
    report["source_fidelity"]={"teacher_stand_01":"original train.py was not frozen; later snapshots do not repair that gap",
        "teacher_stand_02":"manually frozen during execution","teacher_balanced_01":"frozen before rollout","teacher_balanced_02":"frozen before rollout"}
    report["training"]["teacher_balanced_01"]["signal_exit_code"]=130
    (HERE/"phase01_manifest.json").write_text(json.dumps(report,default=str,indent=2))
    print(json.dumps(dict(training={k:v["saved_update"] for k,v in report["training"].items()},validation=report["validation"]),indent=2))


if __name__=="__main__":summarize()
