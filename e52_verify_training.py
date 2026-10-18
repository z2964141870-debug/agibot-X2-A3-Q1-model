"""E52 verify-training: changed-name scope, critic deltas, Adam counters, receipts."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/opt/data/private/yuanqi_ops")
sys.path.insert(0, "/opt/data/private/hechengzhu/ZHY")

from script.a3.checkpoint_store import sha256

OUT = Path("/opt/data/private/yuanqi_ops/data/experiments/a3_zhy_projection_finetune_20261018_E52")
TRAIN = Path("/opt/data/private/yuanqi_ops/data/training/a3_zhy_projection_finetune_20261018_E52")
LOG = Path("/opt/data/private/yuanqi_ops/logs/a3_zhy_projection_finetune_20261018_E52")
SOURCE = Path("/opt/data/private/hechengzhu/ZHY/data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/milestone_4/model_step_000004.pt")


def payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def changed_names(before, after, prefix=None):
    names = []
    for key in before["policy_state_dict"]:
        if not torch.equal(before["policy_state_dict"][key], after["policy_state_dict"][key]):
            names.append(key)
    return names


def value_changed(before, after):
    return any(not torch.equal(before["value_state_dict"][k], after["value_state_dict"][k])
               for k in before["value_state_dict"])


def adam_step(payload):
    steps = {int(v.get("step", 0)) for v in payload["optimizer_state_dict"]["state"].values()}
    return sorted(steps)


report = {}
source = payload(SOURCE)
for arm in ("a", "b", "c"):
    arm_dir = TRAIN / ("arm_" + arm)
    milestones = sorted(int(p.name.split("_")[1]) for p in arm_dir.glob("milestone_*"))
    last = milestones[-1]
    after = payload(arm_dir / ("milestone_" + str(last)) / ("model_step_%06d.pt" % last))
    names = changed_names(source, after)
    actor_changed = [n for n in names if ".decoders.g1_dyn." in n]
    other_actor = [n for n in names if n not in actor_changed]
    adam = adam_step(after)
    restore = json.loads((LOG / ("arm_" + arm) / "step0_optimizer_restore.json").read_text())
    projections = None
    ppath = LOG / ("arm_" + arm) / "projection_steps.jsonl"
    if ppath.exists():
        rows = [json.loads(line) for line in ppath.read_text().splitlines()]
        projections = dict(optimizer_rows=len(rows), param_steps=sum(len(r["projections"]) for r in rows),
                           zero_teacher=sum(1 for r in rows for p in r["projections"] if p["zero_teacher_gradient"]))
    report[arm] = dict(
        milestones=milestones, evaluated_milestone=last,
        changed_actor_other_than_g1_dyn=other_actor[:4] + (["..."] if len(other_actor) > 4 else []),
        changed_actor_other_count=len(other_actor),
        g1_dyn_changed_count=len(actor_changed),
        critic_changed=value_changed(source, after),
        adam_state_steps=adam,
        restore_moments_match_start=restore["optimizer_state_sha256"] == restore["payload_optimizer_state_sha256"],
        projections=projections,
        policy_step=int(after["state"].global_step),
    )
(OUT / "training_verification.json").write_text(json.dumps(report, indent=1))
for arm, row in report.items():
    print(arm.upper(), json.dumps({k: v for k, v in row.items() if k != "restore_moments_match_start"}))
