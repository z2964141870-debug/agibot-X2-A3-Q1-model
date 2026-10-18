"""E52 anchor probe: precompute E45 step4 policy anchor outputs (one GPU forward)."""

import hashlib
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/opt/data/private/yuanqi_ops")
sys.path.insert(0, "/opt/data/private/hechengzhu/ZHY")

from gear_sonic.trl.utils.common import custom_instantiate
from script.a3.cloth_constrained_trainer import restrict_policy

OUT = Path("/opt/data/private/yuanqi_ops/data/experiments/a3_zhy_projection_finetune_20261018_E52")
SOURCE = Path("/opt/data/private/hechengzhu/ZHY/data/training/a3_zhy_cloth_failure_20261013_E45/R19_001_s0/milestone_4/model_step_000004.pt")
ANCHOR = Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt")
TARGETS = Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt")
CONFIG = Path("/opt/data/private/hechengzhu/ZHY/logs/a3_zhy_cloth_capture_20261013_E39/training/runtime_metadata.json")


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    if int(payload["state"].global_step) != 4:
        raise ValueError("E45 source checkpoint step changed")
    config = OmegaConf.create(json.loads(CONFIG.read_text()))
    actor = custom_instantiate(config.algo_config.actor, env_config=config.env_config,
                               algo_config=config.algo_config, _resolve=False)
    actor.load_state_dict(payload["policy_state_dict"], strict=True)
    restrict_policy(actor)
    actor = actor.to("cuda").eval()
    rollout = torch.load(ANCHOR, map_location="cpu", weights_only=False)
    obs = {name: value.to("cuda") for name, value in rollout["all_obs_dict"].items()}
    del rollout
    official = torch.load(TARGETS, map_location="cpu", weights_only=False)["initial"]["mean"]
    with torch.no_grad():
        means = torch.cat([actor.forward({n: v[i:i + 4] for n, v in obs.items()}, is_training=False)
                           for i in range(0, 16, 4)], dim=0).cpu()
    if means.shape != (16, 24, 29) or not torch.isfinite(means).all():
        raise ValueError("Anchor probe coverage or finiteness broken")
    drift = float((means - official).abs().max())
    path = OUT / "anchors_e45_step4.pt"
    torch.save(dict(initial={"mean": means}), path)
    receipt = dict(
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        anchor_observations=str(ANCHOR), official_reference=str(TARGETS),
        max_abs_vs_official_035=drift,
        output=str(path), output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        purpose="E52 start-state anchor identity (cloth gate origin = E45 step4, not 035)",
        determinism_note="Trainer recompute must match within the existing 1e-5 load gate",
        at_utc=__import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    (OUT / "anchors_e45_step4.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in ("max_abs_vs_official_035", "output_sha256")}))


if __name__ == "__main__":
    main()
