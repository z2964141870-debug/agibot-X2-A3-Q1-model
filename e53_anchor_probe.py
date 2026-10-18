"""E53 anchor probe: precompute clean-035 policy anchor outputs (one GPU forward)."""

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

OPS = Path("/opt/data/private/yuanqi_ops")
OUT = OPS / "data/experiments/a3_zhy_rootchannel_20261019_E53"
SOURCE = Path("/opt/data/private/hechengzhu/ZHY/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt")
ANCHOR = Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_gradient_20261012_E11/rollout.pt")
TARGETS = Path("/opt/data/private/hechengzhu/ZHY/data/experiments/a3_zhy_modules_20261012_E12/module_outputs.pt")
CONFIG = Path("/opt/data/private/hechengzhu/ZHY/logs/a3_zhy_cloth_capture_20261013_E39/training/runtime_metadata.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    payload = torch.load(SOURCE, map_location="cpu", weights_only=False)
    config = OmegaConf.create(json.loads(CONFIG.read_text()))
    actor = custom_instantiate(config.algo_config.actor, env_config=config.env_config,
                               algo_config=config.algo_config, _resolve=False)
    actor.load_state_dict(payload["policy_state_dict"], strict=True)
    restrict_policy(actor)
    actor = actor.to("cuda").eval()
    rollout = torch.load(ANCHOR, map_location="cpu", weights_only=False)
    obs = {name: value.to("cuda") for name, value in rollout["all_obs_dict"].items()}
    del rollout
    with torch.no_grad():
        means = actor.forward(obs, None)["mean"] if isinstance(actor.forward(obs, None), dict) else actor.forward(obs, None)
    torch.save({"initial": {"mean": means.detach().cpu()}}, OUT / "anchors_035.pt")
    meta = dict(source=str(SOURCE), source_sha256=sha256(SOURCE),
                anchors_sha256=sha256(OUT / "anchors_035.pt"),
                mean_shape=list(means.shape))
    (OUT / "anchors_035.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta)[:300])


if __name__ == "__main__":
    main()
