"""Replay a captured rollout to isolate pre-update likelihood inconsistency."""

import json
from pathlib import Path

from omegaconf import OmegaConf
import torch

from gear_sonic.trl.utils.common import custom_instantiate
from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.corrected_trial import OFFICIAL
from script.a3.evaluation_common import checkpoint_metadata, preflight
from script.a3.fullchain_support import ROOT
from script.a3.gradient_probe import parameter_digest


def compare(mean, old_mean, std, old_std, actions, old_logprobs):
    delta = mean - old_mean
    logprobs = torch.distributions.Normal(mean, std).log_prob(actions).sum(-1)
    ratio = (logprobs - old_logprobs).exp()
    if not torch.isfinite(ratio).all() or not torch.isfinite(delta).all():
        raise ValueError("Nonfinite replay")
    return dict(action_rmse=delta.square().mean().sqrt().item(), action_max_abs=delta.abs().max().item(),
                changed_frames_over_1e_4=(delta.abs().amax(-1) > 1e-4).sum().item(),
                ratio_mean=ratio.mean().item(), ratio_min=ratio.min().item(), ratio_max=ratio.max().item(),
                max_abs_ratio_minus_one=(ratio-1).abs().max().item())


def main():
    torch.set_num_threads(1)
    output = ROOT / "data/experiments/a3_likelihood_20261010_E09"
    output.mkdir(parents=True, exist_ok=False)
    preflight()
    source = ROOT / "data/experiments/a3_gradient_20261010_E08/rollout.pt"
    record = json.loads(source.with_name("gradient_diagnostic.json").read_text())
    if sha256(source) != record["rollout_sha256"]:
        raise ValueError("Captured rollout changed")
    rollout = torch.load(source, map_location="cpu", weights_only=False)
    config = OmegaConf.load(OFFICIAL.with_name("model_config.yaml"))
    actor = custom_instantiate(config.algo_config.actor, env_config=config.env_config,
                               algo_config=config.algo_config, _resolve=False)
    payload = torch.load(OFFICIAL, map_location="cpu", weights_only=False)
    actor.load_state_dict(payload["policy_state_dict"], strict=True)
    actor.get_std
    before = parameter_digest(list(actor.named_parameters()))
    rows, means = [], {}
    old_mean, old_std = rollout["old_mu_batch"], rollout["old_sigma_batch"]
    actions, logprobs = rollout["actions"], rollout["logprobs"]
    if rollout["padding_mask"].any():
        raise ValueError("This replay currently requires no padded transitions")
    rows.append(dict(label="stored_distribution_recomputed_cpu",
                     **compare(old_mean, old_mean, old_std, old_std, actions, logprobs)))
    with torch.no_grad():
        for device in ("cpu", "cuda"):
            actor.to(device)
            obs = {key: value.to(device) for key, value in rollout["all_obs_dict"].items()}
            precision_modes = (False,) if device == "cpu" else (True, False)
            for tf32 in precision_modes:
                torch.backends.cuda.matmul.allow_tf32 = tf32
                torch.backends.cudnn.allow_tf32 = tf32
                for training, auxiliary, frames in ((False, False, False), (True, True, False),
                                                    (False, False, True)):
                    actor.train(training)
                    if frames:
                        outputs = [actor.forward({key: value[:, step:step+1] for key, value in obs.items()},
                                                 is_training=auxiliary)
                                   for step in range(actions.shape[1])]
                        mean = torch.cat(outputs, dim=1).cpu()
                    else:
                        mean = actor.forward(obs, is_training=auxiliary).cpu()
                    label = f"{device}_tf32{tf32}_train{training}_aux{auxiliary}_frames{frames}"
                    means[label] = mean
                    rows.append(dict(label=label, **compare(mean, old_mean, actor.get_std.cpu(),
                                                           old_std, actions, logprobs)))
        actor.cpu()
        after = parameter_digest(list(actor.named_parameters()))
        if after != before:
            raise ValueError("Replay changed actor parameters")
    capture = output / "action_means.pt"
    torch.save(means, capture)
    result = dict(status="complete", optimizer_steps=0, rollout_sha256=sha256(source),
                  checkpoint_sha256=checkpoint_metadata(OFFICIAL)["sha256"],
                  parameters_before_sha256=before, parameters_after_sha256=after,
                  code_sha256=sha256(Path(__file__)), replay_count=len(rows), frames=actions.shape[0]*actions.shape[1],
                  rows=rows, means_sha256=sha256(capture), backup_status="LOCAL_ONLY",
                  scope="CAPTURED_INITIAL_ROLLOUT_LIKELIHOOD_DIAGNOSTIC_NOT_POLICY_IMPROVEMENT")
    atomic_json(output / "likelihood_diagnostic.json", result)
    atomic_json(ROOT / "data/manifests/a3_likelihood_20261010.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
