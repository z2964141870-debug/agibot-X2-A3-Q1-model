"""Probability replay on the identical runtime actor and observation metadata."""

import os
from pathlib import Path

from omegaconf import OmegaConf
import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.gradient_probe import GradientProbeTrainer, parameter_digest, cpu_tree
from script.a3.likelihood_probe import compare


class RuntimeLikelihoodTrainer(GradientProbeTrainer):
    def _compute_loss(self, forward_results, mb_rollout_data):
        if self.state.global_step or self._optimizer_step_ordinal or self.optimizer.state:
            raise ValueError("Probe must not update parameters")
        model = self.accelerator.unwrap_model(self.model)
        policy = model.policy
        before = parameter_digest(list(policy.named_parameters()))
        rollout = self._probe_rollout
        if rollout["padding_mask"].any():
            raise ValueError("Unexpected padded transitions")
        source = cpu_tree(rollout)
        old_mean, old_std = source["old_mu_batch"], source["old_sigma_batch"]
        actions, logprobs = source["actions"], source["logprobs"]
        output = Path(os.environ["YUANQI_PROBE_OUTPUT"])
        rows = [dict(label="stored_distribution_recomputed",
                     **compare(old_mean, old_mean, old_std, old_std, actions, logprobs))]
        indices = mb_rollout_data["micro_batch_inds"].cpu()
        rows.append(dict(label="actual_first_minibatch",
            **compare(forward_results["policy_results"]["action_mean"].detach().cpu(),
                      old_mean[indices], forward_results["policy_results"]["action_std"].detach().cpu(),
                      old_std[indices], actions[indices], logprobs[indices])))
        forward_results = None
        means = {}
        with torch.no_grad():
            for device in ("cuda", "cpu"):
                policy.to(device)
                obs = {key: value.to(device) for key, value in source["all_obs_dict"].items()}
                modes = (True, False) if device == "cuda" else (False,)
                for tf32 in modes:
                    torch.backends.cuda.matmul.allow_tf32 = tf32
                    torch.backends.cudnn.allow_tf32 = tf32
                    for auxiliary, frames in ((True, False), (False, False), (False, True)):
                        policy.train(auxiliary)
                        if frames:
                            values = [policy.forward({key: value[:, i:i+1] for key, value in obs.items()},
                                      is_training=auxiliary) for i in range(actions.shape[1])]
                            mean = torch.cat(values, dim=1).cpu()
                        else:
                            mean = policy.forward(obs, is_training=auxiliary).cpu()
                        label = f"runtime_{device}_tf32{tf32}_aux{auxiliary}_frames{frames}"
                        means[label] = mean
                        rows.append(dict(label=label, **compare(mean, old_mean, policy.get_std.cpu(),
                                                               old_std, actions, logprobs)))
            after = parameter_digest(list(policy.named_parameters()))
        if before != after:
            raise ValueError("Probability replay changed actor parameters")
        capture = output / "runtime_capture.pt"
        torch.save(dict(rollout=source, means=means), capture)
        metadata = dict(env_config=OmegaConf.to_container(policy.env_config, resolve=True),
                        algo_config=OmegaConf.to_container(policy.algo_config, resolve=True))
        atomic_json(output / "runtime_model_config.json", metadata)
        result = dict(status="complete", optimizer_steps=0, frames=actions.numel()//29,
            rows=rows, parameters_before_sha256=before, parameters_after_sha256=after,
            capture_sha256=sha256(capture), capture_bytes=capture.stat().st_size,
            runtime_metadata_sha256=sha256(output / "runtime_model_config.json"),
            code_sha256=sha256(Path(__file__)), backup_status="LOCAL_ONLY",
            scope="SAME_RUNTIME_POLICY_FIRST_ROLLOUT_NOT_POLICY_IMPROVEMENT")
        atomic_json(output / "likelihood_diagnostic.json", result)
        print("[YUANQI RuntimeLikelihood] " + str(rows), flush=True)
        raise SystemExit(0)
