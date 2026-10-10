"""One fresh rollout, four gradient probes, and zero optimizer updates."""

import hashlib
import os
from pathlib import Path

import torch

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.verified_finetune import VerifiedFineTuneTrainer


def summary(tensor):
    value = tensor.detach().float()
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite diagnostic tensor")
    return dict(shape=list(value.shape), mean=value.mean().item(),
                std=value.std(unbiased=False).item(), min=value.min().item(),
                max=value.max().item())


def parameter_digest(named):
    digest = hashlib.sha256()
    for name, parameter in named:
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def group(name):
    if name.startswith("value_model."):
        return "critic"
    if name.endswith(".std") or name == "policy.std":
        return "actor_std"
    if "encoder" in name:
        return "actor_encoder"
    if "decoder" in name:
        return "actor_decoder"
    return "actor_other"


def gradient_geometry(named, gradients):
    rows = {}
    for bucket in sorted({group(name) for name, _ in named}):
        indices = [i for i, (name, _) in enumerate(named) if group(name) == bucket]
        norms = {}
        for term, values in gradients.items():
            squares = sum(values[i].double().square().sum().item()
                          for i in indices if values[i] is not None)
            norms[term] = squares ** 0.5
        cosines = {}
        for left, right in (("policy", "aux"), ("policy", "entropy"), ("value", "policy")):
            dot = sum((gradients[left][i].double() * gradients[right][i].double()).sum().item()
                      for i in indices if gradients[left][i] is not None and gradients[right][i] is not None)
            denominator = norms[left] * norms[right]
            cosines[left + "_vs_" + right] = dot / denominator if denominator else None
        rows[bucket] = dict(parameters=sum(named[i][1].numel() for i in indices),
                            gradient_norms=norms, gradient_cosines=cosines)
    return rows


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(item) for item in value]
    return value


class GradientProbeTrainer(VerifiedFineTuneTrainer):
    def _before_optimizer_step(self, *args, **kwargs):
        raise RuntimeError("Gradient probe forbids optimizer.step")

    def load_checkpoint(self, *args, **kwargs):
        result = super().load_checkpoint(*args, **kwargs)
        named = list(self.accelerator.unwrap_model(self.model).named_parameters())
        self._initial_digest = parameter_digest(named)
        self._loaded_hashes = {name: parameter_digest([(name, parameter)]) for name, parameter in named}
        self._loaded_std = self.accelerator.unwrap_model(self.model).policy.std.detach().clone()
        return result

    def _get_rollout_data(self, *args, **kwargs):
        self._probe_rollout = super()._get_rollout_data(*args, **kwargs)
        return self._probe_rollout

    def _compute_loss(self, forward_results, mb_rollout_data):
        if self.state.global_step != 0 or self._optimizer_step_ordinal != 0 or self.use_symmetry:
            raise ValueError("Probe requires an untouched nonsymmetry initial rollout")
        model = self.accelerator.unwrap_model(self.model)
        all_named = list(model.named_parameters())
        output = Path(os.environ["YUANQI_PROBE_OUTPUT"])
        output.mkdir(parents=True, exist_ok=True)
        changed = [name for name, parameter in all_named
                   if parameter_digest([(name, parameter)]) != self._loaded_hashes[name]]
        clamped = self._loaded_std.clamp(model.policy.algo_config.std_clamp_min,
                                        model.policy.algo_config.std_clamp_max)
        atomic_json(output / "pre_gradient_parameter_changes.json", dict(changed=changed,
                    loaded_sha256=self._initial_digest, before_gradient_sha256=parameter_digest(all_named),
                    std_clamp_expected=torch.equal(clamped, model.policy.std)))
        if changed and (changed != ["policy.std"] or not torch.equal(clamped, model.policy.std)):
            raise ValueError("Unexpected parameter changes before optimization; receipt preserved")
        before_gradient_digest = parameter_digest(all_named)
        named = [(name, parameter) for name, parameter in all_named if parameter.requires_grad]
        parameters = [parameter for _, parameter in named]
        first = mb_rollout_data["micro_batch_inds"].tolist()
        remaining = [i for i in range(self.args.local_batch_size) if i not in first]
        size = len(first)
        batches = [first] + [remaining[i:i + size] for i in range(0, len(remaining), size)]
        records = []
        for index, indices in enumerate(batches):
            if index:
                mb_rollout_data = self._get_mb_rollout_data(
                    self._probe_rollout, torch.tensor(indices, device=self.accelerator.device))
                forward_results = self._forward_model(self.model, mb_rollout_data)
            losses = super()._compute_loss(forward_results, mb_rollout_data)
            ppo = losses["ppo_loss_dict"]
            coefficient = self.config.get("ppo_loss_coef", 1.0)
            terms = dict(policy=ppo["pg_loss"] * coefficient,
                         value=ppo["vf_loss"] * self.args.vf_coef * coefficient,
                         entropy=ppo["entropy_loss"] * self.entropy_coef * coefficient,
                         aux=losses["aux_loss_dict"]["total_aux_loss"])
            if not torch.allclose(sum(terms.values()), losses["loss"], atol=1e-6, rtol=1e-5):
                raise ValueError("Loss decomposition does not reconstruct the actual objective")
            gradients = {}
            for term, loss in terms.items():
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                gradients[term] = (torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                                   if loss.requires_grad else [None] * len(parameters))
                if any(value is not None and not torch.isfinite(value).all() for value in gradients[term]):
                    raise ValueError("Nonfinite gradient")
            values, returns = mb_rollout_data["mb_values"], mb_rollout_data["mb_return"]
            mask = ~mb_rollout_data["mb_padding_mask_p1"]
            v, r = values[mask], returns[mask]
            variance = r.var(unbiased=False)
            explained = (1 - (r - v).var(unbiased=False) / variance).item() if variance > 0 else None
            aux = losses["aux_loss_dict"]
            records.append(dict(batch=index, environment_indices=indices,
                losses={key: value.item() for key, value in terms.items()},
                aux_raw={key: value.item() for key, value in aux["aux_losses_dict"].items()},
                aux_coefficients={key: float(value) for key, value in aux["aux_loss_coef"].items()},
                gradient_geometry=gradient_geometry(named, gradients),
                initial_value_explained_variance=explained,
                raw_advantage=summary(r - v), normalized_advantage=summary(mb_rollout_data["mb_advantage"]),
                returns=summary(r), initial_values=summary(v), ratio=summary(ppo["ratio"]),
                policy_result_keys=sorted(forward_results["policy_results"])))
            del gradients, losses, forward_results
        capture = output / "rollout.pt"
        torch.save(cpu_tree(self._probe_rollout), capture)
        final_digest = parameter_digest(all_named)
        if final_digest != before_gradient_digest or self._optimizer_step_ordinal != 0 or self.optimizer.state:
            raise ValueError("Probe changed parameters or optimizer state")
        result = dict(status="complete", optimizer_steps=0, global_step=0,
            parameters_before_sha256=before_gradient_digest, parameters_after_sha256=final_digest,
            loaded_parameters_sha256=self._initial_digest, rollout_changed_parameters=changed,
            rollout_sha256=sha256(capture), rollout_bytes=capture.stat().st_size,
            batches=records, parameter_groups={name: group(name) for name, _ in named},
            scope="FRESH_INITIAL_ROLLOUT_NOT_HISTORICAL_CAUSAL_PROOF_OR_POLICY_ACCEPTANCE",
            backup_status="LOCAL_ONLY")
        atomic_json(output / "gradient_diagnostic.json", result)
        print("[YUANQI GradientProbe] complete with zero optimizer steps", flush=True)
        raise SystemExit(0)
