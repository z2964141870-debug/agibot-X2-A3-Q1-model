"""CPU control-flow probes and read-only R05 evidence; no robot-policy updates."""

import ast
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from script.a3.checkpoint_store import atomic_json, sha256
from script.a3.fullchain_support import ROOT, VENDOR
from script.a3.regression_diagnostic import CHECKPOINTS, TRAINING


OUTPUT = ROOT / "data/experiments/a3_optimizer_20261010_E02"


def source_record(function):
    path = Path(inspect.getsourcefile(function))
    _, line = inspect.getsourcelines(function)
    return dict(path=str(path), line=line, sha256=sha256(path),
                qualified_name=function.__qualname__)


def legacy_resume_assignments(controller, checkpoint):
    path = ROOT / "script/a3/resumable_sonic.py"
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ResumableTrainer")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "load_checkpoint")
    statements = [n for n in method.body if ast.unparse(n).startswith((
        'self.optimizer.load_state_dict(checkpoint[',
        'self.args.learning_rate = float(self.optimizer.param_groups[0]'))]
    if len(statements) != 2:
        raise ValueError("Legacy resume LR statements changed; inspect before replay")
    code = compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), str(path), "exec")
    exec(code, {"self": controller, "checkpoint": checkpoint})
    return dict(path=str(path), sha256=sha256(path), lines=[n.lineno for n in statements],
                scope="ONLY_ACTUAL_OPTIMIZER_RESTORE_AND_ARGS_LR_ASSIGNMENTS_NOT_SIMULATOR_RESUME")


def make_optimizer(lrs):
    parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in lrs]
    optimizer = torch.optim.AdamW([dict(params=[p], lr=lr) for p, lr in zip(parameters, lrs)],
                                  weight_decay=0.0)
    return optimizer


def controller_for(optimizer, adaptive_lr, config):
    return SimpleNamespace(args=SimpleNamespace(learning_rate=adaptive_lr), optimizer=optimizer,
                           desired_kl=config["desired_kl"], adaptive_lr_min=config["adaptive_lr_min"],
                           adaptive_lr_max=config["adaptive_lr_max"])


def lr_probe(trainer, config, checkpoint):
    from transformers import get_scheduler
    optimizer = make_optimizer([config["actor_learning_rate"]] * 2)
    scheduler = get_scheduler("constant", optimizer=optimizer)
    controller = controller_for(optimizer, config["actor_learning_rate"], config)
    identical_gaussian_kl = float(torch.log(torch.tensor(1.0 + 1e-5)) * 29)
    # A synthetic KL sequence tests the control path, not historical R05 minibatches.
    sequence = [identical_gaussian_kl] + [0.03] * 19
    steps = []
    for index, kl in enumerate(sequence):
        trainer._adjust_learning_rate_based_on_kl(controller, torch.tensor(kl), optimizer)
        before = [group["lr"] for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["params"][0].grad = torch.ones_like(group["params"][0])
        optimizer.step()
        optimizer.zero_grad()
        steps.append(dict(index=index + 1, synthetic_kl=kl, optimizer_step_lrs=before,
                          args_learning_rate=controller.args.learning_rate))
    before_scheduler = [group["lr"] for group in optimizer.param_groups]
    scheduler.step()
    after_scheduler = [group["lr"] for group in optimizer.param_groups]
    if after_scheduler == before_scheduler:
        raise ValueError("Expected constant scheduler LR overwrite was not reproduced")
    resume_rows = []
    saved_lr = float(checkpoint["args"].learning_rate)
    saved_groups = [float(g["lr"]) for g in checkpoint["optimizer_state_dict"]["param_groups"]]
    # Preserve the real checkpoint LR discrepancy, using only scalar optimizer state.
    synthetic_checkpoint = dict(optimizer_state_dict=optimizer.state_dict())
    for group, rate in zip(synthetic_checkpoint["optimizer_state_dict"]["param_groups"], saved_groups):
        group["lr"] = rate
    for label, kl in (("neutral", 0.01), ("high", 0.03), ("low", 0.002)):
        continuous_optimizer = make_optimizer(saved_groups)
        continuous = controller_for(continuous_optimizer, saved_lr, config)
        resumed_optimizer = make_optimizer(saved_groups)
        resumed = controller_for(resumed_optimizer, saved_lr, config)
        provenance = legacy_resume_assignments(resumed, synthetic_checkpoint)
        restored_args = resumed.args.learning_rate
        trainer._adjust_learning_rate_based_on_kl(continuous, kl, continuous_optimizer)
        trainer._adjust_learning_rate_based_on_kl(resumed, kl, resumed_optimizer)
        resume_rows.append(dict(branch=label, synthetic_kl=kl, saved_args_lr=saved_lr,
                                saved_optimizer_lrs=saved_groups, legacy_restored_args_lr=restored_args,
                                uninterrupted_next_step_lrs=[g["lr"] for g in continuous_optimizer.param_groups],
                                legacy_resume_next_step_lrs=[g["lr"] for g in resumed_optimizer.param_groups]))
    return dict(steps=steps, before_scheduler_lrs=before_scheduler, after_scheduler_lrs=after_scheduler,
                args_after_scheduler=controller.args.learning_rate, resume_branches=resume_rows,
                legacy_resume_source=provenance, identical_gaussian_kl_with_vendor_epsilon=identical_gaussian_kl,
                scope="SYNTHETIC_KL_CPU_SCALAR_OPTIMIZERS_NOT_RECONSTRUCTED_HISTORICAL_STEP_LRS")


def auxiliary_probe(trainer):
    main_leaf = torch.tensor(3.0, requires_grad=True)
    auxiliary_leaf = torch.tensor(5.0, requires_grad=True)
    controller = SimpleNamespace(config={"compute_aux_loss": True}, compute_imgaug_bc_loss=False,
                                 _compute_ppo_loss=lambda *_: {"ppo_loss": main_leaf.square()})
    forward = {"policy_results": {"aux_losses": {"a3_fast_g1_latent": auxiliary_leaf.square()},
                                   "aux_loss_coef": {"a3_fast_g1_latent": 1.0}}}
    result = trainer._compute_loss(controller, forward, {})["loss"]
    main_grad, auxiliary_grad = torch.autograd.grad(result, (main_leaf, auxiliary_leaf), allow_unused=True)
    forward["policy_results"]["aux_losses"]["a3_fast_g1_latent"] = auxiliary_leaf.square() * 100
    changed = trainer._compute_loss(controller, forward, {})["loss"]
    if auxiliary_grad is not None or result.item() != changed.item():
        raise ValueError("Auxiliary omission changed; inspect pinned trainer")
    return dict(total_loss=result.item(), total_after_auxiliary_100x=changed.item(),
                main_leaf_gradient=main_grad.item(), auxiliary_leaf_gradient=None,
                flag_compute_aux_loss=True, config_coefficient=1.0,
                source=source_record(trainer._compute_loss),
                scope="ACTUAL_PINNED_LOSS_METHOD_WITH_SENTINEL_TENSORS_NOT_POLICY_GRADIENT_ATTRIBUTION")


def optimizer_groups_probe(trainer):
    from transformers import Trainer, TrainingArguments
    if trainer.create_optimizer is not Trainer.create_optimizer:
        raise ValueError("Optimizer construction is no longer inherited from Transformers")
    model = torch.nn.Module()
    model.policy = torch.nn.Linear(2, 1)
    model.value_model = torch.nn.Linear(2, 1)
    arguments = TrainingArguments(output_dir=str(OUTPUT / "unused_trainer_output"), use_cpu=True,
                                  learning_rate=2e-5, report_to=[], disable_tqdm=True)
    harness = Trainer(model=model, args=arguments)
    optimizer = harness.create_optimizer()
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = [dict(lr=g["lr"], weight_decay=g["weight_decay"],
                   parameters=[names[id(p)] for p in g["params"]]) for g in optimizer.param_groups]
    if not all(any(n.startswith("policy.") for n in g["parameters"]) and
               any(n.startswith("value_model.") for n in g["parameters"]) for g in groups):
        raise ValueError("Expected joint actor/value grouping not reproduced")
    return dict(groups=groups, source=source_record(trainer.create_optimizer),
                scope="ACTUAL_INHERITED_CONSTRUCTOR_ON_TINY_MODEL_NO_TRAIN_CALL")


def config_differences(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        rows = []
        for key in sorted(old.keys() | new.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in old or key not in new:
                rows.append(dict(path=path, official=old.get(key), r05=new.get(key)))
            else:
                rows.extend(config_differences(old[key], new[key], path))
        return rows
    return [] if old == new else [dict(path=prefix, official=old, r05=new)]


def scalar_evidence():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    directory = TRAINING / "R05_001_s0/tensorboard"
    events = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    events.Reload()
    tags = [tag for tag in events.Tags()["scalars"]
            if any(term in tag.lower() for term in ("loss", "kl", "clip", "grad", "noise", "lr", "ratio"))]
    return dict(tags=tags, scalars={tag: [dict(step=e.step, value=e.value) for e in events.Scalars(tag)]
                                   for tag in tags},
                sources=[dict(path=str(p), sha256=sha256(p)) for p in directory.glob("events*")])


def checkpoint_evidence():
    reference = ROOT / "data/experiments/a3_regression_20261010_E01"
    with np.load(reference / "fixed_input_outputs.npz", allow_pickle=False) as values:
        arrays = {key: values[key].copy() for key in values.files if key.endswith("_actions")}
    rows = []
    baseline_std = None
    for step, directory in CHECKPOINTS.items():
        path = TRAINING / directory / f"model_step_{step:06d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["policy_state_dict"]
        if "std" not in state:
            raise ValueError("Expected direct Gaussian std parameter")
        raw_std = state["std"].cpu().numpy().copy()
        std = np.clip(raw_std, 0.001, 0.5)
        if step == 0:
            baseline_std = std.copy()
        actions = arrays[f"{step}_actions"]
        delta = arrays["official_actions"] - actions
        exact_kl = (np.log(std / baseline_std) +
                    (baseline_std ** 2 + delta ** 2) / (2 * std ** 2) - 0.5).sum(axis=-1)
        rows.append(dict(step=step, checkpoint=str(path), sha256=sha256(path),
                         std_raw_min=float(raw_std.min()), std_raw_max=float(raw_std.max()),
                         std_effective_mean=float(std.mean()), clamp_changed_elements=int(np.sum(raw_std != std)),
                         gaussian_kl_official_to_current_on_fixed_inputs_mean=float(exact_kl.mean()),
                         gaussian_kl_official_to_current_on_fixed_inputs_max=float(exact_kl.max()),
                         args_learning_rate=float(payload["args"].learning_rate),
                         optimizer_groups=[dict(lr=g["lr"], weight_decay=g["weight_decay"],
                                                parameter_count=len(g["params"]))
                                           for g in payload["optimizer_state_dict"]["param_groups"]]))
        del payload
    return dict(rows=rows, input_sha256=json.loads((reference / "fixed_input_diagnostic.json").read_text())["input_sha256"],
                scope="100_IDENTICAL_OFFICIAL_WALK_INPUTS_NOT_R05_ROLLOUT_KL_OR_LOSS_REPLAY")


def main():
    torch.set_num_threads(1)
    sys.path.insert(0, str(VENDOR.resolve()))
    from gear_sonic.trl.trainer.ppo_trainer import TRLPPOTrainer
    OUTPUT.mkdir(parents=True, exist_ok=True)
    state_path = OUTPUT / "state.json"
    atomic_json(state_path, dict(status="running", training_allowed=False, device="cpu"))
    try:
        official_path = ROOT / "data/models/a3_official_035/checkpoints/035_step200000/config.yaml"
        trial_path = TRAINING / "R05_001_s0/config.yaml"
        official = yaml.safe_load(official_path.read_text())
        trial = yaml.safe_load(trial_path.read_text())
        config = trial["algo"]["config"]
        checkpoint = torch.load(TRAINING / "R05_001_s0/model_step_000002.pt", map_location="cpu", weights_only=False)
        previous = json.loads((ROOT / "data/experiments/a3_regression_20261010_E01/update_audit.json").read_text())
        source = Path(inspect.getsourcefile(TRLPPOTrainer))
        if sha256(source) != previous["source"]["sha256"]:
            raise ValueError("Pinned PPO source changed since E01")
        result = dict(scope="CPU_CONTROL_AND_READ_ONLY_EVIDENCE_NOT_CAUSAL_POLICY_ACCEPTANCE",
                      training_allowed=False, generated_utc=datetime.now(timezone.utc).isoformat(),
                      vendor_source=dict(path=str(source), sha256=sha256(source)),
                      config_sources=[dict(path=str(p), sha256=sha256(p)) for p in (official_path, trial_path)],
                      configuration_differences=config_differences(official, trial),
                      lr_control=lr_probe(TRLPPOTrainer, config, checkpoint),
                      auxiliary_loss=auxiliary_probe(TRLPPOTrainer),
                      optimizer_construction=optimizer_groups_probe(TRLPPOTrainer),
                      first_two_updates_scalars=scalar_evidence(), checkpoints=checkpoint_evidence(),
                      diagnostic_source=dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__))),
                      limitations=["Historical minibatch inputs, KLs and pre-step LRs were not saved",
                                   "Small CPU probes do not identify the cause of measured policy regression",
                                   "Simulator collection and robot-policy optimization were not run"])
        atomic_json(OUTPUT / "diagnostic.json", result)
        atomic_json(state_path, dict(status="complete", training_allowed=False,
                                    finished_utc=datetime.now(timezone.utc).isoformat()))
        atomic_json(ROOT / "data/manifests/a3_optimizer_20261010_E02.json", result)
        print(json.dumps({key: result[key] for key in ("lr_control", "auxiliary_loss", "optimizer_construction")}, indent=2))
    except Exception as error:
        atomic_json(state_path, dict(status="failed", error=str(error), training_allowed=False))
        raise


def finalize():
    receipt_path = OUTPUT / "contracts.json"
    receipt = json.loads(receipt_path.read_text())
    result = json.loads((OUTPUT / "diagnostic.json").read_text())
    if not receipt["successful"] or receipt["tests_run"] != 9:
        raise ValueError("Corrected trainer contracts did not pass")
    result["corrected_entry"] = dict(
        target="script.a3.verified_finetune.VerifiedFineTuneTrainer",
        sources=[dict(path=str(ROOT / path), sha256=sha256(ROOT / path)) for path in (
            "script/a3/verified_finetune.py", "script/a3/test_verified_finetune.py")],
        contracts=receipt, contracts_sha256=sha256(receipt_path),
        official_auxiliary_trainer=source_record(__import__(
            "gear_sonic.trl.trainer.ppo_trainer_aux_loss", fromlist=["TRLAuxLossPPOTrainer"]
        ).TRLAuxLossPPOTrainer._compute_loss),
        integration_status="CPU_CONTRACTS_PASSED_FULL_ISAAC_TRAINER_NOT_STARTED",
        training_started=False, policy_improvement_verified=False)
    log_root = ROOT / "logs/a3_optimizer_20261010_E02"
    result["logs"] = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha256(path))
                      for path in sorted(log_root.glob("*.log")) if path.name != "finalize.log"]
    result["diagnostic_source"] = dict(path=str(Path(__file__).resolve()), sha256=sha256(Path(__file__)))
    atomic_json(OUTPUT / "diagnostic.json", result)
    atomic_json(ROOT / "data/manifests/a3_optimizer_20261010_E02.json", result)
    print(json.dumps(result["corrected_entry"], indent=2))


if __name__ == "__main__":
    if sys.argv[1:] == ["--finalize"]:
        sys.path.insert(0, str(VENDOR.resolve()))
        finalize()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit("Usage: python -m script.a3.optimizer_diagnostic [--finalize]")
