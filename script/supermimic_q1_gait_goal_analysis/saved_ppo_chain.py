"""Inspect saved PPO tensors, Adam counters and logs without policy execution."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "project_tools"))
from prepare_q1_gait_review import identity, write_csv

ACTOR = {
    "head.net.0.weight": (256, 1420), "head.net.0.bias": (256,),
    "head.net.2.weight": (128, 256), "head.net.2.bias": (128,),
    "head.net.4.weight": (14, 128), "head.net.4.bias": (14,),
    "distribution.std_param": (14,),
}
CRITIC = {
    "mlp.0.weight": (256, 1503), "mlp.0.bias": (256,),
    "mlp.2.weight": (128, 256), "mlp.2.bias": (128,),
    "mlp.4.weight": (1, 128), "mlp.4.bias": (1,),
}
SCRIPT_SOURCES = (
    "supermimic_q1_gait_goal/train.py", "supermimic_q1_gait_goal/trial_support.py",
    "supermimic_q1_gait_goal/reward_candidates.py", "supermimic_q1_gait_goal_review/train.py",
    "supermimic_q1_contact_comp/contact_adapter.py", "supermimic_q1_gait_comp/gait_adapter.py",
    "supermimic_q1_gait_comp/rewards.py", "supermimic_q1_gait_comp/overnight.py",
    "supermimic_q1_balance/balance_adapter.py", "supermimic_q1_balance/control.py",
    "supermimic_q1_ppo/adapter.py",
)
RSL_SOURCES = ("algorithms/ppo.py", "modules/distribution.py", "storage/rollout_storage.py")
REWARD_PARTS = ("original_reward", "stance_bonus", "swing_bonus", "stationary_bonus",
                "smooth_cost", "residual_cost")
CHECKPOINT_KEYS = ("format", "actor", "critic", "optimizer", "source_base", "training_seconds",
                   "iteration", "upper_reference_alpha", "user_accepted_checkpoint",
                   "maximum_training_seconds", "code_commit")


def counts(iteration):
    if type(iteration) is not int or iteration < 0:
        raise ValueError("Expected a nonnegative integer iteration")
    return 20 * max(iteration - 10, 0), 20 * iteration


def finite_tensor(value, shape):
    if (not torch.is_tensor(value) or value.device.type != "cpu" or tuple(value.shape) != shape
            or not value.is_floating_point() or not bool(torch.isfinite(value).all())):
        raise ValueError("Expected a finite CPU floating tensor with the exact contract shape")


def audit_checkpoint(saved, initial, label):
    if tuple(saved) != CHECKPOINT_KEYS or saved["format"] != "Q1_GAIT_GOAL01_PPO_V1":
        raise ValueError("Unexpected checkpoint format or field inventory")
    expected_actor, expected_critic = counts(saved["iteration"])
    optimizer = saved["optimizer"]
    groups = optimizer["param_groups"]
    if (len(groups) != 1 or groups[0]["params"] != list(range(13))
            or not 1e-5 <= groups[0]["lr"] <= 5e-5 or groups[0]["weight_decay"] != 0
            or groups[0]["amsgrad"] or groups[0]["maximize"]
            or tuple(groups[0]["betas"]) != (.9, .999) or groups[0]["eps"] != 1e-8):
        raise ValueError("Unexpected frozen Adam group contract")
    if tuple(saved["actor"]) != tuple(ACTOR) or tuple(saved["critic"]) != tuple(CRITIC):
        raise ValueError("Cannot map serialized optimizer IDs to the frozen parameter order")
    expected_ids = {i for i in range(13) if i != 6 and (expected_actor if i < 7 else expected_critic)}
    if set(optimizer["state"]) != expected_ids:
        raise ValueError("Adam state inventory does not match critic-only warm-up and fixed std")
    rows = []
    for kind, shapes in (("actor", ACTOR), ("critic", CRITIC)):
        for name, shape in shapes.items():
            parameter_id = len(rows)
            value, origin = saved[kind][name], initial[kind][name]
            finite_tensor(value, shape)
            finite_tensor(origin, shape)
            delta = value.double() - origin.double()
            initial_norm = float(origin.double().norm())
            expected = 0 if parameter_id == 6 else expected_actor if kind == "actor" else expected_critic
            state = optimizer["state"].get(parameter_id)
            moment_rms = variance_rms = None
            step = 0
            if state is not None:
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("Unexpected Adam state fields")
                finite_tensor(state["step"], ())
                step = float(state["step"])
                for key in ("exp_avg", "exp_avg_sq"):
                    finite_tensor(state[key], shape)
                if bool((state["exp_avg_sq"] < 0).any()):
                    raise ValueError("Negative Adam second moment")
                moment_rms = float(state["exp_avg"].double().square().mean().sqrt())
                variance_rms = float(state["exp_avg_sq"].double().square().mean().sqrt())
            if step != expected:
                raise ValueError("Saved Adam step disagrees with actual iteration/warm-up contract")
            if parameter_id == 6 and not torch.equal(value, torch.full_like(value, .1)):
                raise ValueError("Frozen latent standard deviation changed")
            if expected == 0 and bool((delta != 0).any()):
                raise ValueError("An unupdated parameter differs from initial")
            rows.append(dict(label=label, iteration=saved["iteration"], kind=kind, name=name,
                optimizer_parameter_id=parameter_id, shape="x".join(str(v) for v in shape), elements=value.numel(),
                parameter_l2=float(value.double().norm()), initial_l2=initial_norm,
                changed_elements=int(torch.count_nonzero(delta)), delta_from_initial_l2=float(delta.norm()),
                relative_delta_l2=float(delta.norm()) / initial_norm if initial_norm else None,
                delta_abs_max=float(delta.abs().max()), adam_step=step, expected_adam_step=expected,
                adam_first_moment_rms=moment_rms, adam_second_moment_rms=variance_rms))
    return rows


def audit_logs(records):
    if not records:
        raise ValueError("Training log is empty")
    rows, last_seconds = [], 0.
    expected_entropy = 14 * (.5 * math.log(2 * math.pi * math.e) + math.log(.1))
    for iteration, record in enumerate(records, 1):
        actor, critic = counts(iteration)
        numeric = [v for v in record.values() if isinstance(v, (int, float))]
        if (record["step"] != iteration or record["transitions"] != 4096 * 24
                or record["actor_optimizer_steps"] != actor or record["critic_optimizer_steps"] != critic
                or record["wm_optimizer_steps"] != 0 or not all(math.isfinite(v) for v in numeric)
                or not last_seconds < record["training_seconds"] < 1200
                or not 1e-5 <= record["learning_rate"] <= 5e-5
                or abs(record["ppo_entropy"] - expected_entropy) > 1e-5):
            raise ValueError("Training log violates counters, clock, finite values or fixed distribution")
        transitions = record["transitions"]
        mean_alpha = record["alpha_sum"] / transitions
        alpha0_fraction = 1 - mean_alpha / .5
        if not 0 <= alpha0_fraction <= 1 or record["residual_squared_sum"] < 0:
            raise ValueError("Invalid binary cap mixture or residual sum")
        if iteration <= 10 and any(record[key] != 0 for key in
                ("policy_update_kl_mean", "policy_update_kl_p99", "latent_mean_rms")):
            raise ValueError("Critic-only warm-up shows a nonzero mean-policy update")
        summed = sum(record[key + "_sum"] * (-1 if key.endswith("cost") else 1) for key in REWARD_PARTS)
        discrepancy = record["reward_sum"] - summed
        if not math.isclose(record["reward_sum"], summed, rel_tol=1e-6, abs_tol=1e-3):
            raise ValueError("Reward component sums do not reconstruct the logged total")
        rows.append(dict(iteration=iteration, training_seconds=record["training_seconds"],
            actor_updates=actor, critic_updates=critic, transitions=transitions,
            **{key + "_per_transition": record[key + "_sum"] / transitions for key in REWARD_PARTS},
            reward_per_transition=record["reward_sum"] / transitions,
            component_sum_discrepancy=discrepancy,
            sampled_residual_rms_rad=math.sqrt(record["residual_squared_sum"] / (transitions * 14)),
            mean_upper_alpha=mean_alpha, alpha0_fraction_under_binary_cap_contract=alpha0_fraction,
            artificial_timeouts=record["artificial_timeouts"], hard_terminations=record["hard_terminations"],
            completed_episodes=record["completed_episodes"], learning_rate=record["learning_rate"],
            first_timestep_policy_kl_mean=record["policy_update_kl_mean"],
            first_timestep_policy_kl_p99=record["policy_update_kl_p99"],
            first_timestep_latent_mean_rms=record["latent_mean_rms"],
            ppo_value_loss=record["ppo_value"], ppo_surrogate_loss=record["ppo_surrogate"],
            latent_gaussian_entropy=record["ppo_entropy"]))
        last_seconds = record["training_seconds"]
    return rows


def zero_mean_residual_reference(std, radius):
    if not math.isfinite(std) or not math.isfinite(radius) or std < 0 or radius <= 0:
        raise ValueError("Expected finite nonnegative std and positive radius")
    nodes, weights = np.polynomial.hermite.hermgauss(80)
    residual = radius * np.tanh(math.sqrt(2) * std * nodes)
    return float(math.sqrt(np.dot(weights, residual ** 2) / math.sqrt(math.pi)))


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Explicitly hide CUDA for this saved-tensor-only analysis")
    torch.set_num_threads(2)
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a prior saved analysis")
    tracked = {}

    def remember(path, expected=None):
        current = identity(path)
        if expected is not None and any(current[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError(f"Historical source identity differs: {path}")
        tracked[str(path)] = current
        return current

    delivery_path = args.root / "data/manifests/supermimic_q1_gait_goal_ppo_delivery_20261030.json"
    remember(delivery_path)
    delivery = json.loads(delivery_path.read_text())
    history = {str(args.root / item["relative_path"]): item
               for item in delivery["small_artifacts_three_way"]["verified_files"]}
    plan_path, selection_path, log_path, receipt_path = [args.ppo / name for name in
        ("plan.json", "run/checkpoint_selection.json", "run/training_evidence.jsonl", "run/receipt.json")]
    for path in (plan_path, selection_path, log_path, receipt_path):
        remember(path, history[str(path)])
    plan, selection, receipt = [json.loads(path.read_text()) for path in (plan_path, selection_path, receipt_path)]
    frozen = {item["path"]: item for item in plan["source"]}
    package = args.root / "data/environments/supermimic_q1_hp_20261026/lib/python3.11/site-packages/rsl_rl"
    sources = [args.root / "script" / name for name in SCRIPT_SOURCES] + [package / name for name in RSL_SOURCES]
    frozen_sources = [remember(path, frozen[str(path)]) for path in sources]
    remember(Path(__file__).resolve())
    remember(HERE.parent / "project_tools/prepare_q1_gait_review.py")
    if (plan["line"] != "ppo" or plan["automatic_retry"] or plan["automatic_extension"]
            or selection["user_accepted_checkpoint"] is not None or selection["goal_complete"]
            or receipt["outcome"] != "physical_regression"):
        raise ValueError("Original independent trial identity/decision differs")
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    log_rows = audit_logs(records)
    parameter_rows, checkpoints, initial = [], [], None
    if [item["label"] for item in selection["candidates"]] != ["initial", "stage_01", "stage_02"]:
        raise ValueError("Expected the original three saved candidates")
    for candidate in selection["candidates"]:
        label, historical = candidate["label"], candidate["checkpoint"]
        path = args.ppo / "run" / (label + ".pt")
        if historical["path"] != str(path):
            raise ValueError("Checkpoint is outside the original candidate path")
        checkpoint_identity = remember(path, historical)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["code_commit"] != plan["code_commit"] or saved["user_accepted_checkpoint"] is not None
                or saved["upper_reference_alpha"] != .5 or saved["maximum_training_seconds"] != 1200.
                or saved["training_seconds"] != candidate["training_seconds"]):
            raise ValueError("Checkpoint metadata differs from the frozen independent trial")
        if initial is None:
            if saved["iteration"] != 0 or saved["training_seconds"] != 0:
                raise ValueError("Initial checkpoint is not pretraining")
            initial = saved
        elif (saved["source_base"] != initial["source_base"]
                or saved["training_seconds"] != records[saved["iteration"] - 1]["training_seconds"]):
            raise ValueError("Teacher identity or checkpoint/log clock differs")
        parameter_rows.extend(audit_checkpoint(saved, initial, label))
        checkpoints.append(dict(label=label, identity=checkpoint_identity, iteration=saved["iteration"],
            training_seconds=saved["training_seconds"], actor_updates=counts(saved["iteration"])[0],
            critic_updates=counts(saved["iteration"])[1], checkpoint_fields=list(saved),
            optimizer_group=saved["optimizer"]["param_groups"][0], source_base=saved["source_base"]))
    last = log_rows[-1]
    if (last["iteration"] != checkpoints[-1]["iteration"] or last["actor_updates"] != receipt["actor_optimizer_steps"]
            or last["critic_updates"] != receipt["critic_optimizer_steps"]
            or last["training_seconds"] != receipt["training_seconds"]):
        raise ValueError("Final checkpoint, logs and original training receipt disagree")
    if list(tracked.values()) != [identity(Path(path)) for path in tracked]:
        raise ValueError("Input sources changed during saved-only analysis")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "checkpoint_parameters.csv", parameter_rows)
    write_csv(args.output / "training_iterations.csv", log_rows)
    summary = dict(status="SAVED_PPO_LEARNING_CHAIN_REVIEW_READY", captured_utc=datetime.now(timezone.utc).isoformat(),
        source_base_commit=args.source_base_commit, sources=list(tracked.values()),
        historically_frozen_learning_sources=frozen_sources, checkpoints=checkpoints,
        checkpoint_parameter_rows=len(parameter_rows), training_iterations=len(log_rows),
        transitions=sum(row["transitions"] for row in log_rows), first=log_rows[0], last=last,
        log_field_inventory=sorted(records[0]),
        warmup=dict(iterations=10, actor_updates=0, critic_updates=200,
            sampled_residual_rms_rad_range=[min(row["sampled_residual_rms_rad"] for row in log_rows[:10]),
                                            max(row["sampled_residual_rms_rad"] for row in log_rows[:10])]),
        log_ranges={key: [min(row[key] for row in log_rows), max(row[key] for row in log_rows)] for key in
            ("learning_rate", "mean_upper_alpha", "alpha0_fraction_under_binary_cap_contract",
             "sampled_residual_rms_rad", "latent_gaussian_entropy")},
        artificial_timeouts=sum(row["artificial_timeouts"] for row in log_rows),
        zero_mean_fixed_distribution_reference=dict(latent_std=.1, radius_rad=.15,
            expected_residual_rms_rad=zero_mean_residual_reference(.1, .15),
            method="80-node Gauss-Hermite quadrature of .15*tanh(N(0,.1^2)); no draws or policy calls"),
        verified_source_contract=dict(actor_features=1420, critic_features=1503, residual_dimensions=14,
            steps_per_environment=24, control_dt_s=.02, learning_epochs=5, minibatches=4,
            gamma=.99, gae_lambda=.95, clip_param=.2, max_grad_norm=1.,
            gradient_clipping="actor and critic separately", fixed_latent_std=.1, entropy_coefficient=0.,
            learning_rate_schedule="adaptive KL .01, pre-step hook clamps to [1e-5,5e-5]",
            advantage_normalization="whole rollout", artificial_timeout_bootstrap="V(actual terminal next observation)",
            training_upper_alpha="binary 0/.5 caps; reset draws 25% alpha0; head remains active in both",
            evaluation_upper_alpha=.5, evaluation_stochastic=False,
            log_kl_scope="4096 observations at the first timestep, before/after the full update"),
        missing_saved_training_channels=["per-transition training observations and latent actions",
            "per-transition reward parts/phase/contact", "values/returns/raw and normalized advantages",
            "minibatch ratios/clip fraction/adaptive KL/rates", "actor/critic gradients and clipping norms",
            "matched phase-conditioned reward/advantage/mean-policy gradient contributions"],
        new_integration_steps=0, policy_inference_calls=0, random_draws=0, optimizer_updates=0,
        goal_complete=False, user_accepted_checkpoint=None,
        limits=["Adam counters/moments and changed parameters prove updates, not useful physical compensation.",
            "The saved fixed-std parameter has no Adam state by design; the six head parameters do have state.",
            "First-timestep logged KL is not the adaptive per-minibatch KL or a clipping/gradient diagnostic.",
            "Reward/loss changes use evolving stochastic state and episode mixtures, not matched physical conditions.",
            "The zero-mean quadrature is a distribution-only reference, not measured exploration efficacy or a cause.",
            "Training mixture/noise differences are adaptation hypotheses, not isolated explanations of regression.",
            "24 collected steps use critic bootstrap; .48 seconds is not a hard planning horizon.",
            "No reconstructed critic quality/TD error/advantage attribution is possible without the missing training tensors.",
            "New physical measurements, policy inference or training require a new explicit decision."])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in
        ("status", "training_iterations", "transitions", "checkpoint_parameter_rows", "warmup", "last")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "ppo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-base-commit", required=True)
    run(parser.parse_args())
