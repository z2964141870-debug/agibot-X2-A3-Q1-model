#!/usr/bin/env python3
"""One-step policy-gradient lesson; no PPO, robot dynamics, or hardware.

Observation: one fixed situation, represented by [1].
Actions: move a toy angle by -1 or +1 degree, then end the episode.
Reward: reduction of distance to the target angle.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(7)
    torch.set_num_threads(1)
    dtype = torch.float64
    batch_size = 64
    learning_rate = 0.1

    # Two trainable weights, initially giving each action probability 0.5.
    policy = nn.Linear(1, 2, bias=False, dtype=dtype)
    nn.init.zeros_(policy.weight)
    optimizer = torch.optim.SGD(policy.parameters(), lr=learning_rate)
    observation = torch.ones(batch_size, 1, dtype=dtype)
    initial_weights = policy.weight.detach().clone()

    def snapshot():
        with torch.no_grad():
            probs = policy(observation[:1]).softmax(dim=-1)[0]
            return {
                "weights": policy.weight.flatten().tolist(),
                "max_abs_change_from_initial": (
                    policy.weight - initial_weights
                ).abs().max().item(),
                "probability_of_plus_1_degree": probs[1].item(),
            }

    before = snapshot()

    # Collect 64 independent one-step episodes. Scoring doesn't update weights.
    distribution = Categorical(logits=policy(observation))
    actions = distribution.sample()
    angle, target = 10.0, 30.0
    possible_steps = torch.tensor([-1.0, 1.0], dtype=dtype)
    next_angle = angle + possible_steps[actions]
    rewards = abs(target - angle) - (target - next_angle).abs()
    after_reward = snapshot()

    # In a one-step episode return == reward. This is REINFORCE, not PPO.
    # The environment reward is held fixed; gradients go through log probability.
    loss = -(distribution.log_prob(actions) * rewards.detach()).mean()
    optimizer.zero_grad()
    loss.backward()
    after_backward = snapshot()
    after_backward["gradient"] = policy.weight.grad.flatten().tolist()

    optimizer.step()
    after_step = snapshot()

    # Exact evaluation over both actions: no sampling noise, no optimizer call.
    weights_before_evaluation = policy.weight.detach().clone()
    with torch.no_grad():
        possible_rewards = abs(target - angle) - (
            target - (angle + possible_steps)
        ).abs()
        evaluation_probs = policy(observation[:1]).softmax(dim=-1)[0]
        expected_reward = (evaluation_probs * possible_rewards).sum().item()
    evaluation_change = (
        policy.weight.detach() - weights_before_evaluation
    ).abs().max().item()

    checks = {
        "reward_computation_preserves_weights": (
            after_reward["max_abs_change_from_initial"] == 0.0
        ),
        "backward_preserves_weights": (
            after_backward["max_abs_change_from_initial"] == 0.0
        ),
        "backward_produces_nonzero_gradient": (
            policy.weight.grad.abs().max().item() > 0.0
        ),
        "optimizer_changes_weights": (
            after_step["max_abs_change_from_initial"] > 0.0
        ),
        "rewarded_action_more_likely_in_this_example": (
            after_step["probability_of_plus_1_degree"]
            > before["probability_of_plus_1_degree"]
        ),
        "evaluation_preserves_weights": evaluation_change == 0.0,
    }
    result = {
        "experiment": "lesson01_reward_vs_update",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "algorithm": "one-step REINFORCE; not PPO or robot dynamics",
        "seed": 7,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "toy_angle_degrees": angle,
        "toy_target_degrees": target,
        "action_counts_minus_plus": torch.bincount(actions, minlength=2).tolist(),
        "reward_per_action_minus_plus": possible_rewards.tolist(),
        "sample_mean_reward_before_update": rewards.mean().item(),
        "before": before,
        "after_reward": after_reward,
        "after_backward": after_backward,
        "after_optimizer_step": after_step,
        "evaluation": {
            "method": "exact expectation over the two actions at the same start",
            "expected_reward_before_update": 0.0,
            "expected_reward_after_update": expected_reward,
            "max_abs_weight_change_during_evaluation": evaluation_change,
        },
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(checks.values()):
        raise SystemExit("FAIL: inspect the recorded checks")
    print("PASS: all six teaching checks passed")


if __name__ == "__main__":
    main()
