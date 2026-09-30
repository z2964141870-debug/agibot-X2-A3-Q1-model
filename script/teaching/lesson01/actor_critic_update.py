#!/usr/bin/env python3
"""固定一条教学经历，演示一次 PPO 裁剪策略项和价值损失的参数更新。

回报由参数给定；不采集环境、不运行机器人，也不是完整 PPO 训练。
两个网络都只有一层，固定输入为 1，便于手算梯度。
"""

import argparse
import json

import torch
from torch import nn


def run_update(return_target=14.0):
    torch.set_num_threads(1)
    observation = torch.tensor([[1.0]], dtype=torch.float64, device="cpu")
    actual_return = observation.new_tensor(return_target)

    # Actor 输出两个动作的分数，softmax 将分数变成概率。
    actor = nn.Linear(1, 2, bias=False, dtype=torch.float64, device="cpu")
    critic = nn.Linear(1, 1, bias=False, dtype=torch.float64, device="cpu")
    with torch.no_grad():
        actor.weight.zero_()        # 两个动作最初各有 50% 概率。
        critic.weight.fill_(10.0)  # 输入为 1，因此初始预测为 10。

    actor_optimizer = torch.optim.SGD(actor.parameters(), lr=0.1)
    critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
    actor_before = actor.weight.detach().clone()
    critic_before = critic.weight.detach().clone()

    # 假设记录的经历选择了动作 0，之后得到 actual_return。
    probability = actor(observation).softmax(dim=-1)[0, 0]
    value = critic(observation).squeeze()
    old_probability = probability.detach().clone()  # 保存采集时的概率。
    advantage = (actual_return - value).detach()    # 对 Actor 固定这个评分。

    # Critic：让预测靠近这条经历的回报目标。
    critic_loss = (value - actual_return).square()

    # Actor：优势进入 PPO 裁剪目标；取负号后交给梯度下降。
    ratio = probability / old_probability
    actor_loss = -torch.minimum(
        ratio * advantage,
        ratio.clamp(0.8, 1.2) * advantage,
    )

    actor_optimizer.zero_grad()
    critic_optimizer.zero_grad()
    actor_loss.backward()   # 算出 Actor 参数的梯度，尚未改参数。
    actor_gradient = actor.weight.grad.detach().clone()
    actor_leaves_critic_alone = critic.weight.grad is None
    critic_loss.backward()  # 算出 Critic 参数的梯度，尚未改参数。
    critic_leaves_actor_alone = torch.equal(actor.weight.grad, actor_gradient)
    backward_keeps_parameters = (
        torch.equal(actor.weight, actor_before)
        and torch.equal(critic.weight, critic_before)
    )

    actor_optimizer.step()   # 参数 = 参数 - 学习率 × 梯度。
    critic_optimizer.step()  # 本例 Critic：10 - 0.1 × (-8) = 10.8。

    with torch.no_grad():
        probability_after = actor(observation).softmax(dim=-1)[0, 0].item()
        value_after = critic(observation).item()

    return {
        "return_target": actual_return.item(),
        "advantage": advantage.item(),
        "learning_rate": 0.1,
        "clip_epsilon": 0.2,
        "old_probability": old_probability.item(),
        "actor_loss_before": actor_loss.item(),
        "critic_loss_before": critic_loss.item(),
        "actor_gradient": actor_gradient.flatten().tolist(),
        "critic_gradient": critic.weight.grad.item(),
        "actor_weights_before": actor_before.flatten().tolist(),
        "actor_weights_after": actor.weight.detach().flatten().tolist(),
        "probability_before": probability.item(),
        "probability_after": probability_after,
        "critic_prediction_before": value.item(),
        "critic_prediction_after": value_after,
        "critic_loss_after": (value_after - actual_return.item()) ** 2,
        "backward_keeps_parameters": backward_keeps_parameters,
        "actor_backward_does_not_update_critic_gradient": actor_leaves_critic_alone,
        "critic_backward_preserves_actor_gradient": critic_leaves_actor_alone,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--return-target", type=float, default=14.0)
    args = parser.parse_args()
    print(json.dumps(run_update(args.return_target), ensure_ascii=False, indent=2))
