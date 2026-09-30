# 第一课 · 第七小节：学习信号怎样变成参数更新

记录日期：2026-09-30。阶段：L01-S07。状态：Agent 已在 hp3090 的 CPU 上运行教学代码，正、负、零优势对照通过；学员尚未独立运行，理解待确认。不是完整 PPO 或机器人训练验收。

## 目标与范围

用户问“让预测从 10 向 14 调整的信号具体是什么”，并要求用一份简单代码解释 Critic 和 Actor 如何更新。本节把信号具体落实为损失对参数的梯度，并区分计算梯度与执行更新。

完整脚本：[actor_critic_update.py](../script/teaching/lesson01/actor_critic_update.py)。只依赖已有 PyTorch，使用 CPU；回报是给定的教学标签，不来自本次机器人或环境交互。每次执行都创建全新的小网络，只更新一次。

## 完成内容与教学代码

固定观测为 `[1]`。Actor 是无偏置的一层线性网络，输出两个动作的分数，经 softmax 变成概率；初始两个权重都是 0，因此两个动作各有 50% 概率。Critic 同样只有一层，唯一权重初始化为 10，因此初始预测是 `10×1=10`。

假设存下的经历选择了动作 0，回报标签为 14。动作编号和回报都固定，是为了把更新过程与采样随机性分开看；不能把这组数字当成机器人实测成绩。

脚本先计算两个网络的输出，并保存采集参照 `old_probability`。核心更新如下，网络、优化器与观测的创建见完整脚本：

```python
advantage = (actual_return - value).detach()  # 14 - 10 = +4
critic_loss = (value - actual_return).square()

ratio = probability / old_probability
actor_loss = -torch.minimum(
    ratio * advantage,
    ratio.clamp(0.8, 1.2) * advantage,
)

actor_optimizer.zero_grad()
critic_optimizer.zero_grad()
actor_loss.backward()   # 将梯度写到 Actor 参数的 .grad。
critic_loss.backward()  # 将梯度写到 Critic 参数的 .grad。
actor_optimizer.step()  # 才在这里真正修改 Actor 参数。
critic_optimizer.step()
```

`zero_grad()` 清除上次累计的梯度；`backward()` 通过计算过程求梯度；本例 SGD 的 `step()` 使用“参数减去学习率乘梯度”修改权重。用两个优化器是为了清楚展示两个网络的作用，并不要求所有实际 PPO 实现都采用两个优化器。

### Critic 的信号具体是 −8

令唯一权重为 `w`，输入为 1，故预测 `value=w`。对本次样本：

```text
预测误差：10 - 14 = -4
平方损失：(10 - 14)^2 = 16
该损失对 w 的梯度：2 × (10 - 14) × 1 = -8
学习率：0.1
新权重：10 - 0.1 × (-8) = 10.8
重新前向计算的新预测：10.8 × 1 = 10.8
```

梯度 −8 表示在当前点，增大这个权重会使此样本的损失下降；更新时减去负梯度，权重就增大。程序不是直接把预测输出改成 14，而是修改权重后重新计算预测。真实网络会有很多权重和对应梯度，本例因输入为 1，权重与预测值恰好数值相同。

这次向 14 调整仍是单样本拟合；从多次经历学习的目标是平均回报，不能把 14 当成永远固定的目标。

### Actor 的信号是两个权重的梯度

这条样本的优势固定为 +4。第一次更新前，新旧策略相同，概率比为 1，处于未裁剪区域。PyTorch 算出的两个 Actor 权重梯度是 `[-2, +2]`。

用同样的学习率 0.1 更新后，两个权重从 `[0, 0]` 变为 `[0.2, -0.2]`。重新计算 softmax，动作 0 的概率从 50% 变为约 59.87%。改变的是生成动作分布的网络参数；没有把动作幅度直接改成这些数值，也没有把概率直接加上优势 4。

Actor 目标前的负号让梯度下降对应于提高原来的策略目标。Critic 用预测误差学习，Actor 用优势指导概率变化；两种损失的梯度不应混为同一个数。

### 两处固定数据的处理

- `old_probability = probability.detach().clone()` 模拟保存采集时的动作概率；本例只有一次更新。若之后对同批数据多次更新，应继续使用这个保存值，不能每次重新计算分母。
- `advantage = (actual_return - value).detach()` 将评分作为 Actor 本次更新的固定数字，不让 Actor 的损失反向改动 Critic 来改变这个评分。Critic 仍通过自己的 `critic_loss` 获得梯度；`detach()` 没有让它永久停止学习。

## 实际验证结果

三个独立执行只改变回报标签，每次均重新初始化权重，学习率 0.1、裁剪参数 0.2，其余设置相同。

| 回报标签 | 优势 | Actor 两个权重梯度 | 动作 0 概率 | Critic 权重梯度 | Critic 预测 |
| --- | --- | --- | --- | --- | --- |
| 14 | +4 | −2、+2 | 50% → 59.868766% | −8 | 10 → 10.8 |
| 6 | −4 | +2、−2 | 50% → 40.131234% | +8 | 10 → 9.2 |
| 10 | 0 | 0、0 | 50% → 50% | 0 | 10 → 10 |

正负两例的 Critic 样本平方误差均从 16 降为约 10.24；零优势例的预测误差本来为 0，保持为 0。

每例检查六项行为：反向传播不改变权重、Actor 梯度不传入 Critic、Critic 反向传播不改变 Actor 梯度、动作概率方向正确、Critic 更新符合手算 SGD、样本预测误差改善或保持为零。三例共 18 项均通过。这里最后一个零更新结论仅适用于本例的损失和优化器设置，不能推广到含其他损失、动量或共享参数的所有系统。

首次远程执行封装出现字符串换行转义错误，Python 在解析阶段退出，尚未上传脚本或执行实验；修正封装后完成上述运行。失败和修复后的原始输出均保存在发布日志中。

## 关键决定及局限

- 使用一层网络、固定输入 1、显式初始化、float64 和无动量 SGD，让参数变化能直接手算；不安装依赖或占用 GPU 做长训练。
- Actor 使用 PPO 的裁剪策略项，Critic 使用基本平方损失。示例没有采集 rollout、计算 GAE、优势归一化、价值裁剪、熵奖励或多轮训练，不等于完整 PPO。三个对照的更新前比值都为 1，没有验证平台外的裁剪分支。
- 正优势下的概率增加只针对本例未处于裁剪平台时的一次更新；与之前“进入平台的样本策略项梯度为零”的结论一致。
- 未启动仿真、控制真机或生成训练 checkpoint；不能据此标记 P0 通过，也不能记为学员已独立完成训练。

## 版本、可复现命令与产物

- 主机与目录：`ssh hp3090`，`/media/yu/FAFF-E9771/YUANQI`。
- 解释器：`/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python`；本轮实际读取版本为 Python 3.11.15、PyTorch 2.8.0+cu128；实验设备为 CPU。
- 执行完成记录时间：2026-09-30 15:54:31 UTC，即北京时间 23:54:31。
- 脚本 SHA-256：`e6e590543bf2856bec737aa7592a2ff887328f3b416ac1f0663f75f2291bfac6`。
- 结果：`data/teaching/lesson01/actor_critic_update_20260930.json`；SHA-256：`ff7683e783f87d2004acede14008090ae0cc39b668486273d5234c5527cfc86a`。
- 运行日志：`logs/lesson01_actor_critic_update_run_20260930.log`。结果和日志已在服务器与本地镜像保存；运行脚本本身打印 JSON，本轮执行封装将三个结果汇入上述结果文件。

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/teaching/lesson01/actor_critic_update.py --return-target 14
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/teaching/lesson01/actor_critic_update.py --return-target 6
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/teaching/lesson01/actor_critic_update.py --return-target 10
```

## 保存与同步状态

编辑前服务器 HEAD 与 GitHub main 均为 `d0d89a3608ed41ce86153421d505462b280a4f16`，工作区干净；相关既有文档在本地与服务器摘要一致。新增脚本、此 README、阶段总结、两个报告入口和脚本索引按规范审核后提交推送，实际 diff、负载检查与发布核验保留在 `logs/lesson01_actor_critic_update_publish_20260930.log`。

小型原始结果和日志保留在被忽略的 `data/`、`logs/`，Git 跟踪代码、结论与上述 SHA-256。无大型模型或百度网盘传输。

## 下一步与恢复入口

当前先让学员对照代码看清“损失 → 梯度 → 参数更新”，以及 +4 的优势与 −8 的 Critic 梯度分别是什么；已执行对照不等于学员已理解。随后把这段代码放回完整采集与训练流程，保持定期回顾，不继续堆叠新概念。

恢复先读[当前状态](README_CURRENT_STATE.md)、[阶段总结](README_LESSON01_SUMMARY_20260930.md)及本文。工程主线仍是完成首次完整训练、保存和重载闭环。
