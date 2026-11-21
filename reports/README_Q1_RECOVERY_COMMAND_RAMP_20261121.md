# Q1 Recovery 命令级斜坡复核

更新于 2026-11-21。下文保留 Mac 历史试验，原接力 2/5 因参考四元数顺序与步宽取列有误而撤回。原稿将肩俯仰峰值误写为腰肘。3090 的正确入口为 `ssh yu-3090`。实机前的工程判断、非等价场景与精确命令级积分的复核见[正式结果](README_Q1_RECOVERY_3090_20261121.md)。

记录日期：2026-11-21。阶段 / 实验编号：hardware-consistent-08。状态：部分完成。

## 目标与范围

- 验证 Q1 的 0.2 rad 约束应当作用在 500 Hz 低层单帧命令，还是作用在 50 Hz 策略输出。
- 检查命令级斜坡能否解决当前起身过猛、类似鲤鱼打挺的工程风险。
- 保持 formal_03 Recovery ONNX 权重不变，不发送任何真机电机命令。

3090 的 SSH 入口在本轮持续超时，因此本轮回放使用 Mac 上保存的 formal_03 权重和 Q1 机器人资产。为了补足本地缺失的训练场景生成器，临时用发布包中的硬地 XML 和 Q1 v3 网格构造了等价的离线测试场景。结果用于筛选实现方向，不能替代 yu-3090 上的正式复核，也不能证明真机 Recovery 已经可用。

## 完成内容

- 修正 `hardware_consistent_replay.py` 的输出目录生命周期和 E1 参数传递。
- 将 0.2 rad 约束实现为每个 20 ms 策略周期内的 10 次 500 Hz 命令更新，每条命令最多变化 0.2 rad。
- 将目标先限制在 Q1 关节范围，再进入命令级斜坡。
- 增加可选的上肢命令级斜坡参数，用于筛查全局减速是否能抑制躯干和手臂的突跳。
- 将修订后的脚本同步到本地工作树 `/Users/yu/project/RL_locomotion/h003_recovery_expert/hardware_consistent_replay.py`。

脚本与报告副本 SHA-256 均为 `7fae72907fd5b17db2e6e8f65f60d87804ecc9bae5ad6a43dcb82484ce8c6908`。使用的 formal_03 ONNX SHA-256 为 `0231328d60f07297b38a4079e01167cf9fec65fdb2c89302144ee65e0011eb92`。

## 结果

| 条件 | E1 静态起身 | 动态接力 | 峰值关节速度估计 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 50 Hz 推理级 0.2 rad 限幅 | 0/24 | 0/5 | 20.82 rad/s | 起身目标在错误层级被压缩 |
| 不做推理级限幅 | 24/24 | 2/5 | 27.69 rad/s | 恢复能力存在，但动态接力仍有限 |
| 500 Hz 命令级 0.2 rad 斜坡 | 24/24 | 2/5 | 27.90 rad/s | 解决静态失败，未降低起身速度 |
| 命令级全关节 0.1 rad 斜坡 | 0/24 | 0/5 | 28.28 rad/s | 统一减速导致策略无法完成起身 |
| 命令级全关节 0.05 rad 斜坡 | 4/24 | 0/5 | 27.68 rad/s | 成功率仍不可接受 |
| 上肢 0.1 或 0.15 rad、下肢 0.2 rad | 0/24 | 0/5 | 28.12–29.40 rad/s | 简单上肢减速同样破坏起身 |

命令级 0.2 rad 斜坡恢复了 E1 的 24/24，但峰值速度与不做推理级限幅时接近。当前最高速度主要出现在腰部和肘部关节，单纯降低这些关节的全程命令步长会使 E1 失败。

## 关键决定及依据

- 保留命令级斜坡作为真机桥接的必要实现。现有离线结果表明，50 Hz 推理级限幅会把可起身的策略变成 0/24，不能直接沿用。
- 暂不把全局低通或统一上肢降速作为候选方案。它们在本轮固定 E1 和动态接力中均未达到工程可用门槛。
- 起身过猛需要阶段相关的轨迹约束或重新导出的 Recovery 策略。需要让翻身、支撑、离地和站稳阶段使用不同的目标变化约束，不能用一个固定缩放覆盖整个恢复过程。

## 版本、命令与证据

本地脚本检查通过。

```sh
python3 -m py_compile reports/recovery/hardware_consistent_replay.py
```

本轮本地回放使用以下入口。

```sh
Q1_RECOVERY_RUN_ROOT=/Users/yu/project/RL_locomotion/h003_recovery_expert \
Q1_V2B_BUNDLE=/tmp/q1_bundle_local \
Q1_RECOVERY_EXPERT=/Users/yu/project/RL_locomotion/h003_recovery_expert/evidence/formal_03/q1_recovery_expert.onnx \
Q1_SOURCE_ROOT=/Users/yu/project/RL_locomotion/h003_recovery_expert/dependencies \
PYTHONPATH=/tmp/q1_h003_local \
/Users/yu/project/RL_locomotion/.venv/bin/python \
/Users/yu/project/RL_locomotion/h003_recovery_expert/hardware_consistent_replay.py \
/tmp/hardware_consistent_08_command_ramp_formal_v2 \
--expert /Users/yu/project/RL_locomotion/h003_recovery_expert/evidence/formal_03/q1_recovery_expert.onnx \
--tracks /Users/yu/project/RL_locomotion/h003_recovery_expert/evidence/formal_03/input_tracks \
--state-delay-ms 11 \
--command-ramp \
--command-frame-step-rad 0.2 \
--e1-steps 600
```

结果目录保存在 `/tmp`，包括 `summary.json`、逐试次 `npz` 和 `manifest.json`。

- `/tmp/hardware_consistent_08_command_ramp_formal_v2`
- `/tmp/hardware_consistent_08_inference_clamp_formal`
- `/tmp/hardware_consistent_08_no_inference_clamp_formal`
- `/tmp/hardware_consistent_08_command_ramp_005_formal`
- `/tmp/hardware_consistent_08_command_ramp_010_formal`
- `/tmp/hardware_consistent_08_upper_body_010_formal`
- `/tmp/hardware_consistent_08_upper_body_015_formal`

3090 连接尝试记录为 `ssh hp3090` 到 `100.70.125.121:22` 超时。本轮没有上传脚本，没有启动远端作业，也没有真机输出。

## 结果边界

- 已证明的是离线实现层面的差异。推理级限幅与命令级斜坡在同一套本地回放中表现完全不同。
- 尚未证明 500 Hz 命令级斜坡在 yu-3090 的正式 formal_03 环境中保持同样的成功率。
- 尚未证明 Q1 真机 Recovery 可以安全起身。当前 Q1 真机证据仍只有官方 MC 起身，Recovery ONNX 没有进入低层带电控制。
- 这轮结果没有解决起身动作的视觉突跳。它只定位了一个必须修复的桥接层问题。

## 下一步与恢复入口

1. 3090 恢复后，先运行原 formal_03 模型的 500 Hz 命令级斜坡复核，保留 0.2 rad 单命令限制。
2. 取回并比较已有 `hold_v2`、`slow_v2`、`slow_v3` 候选。比较 E1、动态接力、最大速度、最大加速度、穿透和站稳时间。
3. 若候选仍有类似鲤鱼打挺，再做按 Recovery 阶段划分的目标变化约束或重新导出，不再尝试全程统一减速。
4. 真机验证前保持吊架、阻尼退出、低层反馈和零电机默认流程。没有通过正式仿真和低层安全门，不进入带电 Recovery。

恢复时先读取本文件、`reports/README_CURRENT_STATE.md`、`reports/recovery/hardware_consistent_replay.py` 和 `/Users/yu/project/RL_locomotion/h003_recovery_expert/README.md`。
