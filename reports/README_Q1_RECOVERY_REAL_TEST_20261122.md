# Q1 Recovery V2 真机测试报告

日期：2026-11-22。测试入口为 Recovery V2 纯自主模式，使用 `--recovery-offline`。两轮均在掉索和人工扶持条件下进行。

## 结果摘要

| 轮次 | 运行时长 | 结果 | 归因 |
| --- | ---: | --- | --- |
| 第 1 轮 | 15.9 s | 自主起身成功 | 用户现场确认 Recovery 自主站起完成 |
| 第 2 轮 | 59.1 s | 起身流程中止 | 用户现场确认 Q1 手部硬件断裂，无法继续完成起身 |

第二轮不能计为 Recovery 策略失败。中止原因是执行机构硬件故障，硬件故障发生后继续施加起身动作没有意义。

## 采集文件

第 1 轮目录：

`/home/yu/projects/Q1_deploy/real_logs/q1_recovery_v2_real_20261122/`

- `state_500hz.jsonl`：7,940 条
- `policy_50hz.jsonl`：795 条
- `command_500hz.jsonl`：7,939 条
- 异步写盘丢帧：0

第 2 轮目录：

`/home/yu/projects/Q1_deploy/real_logs/q1_recovery_v2_real_20261122_run02/`

- `state_500hz.jsonl`：28,425 条
- `policy_50hz.jsonl`：2,843 条
- `command_500hz.jsonl`：28,431 条
- 异步写盘丢帧：0
- 退出尾段包含阻尼命令记录

FGP 文件为空是预期行为。Recovery offline 不使用 FGP 输入。

## 数值检查

第 1 轮状态和 policy 数值均为有限值。关节速度最大约 `9.71 rad/s`，policy 估算力矩最大约 `33.82 Nm`。

第 2 轮状态和 policy 数值均为有限值。关节速度最大约 `19.16 rad/s`，policy 估算力矩达到 `36 Nm` 限幅。该轮已有手部硬件故障，后段数值不用于判断完整动作质量。

两轮状态记录间隔约 2 ms，policy 约 20 ms，说明控制循环和采集器运行正常。第 2 轮退出命令尾段记录 6 条阻尼帧。

## 结论边界

本轮提供了 Recovery V2 在 Q1 真机上至少一次自主起身成功的证据，说明从仿真到真机的控制链和 Recovery owner 可以工作。样本量为 1 个成功轮次，不能据此给出不同倒姿、重复性、柔和度或硬件耐久性的统计结论。

第二轮断手必须单独登记为硬件故障。继续真机 Recovery 测试前，应先完成手部机构检查和更换，保留两轮原始数据，不用第二轮故障样本否定第一轮成功结果，也不把第一轮结果扩大成正式验收。

## 版本与入口

- Recovery V2 候选：`/home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/`
- 退出零力矩选项：`--exit-passive`
- 相关 Q1 分支修复提交：`56273612`
- 原始数据：本机 `LOCAL_ONLY`
