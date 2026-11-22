# Q1 Planner / human_smpl 切换真机数据报告

日期：2026-11-22。运行入口为 Q1 V2 候选桥接的 `--keyboard-switchable`。本轮没有启用 Recovery。

## 运行范围

- 运行时间：2026-11-22 15:26:31.298 到 15:29:49.041，约 197.74 秒。
- 控制模式顺序：Planner → human_smpl → Planner。
- 机器人数据来自 Q1 HAL 反馈，命令通过 `/aima/hal/joint/command` 发布。
- FGP 来自 `http://127.0.0.1:8090` 的实时衣服服务。
- 录制目录：`/home/yu/projects/Q1_deploy/real_logs/q1_switchable_record_20261122/`
- 启动配置：`/home/yu/projects/Q1_deploy/real_logs/q1_switchable_record_20261122/session.json`

## 文件完整性

| 文件 | 记录数 | 文件大小 | SHA-256 |
| --- | ---: | ---: | --- |
| `fgp.jsonl` | 5,855 | 38,924,514 B | `2dac7a8e436c24a1f220be3d279f1ca02901cc4f22843ab9d346ef9327371a35` |
| `state_500hz.jsonl` | 98,861 | 174,389,304 B | `34211bd24e32c114084d8a72b8a7fd9cb17d55c2165e157d0fa608eb58b9a14c` |
| `policy_50hz.jsonl` | 9,887 | 109,289,656 B | `cd2d0984dd89345254691cc206007312fb1d42214dc0c02dc8bfaaa1ca9dc451` |
| `command_500hz.jsonl` | 98,866 | 140,831,446 B | `241d158cd0b02f5b03d15b2b8c4d486ab272676c71a91d4cba018df97fcc33e7` |

`summary.json` 中异步写盘的 `dropped` 为空，`queue_unfinished_tasks=0`。四路文件均为有限数值记录，数组维度符合 Q1 22 关节、`obs_80` 和 `human_smpl` 430 维输入。

## 频率与同步

| 流 | 实际记录 | 时间间隔中位数 | P95 | 最大间隔 |
| --- | ---: | ---: | ---: | ---: |
| FGP 新接受帧 | 5,855 | 31.05 ms | 42.23 ms | 88.05 ms |
| HAL 状态 | 98,861 | 1.996 ms | 3.135 ms | 26.82 ms |
| policy | 9,887 | 19.999 ms | 21.815 ms | 33.10 ms |
| 实际命令 | 98,866 | 1.977 ms | 4.690 ms | 17.68 ms |

状态、policy 和命令流均没有超过 50 ms 的时间间隔。policy 记录和最近命令的目标位置逐条对齐误差为 0。命令比状态多 5 条，对应退出时的阻尼帧。

## 模式与 FGP 有效性

| 模式 | 状态记录 | policy 记录 | 命令记录 |
| --- | ---: | ---: | ---: |
| Planner | 28,401 | 2,841 | 28,401 |
| human_smpl | 70,460 | 7,046 | 70,460 |
| damping | 0 | 0 | 5 |

human_smpl policy 中有有效 FGP 元数据的记录为 6,888/7,046，约 97.76%。共有 12 个短暂无有效 FGP 的区间，单段约 0.22–0.26 秒。缺失区间已经在 `policy_50hz.jsonl` 中以 `fgp=null` 记录，不能直接删除后拼接。

FGP 接受帧来自单一会话和标定代次 11，序号范围为 88,356–94,211。接受帧序号出现 1 次跳跃。policy 侧出现多个参考 segment，说明实时消费过程中发生过短暂失效与重新锚定。FGP 服务自身的长期日志还受到 256 MiB 日志上限影响，不能用它替代本次桥接目录中的同步记录。

## 数值与安全检查

- `state_500hz` 的 IMU 矩阵、角速度、关节位置、速度和力矩反馈均为有限值。
- 最大关节速度反馈约 `14.41 rad/s`。
- 反馈力矩最大约 `36.10 Nm`，发现 1 个样本超过配置努力上限约 `0.10 Nm`，属于反馈值检查结果，不等同于本桥接估算命令越限。
- policy `tau_est` 最大约 `36.00 Nm`，目标位置最大绝对值约 `2.22 rad`。
- 录制中没有 Recovery owner。Recovery 配置保持关闭。

## 数据是否正常

采集器本身正常。四路记录齐全、异步写盘无丢帧、频率和时间对齐满足回放需要。

数据内容存在真实 FGP 短暂失效和参考重锚，属于本轮输入质量的一部分。该目录适合用于：

- 对照 Planner 与 human_smpl 两种 owner 的状态、命令和 policy 输出。
- 分析 FGP 失效前后目标、观测和机器人反馈。
- 训练端按 `fgp` 是否为空、segment 和 `mode` 做有效样本筛选。

该目录不适合作为无缺口的连续 human_smpl 训练集。真机反馈已经录下，当前没有发现本轮对应的板端 bag 文件，训练端应以本目录的 `state_500hz` 和 `command_500hz` 为真机同步记录。

## 交付路径

- 完整数据：`/home/yu/projects/Q1_deploy/real_logs/q1_switchable_record_20261122/`
- 本报告：`/home/yu/projects/Q1_deploy/real_logs/Q1_SWITCHABLE_RUNTIME_DATA_REPORT_20261122.md`
- 代码版本：Q1 分支提交 `b3451dd2` 已包含采集器。后续按键修正和退出诊断提交分别为 `1a49c4f6`、`cc1f0e9f`。

大文件和原始真机记录保持本机 `LOCAL_ONLY`，没有提交到普通 Git 历史。
