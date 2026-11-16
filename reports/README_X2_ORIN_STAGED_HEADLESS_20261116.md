# X2 Orin 分阶段 FGP 录制验收（2026-11-16）

## Done

把已有的分阶段实验路径封装为可重置的 S1 引擎。CANO 继续使用原 TensorRT CUDA Graph，独立 CPU 子进程按序做 LIP、GlobalPose 和原生物理解算。输出保留原输入的时标。用同一段录制 IMU，完成 64 个输出的数值门，再做两次按 30 Hz 释放的较长对照。每轮 240 帧输入产生 211 个同序输出，前 29 帧用于模型历史。最后一轮相对现用单体 S1，最大姿态矩阵分量差 7.153e-7，最大根平移分量差 1.94e-7 m。主动保留一帧在途后执行重置，重置前的输出没有混入新一轮。这里验证的是软件一致性，真人 SMPL 精度尚待独立参照。

| 最后一轮的实际时间 | 单体 S1 | 分阶段 S1，CPU 算完 | 分阶段 S1，父进程收到 |
| --- | ---: | ---: | ---: |
| 平均输出率 | 29.98 Hz | 29.98 Hz | 29.99 Hz |
| 相邻间隔 P95 | 39.16 ms | 38.44 ms | 51.81 ms |
| 相邻间隔最大值 | 43.36 ms | 47.94 ms | 55.76 ms |
| 录制释放到该阶段 P95 | 39.32 ms | 38.14 ms | 54.91 ms |

CPU 算完到父进程收回的间隔 P95 为 18.30 ms，包含结果封包、进程间传递和客户端取件等待。最长出帧间隔 **55.76 ms** 的两帧在 CPU 侧只隔了 **37.03 ms**。这两帧各自的回收等待为 **0.56 ms** 和 **19.29 ms**。在最慢 21 个间隔中，出帧间隔中位 51.82 ms，对应的 CPU 完成间隔中位 38.24 ms。分阶段 CANO P95 20.17 ms、CPU 重建 P95 17.00 ms、CPU 入队 P95 0.68 ms。各分段 P95 不可直接相加。

第一次长对照的单体释放到结果 P95 为 61.34 ms，分阶段为 55.07 ms。第二次单体降到 39.32 ms，分阶段为 54.91 ms。两次都先跑单体，无法据此判断净延迟收益。两次分阶段的出帧间隔 P95 都更高，因此本轮未切换实时衣服入口。所有轮次仅回放录制 IMU，未启动 BLE、策略反馈、MC 或电机发布。Orin 子进程均正常退出，原预览锁已释放。

- [64 输出原件](../logs/session_records/x2_orin_staged_headless_parity_20261116.json)
- [第一轮 paced 原件](../logs/session_records/x2_orin_staged_headless_paced_20261116.json)
- [第二轮 paced 与计时原件](../logs/session_records/x2_orin_staged_headless_timing_20261116.json)
- [422 行逐帧计时压缩件](../data/analysis/X2_ORIN_STAGED_HEADLESS_20261116/stage_trace.jsonl.gz)保存在 Mac 主根，Git 保留小回执。压缩件 51595 B，SHA-256 为 `44f1870e08901de8de40c25feb27bffac48f76301b8931da0ce5b3362e5aa3a7`，与板端一致。
- 两个新增文件在 Orin 临时 overlay `/home/run/x2-fgp-stage-candidate.AMQOAO` 上验收，安装目录未替换。初次启动因当前目录遮蔽 overlay 未导入模块，在 overlay 工作目录重试通过。Mac 27 项预览/流水线、19 项 MorphMimic 测试通过。

## To do

先让结果接收独立于 CANO 主循环，并保留 CPU 完成、结果收回、HumanFrame 发送三个时点。用固定输入做两种收集方式的配对对照，判断能否缩短实际出帧长尾，再接无网页穿戴终端和真实反馈只读测试。届时同时核对 SMPL 数值、时间戳归属、参考年龄和目标跳变。
