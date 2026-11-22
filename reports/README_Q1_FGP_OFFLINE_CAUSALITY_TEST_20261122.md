# Q1 FGP 遥操离线因果对照

日期：2026-11-22

输入使用真机录制：

`/home/yu/projects/Q1_deploy/real_logs/q1_fgp_control_v2_real_20261122_run02/fgp.jsonl`

对照片段从录制开始后的 `175.3 s` 开始，使用相同的 human_smpl ONNX、相同 MuJoCo 场景和 2 ms 物理步长。

| 适配器 | 根反馈 | 仿真时长 | 结果 | 最大倾角 | target error P95 |
| --- | --- | ---: | --- | ---: | ---: |
| 旧 FGP adapter | CGFA | 12.00 s | 失稳 | 67.31° | 0.685 rad |
| 旧 FGP adapter | zeroed | 66.58 s | 失稳 | 66.58° | 0.543 rad |
| A2 FGP V2 | CGFA | 119.72 s | 完成 | 13.50° | 0.515 rad |
| A2 FGP V2 | zeroed | 68.32 s | 失稳 | 68.04° | 0.563 rad |
| A2 FGP V2 | sim-truth | 119.72 s | 完成 | 14.46° | 0.556 rad |

输出目录：

- `verification/q1_fgp_run02_sim_cgfa`
- `verification/q1_fgp_run02_sim_zeroed`
- `verification/q1_fgp_run02_sim_v2_cgfa`
- `verification/q1_fgp_run02_sim_v2_zeroed`
- `verification/q1_fgp_run02_sim_v2_truth`

## 结论

1. 旧适配器的根反馈会显著放大真实 FGP 录制中的失稳。
2. A2 的 CGFA 根反馈和 MuJoCo 真值反馈表现接近，A2 的接触锚定没有表现为主要失稳源。
3. 去掉根反馈后 A2 反而在约 `68.3 s` 失稳，说明 human_smpl 策略需要正确的根运动和接触反馈。
4. 当前真机“能迈步但不自然”的剩余问题优先指向 human_smpl encoder / policy 的步态相位与真机 PD 跟踪，而非 FGP HTTP 断流。

这组测试没有修改 FGP 服务端，也没有发送真机命令。
