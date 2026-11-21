# Q1 FGP `human_smpl` 仿真遥操验证

**日期**：2026-11-21
**场景**：Q1 MuJoCo hard-floor，Pure V2b `human_smpl` ONNX
**输入**：保存的 FGP SMPL24 JSONL
**控制**：50 Hz 策略，MuJoCo 5 ms 子步，Q1 部署安全位置目标

## 主结果

主输入文件为：

```text
/home/yu/projects/FGP_mocap/results/live_device_status_r1/smpl.jsonl
```

仿真命令使用 1 秒默认站姿预热：

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=code /home/yu/projects/SuperMimicHP/.venv/bin/python \
code/sim_q1_fgp_human_smpl.py \
  --smpl-jsonl /home/yu/projects/FGP_mocap/results/live_device_status_r1/smpl.jsonl \
  --duration 3 \
  --prestand-s 1 \
  --video \
  --output verification/q1_fgp_human_smpl_sim_20261121
```

结果文件：

- [report.json](q1_fgp_human_smpl_sim_20261121/report.json)
- [trace.npz](q1_fgp_human_smpl_sim_20261121/trace.npz)
- [q1_fgp_human_smpl_sim.mp4](q1_fgp_human_smpl_sim_20261121/q1_fgp_human_smpl_sim.mp4)

| 指标 | 结果 |
|---|---:|
| FGP 50 Hz tick | `31` |
| policy steps | `31` |
| 仿真时间 | `1.62 s` |
| 终止原因 | `tilt_or_height_limit` |
| 最大倾角 | `66.176°` |
| 最大关节速度 | `12.8368 rad/s` |
| 目标误差 P95 | `1.2583 rad` |
| 单步目标变化最大值 | `0.2 rad` |
| 末段双脚接触比例 | `0.129` |
| 末端根位置 | `[-0.34534, 0.08442, 0.20095] m` |

## 交叉检查

- 不加站姿预热、从 `default_q` 直接接入时，约 `0.86 s` 触发倾角/高度终止。
- 使用官方面朝上起身记录末帧作为 Q1 初始关节姿态时，约 `1.10 s` 触发终止。
- 使用 FGP 记录中根部 roll/pitch 接近零的静态片段时，约 `1.78 s` 触发终止。

这三组结果都出现明显目标误差和身体倾倒，说明当前 `human_smpl` 真闭环仿真还没有通过。站姿预热和初始关节姿态会改变失败时间，不能单独解释失败。

## 数据来源边界

`live_device_status_r1/session.json` 的启动参数包含：

```text
fixture=/home/yu/projects/FGP_mocap/assets/recorded_fixture.npz
```

因此本轮使用的是 FGP 真实输出格式和完整 FGP 推理结果的保存回放。该文件不能证明本轮输入来自刚刚采集的真实 BLE 衣服数据。仿真没有启动 FGP BLE，也没有连接或控制 Q1 真机。

## 当前判断

Q1 MuJoCo 场景、`human_smpl` 特征、ONNX 推理和仿真执行链已经接通。当前失败发生在 `human_smpl` 策略进入物理闭环后的稳定性环节，后续需要检查：

1. FGP 初始姿态与训练时人体输入分布的差异。
2. `R_ref`、根 pitch/roll 和 Q1 本体 `obs_80` 的联合对齐。
3. `human_smpl` 输出目标与 Q1 仿真接触动力学的兼容性。
4. 站姿切换阶段的输入预热、动作限幅和策略 handoff。

本轮没有发布任何 `/aima/hal/joint/command`。
