# Q1 部署交接指南：FGP 衣服实时控制优化 V2（适配层升级 + 防跌落保护 + 热切换修复）

> **致部署 Agent（负责 `/home/yu/projects/Q1_deploy/`）**：
> 本文档及配套独立增量包面向 Q1 V2 候选部署目录（[`q1_deploy_v2_candidate_20261122`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/)）的 `human_smpl`（FGP 衣服实时遥操）控制链路升级。
> 核心升级为 **纯 NumPy / 零 PyTorch / 零 MuJoCo 的端侧适配器、流量整形客户端、回退重锚守卫与桥接热切换修复（方案 `A2`）**；冻结主干 `q1_h002_token_head.onnx` 与全部现有 ONNX 兄弟模态均 **100% 兼容不变**。

---

## 一、 结论摘要、入选方案与验证边界

### 1.1 预注册晋级评审与最终入选方案（`A2` 胜出，零 ONNX 回归风险）

我们在 5 套 MuJoCo 闭环基准中严格对比了四组配置，并对“新训练 V2 暖启动近端微调 ONNX（`A3`）”施加了严格的 5 项全过晋级门槛：

| 候选方案 | 核心组成 | 晋级评审结论 |
| --- | --- | --- |
| `A0_ProdOld`（线上基线） | 旧 `FGPSMPLStream` + `get_latest()` + 无重锚 `planner` 回退 + V2 ONNX | 今日真实 FGP 在 `18.82 s` 跌倒；保留集 `cal2`/`cal5` 均跌倒；Suite 5 在首个单帧闪烁后 `7.48 s` 跌倒 |
| `A1_SmoothV2`（无断流无去偏对照组） | 去掉断流与阶梯的原始 FGP 特征 + V2 ONNX + 500 Hz 斜坡 | 虽走完 `141.00 s`，但跟随系统倾角持续前倾（`pitch p50 = +11.26°`、`tilt p95 = 18.57°`、`arm = 14.14°`、`v_xy_corr = 0.205`） |
| **`A2_AdapterV2_V2ONNX`（最终入选交付）** | **`FGPSMPLStreamV2` + `FGPHttpSourceV2` + `Q1FGPFallbackGuard` + 热切换不重置 tracker + 冻结 V2 `q1_tok_human_smpl.onnx` (`7dab4d9d...`)** | **实机录制 `140.92/140.92 s`（`tilt max 15.92°`、`pitch +0.53°`、`arm 7.22° (+0.893)`、`v_xy_corr 0.573`），保留集 `cal2`/`cal5` 全过，8 项原语 `8/8`，Suite 5 含 400 ms 真断流 `40.0/40.0 s` 全过；且完全复用已部署 V2 ONNX，零回归风险** |
| `A3_AdapterV2_FGPV2ONNX`（备选微调） | `A2` 适配栈 + V2 暖启动近端微调 ONNX（`1927` 片段 / `3000` 步，`final_l_prox=0.47065`，`sha256: 318b3ae5...`） | 虽在 `50-HO`（`94%->100%`）与 `20-STR`（`85%->100%`）存活率提高，但**未通过 Suite 4 三项硬门槛**：`50-HO body_pos_rmse = 6.20 cm`（`> 6.1 cm` 门槛且劣于 V2 `5.78 cm`）、`20-STR body_pos_rmse = 9.22 cm / slip = 350.2 mm/s`（劣于 V2 `7.17 cm / 263.8 mm/s`）、`8-BVH` 零 IK 跌倒 1 例（`87.5%`，`lafan_walk2_subject1` 在 `267/280` 跌倒，`xy=46.24 cm > 42.0 cm`，而 V2 为 **`100% (8/8)`**）；且在今日实机小抬腿（`lift_3 3.58 vs 7.19 cm`）与速度追踪（`0.538 vs 0.573`）弱于 `A2` |

因此按预注册晋级规则，**最终正式交付方案锁定为 `A2_adapter_v2_frozen_v2_onnx`**（[`q1_tok_human_smpl.onnx`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/onnx/q1_tok_human_smpl.onnx)，`sha256: 7dab4d9d056e797c1ae8631d6c15ffd5fc3324234c86b821b253fcbb46adfb0e`）。

### 1.2 与现部署 V2 候选包的 100% 冻结兼容性

以下已部署 ONNX 权重、元数据与 `codec_fingerprint` 保持 **100% 不变**：
- `q1_h002_token_head.onnx`: `sha256 = e0455c0b50db1c1fdc8aed1e0d283820d3fd64174b261093d99dc4430ff3e80f`，`codec_fingerprint = 33f64eaefe80c12b35d7bd0b8bfbf641a3a8d463e9979ea448b7ba79788f13f8`
- `q1_tok_human_smpl.onnx`: `sha256 = 7dab4d9d056e797c1ae8631d6c15ffd5fc3324234c86b821b253fcbb46adfb0e`（与现部署候选包逐字节一致）
- `q1_tok_planner_cmd.onnx`（Lateral Strafe V2）: `sha256 = ce0aa0a49666515e0b47dd200995fecc8980cd3bcdd41149fbc4824fb92679e5`
- `q1_tok_robot_pose.onnx`: `sha256 = 9ee7a429347d0c9076657c6bb47e4635ee6b149cdc2504370fbfcecf09e6e6d9`
- `q1_tok_vr_3pt.onnx`: `sha256 = ff0586884a3f1b0aa33883145945be8f32e9cd91110891ffc1d7b41aa97e06ba`
- `q1_recovery_expert_v2.onnx`: `sha256 = 5d38cfcb3c31ad1103bd8897d4c8fad97472a04d5e258106732dd3a409b842ee`

### 1.3 验证边界声明（重要）

- **已验证范围**：2 ms 物理步长 MuJoCo 闭环仿真（5 大套件、含今日真实 `140.92 s` FGP 时序与 12 次实机断流重放、保留集 `-22°` 偏置 FGP、`0.70x` 低摩擦 + `20 ms` 延迟 + `400 ms` 真断流切换压测）+ 独立进程 `python -I` 单线程 CPU ONNX Runtime 冒烟验证（零 PyTorch / 零 MuJoCo）。
- **未验证范围**：**尚未在物理 Q1 真机硬件上运行实测**。上机前仍需先做平地吊架保护小幅度试测，并遵守第四、五节接入与标定检查流程。

---

## 二、 今日真机轨迹根因取证摘要

对 [`q1_switchable_record_20261122`](file:///home/yu/projects/Q1_deploy/real_logs/q1_switchable_record_20261122/) 同步 4 路日志的逐帧分析与 MuJoCo 闭环复现确认了 **4 个原始根因 + 1 个压测新发现的桥接热切换缺陷**：

1. **FGP 服务端单帧丢失即触发假 `outage`**：[`service.py`](file:///home/yu/projects/FGP_mocap/fgp_linux/service.py#L269-L287) 在 `max_skew_ms=40` 且 `missing_ticks == 1` 时立刻广播 `status="outage"`，导致今日实机共 `12` 次 `0.22–0.26 s` 假断流；旧客户端每次立刻 `invalidate()` 清空 `11` 帧历史并重锚。
   - **V2 对策**：[`FGPSMPLStreamV2`](file:///home/yu/projects/agibot-X2-A3-Q1-model/script/q1_fgp_control_v2/q1_fgp_smpl_v2.py#L170-L182) 增加 `outage_grace_ms = 220.0`，同 `(session_id, calibration_id)` 下短暂断流忽略 `invalidate()`，保持段号 `segment` 与 11 帧滑窗连续。
2. **桥接 `fgp=null` 回退 `planner_cmd` 漏掉 `_reanchor_reference()`（实机险摔主因）**：原 [`q1_ros2_lowlevel_bridge.py`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/code/q1_ros2_lowlevel_bridge.py#L690-L702) 在 `active_mode=="human_smpl"` 且 `live_fgp=False` 时调用 `_base_step(..., neutral=True)`，但行 693 仅对 `active_mode=="planner"` 重锚，导致混合旧 `yaw_plan`/`p_plan_xy` 与实机 `CGFA` 状态（注入最大 `0.250 m` XY 与 `112.3°` yaw 跳变，直接触发真机 `t=43.39 s` `58.7°` 大倾角，并在 MuJoCo 中于 `18.82 s` 复现跌倒）。
   - **V2 对策**：[`Q1FGPFallbackGuard`](file:///home/yu/projects/agibot-X2-A3-Q1-model/script/q1_fgp_control_v2/q1_fgp_client_v2.py#L209-L230) 强制追踪实际执行分支，只要从 `human_smpl` 回退到 `planner` 必先单次调用 `_reanchor_reference()`。
3. **`get_latest()` 无节拍缓冲导致 30→50 Hz 阶梯采样**：旧接口仅 `20.8%` 为正常 `20 ms` 间隔，出现 `41.0%` `0 ms` 重复帧与 `35.4%` `40 ms` 跳变。
   - **V2 对策**：[`FGPHttpSourceV2.pop_paced()`](file:///home/yu/projects/agibot-X2-A3-Q1-model/script/q1_fgp_control_v2/q1_fgp_client_v2.py#L128-L153) 采用 `max_buffer_ticks = 3` 有界 FIFO 匀步消费，短缺口（`<= 220 ms`）内平滑衰减速度与角速度保持（`_decay_tick_motion`），将正常 `20 ms` 采样率从 `20.8%` 提升至 `83.6%`。
4. **FGP 世界/腰带标定倾角偏置与支撑脚打滑**：今日实机出现 `yaw=±180°` 时 `+10.8°` 俯仰偏差（将 `pelvis->torso_x` 推至 `+0.187`，即 `+1.62σ` / P95 `3.35σ` 域外）；保留集 `live_20261122_141319` `cal_id=2` 甚至存在 `-22° ~ -31°` 安装倾角；且支撑脚水平滑移达骨盆速率 `~45%`。
   - **V2 对策**：[`FGPSMPLStreamV2._condition_sample()`](file:///home/yu/projects/agibot-X2-A3-Q1-model/script/q1_fgp_control_v2/q1_fgp_smpl_v2.py#L196-L326) 在静止直立 Calm Gate（`<16°` 躯干、`w_xy < 0.7 rad/s`）下估计慢变 `torso_bias_xy` 与 `pelvis_rel_bias_xy`，**仅修正上身 `_UPPER_BODY_BIS` 与骨盆姿态，保持重力对齐双腿 `1..6` 不弯曲**；同时基于最低脚踝接触权重（`vel_stance_beta=0.60`、`deadband 0.025..0.060 m/s`）抑制地面滑移漂移。
5. **压测新发现：桥接热切换调用 `self.tracker.reset(q)` 清空历史与支撑 PD 力矩**：[`q1_ros2_lowlevel_bridge.py`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/code/q1_ros2_lowlevel_bridge.py#L313-L317) 在 `planner <-> human_smpl` 热切换及重锚时调用 `self.tracker.reset(q)`，在半空中清空 `10` 帧 `obs_hist` 并重启 `engage_steps=5`（`100 ms` 目标向实测角 `q` 衰减）。热切换必须仅重置 `CGFA`/参考系（`reset_tracker=False`），保留 `tracker.history` 连续。

---

## 三、 5 套件 MuJoCo 闭环消融数据（[`q1_fgp_control_v2_eval_20261122.json`](file:///home/yu/projects/agibot-X2-A3-Q1-model/data/manifests/q1_fgp_control_v2_eval_20261122.json)）

以下数字全部逐项核对自 [`q1_fgp_control_v2_eval_20261122.json`](file:///home/yu/projects/agibot-X2-A3-Q1-model/data/manifests/q1_fgp_control_v2_eval_20261122.json)（2 ms MuJoCo、单线程 CPU ORT）：

### 3.1 Suite 1：今日真实 FGP 录制（`140.92 s` 真实到达时刻 + 12 次实机断流闪烁）

| 组别 | 存活时间 | Tilt p50 / p95 / max | Pitch p50 | 去偏躯干误差 | 去偏手臂误差 (corr) | 去偏腿部误差 (corr) | 根速度相关性 `v_xy_corr` | 6 次抬脚峰值 (cm) | Reanchor / Null / PD OOB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `A0_ProdOld` | **18.84 s（18.82s 跌倒）** | 3.38° / 6.79° / **70.46°** | +1.70° | 2.92° | 9.97° (+0.687) | 7.34° (+0.059) | 0.257 | `[5.12]` | 2 / 27 / 0 |
| `A1_SmoothV2` | 141.00 / 141.00 s | 11.48° / 18.57° / 45.30° | **+11.26°** | 10.64° | 14.14° (+0.804) | 8.30° (+0.349) | 0.205 | `[5.17, 4.26, 8.07, 1.64, 3.47, 5.77]` | 0 / 0 / 0 |
| **`A2_AdapterV2_V2ONNX`（入选）** | **140.92 / 140.92 s** | **2.12° / 6.76° / 15.92°** | **+0.53°** | **2.53°** | **7.22° (+0.893)** | **7.47° (+0.372)** | **0.573** | **`[5.00, 5.36, 7.19, 0.84, 4.31, 5.90]`** | **0 / 0 / 0** |
| `A3_AdapterV2_FGPV2ONNX` | 140.92 / 140.92 s | 2.08° / 7.06° / 14.12° | +0.34° | 2.43° | 6.49° (+0.909) | 7.49° (+0.359) | 0.538 | `[5.13, 5.63, 3.58, 1.04, 4.43, 3.84]` | 0 / 0 / 0 |

### 3.2 Suite 2：保留集未见真实 FGP 录制（[`live_20261122_141319/smpl.jsonl`](file:///home/yu/projects/FGP_mocap/results/live_20261122_141319/smpl.jsonl)）

| 片段 | 组别 | 存活时间 | Tilt p50 / max | Pitch p50 | 手臂误差 (corr) | 腿部相关性 `leg_corr` | PD OOB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cal2_bias_neg22deg`（`-22°` 安装偏置） | `A0_OldAdapter_V2ONNX` | **76.04 / 104.12 s（跌倒）** | 2.81° / **74.21°** | +0.30° | 8.78° (+0.860) | +0.345 | 0 |
| `cal2_bias_neg22deg`（`-22°` 安装偏置） | **`A2_AdapterV2_V2ONNX`（入选）** | **104.12 / 104.12 s** | **2.87° / 11.65°** | +0.90° | 8.62° (+0.894) | **+0.663** | **0** |
| `cal2_bias_neg22deg`（`-22°` 安装偏置） | `A3_AdapterV2_FGPV2ONNX` | 104.12 / 104.12 s | 2.60° / 12.12° | +0.33° | 7.52° (+0.898) | +0.690 | 0 |
| `cal5_dynamic_112s`（动态长序列） | `A0_OldAdapter_V2ONNX` | **103.48 / 112.34 s（跌倒）** | 2.63° / **66.35°** | +0.26° | 11.30° (+0.875) | +0.301 | 0 |
| `cal5_dynamic_112s`（动态长序列） | **`A2_AdapterV2_V2ONNX`（入选）** | **112.34 / 112.34 s** | **2.64° / 7.54°** | +0.82° | 11.67° (+0.835) | +0.313 | **0** |
| `cal5_dynamic_112s`（动态长序列） | `A3_AdapterV2_FGPV2ONNX` | 112.34 / 112.34 s | 2.09° / 10.10° | +0.41° | 11.47° (+0.834) | +0.315 | 0 |

### 3.3 Suite 3：8 项高难非行走 / 步态原语

| 原语（总步数 / 参考脚高差） | `A2_V2ONNX`（入选）：完成 / TiltMax / 腿误 / 臂误 / 脚高差 | `A3_FGPV2ONNX`：完成 / TiltMax / 腿误 / 臂误 / 脚高差 |
| --- | --- | --- |
| `kick` (250 / 24.71 cm) | **250/250**, 18.03°, 17.72°, 17.87°, 7.53 cm (`oob=0`) | 250/250, 16.36°, 17.51°, 16.36°, 9.70 cm (`oob=0`) |
| `syn_squat_deep` (250 / 0.0 cm) | **250/250**, **42.62°**, 9.96°, 13.71°, 16.48 cm (`oob=0`) | 250/250, 50.18°, 10.03°, 13.80°, 20.06 cm (`oob=0`) |
| `lock_step` (250 / 5.47 cm) | **250/250**, 11.31°, 8.66°, 17.85°, 4.99 cm (`oob=0`) | 250/250, 9.69°, 8.24°, 15.32°, 4.35 cm (`oob=0`) |
| `turn_walk_360` (250 / 8.93 cm) | **250/250**, 9.18°, 5.99°, 4.12°, 8.79 cm (`oob=0`) | 250/250, 10.09°, 5.27°, 4.14°, 8.88 cm (`oob=0`) |
| `static_stand_50hz` (500 / 0.0 cm) | **500/500**, 2.50°, 4.67°, 12.45°, 0.79 cm (`oob=0`) | 500/500, 2.55°, 4.59°, 12.18°, 0.82 cm (`oob=0`) |
| `vr3pt_arm_wave_50hz` (600 / 0.0 cm) | **600/600**, 5.59°, 9.62°, 8.96°, 4.01 cm (`oob=0`) | 600/600, 4.64°, 7.14°, 9.73°, 2.34 cm (`oob=0`) |
| `ho_phuma_walk_009_50hz` (199 / 5.59 cm) | **199/199**, 22.84°, 27.96°, 12.16°, 10.14 cm (`oob=0`) | 199/199, 14.27°, 26.96°, 14.22°, 6.61 cm (`oob=0`) |
| `ho_lafan_walk2_subject3_50hz` (399 / 8.10 cm) | **399/399**, 11.42°, 10.16°, 6.57°, 8.49 cm (`oob=0`) | 399/399, 11.13°, 10.81°, 6.37°, 8.33 cm (`oob=0`) |

### 3.4 Suite 4：`H002-SONIC v2` 50 保留片段（名义 + 20 极限扰动）与 8 条零 IK 原始 BVH（ONNX 选型依据）

| ONNX 分词器版本 | 50-HO 存活 / Return / BodyRMSE / Slip | 20-STR (0.55x+推力) 存活 / BodyRMSE / Slip | 8-BVH 零 IK 存活 / RootXY RMSE |
| --- | --- | --- | --- |
| **`V2_HumanSMPL_ONNX` (`7dab4d9d...`, 入选)** | 94.0% / 0.760 / **5.78 cm** / **196.2 mm/s** (`oob=0%`) | 85.0% / **7.17 cm** / **263.8 mm/s** (`oob=0%`) | **100.0% (8/8)** / **44.68 cm** (`oob=0%`) |
| `FGP_V2_HumanSMPL_ONNX` (`318b3ae5...`, 未晋级) | 100.0% / 0.761 / 6.20 cm / 223.3 mm/s | 100.0% / 9.22 cm / 350.2 mm/s | **87.5% (7/8, `walk2` `267/280` 跌倒)** / 46.24 cm |

### 3.5 Suite 5：`40 s` 连续热切换（`Planner <-> HumanSMPL` + `0.70x` 低摩擦 + `20 ms` 延迟 + 4 闪烁 + `400 ms` 真断流）

| 组别 | 40 s 全程完成 | 存活时间 | Tilt p50 / p95 / max | `18.0–20.0 s` 断流恢复窗口最大倾角 | PD OOB |
| --- | --- | ---: | ---: | ---: | ---: |
| `A0_ProdOld` | False | **7.50 s（7.48 s 跌倒，首次 `7.00 s` 单帧闪烁触发）** | 3.82° / 10.61° / **67.51°** | N/A（未活到 18 s） | 0 |
| **`A2_AdapterV2_V2ONNX`（入选）** | **True** | **40.00 / 40.00 s** | **2.48° / 7.81° / 16.86°** | **6.57°** | **0** |
| `A3_AdapterV2_FGPV2ONNX` | True | 40.00 / 40.00 s | 2.33° / 5.94° / 12.25° | 5.40° | 0 |

---

## 四、 部署 Agent 精确接入指南（面向 [`q1_deploy_v2_candidate_20261122`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/)）

### 4.1 增量包拷贝与配置合并清单

解压增量包 `/home/yu/projects/SuperMimicHP/runs/q1_fgp_human_smpl_v2_20261122/q1_fgp_control_addon_v2_20261122.zip`（或直接使用已解压校验目录 `deploy_bundle_fgp_v2_20261122/`）后，按表合入目标目录：

| 包内相对路径 | 作用与状态 | 目标动作（目标候选目录内） |
| --- | --- | --- |
| `code/q1_fgp_smpl_v2.py` | 新增 `FGPSMPLStreamV2`（断流宽限、上身去偏、防打滑速度条件化） | **新增复制**至 `code/q1_fgp_smpl_v2.py` |
| `code/q1_fgp_client_v2.py` | 新增 `FGPHttpSourceV2`（`pop_paced` + `status()` 启动可用）与 `Q1FGPFallbackGuard` | **新增复制**至 `code/q1_fgp_client_v2.py` |
| `code/smoke_test_fgp_v2_addon.py` | 零 PyTorch/MuJoCo 独立冒烟回归校验脚本 | **新增复制**至 `code/smoke_test_fgp_v2_addon.py` |
| `onnx/q1_tok_human_smpl.onnx` | `sha256: 7dab4d9d...`（与现 V2 候选逐字节一致） | 核对 SHA-256 一致，无需改动 |
| `onnx/q1_h002_token_head.onnx`, `q1_tok_planner_cmd.onnx`, `q1_tok_robot_pose.onnx` | 冻结兄弟模型，只读随包自检 | 与现候选完全一致，无需覆盖 |
| `code/q1_v2b_runtime.py`, `q1_cgfa_obs_np.py`, `q1_kinematics_np.py`, `robot_model/supermimic.xml` | 现部署公共底座只读拷贝 | 与现候选一致，无需覆盖 |
| `config/q1_deploy_config.json` | 新增 `"fgp_v2"` 参数记录块，`onnx_files["q1_tok_human_smpl.onnx"]` 保持 `7dab4d9d...` | 将包内 `"fgp_v2"` 字段合并进目标 `config/q1_deploy_config.json` |

### 4.2 精确最小代码补丁：[`code/q1_ros2_lowlevel_bridge.py`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/code/q1_ros2_lowlevel_bridge.py)

请按以下 **4 处精准修改**更新 [`q1_ros2_lowlevel_bridge.py`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/code/q1_ros2_lowlevel_bridge.py)：

1. **导入并初始化 `FGPHttpSourceV2` 与 `Q1FGPFallbackGuard`**（行 `38`、行 `189-201`、行 `573-582`）：
   - 行 `38` 替换为：
     ```python
     from q1_fgp_client_v2 import FGPHttpSourceV2, Q1FGPFallbackGuard
     ```
   - 行 `189-197` 实例化改为（保留旧 `FGPHttpSource` 仅作回滚备份）：
     ```python
     self.fgp = (
         FGPHttpSourceV2(
             fgp_url,
             dt=float(self.cfg["dt_s"]),
             standing_height_m=fgp_standing_height,
             fresh_ms=220.0,
             hold_grace_ms=220.0,
             max_buffer_ticks=3,
             on_sample=self._record_fgp,
         )
         if self.human_smpl else None
     )
     self.fgp_guard = Q1FGPFallbackGuard()
     ```
   - 在 `_reset_anchor()`（行 `574` `_reanchor_reference` 之后）增加：
     ```python
     self.fgp_guard.reset("planner")
     ```

2. **用 `pop_paced(now_ns=now_ns)` 替代 `get_latest()`，并统一 `220 ms` 新鲜度判据**（行 `683-686`）：
   ```python
   now_ns = time.monotonic_ns()
   if self.fgp is not None:
       queued, _, _ = self.fgp.pop_paced(now_ns=now_ns)
   else:
       queued = None
   live_fgp = queued is not None
   ```
   *(若同时启用 `--enable-recovery` (`recovery_v2=True`)，行 `711` 建议改为 `stamp_ns = now_ns if live_fgp else (None if self.human_smpl else now_ns)`，从而与 `FGPHttpSourceV2` 的 `220 ms` 保持宽限协同工作：短闪烁由平滑保持渡过而不提前触发离线接管，超过 `220 ms` 真实断流则立即交出控制权或安全回退。)*

3. **在 `base_step()` 内强制使用 `Q1FGPFallbackGuard` 防范 `fgp=null` 混用坐标系**（行 `690-702`）：
   ```python
   def base_step():
       branch, need_reanchor = self.fgp_guard.resolve_branch(active_mode, live_fgp)
       if branch == "human_smpl":
           return self._human_smpl_step(R, omega, q, qd, queued.tick)
       if need_reanchor:
           self._reanchor_reference(R, omega, q, qd)
       return self._base_step(
           R,
           omega,
           q,
           qd,
           neutral=(active_mode == "human_smpl" or self.recovery_offline),
       )
   ```

4. **移除半空热切换/重锚时的 `self.tracker.reset(q)`**（行 `313-317`、行 `637-638`、行 `695`）：
   - `_apply_pending_mode()`（行 `313-317`）：
     ```python
     if requested == "human_smpl":
         if queued is None or not live_fgp:
             ...
             return
         self._reset_fgp_segment(R, omega, q, qd, queued.tick, reset_tracker=False)
     else:
         self._reanchor_reference(R, omega, q, qd)
         self.fgp_guard.reset("planner")
     ```
     **禁止调用 `self.tracker.reset(q)`！**
   - `_human_smpl_step()`（行 `637-638`）：改为：
     ```python
     if tick.segment != self.fgp_segment:
         self._reset_fgp_segment(R, omega, q, qd, tick, reset_tracker=False)
     ```
   - `self.tracker.reset(q)` **仅允许保留在初始上电 `_reset_anchor()`（行 `575`）中**。

### 4.3 合并后本地隔离校验命令（零 PyTorch/MuJoCo）

合入 [`q1_deploy_v2_candidate_20261122`](file:///home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/) 后，请在该目录下执行（合并目录无独立 `SHA256.json`，因此**不带** `--require-manifest`）：
```bash
cd /home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yu/projects/SuperMimicHP/.venv/bin/python -I code/smoke_test_fgp_v2_addon.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yu/projects/SuperMimicHP/.venv/bin/python code/smoke_test_deploy_bundle.py
```
确认输出包含 `FGP_V2_ADDON_SMOKE_PASS`、`oob_total: 0`、`torch_loaded: false`、`mujoco_loaded: false`、`fallback_guard_verified: true`。

---

## 五、 可选 FGP 服务端防抖补丁、真机四向标定 SOP 与回滚方案

### 5.1 可选 FGP 服务端 2 行防抖补丁（仅在非采集停机窗口施工）

当前实时衣服服务进程运行于 [`/home/yu/projects/FGP_mocap`](file:///home/yu/projects/FGP_mocap/)。本增量包通过 `outage_grace_ms = 220.0` 已能在客户端完全吸收单帧丢包抖动；如需同时从源头消除 30 Hz 单拍 `outage` 日志风暴，可在空闲维护窗口对 [`fgp_linux/service.py`](file:///home/yu/projects/FGP_mocap/fgp_linux/service.py#L282-L287) 做 2 行微调：
- 将行 `284` 的 `if calibration is not None and missing_ticks == 1:` 改为 `missing_ticks >= 4:`（并保持行 `286` `missing_ticks == 4` 时才 `new_generation("outage")`），使单拍偶发丢帧先进入短暂保持而不广播 `outage`。

### 5.2 真机四向（`0° / 90° / 180° / -90°`）衣服标定上机验收 SOP

1. **上机前静态立正锁定 `2 s`**：穿戴好 FGP 衣服后，操作员自然竖直立正静止 `~2 s`，待 `FGPSMPLStreamV2` Calm Gate（躯干倾角 `<16°` 且横滚俯仰角速度 `<0.7 rad/s`）收敛残余俯仰/横滚安装偏置。
2. **四向转体对比只读检查（建议先跑 `--fgp-readonly`）**：分别面朝 `0°`、`90°`、`180°`、`-90°` 各静立 `3 s` 并小步踏步，观察日志中实机响应无单向固定前倾且断流闪烁时不出现 `reanchor`。
3. **吊架保护切换验收**：`--keyboard-switchable` 模式下先站稳，按 `t` 切入 `human_smpl` 做小幅度挥臂、慢走及单腿轻抬，再按 `p` 切回 `planner` 验证无半空力矩跌落；确认全程退出阻尼与 `PASSIVE_DEFAULT` 正常。

### 5.3 一键回滚方案

如实测需退回改动前版本，只需恢复修改前的 `q1_ros2_lowlevel_bridge.py` 与 `q1_deploy_config.json`（`onnx/q1_tok_human_smpl.onnx` `7dab4d9d...` 未被修改）：
```bash
cd /home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122
cp code/q1_ros2_lowlevel_bridge.py.bak_pre_fgp_v2 code/q1_ros2_lowlevel_bridge.py
cp config/q1_deploy_config.json.bak_pre_fgp_v2 config/q1_deploy_config.json
```

---

## 六、 部署包完整性校验与隔离冒烟实测摘要（构建验证通过）

以下数据由 [`build_fgp_deploy_addon_v2.py`](../script/q1_fgp_control_v2/build_fgp_deploy_addon_v2.py) 自动构建、校验压缩包并用零 PyTorch/零 MuJoCo 的隔离进程 `python -I` 实测生成；完整机器可读记录见 [`q1_fgp_deploy_addon_v2_manifest_20261122.json`](../data/manifests/q1_fgp_deploy_addon_v2_manifest_20261122.json)（`/home/yu/projects/Q1_deploy/` 与 `/home/yu/projects/FGP_mocap/` 全程只读未修改）：

- **压缩包与校验和**：
  - **Zip 路径**：`/home/yu/projects/SuperMimicHP/runs/q1_fgp_human_smpl_v2_20261122/q1_fgp_control_addon_v2_20261122.zip`（`36,457,648` bytes）
  - **Zip SHA-256**：`824c2eeec9b549536966b0c27711cf39be9b7f52bb3f9205dc2105325feeef7f`
  - **SHA-256 文件**：`/home/yu/projects/SuperMimicHP/runs/q1_fgp_human_smpl_v2_20261122/q1_fgp_control_addon_v2_20261122.zip.sha256`（实测 `sha256sum -c` 输出：`q1_fgp_control_addon_v2_20261122.zip: OK`）
  - **解压目录**：`/home/yu/projects/SuperMimicHP/runs/q1_fgp_human_smpl_v2_20261122/deploy_bundle_fgp_v2_20261122/`
  - **入选模态变体**：`A2_adapter_v2_frozen_v2_onnx`

- **包内全部 13 个文件 SHA-256 清单**：

| 包内相对路径 | SHA-256 |
| --- | --- |
| `DEPLOY_HANDOFF.md` | `8df7c334d21163770991e9d66f619d5d9d911bef2a02a2c3e381705840afd42d` |
| `code/q1_fgp_smpl_v2.py` | `5a53e4248c940a0aca296fe612228e22b5c854f6d54dcd1952cd1489ab9001b6` |
| `code/q1_fgp_client_v2.py` | `84ff596cbf587eff41421c1b1fbfd37af26eb93be0052451a2be6d1962b2cfef` |
| `code/smoke_test_fgp_v2_addon.py` | `88cd975ead45df23a152549781bf65d4e4cba47c8697ed0b60bb407f05850913` |
| `code/q1_cgfa_obs_np.py` | `2f03e42598d3270094f18729bdd316712d7ad0a3505815d299f5e9383c2ca99c` |
| `code/q1_kinematics_np.py` | `7c9a33862fd5e287d74d93b20b7cf2d2f5995e5dedb00eb4ecdb70910df7a6b5` |
| `code/q1_v2b_runtime.py` | `f913fee386d177f95df00696cac4e140b22643d5882283274fb6be961f198ea8` |
| `config/q1_deploy_config.json` | `84b2cea4876a3769a6f7901172d58b0bc38b305df6e975aa1ef2b28cb108d414` |
| `onnx/q1_h002_token_head.onnx` | `e0455c0b50db1c1fdc8aed1e0d283820d3fd64174b261093d99dc4430ff3e80f` |
| `onnx/q1_tok_human_smpl.onnx` | `7dab4d9d056e797c1ae8631d6c15ffd5fc3324234c86b821b253fcbb46adfb0e` |
| `onnx/q1_tok_planner_cmd.onnx` | `ce0aa0a49666515e0b47dd200995fecc8980cd3bcdd41149fbc4824fb92679e5` |
| `onnx/q1_tok_robot_pose.onnx` | `9ee7a429347d0c9076657c6bb47e4635ee6b149cdc2504370fbfcecf09e6e6d9` |
| `robot_model/supermimic.xml` | `47bfa08baed1a3482e575a6489b29c2fc1c720ba1f359f636a76c40a5e9da9df` |

- **隔离 `python -I` 冒烟实测结果**（单线程 CPU ONNX Runtime，`torch_loaded = false`、`mujoco_loaded = false`、`oob_total = 0`）：
  - **独立解压模式（`--require-manifest`，校验 `SHA256.json` 全部 `13` 文件）**：
    - `human_smpl`: `p50 = 0.6852 ms` / `p95 = 0.8836 ms`
    - `planner_cmd`: `p50 = 0.4994 ms` / `p95 = 0.6577 ms`
    - 启动即新鲜 `status()["fresh"] = true`；`transient_outages_ignored = 3`、`final_segment = 1`；`debiased_pitch_deg = -0.0°`、`debiased_torso_x = 0.0`、`antiskate_vx_m_s = 0.0194 m/s`；`fallback_guard_verified = true`
  - **模拟合入模式（移除 `SHA256.json`，仅校验 `q1_deploy_config.json` 与 4 个 ONNX 哈希）**：
    - `human_smpl`: `p50 = 0.6900 ms` / `p95 = 0.9395 ms`
    - `planner_cmd`: `p50 = 0.5037 ms` / `p95 = 0.6834 ms`
    - 全部功能断言同样通过（`transient_outages_ignored = 3`、`final_segment = 1`、`debiased_pitch_deg = -0.0°`、`debiased_torso_x = 0.0`、`antiskate_vx_m_s = 0.0194 m/s`、`fallback_guard_verified = true`）
