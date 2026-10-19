# G0 — 参考可达性审计（门 2.0 硬前置）最终报告

合同：GATE2_CONTRACT.md v1.0 §0 · 平台：**ZHY MuJoCo 线**（platform: zhy）· 2026-10-19
Checkpoint：035_step200000（E26 同款，data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt）
执行：exp-operator · 30/30 rollouts 通过，零训练零候选 · code @ a7c15d7

## 一句话结论

**zzx_111505 fall@5.66 s 按预登记规则触发 infeasible_suspect**（3/3 评测批次在同一 ±2 s 窗内终止；窗口顶关节 right_hip_roll）；同时发现**评测器逐位确定**（30 rollouts metrics SHA 全同）——"3 seed"是同协议复现而非独立采样，底噪 σ≡0，abs 容差维持 max(0.010, 3σ)=0.010。

## 1. 协议忠实度（首要验收键）

- 复用 E26 精确协议（csv_source_fps=30、frame_stride=1、batch_once、action_delay_ms=0、同 MJCF、同 checkpoint）。
- **cloth6 mean full = 0.1413798 rad —— 与 E26 registration 记录值逐位一致**（协议复现零漂移）。
- zzx 逐段：full 0.326613（E26 记录 0.32661）、before_first_fall 0.242819（HP11 记录 0.2428）——双平台历史记录同时复现。

## 2. zzx_111505 判读（首个复核对象，明确回答）

**答案：infeasible_suspect 成立（按预登记规则），建议处置 gate_excluded_window，参考修订列候选。**

| 证据 | 数值 |
| --- | --- |
| 终止点 | 3/3 批次 first_fall = 5.66 s（逐位相同；E26 fall_tick=283 一致） |
| 窗口 [3.66, 7.66 s] 顶关节 | right_hip_roll 铰链误差 **1.096 rad**（bff 全程 0.748，为其余五布料顶关节的 ~6–8 倍） |
| 同段参考特性 | 关节速度 RMS 0.5 rad/s（温和）；支撑相 L 0.27 / R 0.36（行走单支撑段，非特殊姿态） |
| 全关节 bff 排行 | right_hip_roll 0.748 ≫ right_shoulder_pitch 0.373 > left_wrist_yaw 0.339 |

判读：发散集中在**单一髋滚转关节**，参考速度温和、步态相正常 → 不是"速度不可行"签名，而是**参考该段 right_hip_roll 轨迹与 035 能力失配**（035 无法执行 → 群体饱和点）。注意：鉴于确定性评测器，"3 seed"证据本质是同协议逐位复现；独立采样意义下的群体性验证需注入观测噪声（见 §3 决策项）。

**处置建议（二选一，供 lead 拍板）**：推荐 **gate_excluded_window** 登记 zzx [3.66, 7.66] s（该窗不参与 H 与 guard 统计，报告如实记录；快门第 3 动作按 §6-2 预登记换 fjw_140008）；"修参考（right_hip_roll 段重定向/平滑）"列入参考合同修订候选，走新合同。若 lead 选修参考，则 zzx 全动作保留、以修订参考重新基线。

## 3. 底噪表（门 2.0 abs 容差输入）——含关键发现

**发现：ZHY 评测管线逐位确定。** joint-reference-mode 关闭（E26 协议）+ 确定性 rollout ⇒ 30 rollouts 的 metrics.json SHA256 **全部相同**（seed 0/1/2 仅名义区分，joint_noise_seed 在 std=0 时不生效）。因此：

| 指标 | 3 seed σ | 3σ | abs 容差 = max(0.010, 3σ) |
| --- | --- | --- | --- |
| cloth6_rmse（full） | 0.0 | 0.0 | **0.010 rad**（占位值维持） |
| official4 各段 rmse | 0.0 | 0.0 | 0.010 |
| world_drift_ratio（full/bff） | 0.0 | 0.0 | 0.02（HP11 口径维持） |
| per_joint_guard_max（full/bff） | 0.0 | 0.0 | 0.010 |

**门 2.0 规格决策项（上报 lead）**：若门 2.0 需要真·独立重复方差，须在 VerificationSpec 显式启用 joint_noise_std>0（评测器已支持 --joint-noise-std/--joint-noise-seed）；当前 E26 口径下 repeats=3 不提供方差信息，只提供复现验证。

## 4. per_joint_guard_max 逐关节基线表

工件：tables/per_joint_baseline.json —— 10 动作 × 29 关节 ×（full + before_first_fall）RMSE（3 批均值，逐位确定故均值=单值）。样例（bff 顶关节）：
- zzx_111505：right_hip_roll **0.748**（异常段）
- 其余九动作 bff 顶关节 0.05–0.21 rad（健康带）

## 5. 终止点分布

- **唯一倒地动作：zzx_111505（3/3 批次 @5.66 s）**；其余 9 动作 27/30 rollouts 全程无终止（含官方 4 段全部完整走完）。
- official4 步数（ZHY 实测/段）：001_walk_front 6607、029_rotate_place 10415、048_balance 2549、050_march 3532（合计 23103 步/批次；HP11 记录的 5775 为 Isaac 口径，不可比）。

## 6. ZHY 线 official4 首读（平台绑定标定警示）

| 动作 | joint rmse（full） | world_drift_ratio（full） |
| --- | --- | --- |
| 048_balance_right_singleleg | 0.0516 | 0.994 |
| 050_march_place_slow | 0.0566 | **1.491** |
| 001_walk_front_slow | 0.0817 | 1.018 |
| 029_rotate_place_slow | 0.0850 | **1.143** |

**警示**：march 1.491 / rotate 1.143 超出 HP11 经验带（≤1.10）。GATE2 §1.3 的 world_drift_ratio guard（base 1.00–1.03、上限 1.10）**不可直接移植 ZHY 线**——035 本尊在 ZHY 官方长程段即出带。平台绑定条款（禁止跨平台换算）要求 ZHY guard 按本表重标定（per-action 或按动作族分带），lead 裁定。

## 7. Receipts 与工件链

- registration.json（code SHA256: g0_runner/g0_analysis）、batch_seed{0,1,2}_{cloth,official}.json（6/6 passed, 0 失败）、g0_eval_complete.json
- preflight：nvidia-smi 空闲 3×/5s（util 0% / 14 MiB）+ cgroup memory.max = 17,179,869,184（恰 16 GiB）通过
- 原始工件（repo 侧新命名空间，E52/E53 先例）：hechengzhu ZHY data/experiments|logs/a3_zhy_reachability_audit_20261019_G0/（30×metrics/timeseries）
- 台账镜像（yuanqi_ops）：data/experiments/a3_zhy_reachability_audit_20261019_G0/tables/{noise_floor,termination_distribution,per_joint_baseline,rollout_rows,g0_verdict}.json
- 失败归档：failed_launch_archive/1st_batch_seed0_cloth.json（路径守卫修正，预训练期，无污染）
- 代码：g0_runner.py / g0_analysis.py @ a7c15d7；单次运行纪律满足（无重跑）
