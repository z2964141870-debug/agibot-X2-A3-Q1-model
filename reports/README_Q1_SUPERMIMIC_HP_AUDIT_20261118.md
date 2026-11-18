# Q1 SuperMimicHP 零延迟遥操复现与真机鸿沟定量审计

日期：2026-11-18（北京时间）｜节点：`yu-3090`（`i7-12700KF / 46GiB / RTX 3090 24GB`，`/` 余 `810G`）｜分支：`Q1`

> 审计对象：`humanplus-ai/SuperMimicHP`（`feat/x2-embodiment` @ `27c3e2cfa6a29c8d003e3fcd63fbf5dd37e8b190`，发布标签 `v1.0.0-teleop`）与 `PrimebotTech/primebot-q1-urdf`（`9657817ba375f41b0e5d2b62d8d6b72b757cdf4f`）。

---

## 一、Done / To Do

### Done
1. **远端 `SuperMimicHP` 环境与权重复现基座就位**：完成 `/home/yu/projects/SuperMimicHP/.venv`（`Python 3.13.18`、`torch 2.14.1+cu130`、`CUDA=True`、`mujoco 3.11.0`、`unilab 1.5.0`、`unisim-core 2.1.0`），并在远端校验通过 Q1/X2 共 6 个 `v1.0.0-teleop` 预训练权重（`q1_base_soma1k.pt`、`q1_scheme_a_1k.pt`、`q1_scheme_b_66hz_1k.pt` 及对应 X2 权重；注：上游 `tools/download_weights.py` 内嵌的两条 Q1 哈希相对实际 Release `checksums.sha256` 陈旧，本次以 Release 清单校验通过）。
2. **Q1 资产标定与 LAFAN1 重定向质检**：提取官方 STL 网格后执行 `tools/prepare_q1_assets.py --validate-only`，FK 32 次随机试算最大误差 `[1.33e-7, 3.08e-7, 4.09e-8, 9.97e-8]`（`< 2e-5` 门槛），名义根高 `0.42132 m`；将 `walk1_subject1`~`walk4_subject1`（各 240 帧 / 30Hz）通过外置 `GMR`（临时适配其 `load_bvh_file` 三元组返回而不改主仓文件）重定向到 Q1，`tools/check_q1_motions.py` 全部零失败通过（`soft_penalty=0.000844`，`hard_violation=0.0`，`velocity_exceed=0.0`）。
3. **闭环 4 片段测试 + 2x2 视频渲染 + 真机鸿沟定量审计**：生成对比视频 `q1_walk1_teleop_compare.mp4`（`1.1 MB`）、四合一诊断图 `deploy_audit_summary.png`（`221 KB`）及完整指标 JSON `audit_metrics.json`，量化对比 `Oracle (+200ms)`、`Causal ZOH (0ms)`、`Scheme A (0ms, 50Hz)`、`Scheme B (0ms, 66.7Hz)` 与 `root_xy` 特权消融、PD 限位/跳变、足底滑移及抬脚高度。

### To Do
1. **建立纯本体 `78D`（去 `root_xy`）Q1 可部署 Baseline + ONNX Sim2Sim 闭环**：在 `yu-3090` 上以 `observation_mode=legacy`（移除仿真器全球坐标反馈）重训干净 Baseline 并补齐 `export_onnx.py` 与首帧渐变/PD 软限幅沙盒，作为后续硬件落地底座与论文公平对照基线。
2. **在干净无特权基线上锁定 1 个真正有学术壁垒的主创新点开展对照消融**（见第五节战略定位与三个创新候选）。

---

## 二、核心实验结果与定量诊断（4 段 LAFAN1 行走 `walk1_subject1`~`walk4_subject1`，各 8s）

### 1. 零延迟因果闭环对比（Oracle `+200ms` vs. `ZOH 0ms` vs. `Scheme A/B 0ms`）

| 方案 | 未来看帧 | 频率 | 存活率 | NormReturn | Body RMSE (cm) | Root RMSE (cm) | Joint RMSE (°) | 步间跳变 p99 / Max (°) | 单步推理 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `base_oracle` | `+200ms` | 50 Hz | 100% | 0.8894 | 4.35 | 9.75 | 8.41 | 10.65 / 61.60 | 1.83 |
| `base_zoh` | `0ms` (ZOH) | 50 Hz | 100% | 0.8369 | 4.69 | 11.31 | 9.73 | **21.51 / 81.32** | 2.07 |
| **`scheme_a`** | **`0ms` (因果历史)** | 50 Hz | **100%** | **0.8925** | **4.35** | **9.36** | 8.50 | **9.66 / 40.45** | **2.09** |
| **`scheme_b`** | **`0ms` (因果历史)** | **66.7 Hz** | **100%** | 0.8919 | **4.34** | **8.72** | **8.37** | 10.79 / 40.28 | **2.15** |

- 在 `walk1_subject1` 2x2 对比视频渲染中额外测得 `SchemeA+WM`（在线 1 步世界模型梯度细化）：`NormReturn=0.9085`、`Body RMSE=3.68 cm`、`Root RMSE=3.94 cm`，但 CPU 平均单步延迟升至 **`17.86 ms`**（逼近 `20 ms` 50Hz 控制周期上限），不适合直接用于真机实控；纯 `Scheme A / B` 在 `~2.1 ms` 推理开销下即可持平或小幅反超 `+200ms Oracle`。

### 2. 真机鸿沟一：`root_xy` 仿真器特权通道消融（最大部署阻塞点）

`SuperMimicHP` 对 Q1/X2 默认开启 `observation_mode="root_xy"`，观测末 2 维直接注入仿真器真值全球水平偏差 `heading_t @ (ref_root_p - sim_root_p)[:2]`（仅 G1 使用纯本体 `legacy` 模式）。消融实验如下：

| 配置 (`obs[:, 78:80]`) | 存活率 | NormReturn | Body RMSE (cm) | **Root RMSE (cm)** | 最大 PD 超限 (°) | 力矩利用率 p99 | 支撑滑移均值 (mm/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `base_oracle` | 100% | 0.8894 | 4.35 | 9.75 | 27.31° | 54.8% | 162.6 |
| **`base_zeroed` (置零)** | **75%** (`walk2` 第 296 步倒地) | 0.7087 | 6.53 | **87.51** | **92.27°** | **84.5%** | **280.7** |
| **`base_drift` (`+0.15m/s` 漂移)** | 100% | 0.7313 | 6.29 | **82.64** | 19.70° | 64.6% | 281.0 |
| `scheme_a` | 100% | 0.8925 | 4.35 | 9.36 | 18.98° | 56.8% | 155.8 |
| **`scheme_a_zeroed` (置零)** | **75%** (`walk2` 第 292 步倒地) | 0.7370 | 5.80 | **74.05** | **93.18°** | **86.2%** | **250.4** |
| **`scheme_a_drift` (`+0.15m/s` 漂移)** | 100% | 0.7431 | 6.05 | **79.32** | 25.42° | 64.6% | 294.7 |

**结论**：发布版 Q1 权重高度依赖仿真器全球 XY 纠偏信号，无外部动捕真值时不能直接部署上机；重训移除 `root_xy`（切换纯本体 `78D` `legacy` 观测）是硬件部署的必要前置条件。

### 3. 真机鸿沟二：PD 安全限位、首帧跳变与足地接触指标

- **首帧与限位越界**：即使在 Oracle 真值下，相对标称 `default_q` 的第 0 步目标跳变峰值达 **`76.36°~79.07°`**（相对动作首帧 `q_init` 为 `12.8°~17.8°`）；正常闭环中 `0.33%~0.46%` 的 `(step, joint)` 目标超过 MJCF 硬件限位，最大超限 **`18.98°`（Scheme A）~ `27.31°`（Base）**，集中于 **`right_knee`、`left/right_ankle_pitch`、`left/right_ankle_roll`**（实机上层导出与 C++ 执行器必须做首帧插值与软限幅）。
- **摆动抬脚与支撑滑移**（通过 `CollisionGeometry` 直接测脚底板网格最低点）：参考动作自身为 `swing_p95=25.1 mm`、`stance_slip_mean=66.1 mm/s`；策略闭环测得 `base_oracle` `15.7 mm / 162.6 mm/s`、`scheme_a` `19.5 mm / 155.8 mm/s`、`scheme_b` `21.2 mm / 144.2 mm/s`。高频 66.7Hz 在抬脚与抑滑移上略有优势，但均仍显著高于参考滑移基准。
- **开环固定 PD 站立不可行**：`prepare_q1_assets.py` 实证 `stand_pass=false`（3s 内塌陷至 `0.059 m`），硬件需配套保护阻尼/闭环站立状态机。
- **输入接口差距**：当前 `goal` 消费 59D 重定向机器人关节角及差分速度，尚无 SMPL 直出前端；且仓库内缺失 `export_onnx.py` 与 `Sim2Sim` 执行器代码。

---

## 三、复现命令与本地工件清单（`yu-3090`）

```bash
cd /home/yu/projects/SuperMimicHP
MUJOCO_GL=egl .venv/bin/python tools/record_teleop_comparison_video.py \
  --robot q1 --motion src/unisupermimic/assets/motions/q1/lafan1_showcase/walk1_subject1.csv \
  --base-ckpt checkpoints/q1_base_soma1k.pt --scheme-a-ckpt checkpoints/q1_scheme_a_1k.pt \
  --scheme-b-ckpt checkpoints/q1_scheme_b_66hz_1k.pt \
  --out-mp4 runs/deploy_readiness_audit/q1_walk1_teleop_compare.mp4 --max-seconds 6.0 --fps 15

CUDA_VISIBLE_DEVICES="" WARP_DISABLE_CUDA="1" MPLBACKEND=Agg \
  .venv/bin/python runs/deploy_readiness_audit/audit_deploy_gap.py
```

- 证据清单与全部指标 SHA-256：`data/manifests/q1_supermimic_hp_audit_evidence_20261118.json`
- 审计脚本入库副本：`script/q1_supermimic_hp_audit/audit_deploy_gap.py`
- 会话记录：`logs/session_records/q1_supermimic_hp_audit_20261118.json`
- 远端大体积媒体与权重留存路径（仅存于 `yu-3090`，不入 Git）：`/home/yu/projects/SuperMimicHP/runs/deploy_readiness_audit/{deploy_audit_summary.png, q1_walk1_teleop_compare.mp4, q1_walk1_teleop_compare.png, audit_metrics.json}` 与 `/home/yu/projects/SuperMimicHP/checkpoints/*.pt`。

---

## 四、调研结论：零延迟遥操的创新边界，以及“推复现”还是“推创新”

1. **学术界零延迟格局与创新红线**：
   - `SONIC`、`SuperMimic` 原版与 `MorphMimic` 均依赖 `180~200 ms` 未来看帧缓冲；`TWIST / TWIST2`（2025）、`GigaBrain-WBC` 与 `SuperMimicHP (Scheme A/B)`（2026-11）已将“未来教师 → 因果历史潜空间蒸馏 / 66.7Hz 高频因果执行”做成公开基线。
   - **因此，“去 `root_xy` 重训 Q1 Baseline + ONNX 导出部署 + 普通延迟随机化”是必须补齐的工程可部署底座，但不构成充分的论文原创点**（结合本项目 `reports/research/README_IDEAS.md`：普通记忆/融合/延迟随机化已有大量先例，不能单独宣称原创）。
2. **为什么仍要保留最小复现工作作为前置底座？**
   - 参考本仓库 `11/08` Q1 低延迟总结（`Q1_LOW_LATENCY_RECAP_20261109`）：此前在未固化的 32k 旧底座上连续做 20+ 轮 `f0`、辅助下一状态、世界模型蒸馏，虽然离线拟合指标改善 `~36%`，但闭环仍出现 `0/7` 提前终止或全部选 `update 0`。**没有无特权、闭环 100% 稳住的干净 Baseline，任何创新消融都会被底层 Bug 或特权依赖掩盖。**
3. **建议节奏与三大可聚焦主创新点**（最小复现打底 + 同步锁定 1 个主创新点）：
   - **方向 A（直击无动捕漂移与足底滑移死穴）**：接触置信度门控的支撑脚局部锚定零延迟遥操（利用支撑脚短暂局部锚点代替全局 `root_xy`，同时压低无动捕飘移与支撑滑移）；
   - **方向 B（利用团队独有“衣服 FGP + 双目头环”异步多源硬件壁垒，延续 `H001`）**：观测年龄与可见性感知的异步可穿戴全身控制（在 `~23–30Hz` 衣服断流与头环手部遮挡恢复时抑制 `26°~39°` 关节突跳）；
   - **方向 C（发挥 SuperMimic 可微世界模型优势并消除 `17.9ms` 在线开销）**：基于可微世界模型接触/失稳动作雅可比 $\nabla_a J$ 的各向异性因果蒸馏（只在高度敏感的支撑腿关节相位严控因果误差）。
