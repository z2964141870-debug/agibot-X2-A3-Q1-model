# Q1 柔和自主 Recovery V2：全姿态独立起身训练与保留种子验收报告

2026-11-22，yu-3090（实验阶段 `2026-11-21` 至 `2026-11-22`）。状态：**核心功能、时限与回归项达标，但 3 项所持控制权严格柔和度峰值 gate 未全过，严格工程验收未完全通过，暂不替换真机部署模型**。

本轮工作严格延续并升级 [README_Q1_SOFT_RECOVERY_20261121.md](README_Q1_SOFT_RECOVERY_20261121.md)，目标契约见 [acceptance_plan.json](../script/q1_soft_recovery/acceptance_plan.json)。所有训练、标定与验收均在 `2 ms` MuJoCo `3.11.0` 物理、`20 ms`（50 Hz）控制、`500 Hz` 发送器（`default_q ± 2.0 rad`，单 `2 ms` 步最大增量 `0.2 rad`）、`11 ms` 假设观测延迟、冻结限矩 PD 及 `96` 对显式前臂/腕部-腿部碰撞下执行。`/home/yu/projects/Q1_deploy/`、`eval_core.py`、`evaluate.py`、`q1_autonomous_recovery.py` 及 `acceptance_plan.json` 全程只读未改。

---

## 一、 严格验收总览（保留未见种子，一次性盲测）

使用独立脚本 [run_final_acceptance.py](../script/q1_soft_recovery/run_final_acceptance.py) 锁定单一候选 `/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/unified_v3_08/expert.onnx` 与 `CalmUprightGate`（`h_min=0.355`, `rms_leg_max=0.35`, `calm_ticks=4`, `blend_ticks=16`），在完全未参与调参的保留种子集上一次性执行全部 120 静态倒姿、30 受推倒地、回归校验及原速视频渲染，结果存于 `/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/final_acceptance_20261121/final_acceptance_report.json` 及 [final_acceptance_manifest_20261121.json](../script/q1_soft_recovery/final_acceptance_manifest_20261121.json)。

| 验收条目 | 契约要求 | 实测结果 | 判定 |
| --- | --- | --- | --- |
| 静态未见总成功率（`84101–84105`） | `>= 114 / 120`（95%） | **`116 / 120`（96.7%）**；同时柔和 `114 / 120`（95.0%） | **PASS** |
| 分姿态及时成功 | 俯/仰/侧各 `>= 36 / 40`（90%） | 俯卧 **`40/40`**、仰卧 **`38/40`**、侧卧 **`38/40`** | **PASS** |
| 静态站稳确认时限 | 接管后 `<= 15 s` | 全部 116 例成功试次 **`<= 14.42 s`**（超时失败 `84101/T15` 单独记失败） | **PASS** |
| 静态合法初态与误交还 | 120/120 合法起态、误交还 `= 0` | **`120 / 120`** 合法起态（初态穿地 `<= 3.81 mm`，臂腿 `0.00 mm`；保留 1 次初始化剔除记录），误交还 **`0`** | **PASS** |
| 受推实际倒地数（`84111–84113`） | 30 次中 `>= 20` 次倒地 | **`30 / 30`** 实际倒地 | **PASS** |
| 受推恢复率与先倒地检查 | 恢复率 `>= 90%`，推前倒地 `= 0`，误交还 `= 0` | **`30 / 30`（100.0%）**恢复交还并稳站 >= 1s，推前倒地 **`0`**，误交还 **`0`** | **PASS** |
| 所持控制权峰值（含失败试次）——关节角速度 | `peak_qd <= 12 rad/s` | 静态 **`120/120 <= 11.70`（PASS）**；受推总持有期 **`15/30 <= 12`**，最坏 **`22.91 rad/s`**（`84113/T06` 于 `cap+0` 继承倒地初速度） | **FAIL** |
| 所持控制权峰值（含失败试次）——地面穿透 | `ground_mm <= 10 mm` | 静态 **`119/120 <= 10`**（仅 `84102/T23 = 10.06 mm`）；受推 **`26/30 <= 10`**，最坏 **`11.37 mm`**（`84113/T05 @ cap+5`） | **FAIL** |
| 所持控制权峰值（含失败试次）——臂腿穿透 | `arm_leg_mm <= 2 mm` | 静态 **`119/120 <= 2`**（仅 `84102/T18 = 2.43 mm` @ `tick=0`）；受推 **`22/30 <= 2`**，最坏 **`12.21 mm`**（`84111/T06 @ cap+8`） | **FAIL** |
| 控制权/离线/行走/二次受推回归 | 离线持控、无遥控推进、平地零接管、二次倒地可恢复、陈旧/非单调时钟拒绝、单次 reanchor | `offline_84101` `3/3`、`flat_84101` `0` 接管、`double_84111` **`5/5`** 双阶段恢复、`check_soft_ownership` `11/11` 全过 | **PASS** |

> [!WARNING]
> **工程结论**：本模型在 120 静态倒姿（116/120，15s 内无误交还）、30 受推（30/30 恢复、0 推前倒地）和全套控制权回归上均达到或超出成功率门槛；但在所持控制权期间全时段峰值约束（含失败试次及受推坠地撞击首 0.8 秒 `[cap, cap+40)` 区间）上未同时满足 `qd <= 12`、`ground <= 10 mm`、`arm_leg <= 2 mm`。因此按契约判定 **`overall_strict_acceptance_pass = false`**，仅保留为仿真验证阶段成果，**不替换部署包真机模型，不执行带电真机测试**。

---

## 二、 算法架构与训练迭代链路

源码位于 [script/q1_soft_recovery/](../script/q1_soft_recovery/)，完整链路分为数据池构造、Stage A（侧卧着地防碰 + 翻身）、Stage B（仰卧快速连续起身）、Stage C（单 ONNX 状态机融合）以及外层安全包装器。

1. **精确 2 ms 训练与验证池（[build_v2_pools.py](../script/q1_soft_recovery/build_v2_pools.py)、[expand_side_pool.py](../script/q1_soft_recovery/expand_side_pool.py)）**：
   - 使用与 `evaluate.py` 完全同构的 `50 × 2 ms` 沉降与碰撞筛选构造初始状态，C++ 持久向量植物 `Plant` 与独立 Python 参考闭环 `45` 拍对比验证 `early25 obs_74d max_diff = 2.02e-7`、`early25 state_57d max_diff = 1.22e-7`。
   - 训练集基础种子 `71001–71025`（俯/仰/侧各 200），独立验证集种子 `71026–71029`（各 32，全程未变）。随后将侧卧训练池扩增至 `side_train_800.npz`（`[800, 79]`，新增种子至 `71104`），并验证 `roll.npz` 俯卧翻身可达 `200/200`（训练）及 `32/32`（验证，峰值 `(8.18 rad/s, 8.26 mm, 0.00 mm)`）。
2. **Stage A：左右对称规范化侧卧两段策略（[train_side_v3.py](../script/q1_soft_recovery/train_side_v3.py)）**：
   - `SideLandingNet`（A1，`75->128->128->22`，拍 `0–11` 着地卸载与避碰，带 `-0.03*qd` 速度阻尼）：在 `side_v2_06` 中以 `0.10 mm` 臂腿严格屏障 + Top-64 难例过采样（70%）训练，于 `update=160`（122,880 拍）得到 `best_landing.pt`，在未见 `side_val` 上达 **`soft_12 = 32/32`**，峰值 `(10.63 rad/s, 8.80 mm, 1.72 mm)`，相对超限 `0.0`。
   - `SideSettledRollNet`（A2，4 帧 `74->64` 编码器 + `257->192->128->22`，拍 `12–149`，滚向对称宽臂初态 `symmetrize_goal(cmd0)`）：在 `side_v2_07` 冻结上述 `best_landing.pt` 上微调，验证采用 `220` 拍全流程闭环 + `(age>=12 & up_x<-0.75)->roll.npz` 动态俯卧路由 + Top-64 难例重采样，于 `update=300`（614,400 拍）得到 `best_side_v3.pt`：**`side_val` 端到端 `supine=32/32`、`simultaneous_soft=32/32`、`flat_supine=32/32`**（13 例动态借道 `roll.npz`），峰值 `(10.63, 8.80, 1.72)`，ORT 等价误差 `1.19e-7`。
3. **Stage B：快速闭环仰卧起身（[train_supine_v3.py](../script/q1_soft_recovery/train_supine_v3.py)）**：
   - `SupineRiseNetV3`（6 帧 `74->48` 编码器 + `337->256->256->128->22`）：以分段加速官方参考表于 ~`6.8 s`（tick 341）平滑收敛至 `STAND`，采用 `t=0 / 160 / 235` 三层状态复位课程先在 `supine_v2_03` 训练 `1500` 更新（`6,912,000` 拍，最佳 `upd=1400` 达 `92/96` 静立、`87/96` 柔和），再于 `supine_v2_04` 降低探索噪声微调（`max_logstd=-2.1`），于 `update=1`（`4,608` 拍）取得最佳模型：在 96 例三源验证集（仰卧 32 + 俯卧翻身 32 + 侧卧翻身 32）上达 **`quiet_stand=94/96`、`simultaneous_soft=89/96`**（仰卧 `32/32` 柔和、俯转仰 `32/32` 柔和、侧转仰 `30/32` 静立 / `25/32` 柔和），ORT 误差 `2.38e-7`。
4. **Stage C：统一状态机单 ONNX（[unified_v3.py](../script/q1_soft_recovery/unified_v3.py)）与外层监督器（[eval_v2.py](../script/q1_soft_recovery/eval_v2.py)）**：
   - 导出单一有状态 ONNX `unified_v3_08/expert.onnx`：整合 Mode `12`（空中/剧烈翻转软着陆）、Mode `10`（`SideV3Combined`，达成 `up_x>0.84 & w_norm<1.45` 立即转入仰卧或 `up_x<-0.75` 转入俯卧翻身）、Mode `8`（`roll.npz` 俯卧翻身直连 Mode `6` `pt_6=0`）、Mode `14`（`40` 拍清臂 `PreAlign`）、Mode `6`（`SupineRiseNetV3` + 早期失败 `FastRetry` 快速重调度：`pt_6∈[20,95] & up_x<-0.55`、`pt_6>=260 & up_z<0.35` 或 `pt_6>=360 & up_z<0.65` 重置回模式 `0`）。
   - 外层 `SoftRecoveryOwner` 包装器增加 `CalmUprightGate`（`h>0.355`, `-up_z>0.965`, `rms_leg<0.35`, `|w|<0.70`, `qd_rms<0.95` 连续 `4` 拍）、`16` 拍余弦过渡至 V2b 站立控制器与腿速软限幅（`8.5 rad/s` 增益 `0.5`），以及推倒瞬间的零腿刚度宽肩外展 `protect` 保护阶段（在进入 `protect` 早退分支前同样强制校验有限本体输入与严格单调递增时间戳）。

---

## 三、 开发确认基准（`dev_eval_07`，与盲测严格隔离）

锁定参数后，先于开发种子 `92017 / 92019 / 92117 / 92023` 跑通完整 DEV 冻结测试 `/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/dev_eval_07/dev_report.json`：

- **静态开发集**：`92017` `24/24` 及时（`23/24` 柔和，`gnd=10.42 mm`）、`92019` **`24/24` 及时且 `24/24` 同时柔和**（`max=(11.30, 9.15, 0.00)`）、`92117` `24/24` 及时（`23/24` 柔和，`T20` tick-0 子步 `self=3.91 mm`）；合计 **`72/72`（100%）及时、`70/72`（97.2%）同时柔和**，确认耗时 `<= 13.24 s`，误交还 `0`。
- **受推开发集（`92023`）**：`9/9` 实际倒地 **`9/9` 恢复**，`0` 推前倒地，`0` 误交还；持有期全段最坏 `(17.79 rad/s, 9.49 mm, 7.76 mm)`，其中 `cap+40` 之后主动起身段 **`9/9` 同时柔和**（最坏 `(11.37 rad/s, 9.49 mm, 0.00 mm)`）。
- **开发回归**：`offline_92017` `3/3`（释放 0、遥操调用 0、末帧站立 3）；`flat_92017` `2/2`（接管 0）；`double_92023` 4 例双倒恢复 1 例；`check_soft_ownership` 全部 11 项确定性检查通过。

---

## 四、 未见保留种子最终验收详表（`final_acceptance_20261121`）

### 4.1 静态 120 倒姿分种子与分姿态明细

| 种子 / 维度 | 及时成功 | 同时柔和 | 误交还 | 确认最长用时 (成功/含超时) | 所持 `peak_qd` | 所持 `ground_mm` | 所持 `self_mm` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `static_84101` | 22 / 24 | 22 / 24 | 0 | 12.96 s / 15.32 s | 11.54 | 9.75 | 0.41 |
| `static_84102` | 24 / 24 | 22 / 24 | 0 | 10.72 s / 10.72 s | 11.70 | 10.06 | 2.43 |
| `static_84103` | 24 / 24 | 24 / 24 | 0 | 10.90 s / 10.90 s | 11.14 | 9.74 | 0.00 |
| `static_84104` | 23 / 24 | 23 / 24 | 0 | 12.50 s / 12.56 s | 11.64 | 9.81 | 0.00 |
| `static_84105` | 23 / 24 | 23 / 24 | 0 | 14.42 s / 14.42 s | 11.12 | 9.55 | 1.87 |
| 汇总 `prone` | **40 / 40** | **40 / 40** | 0 | 14.42 s | 11.70 | 9.81 | 0.00 |
| 汇总 `supine` | **38 / 40** | **38 / 40** | 0 | 12.50 s / 15.32 s | 10.13 | 9.74 | 0.00 |
| 汇总 `side` | **38 / 40** | **36 / 40** | 0 | 12.96 s | 11.13 | 10.06 | 2.43 |
| **总计（含失败全 trial）** | **116 / 120** | **114 / 120** | **0** | **14.42 s**（成功）/ **15.32 s**（全确认） | **11.70** | **10.06** | **2.43** |

*注：120 个合法初态 `initial_ground_mm <= 3.81 mm`、`initial_arm_leg_self_mm = 0.00 mm`、`initial_all_self_mm = 0.00 mm`；全集 `84102` 生成时被几何筛拒绝 `1` 次随机候选（`reason="settled_overlap"`，完整留存于 `static_84102/rejected_initializations.json`）。静态 120 例中全试次（含交还后 1 s 遥操站立）峰值与恢复期完全一致 `(11.70 rad/s, 10.06 mm, 2.43 mm)`。*

**全部 6 例未同时通过静态严格指标的试次逐条列出：**

1. `84101/T15` (`supine`, **`success=False`**)：`peak_qd=9.10`, `ground=8.61 mm`, `self=0.00 mm`；于 **`15.32 s`** 站稳确认并在 `tick=816` 正常交还，仅超 15 秒时限 0.32 s（终态 `z=0.410 m, up=0.999`）。
2. `84101/T19` (`side`, **`success=False`**)：`peak_qd=9.69`, `ground=8.37 mm`, `self=0.00 mm`；850 拍评估窗内终态已起身直立（`z=0.413 m, up=0.996`），未完成 `standing_confirmed` + 1 s 在线等待窗口。
3. `84104/T12` (`supine`, **`success=False`**)：`peak_qd=9.22`, `ground=8.14 mm`, `self=0.00 mm`；曾于 `12.56 s` 触发 `seen_stable`，但在线接入前静立稳定条件抖动，未能在接管后 850 拍窗内交还（终态 `z=0.411 m, up=0.997`）。
4. `84105/T17` (`side`, **`success=False`**)：`peak_qd=9.05`, `ground=8.12 mm`, `self=0.00 mm`；于 `11.06 s` 曾触发 `seen_stable`，但在接管后 850 拍窗内 V2b 接手站立有微晃导致未满足完整交还稳态（`terminal_standing=False`，终态 `z=0.409 m, up=0.997`）。
5. `84102/T18` (`side`, **`success=True`, 仅柔和超限**)：`confirm=8.90 s`, `peak_qd=10.55`, `ground=7.59 mm`, **`self=2.43 mm`**（发生于 `tick=0` 2 ms 子步的 `SideLandingNet` 起始反射瞬间）。
6. `84102/T23` (`side`, **`success=True`, 仅柔和超限**)：`confirm=10.62 s`, `peak_qd=9.76`, **`ground=10.06 mm`**（超限 `+0.06 mm`，发生于 `tick=493` V2b 站立接手踝部接触 `geom=29`），`self=0.00 mm`。

### 4.2 受推 30 试次（`84111–84113`）恢复与双阶段峰值

| 种子 | 倒地数 | 恢复交还 | 推前倒地 / 误交还 | 所持全段峰值 (`[cap, end)`) | 主动起身段峰值 (`[cap+40, end)`) | 全试次完整峰值（含外力撞击+遥操） |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `push_84111` | 10 / 10 | **10 / 10** | 0 / 0 | `(15.42, 11.21, 12.21)` | **`(11.49, 10.23, 0.49)`** | `(27.08, 18.37, 12.21)` |
| `push_84112` | 10 / 10 | **10 / 10** | 0 / 0 | `(12.87, 10.68, 7.09)` | **`(11.26, 10.68, 0.00)`** | `(34.54, 24.21, 7.09)` |
| `push_84113` | 10 / 10 | **10 / 10** | 0 / 0 | `(22.91, 11.37, 10.90)` | **`(11.21, 9.03, 1.28)`** | `(26.09, 20.33, 10.90)` |
| **总计 (30 例)** | **30 / 30** | **30 / 30 (100%)** | **0 / 0** | **`(22.91, 11.37, 12.21)`** | **`(11.49, 10.68, 1.28)`** | **`(34.54, 24.21, 12.21)`** |

- **超限分布物理归因**：
  - 主动恢复与交还段 `[cap+40, end)` 中，`peak_qd`（**30/30 `<= 11.49`**）和 `self_mm`（**30/30 `<= 1.28 mm`**，其中 `28/30 = 0.00 mm`）**100% 满足柔和限值**，**28/30** 三项同时达标（仅 `84111/T01` 于 `cap+461` 达 `10.23 mm`、`84112/T05` 于 `cap+49` 达 `10.68 mm`）。
  - 严格限值超限集中于倒地撞地首 0.80 s（`[cap, cap+40)`）：`84113/T06` 于接管第 0 拍（`cap+0`）即因外推踢击后的遥操残留腿部角速度出现 `right_ankle_pitch_joint = 22.91 rad/s`（`15/30` 超 `12 rad/s`）；`84113/T05` 于 `cap+5` 摔落触地瞬时出现 `11.37 mm` 地面穿透及 `cap+30` 臂腿接触 `10.90 mm`；`84111/T06` 于 `cap+8` 触地挤压出现 `12.21 mm` 臂腿穿透（共 `8/30` 在 `cap+5..30` 出现 `> 2 mm` 撞地臂腿挤压）。上述首秒撞击动量无法仅靠零刚度退让策略在无气囊物理接触下完全消除。

### 4.3 回归校验详情

- **`offline_84101`**：3/3 试次独立站起并保持，`releases = 0`，`teleop_calls_while_owned = 0`，`terminal_standing = 3/3`；持有期峰值 `(9.48 rad/s, 9.75 mm, 0.00 mm)`（全部柔和）。
- **`flat_84101`**：2/2 正常行走未触发任何接管（`takeovers = 0`），`upright_fraction_min = 1.0`，`trial_peak_qd = 7.94 rad/s`。
- **`double_84111`**：5/5 试次两次受推均倒地，且 **`5/5`（100%）两阶段均完整恢复并交还后稳站 `>= 1 s`**，`wrong_release = 0`，`releases == anchors` 全部断言通过，`releases_confirmed_all = true`；持有期峰值 `(18.45, 11.21, 5.08)`，全试次峰值 `(27.99, 14.83, 6.44)`。
- **`check_soft_ownership`（确定性门控与时钟回归）**：`11/11` 全通过 —— 自动严重倾倒接管、`**protect` 期间重复时间戳 & NaN 本体输入拒绝**、离线保持 `0` 遥操调用、陈旧/超前/早于接管时间戳拒绝、角速度晃动重置 0.5 s 驻留、采样间隔跳跃（`dt=0.70 s`）重置驻留、交还时严格先 `anchor` 再 `teleop`、二次倒地重接管、`reset` 及 30 s 持续倒地不超时误交还。
- **`pd_oob = 0`，`teleop_calls_while_owned = 0`，`max_command_step = 0.200000048 rad`（浮点舍入，合规）**。

---

## 五、 交付物、哈希校验、原速视频与训练曲线

### 5.1 模型与 ONNX 等价性验证

- **最终 ONNX**：`/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/unified_v3_08/expert.onnx`（`1,320,969` bytes）
  - SHA-256: `5d38cfcb3c31ad1103bd8897d4c8fad97472a04d5e258106732dd3a409b842ee`
  - PyTorch vs ONNXRuntime 15 组路由与状态机探针最大绝对误差：`2.384185791015625e-07`（`< 1e-5`）
  - 单线程 CPU ORT 延迟（500 次）：验收实测 `p50 = 0.1776 ms`，`p95 = 0.1830 ms`（导出时实测 `p50 = 0.1851 ms`，`p95 = 0.1898 ms`）
- **上游冻结组件 SHA-256**：
  - `side_v2_06/best_landing.pt`: `b37c4f76e481b9d81af81a18caf0dc083c7f259050406f4afb60d60425245597`
  - `side_v2_07/best_side_v3.pt`: `d785ca56b6a633030e2990974595601c79a984c7cd7aa3511fd4524a7cb45dd1`
  - `supine_v2_04/best.pt`: `c00e3c2fae17904cc6ae48ed0551fc4e659275f513b9a7a8ade7990991e0b035`
  - `roll.npz`: `6fb3cbda24fd6ab12ff61a6b38ca0ed7f0269427b21cf6bb5bbc707aa660792a`
  - 完整文件哈希、脚本与配置快照见 `/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/final_acceptance_20261121/frozen_config_snapshot.json` 与 [final_acceptance_manifest_20261121.json](../script/q1_soft_recovery/final_acceptance_manifest_20261121.json)。

### 5.2 真实训练曲线与数值源文件

全部来自实际闭环 RL 训练过程（同时包含在 3090 归档包中）：
- Stage A1 (`side_v2_06`): `train_curve_a1.png` (`sha256: 478842b1...`), `train_curve_a1.csv` (`sha256: 3d340587...`), `metrics_a1.jsonl`, `val_a1.jsonl`
- Stage A2 (`side_v2_07`): `train_curve_a2.png` (`sha256: b156e121...`), `train_curve_a2.csv` (`sha256: 6d995bd6...`), `metrics_a2.jsonl`, `val_a2.jsonl`
- Stage B 基础 (`supine_v2_03`): `train_curve.png` (`sha256: 7bbe11d5...`), `train_curve.csv` (`sha256: bc8f57b3...`), `metrics.jsonl`, `val_metrics.jsonl`
- Stage B 精调 (`supine_v2_04`): `train_curve.png` (`sha256: 9dbd6ed9...`), `train_curve.csv` (`sha256: 9ca295c8...`), `metrics.jsonl`, `val_metrics.jsonl`

### 5.3 原速 (`1.0x`, `25 fps`) 盲测视频与缩略图

目录 `/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/final_acceptance_20261121/`，第四宫格优先展示失败例或最差峰值例，均已人工抽检 `.jpg` 渲染与叠加数值无误：

| 文件名 | 场景 | 帧数 (时长) | SHA-256 |
| --- | --- | ---: | --- |
| `Q1_recovery_v2_unseen_84101_1x.mp4` (+`.jpg`) | 静态未见 84101 | 447 (17.88 s) | `fd9f4852577b6d6c97b331d1ee98b6c4af03931c75c7b8c997defc7e9f3ecf3b` |
| `Q1_recovery_v2_unseen_84102_1x.mp4` (+`.jpg`) | 静态未见 84102 | 306 (12.24 s) | `4995d83644e8f6b4e48a014912198052fb7168ffa0a4f4d4c715f0d19f044c4d` |
| `Q1_recovery_v2_unseen_84103_1x.mp4` (+`.jpg`) | 静态未见 84103 | 301 (12.04 s) | `be1668aa422a891052e9c794f4bfd1cdee3b53bc1e00163c9d5f6b74c484cbd0` |
| `Q1_recovery_v2_unseen_84104_1x.mp4` (+`.jpg`) | 静态未见 84104 | 426 (17.04 s) | `6eaa68a6fdfddb2d7b8f6c9abafd9bf6c8f5c690049416f4e2c691a7067b4b2e` |
| `Q1_recovery_v2_unseen_84105_1x.mp4` (+`.jpg`) | 静态未见 84105 | 426 (17.04 s) | `8df6f274cfea724a37cab5b9711b3ae89156c98c6d18ebb4aad022271933c59a` |
| `Q1_recovery_v2_unseen_push_84111_1x.mp4` (+`.jpg`) | 受推未见 84111 (`T06,T05,T00,T09`) | 379 (15.16 s) | `f76025ad4345791794f2646ff83f3638696f0b84822c0f3c29e713b56649ae67` |

### 5.4 3090 证据包归档与 Mac 镜像指引

- **3090 证据包**：`/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_final_acceptance_20261121.tar.gz`（`16,967,838` bytes）
- **SHA-256**：`635d7e30523eae3715006151bed19b70a4c77f6992276ecc42ba46718221d782`（已通过 `sha256sum -c q1_soft_recovery_v2_final_acceptance_20261121.tar.gz.sha256` 校验 `OK`）
- **Mac 双份镜像说明**：3090 本机 `~/.ssh/config` 仅配置了 `github.com`，当前沙箱无法直连 Mac 主机，未虚构 Mac 传输完成记录。请在 Mac 终端拉取并用同源哈希校验：

```bash
scp yu@<3090-host>:/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_final_acceptance_20261121.tar.gz* ~/Downloads/
shasum -a 256 -c ~/Downloads/q1_soft_recovery_v2_final_acceptance_20261121.tar.gz.sha256
# 预期: 635d7e30523eae3715006151bed19b70a4c77f6992276ecc42ba46718221d782
```

---

## 六、 复现命令

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export Q1_AUTONOMOUS_ROOT=/home/yu/projects/SuperMimicHP/runs/h003_gait_engineering_01/autonomous_recovery
export Q1_V2B_BUNDLE=/home/yu/projects/SuperMimicHP/runs/h003_recovery_expert/formal_03/pinned/q1_v2b_gantry_deploy_bundle
export MUJOCO_GL=egl
PY=/home/yu/projects/SuperMimicHP/.venv/bin/python
SC=/home/yu/projects/agibot-X2-A3-Q1-model/script/q1_soft_recovery
RUN=/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121

# 1. 开发冻结回归（不可与保留种子混用；需新目录）
"$PY" "$SC/eval_v2.py" "$RUN/unified_v3_08/expert.onnx" "$RUN/dev_eval_repro" --no-videos

# 2. 一次性盲测全套复现（需新输出目录）
"$PY" "$SC/run_final_acceptance.py" --out "$RUN/final_acceptance_repro"
```
