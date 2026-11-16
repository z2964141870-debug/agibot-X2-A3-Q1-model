# 任务转接卡：X2 衣服遥操（FGP 替换 LFP → MorphMimic 替换 SONIC）

- **交接日期**：2026-11-16
- **代码基线**：分支 `Q1`（commit `2bb0dc70` 及本转接卡提交）
- **当前已验证可跑通基线**：**LFP 衣服 → `x2_smpl_wholebody` → SONIC（`s1ft16000_g1` + `_g1_token`）** 已完成真机全身带电闭环（含上肢、连续行走、370°+ 大幅度转身、俯仰恢复），详见 [README_X2_SONIC_LIVE_HANDOFF_20261116.md](README_X2_SONIC_LIVE_HANDOFF_20261116.md)。

---

## 一、接手人先看清「已跑通的现网基线」

### 1. 板端（SoC1 `agi@10.0.1.41`）现网可用资产
- **3 个全前台启动入口（推荐日常直接用）**：
  - 终端 1（衣服 LFP）：`/home/agi/x2_sonic_live/term1_garment.sh`（按 Enter 摆 3 秒 T-Pose；Ctrl-C 自动释放蓝牙）
  - 终端 2（Policy 栈）：`/home/agi/x2_sonic_live/term2_policy.sh`（自动起 `watchdog` + `kplanner` + `--root-heading-band-deg 0` 的 `x2_smpl_token_service`，前台调 `deploy_x2.sh onbot`）
  - 终端 3（运控切换 + 监控）：`/home/agi/x2_sonic_live/term3_switch.sh`（1 Hz 全链监控 + 按 Enter 切换 `kplanner` 站立 ↔ 全身 SONIC）
- **已修复并备份的生产文件**（原件备份为 `*.bak_20261116_pre_sonic_fix`，源码与 diff 在 [script/x2_sonic_live/](../script/x2_sonic_live/README.md)）：
  - `/home/agi/projects/project_tools/x2_smpl_wholebody/garment_intent_adapter.py`：`SMPLXConverter` Y-up → SONIC Z-up 转换（默认 `source_frame="smpl_fk"`）
  - `/home/agi/projects/project_tools/x2_smpl_wholebody/x2_smpl_token_service.py`：`sync_and_tick` 对齐 `time.monotonic()` 时钟
  - `/home/agi/projects/smartwear_v2/garment_udp_v2_reconnect_safe.py`：`SIGINT`/`SIGTERM`/`atexit` 调用 `close_all()` + `bluetoothctl disconnect` 释放 BLE
- **已实测指标（今晚真机闭环 `rec_5122_000`..`rec_5360_024`）**：
  - 连续稳定接管 `247.6 s`（0 次误断连）；
  - 展开转身净跨度人 `-263.9°` / 机器人 `-268.0°`，航向误差均值 `5.2°`（P95 `12.5°`）；
  - 机身倾角平时 `P50=1.59°` / `P95=5.52°`，俯身跟随最大 `38.2°` 后平稳回正到 `2.1°`；
  - **当前短板**：CPU 版 LFP 与 50 Hz 控制共核，导致 `garment -> token_service` 到达率只有 **`15–18.7 Hz`**（均值 `17.2 Hz`）。

---

## 二、Phase 1：用 FGP 替换 LFP（先不动 SONIC 下游）

### 目标与验收标准
用 Orin GPU FGP 替换当前 CPU LFP，在真机同负载（`kplanner --ort-gpu` + 50 Hz `deploy`）下把 `pico_intent` 到达率从 ~17 Hz 提到稳定 ~25–30 Hz，并保持“只读影子验证 → 真机带电行走/转身”全流程不回退。

### 已有仓库基础
- 可行性与协议说明：[README_X2_FGP_SONIC_FEASIBILITY_20261115.md](README_X2_FGP_SONIC_FEASIBILITY_20261115.md)
- Orin FGP 独立结果接收实测：[README_X2_ORIN_FGP_COLLECTOR_20261116.md](README_X2_ORIN_FGP_COLLECTOR_20261116.md)（独立线程回放均值 `~29.99 Hz`，出帧间隔 P95 `40.77–42.19 ms`）
- 适配器与桥接原型：
  - [script/x2_fgp_preview/sonic_preview.py](../script/x2_fgp_preview/sonic_preview.py)（`SonicIntentAdapter`）
  - [script/x2_fgp_preview/sonic_terminal_pipe.py](../script/x2_fgp_preview/sonic_terminal_pipe.py)
  - [script/x2_fgp_preview/staged_engine.py](../script/x2_fgp_preview/staged_engine.py)

### 必踩避坑点（逐条核对）
1. **`SonicIntentAdapter` 必须传 `source_frame="smpl_fk"`**
   - 现版 [sonic_preview.py](../script/x2_fgp_preview/sonic_preview.py#L38) 直接调用 `root_local_joints(positions, quaternions[0])`，而 `root_local_joints()` 函数默认值仍为 `"canonical"`（仅 `GarmentIntentAdapter` 类默认 `"smpl_fk"`）。
   - 不加 `source_frame="smpl_fk"` 会立即重现 Bug 1（`CALIB_SUSPECT knees not below hips`）。
2. **BLE 释放与标定代次隔离**
   - 复用 [term1_garment.sh](../script/x2_sonic_live/term1_garment.sh) / [garment_udp_v2_reconnect_safe.py](../script/x2_sonic_live/onboard_patched/smartwear_v2/garment_udp_v2_reconnect_safe.py) 的 `SIGINT`/`SIGTERM` + `BleV2Manager.close_all()` + `bluetoothctl disconnect` 清理机制。
   - 重置/重标定时必须在消费端按标定代次丢弃旧帧，避免旧回包串过重置边界。
3. **Orin GPU 并发争用验证**
   - 现场 `pc2_kplanner_onnx.py` 开启了 `--ort-gpu`；FGP 的 CANO CUDA Graph + CPU 子进程需在同时跑 `kplanner` + 50 Hz `deploy` + 真实 BLE 时重测出帧间隔 P95 与长尾。

### 执行步骤
1. 先把 FGP 桥接入口接到隔离端口 `:15573`（engage 文件 `/tmp/x2_shadow_pico_intent.engage`），走 [run_shadow_live.sh](../script/x2_sonic_live/run_shadow_live.sh)（`--no-publish` 零电机输出）跑通 `pose → token → pose`：对比 FGP 与 LFP 的 `intent` 到达率、站立通过率、单臂响应与转身平滑度。
2. 只读与审计通过后，复制一份 FGP 版入口替换 [term1_garment.sh](../script/x2_sonic_live/term1_garment.sh) 的调用，保持 `term2_policy.sh` / `term3_switch.sh` 不变，在有人扶护、手握急停下做带电对照验收。
3. 验收时记得在 [term2_policy.sh](../script/x2_sonic_live/term2_policy.sh) 的 `./deploy_x2.sh` 参数中加上 `--log-dir` 保存逐关节 `tick.csv`/`joint_pos.csv`/`target_pos.csv` 方便定量对比。

---

## 三、Phase 2：用 MorphMimic 替换 SONIC

### 目标与当前进度边界
保留已跑通的「`kplanner` 稳站基线 + FGP 衣服前端 + 只读影子验收 + 收工回退」体系，将全身控制策略从 SONIC（`x2_sonic_s1ft16000_g1[_token].onnx` + `x2_smpl_token_service`）切换为 MorphMimic（[script/x2_morph_human/](../script/x2_morph_human/README.md)）。

### 已知未解决阻塞问题（未解禁前**禁止直接带电接管**）
1. **右膝越下限与动作历史单步跳变（实测未闭环）**
   - 见 [README_X2_MORPH_ACTION_HISTORY_CHECK_20261116.md](README_X2_MORPH_ACTION_HISTORY_CHECK_20261116.md)：在真人 `HumanFrame` + 真实反馈离线复放（1238 帧）中，原始 MorphMimic 出现 **`-26.72°` 右膝突跳**，且右膝目标低于下限达 **`1008/1238`** 帧；加入动作历史指数平滑（权重 `0.75/0.50`）虽能压低突跳至 `-0.03°`，但 **右膝低于下限仍有 `1014–1172/1238` 帧**，全程仍存在明显单步变化。
2. **左踝越界对腰部观测与 pelvis 推算高度敏感**
   - 见 [README_X2_GARMENT_GUIDED_TEST_20261116.md](README_X2_GARMENT_GUIDED_TEST_20261116.md)：在保存反馈（实测腰俯仰约 `+20°` vs 策略默认 `0°`）下，左踝超限达 `187/189` 帧；离线同时替换腰俯仰观测与 pelvis 推算后仍有 `14/189` 超下限。由于旧日志为 `PASSIVE_DEFAULT`、缺少 `kplanner` 稳站同源反馈，需要先在 `kplanner` 稳站下重采配对。
3. **骨骼与根坐标系未与 SONIC 做逐项对齐**
   - 见 [README_X2_FGP_SONIC_FEASIBILITY_20261115.md](README_X2_FGP_SONIC_FEASIBILITY_20261115.md#L5)：MorphMimic 的归一化骨架和根变换尚未对照过 `GarmentIntentAdapter` 的静立、单侧抬臂与转身轴向。

### 执行步骤（严格先只读、再离线排查、后上电）
1. **在已确认稳站的 `kplanner` 基线下做首次 FGP + MorphMimic 只读录制**：
   - 使用板端 `~/x2-garment-check`（参考 [README_X2_GARMENT_GUIDED_TEST_20261116.md](README_X2_GARMENT_GUIDED_TEST_20261116.md)）或隔离只读入口，录下同一单调时钟下的真实 `kplanner` 稳站反馈 + FGP 人体流 + MorphMimic 策略输出（全程零电机发布）。
2. **用 `script/x2_morph_human/` 工具定量过关**：
   - 检查站立 → 抬左臂 → 抬右臂 → 回到站立的膝踝目标是否仍在 X2 关节配置区间内、进入/退出边界有无突跳（工具：[review_saved_trace.py](../script/x2_morph_human/review_saved_trace.py)、[replay_waist_paths.py](../script/x2_morph_human/replay_waist_paths.py)、[replay_live_history_sensitivity.py](../script/x2_morph_human/replay_live_history_sensitivity.py)）。
   - 核对 MorphMimic 站立参考/腰俯仰基准、`wrist_bypass`/关节限幅及 `target_lpf`。
3. **只在以下条件全满足后才允许真机带电实测**：
   - FGP `smpl` 帧稳定无跳帧；
   - 只读状态下 `kplanner` 稳站基线切换进出 MorphMimic 时，膝踝不越限、单步无异常突跳、上肢同侧正确跟随；
   - 带电测试严格沿用 `L2+X → 站稳 → R2+X`、保护绳兜而不拉、有人扶护、限时运行与物理急停优先。

---

## 四、交接清单

- [ ] Phase 1.1：对接 FGP → `source_frame="smpl_fk"` → `:15573` 隔离口，跑通 `run_shadow_live.sh` 只读 `pose → token → pose`
- [ ] Phase 1.2：验证 Orin `FGP + kplanner(--ort-gpu) + deploy` 同机负载帧率，切换 `term1_garment.sh` 并带 `--log-dir` 完成 FGP+SONIC 真机行走/转身复验
- [ ] Phase 2.1：核对 FGP → MorphMimic 骨架/根坐标转换与 `GarmentIntentAdapter` 一致性
- [ ] Phase 2.2：在 `kplanner` 稳站下完成 FGP + MorphMimic 只读录制，定位并消除右膝下限越界/单步跳变与左踝越界
- [ ] Phase 2.3：完成只读过关后，在扶护与物理急停保护下开展 MorphMimic 真机闭环测试
