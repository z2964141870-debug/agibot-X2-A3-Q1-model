# Q1 部署交接指南：PlannerCmd 左右横移 V2 + 自主起身 Recovery V2（独立增量部署包）

> **致部署 Agent（负责 `/home/yu/projects/Q1_deploy/`）**：
> 本文档及配套独立部署压缩包包含两项已完成的仿真阶段产出：**① `planner_cmd` 左右横移稳定性标定 V2** 与 **② 全姿态自主起身 Recovery V2**。
> 请**严格先读“准备度红绿灯与安全边界”**，切勿在未确认边界条件与保护措施的情况下直接在真机上启用自主起身。

---

## 一、 部署准备度红绿灯与安全边界

| 模块 | 当前交付产物 | 仿真闭环指标（2 ms 物理 / 11 ms 延迟 / 500 Hz 斜坡） | 准入判定 |
| --- | --- | --- | --- |
| **Task 1：左右横移校准 V2**（冻结 `q1_h002_token_head.onnx`） | `onnx/q1_tok_planner_cmd.onnx` (`ce0aa0a4...`) | 12 速名义横移 **12/12 稳站无跌倒**（旧基线 `+0.10/+0.20` 跌倒）；24 组初态扰动+延迟 **24/24**（旧基线 4/24）；17.5 s 连续左右交替完成，`PD OOB = 0`；单核 ORT `tok+head p50 = 0.534 ms` | **建议合入并做吊架/地面保护实测** |
| **Task 2：自主起身 Recovery V2** | `onnx/q1_recovery_expert_v2.onnx` (`5d38cfcb...`) + `code/q1_soft_recovery_v2_owner.py`（`Q1SoftRecoveryV2Owner`） | 未见保留种子：静态 **116/120（96.7%）** 15 s 内站稳交还（俯 `40/40`、仰 `38/40`、侧 `38/40`，误交还 `0`）；受推 **30/30（100%）** 恢复交还；离线/行走/二次受推回归全过；单核纯模型+封装 `~0.18 ms`。<br>**未全过项（15/18 PASS, 3/18 FAIL）**：受推摔地首 0.8 s `[cap, cap+40)` 继承跌倒速度和触地冲击超过严格软峰值 `12 rad/s / 10 mm / 2 mm`（主动起身段 `[cap+40, end)` 关节速 30/30 `<=11.49`、臂腿 30/30 `<=1.28 mm`），另静态侧卧有 2 例轻微擦边（`2.43 mm` / `10.06 mm`） | **仅用于受保护垫上吊架试测；`recovery.enabled_by_default` 必须保持 `false`，待人工决策后再决定是否作为默认发布模型** |

**关键硬性约束**：
1. 现有 [q1_deploy_config.json](file:///home/yu/projects/Q1_deploy/config/q1_deploy_config.json) 中的 `default_q`、`action_scale`、`kp`、`kd`、`effort_limit_nm`、`q_lo`、`q_hi`、`cgfa_recommended`、`torque_safe_pd` 均**完全冻结不变**，零 PyTorch / 零 MuJoCo 依赖。
2. `Q1SoftRecoveryV2Owner` 对上层要求**两路输入、两项桥接修复**：必须将 `0.2 rad` 限幅移至 500 Hz 发布环，且必须传入单调时钟 `sample_time_s` 与有效参考时间戳 `teleop_stamp_s`（详见第四节）。

---

## 二、 独立部署包位置与文件清单

### 2.1 包路径（不改动原 `/home/yu/projects/Q1_deploy/`）

- **独立压缩包**：`/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/q1_deploy_addon_lateral_and_recovery_v2_20261122.zip`
- **SHA-256 校验文件**：`/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/q1_deploy_addon_lateral_and_recovery_v2_20261122.zip.sha256`
- **已解压校验目录**：`/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/deploy_bundle_v2_20261122/`
- **元数据与实机日志零误触发报告**：[q1_deploy_addon_v2_manifest_20261122.json](../data/manifests/q1_deploy_addon_v2_manifest_20261122.json)

> 如需镜像拷贝至 Mac 或车端控制机：
> ```bash
> scp yu@<3090-host>:/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/q1_deploy_addon_lateral_and_recovery_v2_20261122.zip* ~/Downloads/
> shasum -a 256 -c ~/Downloads/q1_deploy_addon_lateral_and_recovery_v2_20261122.zip.sha256
> ```

### 2.2 相对现有 `/home/yu/projects/Q1_deploy/` 的精准增量对照

以下 **7 个基础文件已通过 SHA-256 核对与现部署完全逐字节一致，无需修改**：
- `code/q1_v2b_runtime.py` (`f913fee3...`)
- `code/q1_recovery_runtime.py` (`f4754486...`)
- `code/recovery.py` (`217ec262...`)
- `code/leg_height.py` (`0e215e8a...`)
- `code/q1_cgfa_obs_np.py` (`2f03e425...`)
- `code/q1_kinematics_np.py` (`7c9a3386...`)
- `robot_model/supermimic.xml` (`47bfa08b...`)

本压缩包内**需要合入或新增的 5 个核心文件**（外加配置参考 `config/q1_deploy_config.json` 与独立验收脚本 `code/smoke_test_deploy_v2_addon.py`）：

| 包内相对路径 | 作用 | SHA-256 | 目标位置（`/home/yu/projects/Q1_deploy/` 下） |
| --- | --- | --- | --- |
| `onnx/q1_tok_planner_cmd.onnx` | 横移 V2 校准后 Planner 分词器（`1,030,628` bytes） | `ce0aa0a49666515e0b47dd200995fecc8980cd3bcdd41149fbc4824fb92679e5` | 替换 `onnx/q1_tok_planner_cmd.onnx`（替换前先备份旧版 `6a989b2b...`） |
| `onnx/q1_recovery_expert_v2.onnx` | Recovery V2 统一状态机 ONNX（`1,320,969` bytes） | `5d38cfcb3c31ad1103bd8897d4c8fad97472a04d5e258106732dd3a409b842ee` | 新增 `onnx/q1_recovery_expert_v2.onnx`（保留旧版 `q1_recovery_expert.onnx` 作对照） |
| `code/q1_autonomous_recovery.py` | 独占控制权 + 0.5 s 测站稳确认 + 单次 reanchor 基类（`5,424` bytes） | `5c99519c69d28e648fe47e84e18c63fdb9f35022c0bd2e30e7b25fb95be3a378` | 新增 `code/q1_autonomous_recovery.py` |
| `code/q1_soft_recovery_v2_owner.py` | Recovery V2 包装器 `Q1SoftRecoveryV2Owner` + `CalmUprightGate` + `protect` 卸载 + 16 拍交接渐变 + 腿速限幅（零 PyTorch/MuJoCo） | 与 `deploy_bundle_v2_20261122/SHA256.json` 一致 | 新增 `code/q1_soft_recovery_v2_owner.py` |
| `code/smoke_test_deploy_v2_addon.py` | 纯 NumPy + ORT 隔离自检（含 `planner_cmd`、`11` 项控制权回归与双模式耗时） | 与 `deploy_bundle_v2_20261122/SHA256.json` 一致 | 可并入 `code/smoke_test_deploy_v2_addon.py` |

*(注：压缩包另随附只读 `onnx/q1_h002_token_head.onnx`、`onnx/q1_tok_robot_pose.onnx` 与 `robot_model/supermimic.xml` 以便 zip 解压后单目录独立运行 `python -I`；现部署包的 `q1_tok_human_smpl.onnx` 与 `q1_tok_vr_3pt.onnx` 保持原有文件不动。)*

---

## 三、 Task 1 部署操作：PlannerCmd 横移 V2

### 3.1 备份与替换

```bash
cd /home/yu/projects/Q1_deploy
# 1. 备份当前线上的 planner ONNX 与配置
cp -n onnx/q1_tok_planner_cmd.onnx onnx/q1_tok_planner_cmd_h003_baseline_bak.onnx
cp config/q1_deploy_config.json config/q1_deploy_config.json.bak_20261122
cp code/q1_ros2_lowlevel_bridge.py code/q1_ros2_lowlevel_bridge.py.bak_20261122

# 2. 解压独立校验（先不要覆盖 Q1_deploy）
ADDON=/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/deploy_bundle_v2_20261122
cp "$ADDON/onnx/q1_tok_planner_cmd.onnx" onnx/q1_tok_planner_cmd.onnx
```

### 3.2 更新 `config/q1_deploy_config.json` 并运行原部署自检

仅更新 `config/q1_deploy_config.json` 中 `onnx_files["q1_tok_planner_cmd.onnx"]` 的哈希与字节数：
```json
    "q1_tok_planner_cmd.onnx": {
      "bytes": 1030628,
      "sha256": "ce0aa0a49666515e0b47dd200995fecc8980cd3bcdd41149fbc4824fb92679e5",
```
更新后立即在 `/home/yu/projects/Q1_deploy` 运行现有五模态自检脚本，确认 5/5 SHA-256 全部匹配且 `oob=0`：
```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yu/projects/SuperMimicHP/.venv/bin/python code/smoke_test_deploy_bundle.py
```

### 3.3 横移实机指令建议

- 左右键盘映射：`a -> (vx=0, vy=+keyboard_vy, wz=0)`（左移），`d -> (vx=0, vy=-keyboard_vy, wz=0)`（右移）。
- `--keyboard-vy` 建议按梯度使用仿真已锁定的精确锚点速度：**`0.05 -> 0.10 -> 0.15 m/s`**。
- **避开未锁定的中间插值点 `+0.08 m/s`**（仿真中 `+0.08` 虽不倒，但刹车余步漂移较锚点明显增大）。

---

## 四、 Task 2 工程接入：Recovery V2 与 `q1_ros2_lowlevel_bridge.py` 必改 3 处

若要在受保护测试中合入 Recovery V2，先把增量代码与模型拷入 `/home/yu/projects/Q1_deploy/`：
```bash
ADDON=/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/deploy_bundle_v2_20261122
cp "$ADDON/onnx/q1_recovery_expert_v2.onnx" /home/yu/projects/Q1_deploy/onnx/
cp "$ADDON/code/q1_autonomous_recovery.py" /home/yu/projects/Q1_deploy/code/
cp "$ADDON/code/q1_soft_recovery_v2_owner.py" /home/yu/projects/Q1_deploy/code/
cp "$ADDON/code/smoke_test_deploy_v2_addon.py" /home/yu/projects/Q1_deploy/code/
```
并将 `config/q1_deploy_config.json` 中 `"recovery"` 段更新为（保持 `"enabled_by_default": false`）：
```json
  "recovery": {
    "enabled_by_default": false,
    "expert_file": "q1_recovery_expert_v2.onnx",
    "expert_sha256": "5d38cfcb3c31ad1103bd8897d4c8fad97472a04d5e258106732dd3a409b842ee",
    "obs_dim": 74,
    "state_dim": 26,
    "owner_class": "Q1SoftRecoveryV2Owner",
    "calm_upright_gate": {
      "h_min": 0.355,
      "up_z_min": 0.965,
      "rms_leg_max": 0.35,
      "w_norm_max": 0.7,
      "qd_rms_max": 0.95,
      "calm_ticks": 4
    },
    "blend_ticks": 16,
    "leg_speed_cap": 8.5,
    "leg_brake_gain": 0.5,
    "sender_ramp_hz": 500,
    "sender_max_increment_rad_per_2ms": 0.2,
    "fall_gate_ticks": 3,
    "stand_dwell_ticks": 25,
    "wide_stand_dwell_ticks": 40,
    "handoff_ticks": 30,
    "source": "q1_deploy_addon_lateral_and_recovery_v2_20261122.zip",
    "strict_acceptance_note": "15/18 strict gates passed (unseen static 116/120 timely & 114/120 soft, unseen push 30/30 recovered); push impact [cap,cap+40) peaks exceed 12/10/2; keep enabled_by_default=false"
  }
```

> [!CAUTION]
> **接入 [q1_ros2_lowlevel_bridge.py](file:///home/yu/projects/Q1_deploy/code/q1_ros2_lowlevel_bridge.py) 时必须修复以下 3 处结构性差异**（否则离线与实机表现将与 2 ms 验收路径严重不一致）：

### 必改 1：用 `Q1SoftRecoveryV2Owner` 替换旧版 `Q1RecoveryBranch`（且须同时允许 `robot_pose`）

旧版桥接只在 `__init__` 加载了 `Q1RecoveryBranch` 且 `modalities` 在 planner 模式下只加载了 `("planner_cmd",)`，注意 **`Q1AutonomousRecovery.__init__` 内部会在 `self.stand = Q1V2bOnnxController(tracker.root, ("robot_pose",))` 自动加载 `q1_tok_robot_pose.onnx`**（无需额外改 `tracker` 的 `modalities`，只需确保 `onnx/q1_tok_robot_pose.onnx` 在 `Q1_deploy/onnx/` 存在即可）。实例化改为：

```python
from q1_soft_recovery_v2_owner import Q1SoftRecoveryV2Owner

if self.enable_recovery:
    expert = self.root / "onnx" / self.cfg["recovery"]["expert_file"]
    self.recovery = Q1SoftRecoveryV2Owner(
        self.tracker,
        expert,
        self.root / "robot_model/supermimic.xml",
        blend_ticks=int(self.cfg["recovery"].get("blend_ticks", 16)),
        leg_speed_cap=float(self.cfg["recovery"].get("leg_speed_cap", 8.5)),
        leg_brake_gain=float(self.cfg["recovery"].get("leg_brake_gain", 0.5)),
    )
```

### 必改 2（最关键）：把 `±0.2 rad` 发送斜坡从 50 Hz 移入 500 Hz 发布循环！

查看现行 [q1_ros2_lowlevel_bridge.py](file:///home/yu/projects/Q1_deploy/code/q1_ros2_lowlevel_bridge.py#L500-L567)：当前代码是在 50 Hz `_infer_once` 末尾直接对 `target` 做 `np.clip(target, self.last_target - 0.2, self.last_target + 0.2)`！
**这相当于每 `20 ms` 才允许动 `0.2 rad`（即 `10 rad/s`），会把侧卧避碰与快速俯仰起身截断 10 倍，导致轨迹畸变！**
正确训练/验收契约是：**`50 Hz` 策略输出 `desired_target = clip(target, default_q - 2.0, default_q + 2.0)`，而在 `500 Hz`（每 `2 ms` 一拍）发布循环里逐步按 `0.2 rad / 2 ms` 逼近 `desired_target`，并用最新发出的 `sent_target` 回填 `prev_action`**：

```python
# 在 _reset_anchor / __init__ 初始化：
self.desired_target = self.default_q.copy()
self.sent_target = self.default_q.copy()

# 在 _infer_once 末尾（50 Hz）：只做 box 裁剪，不在此处截断 0.2 rad！
desired = np.asarray(target, dtype=np.float32)
self.desired_target = np.clip(desired, self.default_q - 2.0, self.default_q + 2.0)

# 在 _control_loop 每次 HAL 发布前（500 Hz / 2 ms 一拍）：
step_target = self.sent_target + np.clip(self.desired_target - self.sent_target, -0.2, 0.2)
step_target = np.clip(step_target, self.default_q - 2.0, self.default_q + 2.0)
self.sent_target = step_target.astype(np.float32)
self.target = self.sent_target
self.prev_action = (self.sent_target - self.default_q) / self.action_scale
msg = self._command(self.sent_target, self.kp, self.kd)
```
*(提示：`last_target` 初值也应在 `_reset_anchor` 同步初始化为测量姿态 `q.copy()` 写入 `desired_target` 与 `sent_target`，避免首拍跳变。)*

### 必改 3：显式传递单调时钟与 `teleop_stamp_s`，并解除 CLI 互斥参数

1. **正确调用签名**：`Q1SoftRecoveryV2Owner.step(...)` 需要两个关键字时间参数（单调时钟秒 `time.monotonic()`）：
   - `sample_time_s = now_mono`（必须在每次 50 Hz `_infer_once` 严格递增）。
   - `teleop_stamp_s`：
     - 若处于“无遥控纯自主站立调试 / 或外部参考断流超过门限”：传 `None`（此时 Recovery 接管后独立起身站立并保持 `owner="recovery"`，绝不误交还、绝不调用 `base_step`）；
     - 若处于正常 `--planner` / `--keyboard` 遥控：传入本拍前最新的操控时间戳（本地常驻 planner 传 `now_mono`；若未来接遥控句柄超时未刷新，应传最后一次有效操作时间从而让 `now - stamp > 0.1` 自动识别为 stale/offline）；
     - 若处于 `--human-smpl` FGP 模式：先拉取 `queued` 帧，若 `(now_ns - queued.arrival_ns) <= 150 ms` 传 `queued.arrival_ns * 1e-9`（与 `sample_time_s = time.monotonic_ns() * 1e-9` 同源单调钟），断流或超龄传 `None`——且 `FGP` 逻辑应在 `base_step()` 内部执行，**绝不能在 Recovery 持控时提前 `return None` 跳过 `self.recovery.step(...)`！**
2. **解除 CLI 互斥**：现行 [q1_ros2_lowlevel_bridge.py](file:///home/yu/projects/Q1_deploy/code/q1_ros2_lowlevel_bridge.py#L573-L580) 把 `--enable-recovery` 和 `--planner / --keyboard / --human-smpl` 放进了同一个 `add_mutually_exclusive_group`。建议将 `--enable-recovery` 改成可叠加的独立布尔开关（并额外增加 `--recovery-offline` 用于强制 `teleop_stamp_s=None` 的纯独立站起测试）。

---

## 五、 上机前与吊架分阶段验收清单

1. **增量包独立完整性自检（零 PyTorch/MuJoCo 隔离验证）**：
   ```bash
   ADDON=/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/deploy_bundle_v2_20261122
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     /home/yu/projects/SuperMimicHP/.venv/bin/python -I "$ADDON/code/smoke_test_deploy_v2_addon.py"
   ```
   合并进 `Q1_deploy` 后再同步运行 `code/smoke_test_deploy_bundle.py` 与 `code/smoke_test_deploy_v2_addon.py`（检查全部 ONNX 哈希一致、`oob=0`、`torch`/`mujoco` 未导入、11 项控制权回归全 `True`）。
2. **第一阶段实测（只合入 Task 1 横移 V2，`--enable-recovery` 关闭）**：
   - 先用现有平地安全流程启动 Pure V2b（站立承重确认 -> `w/s` 前后慢走 -> `--keyboard --keyboard-vy 0.05` 左右小步横移 -> 逐步测试 `0.10`）；确认无横移偏移失稳且紧急 `damping` 退出正常。
3. **第二阶段受保护测试（厚缓冲垫 + 吊架半松保护，确认人工授权后再开 Recovery V2）**：
   - 先运行纯离线站立模式（`teleop_stamp_s=None`），让机器人从平躺（推荐先仰卧、再俯卧、后侧卧）触发自主起身并静立保持，观察站稳确认无异常再连接在线交还；
   - 测试完成后**在退出前发 5 帧阻尼帧（已有 `_publish_damping`），并保留备份一键回滚方案**：
     ```bash
     cd /home/yu/projects/Q1_deploy
     cp onnx/q1_tok_planner_cmd_h003_baseline_bak.onnx onnx/q1_tok_planner_cmd.onnx
     cp config/q1_deploy_config.json.bak_20261122 config/q1_deploy_config.json
     cp code/q1_ros2_lowlevel_bridge.py.bak_20261122 code/q1_ros2_lowlevel_bridge.py
     ```

---

## 六、 部署包校验实测摘要（构建验证通过）

以下结果为构建工具 [build_deploy_bundle_v2.py](../script/q1_soft_recovery/build_deploy_bundle_v2.py) 自动生成并实测通过的验证摘要，完整机器可读记录见 [q1_deploy_addon_v2_manifest_20261122.json](../data/manifests/q1_deploy_addon_v2_manifest_20261122.json)（`/home/yu/projects/Q1_deploy/` 全程只读未修改）：

- **压缩包大小与校验和**：
  - 文件：`/home/yu/projects/SuperMimicHP/runs/q1_soft_recovery_v2_20261121/q1_deploy_addon_lateral_and_recovery_v2_20261122.zip`（`33,509,500` bytes）
  - SHA-256：`642574482ad5c13fceb132d5211bd9da518b7ae2b317e6568ca3ab072e382cf6`（`sha256sum -c q1_deploy_addon_lateral_and_recovery_v2_20261122.zip.sha256` -> `OK`）
  - `code/q1_soft_recovery_v2_owner.py` SHA-256：`d74cdecb5cc8192adf104cf3d9908920f7cc01b36219e648477029ded56c2f38`
  - `code/smoke_test_deploy_v2_addon.py` SHA-256：`ba07d9caada410917b376d46c435873f25a929970a00d29c81f74799c9d27333`
- **封装与训练验收参考实现逐位对齐（`eval_v2.make_calm_controller(16)`）**：
  - 验证步数：`95` 步（覆盖 `protect`、`rise`、`blend_and_leg_brake`、`handback` 全四阶段）
  - `max_target_abs_diff = 0.0`、`max_tau_abs_diff = 0.0`、`max_owner_abs_diff = 0.0`
- **隔离 `python -I` 冒烟实测（单线程 CPU ONNX Runtime，`torch_loaded = false`、`mujoco_loaded = false`）**：
  - 独立解压目录（校验 `SHA256.json` 共 `16` 个文件）：`planner_cmd_v2` `p50 = 0.4184 ms / p95 = 1.1217 ms / oob = 0`；`Q1SoftRecoveryV2Owner` 待命态 `p50 = 0.4876 ms / p95 = 1.1456 ms`、接管态 `p50 = 0.2501 ms / p95 = 0.2673 ms`、`oob = 0`；`11/11` 控制权/时间戳/断流回归项全过（含无遥控 `30 s` 持续倒地不超时）。
  - 模拟合入目录（无独立 `SHA256.json` 仅校验 `q1_deploy_config.json` 与 4 个 ONNX 哈希）：同样通过 `ADDON_SMOKE_ALL_PASS`（`planner_cmd_v2` `p50 = 0.4205 ms / p95 = 1.1373 ms / oob = 0`，`11/11` 控制权回归全过）。
- **现网真机直立日志只读零误触发审计**：
  - `q1_planner_keyboard_monitor_20261121_194500.jsonl`：`8,554` 样本（`171.117 s`），`min_up_z = 0.9838`，`max_fall_count = 0`，`max_imminent_ticks = 0`，`false_takeovers = 0`
  - `q1_official_standup_faceup_20261120.npz`（`BIPED_WALK_RUN` `9.143–14.983 s`，`293` 拍 @ 50 Hz）：`min_up_z = 0.9996`，`false_takeovers = 0`
  - `q1_official_standup_from_mcap.npz`（`BIPED_WALK_RUN` `8.244–14.984 s`，`338` 拍 @ 50 Hz）：`min_up_z = 0.9988`，`false_takeovers = 0`
