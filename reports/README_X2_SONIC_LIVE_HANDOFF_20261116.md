# X2 真人穿戴 SONIC 只读切换联测

2026-11-16 晚。机器人在现场 kplanner 稳站基线下，真人穿衣服，在隔离端口、`--no-publish` 的诊断 SONIC 后端上首次录到完整的 `pose → token → pose`，同侧手臂有响应，撤权后参考来源回到上游。全程零电机输出，没有改动 MC、kplanner、生产站立后端，也没有改动生产 token 服务的端口。

## Done

### 链路与隔离

| 环节 | 本轮实际配置 |
| --- | --- |
| 衣服 | 现有 LFP 入口 `garment_udp_v2_reconnect_safe.py`（`LFP_dense_taichi_ft.onnx`，本地 CPU），约 30 fps；`pico_intent` 改发到 15573，engage 哨兵改为 `/tmp/x2_shadow_pico_intent.engage` |
| 坐标中继 | `garment_sonic_relay.py`：15573 → 15575，把关节从 SMPL Y-up 转成 SONIC Z-up（见 Bug 1） |
| token 服务 | 生产 `x2_smpl_token_service.py` 加时钟修正后的副本（见 Bug 2），`--planner-cmd-port 0`，intent 15575、token 15574、debug 15557 |
| 站立参考 | 打了来源标记补丁的 watchdog 订阅现场 kplanner 的 `:5556`，转发到 15558，每帧写 `wd_pose_provenance` |
| 诊断后端 | 在独立工作区编译的 `x2_deploy_onnx_ref`，带 [sonic_cycle_trace.patch](../script/x2_fgp_preview/sonic_cycle_trace.patch) 和 [sonic_reference_trace.patch](../script/x2_fgp_preview/sonic_reference_trace.patch)，`--no-publish --cycle-trace`，模型对为 `s1ft16000_g1` / `_g1_token`，与现场一致 |

### 结果（`run_live_arm_20261116_215153`）

[audit_sonic_handoff](../script/x2_fgp_preview/audit_sonic_handoff.py) 审计结果：975 个周期，未知头 0，排除行 0，没有找不到对应请求的头变化，两次请求都确认。

| 事件 | 请求时刻 | 实际头确认 | 延迟 | 参考来源 |
| --- | --- | --- | ---: | --- |
| 接入 tracking | 1502.804 s | 1506.992 s 切到 `token` | 4187.7 ms | 切换前 10 槽全部为 `upstream` |
| 撤权回 planner | 1515.316 s | 1516.290 s 切回 `pose` | 973.6 ms | 确认时 10 槽全部为 `upstream` |

- 接入延迟包含衣服端 2 s 静止门、token 服务 1 s 预热和 10 槽取样窗口。
- 撤权前后的 planner 段（281 个和 229 个周期）全部 2810 个和 2290 个槽位都是 `upstream`，没有出现 `hold`、`blend` 或 `cold_idle`。`pose_received_total` 在撤权后从 719 持续增长到 935。按审计约定，`upstream` 标签只证明 watchdog 在转发上游消息，上游是否就是 kplanner 需要另外核实。现场只有一个进程在 5556 上发布，就是 kplanner（PID 15176）。
- 穿戴者抬左臂，左腕 z 坐标从 −0.066 m 升到 +0.497 m，右腕变化 0.023 m。token 段内左肩 pitch 目标为 −37.9°~+21.4°，左肩 roll 为 +7.3°~+60.6°，左肘为 −149.0°~−22.1°；右肘为 −45.5°~−25.0°，没有明显跟随。撤权后各关节回到 planner 段的基线。

### 风险点（只读开环，不能直接当作真机表现）

| 关节 | planner 段 | token 段 | token 段单步最大 | 撤权首步 |
| --- | --- | --- | ---: | ---: |
| waist_roll | +0.3°~+2.5° | **−25.8°~+25.8°**，两端都打到限幅 | 14.45° | 23.37° |
| right_ankle_roll | +2.2°~+3.4° | **−40.1°**~+2.6°，打到腿部限幅 | 2.98° | 36.76° |
| left_elbow | −38.9°~−38.1° | −149.0°~−22.1° | 17.82° | 110.85° |

- 抬左臂时（1513.47–1513.57 s），waist_roll 目标在 5 个周期内从 −25.78° 甩到 +25.78°。
- 撤权首步很大，原因是这个分支在切换时把软启动 ramp 重置到实测关节角。机器人全程没动，实测一直停在站姿，所以目标会直接跳回站姿。真机上实测会跟着走，这一步的大小取决于闭环跟随的情况。
- `--no-publish` 下实测姿态始终停在站姿，反馈和目标越差越远，所以表中的偏差都是开环累积的结果，不是闭环跟踪误差。闭环下是否还会打满限幅，需要另做 MuJoCo 闭环回放来判断。

### 本轮定位的生产链路问题

1. **关节坐标系**：`SMPLXConverter` 的 fast FK 给根节点左乘了 `T=[[1,0,0],[0,0,-1],[0,1,0]]`，`GarmentIntentAdapter.root_local_joints` 去掉根旋转后，得到的关节坐标仍在 SMPL 身体系（x 左、y 上、z 前）。线协议要求 x 前、y 左、z 上。因此每次接入都会被 token 服务的站立检查以 `CALIB_SUSPECT knees not below hips` 释放（`run_live_stand`）。中继按 `M=Rz(90°)·T` 转换后，110 帧实时数据全部通过站立检查，并成功接入（`run_live_fixed`、`run_live_arm`）。11-14 记录里“自然站立映射成扭曲下蹲”可能是同一个原因，待核对。
2. **token 服务时钟**：`TokenServiceCore` 的内部时钟从 0 开始，每个 tick 加 0.02 s，而输入数据用 `time.monotonic()`（约 1000 s 量级）打时间戳。结果输入新鲜度检查永远判为新鲜，取样窗口一直卡在第一帧，输入停止后仍以约 50 Hz 继续发 token（`run_live_fixed`：输入 0 Hz、token 49.9 Hz，19 s 内一直没有撤权）。在循环里把内部时钟对齐到 monotonic 后，撤权正常（`run_live_arm`）。
3. **衣服进程退出**：按 Ctrl-C 后进程以 `Aborted (core dumped)` 退出，BlueZ 仍保持两件衣服 `Connected: yes`，下次启动报 `BleakDeviceNotFoundError`。手动 `bluetoothctl disconnect` 后衣服恢复广播。`run_garment_fg.sh` 在启动前和退出时都会断开这两个连接。

### 五轮录制

| 录制 | 输入 | 结果 |
| --- | --- | --- |
| `run_smoke2_213229` | 保存的 `real_worn_globalpose.jsonl`，kplanner 尚未启动 | 2 次确认（1864.8 ms / 516.4 ms），来源为 `cold_idle`，补丁链路验证可用 |
| `run_live_stand_214240` | 真人，未加中继 | 被站立检查释放，未切头（Bug 1） |
| `run_live_fixed_214511` | 真人，加中继 | 切到 token，但输入停止后一直不撤权（Bug 2） |
| `run_live_cycle_214739` | 真人，加中继和时钟修正，穿戴者处于屈膝姿态 | `ankles not below knees`，站立检查按设计拦截 |
| `run_live_arm_215153` | 真人站立，抬左臂 | 完整 `pose → token → pose`，见上文 |

### 已落实的板端修复与验证

三个生产问题已做成正式补丁并安装到 SoC1（原件均保留为 `.bak_20261116_pre_sonic_fix`，详见 [script/x2_sonic_live/](../script/x2_sonic_live/README.md)）：

- **坐标系修复（`garment_intent_adapter.py`）**：默认 `source_frame="smpl_fk"` 按 `M = Rz(+90°) · T` 将 `SMPLXConverter` 的 Y-up 输出转为 SONIC Z-up，保留 `canonical` 参数兼容；`garment_sonic_relay.py` 标记为 `LEGACY-ONLY WORKAROUND`。
- **token 时钟修复（`x2_smpl_token_service.py`）**：以 `now=time.monotonic()` 初始化 `TokenServiceCore`，每周期经 `sync_and_tick(core, now)` 对齐单调时钟；`token_service_core.py` 算法未改。
- **衣服退出释放蓝牙（`garment_udp_v2_reconnect_safe.py`）**：接入 `SIGINT`/`SIGTERM`、`_STOP_EVENT`、守护线程、`atexit` 及 `shutdown_garment_session`（调 `BleV2Manager.close_all()`、`pico_pub.close(0)` 及 `bluetoothctl disconnect` 兜底）。
- **板端全量验证（`install_onboard.sh`）**：5 份备份及安装后 SHA-256 校验通过；`test_garment_adapter.py`（35/35，含真实 `SMPLXConverter`）、`test_heading_guards.py`（`VERDICT: PASS`）、`test_token_chain.py`（46/46）、[test_onboard_fixes.py](../script/x2_sonic_live/test_onboard_fixes.py)（4 tests OK），以及 `garment_udp_v2_reconnect_safe.py` 的 `py_compile` 与 `--help` 均零失败通过。

### 保存位置

- 本地证据包 `data/analysis/X2_SONIC_LIVE_HANDOFF_20261116/`，内含板端原件快照及 SHA-256、诊断脚本、五轮原始记录和各轮摘要；另有压缩包 `.tgz`，SHA-256 `1a348b03…72db`。
- 脚本在 [script/x2_sonic_live/](../script/x2_sonic_live/README.md)：`build_diag.sh` 把诊断后端编进持久目录 `/home/agi/x2_sonic_live/diag_ws`（避免重启后丢失）；`run_shadow_live.sh` 一条命令完成隔离录制并审计；`run_garment_fg.sh` 是带蓝牙释放的衣服前台入口；`summarize_capture.py` 生成摘要；`install_onboard.sh` 执行带备份安装与全套验证。
- [会话记录](../logs/session_records/x2_sonic_live_handoff_20261116.json)。

## To do

- 用本轮人体流在 X2 MuJoCo 里闭环回放，判断 `waist_roll` 和 `right_ankle_roll` 打满限幅是开环造成的假象还是真实风险（用户先切入真机核验，仿真回放延后）。
- 现场补测右臂单侧、双臂、以及 30 s 以上静止，并看实机动作表现（注意：如需跑已打补丁的 `x2_smpl_token_service.py`，需使用已更新的板端入口启动/重启该服务）。
- 带电前：收紧腰和腿的限幅；撤权回 MC 时加 `--stand-default-pose`；现场有人扶着并能随时急停。
