# X2 Joint Feedback Missing: Read-Only Diagnosis

Observed: 2026-10-23, 16:08-16:18 Asia/Shanghai.

## Finding

The immediate observed fault is an absent EtherCAT HAL process, not merely
joint topics hidden from a remote DDS reader. The historical startup success
does not describe the current operating state. This does not establish
physical hardware damage or identify who requested the HAL shutdown.

## Access and Scope

- Reused the existing Windows bridge tunnel at 127.0.0.1:22041/22040.
- Actual SoC1 develop0 address: 10.0.1.41/24. The supplied 10.1.0.41 was not
  used as a verified robot address.
- Both login banners identify X220025BBZ0029 and release v1.0.10. These
  banners alone do not establish the SDK or full component compatibility.
- SoC0 current log directory: /agibot/data/log/log_20261023_151238.
- No control mode changes, signals, service changes, file uploads, firmware
  changes, custom command publishers, or motor-enable operations were made.
- Diagnostics bypassed the ROS CLI daemon for fresh graph discovery. No
  ROS daemon was stopped or restarted.

## Evidence

1. SoC0 process inspection found MC PID 3195, command
   `./aima-mc-app-main`, and IMU PID 4027. No EtherCAT HAL application was
   found in the process list. PID 2339, identified by the current boot HAL
   log, was absent when checked explicitly. Kernel EtherCAT IRQ threads
   remain visible; these are not proof that the userspace HAL is running.
2. Fresh SoC1 DDS discovery showed joint state topic names, but all four
   leg/waist/arm/head state topics had zero discovered publishers. Leg
   state had three subscriptions, which explains why its topic name remains
   discoverable despite the missing source.
3. All four joint command topics had a discovered publisher named
   `mc_ros2_node3195`. This was graph evidence, not verification of actual
   command delivery, control frequency, or effective motor control.
4. Both IMU topics had publisher `soc0_hal_imu4027`. A subscription-only
   probe on SoC1 received torso=1489 and chest=1487 Imu messages over
   3.000788 s. Thus this observation path was carrying real data; a complete
   DDS discovery/transport outage is not a sufficient explanation.
5. Current boot EtherCAT HAL log explicitly records an orderly shutdown:

```text
2026-10-23 15:41:02.724989  Received shutdown signal.
2026-10-23 15:41:02.824944  Exit Mainloop
2026-10-23 15:41:04.915475  disable ethercat slave node: x2_upper_ethercat_node success.
2026-10-23 15:41:07.000472  disable ethercat slave node: x2_leg success.
2026-10-23 15:41:07.100588  ethercat disable servo for all slave nodes.
2026-10-23 15:41:07.300743  ethercat controller stop success.
2026-10-23 15:41:07.749286  Shutdown success.
```

Source: SoC0
`/agibot/data/log/log_20261023_151238/hal_ethercat/hal_ethercat_x2.log`.
The excerpts above are selected timestamp/message fields; consult the
original for full severity, thread and source-location fields.

Earlier entries in that same file show controller startup and slave enable
at 15:13. They cannot establish controller health after the 15:41 shutdown.

## Limits and Next Step

- No evidence collected here proves hardware damage, a software handshake
  corruption, or the sender of the shutdown signal. The log does not name
  the signal number or the caller. Do not attribute it to a person/agent or
  claim it was a spontaneous crash.
- The short IMU test does not clear all DDS faults or prove joint transport
  will be healthy after HAL recovery.
- Checked housekeeper log entries around 15:41 report service-property
  updates but do not identify the shutdown requester.
- First reconcile whether the 15:41 shutdown was intentional and whether
  another operator/agent is maintaining the robot. Then inspect the official
  service supervisor and its recovery procedure. Starting HAL can re-enable
  motors, so recovery requires a separate supported, operator-coordinated
  step; do not automatically restart HAL, MC, or the whole robot.
- Do not retune Sonic, remap joints, clear shared memory, or upgrade firmware
  to address this missing-process symptom before resolving service ownership.

The two diagnostic SSH shells were closed normally after inspection; the
pre-existing bridge tunnel was left untouched. No files were changed on the
robot. This report is local documentation and has not been pushed to GitHub
as part of this diagnostic task.


---

## ✅ 结案（2026-10-23 18:20, 首走成功后）

三层真相, 全部实证:

1. **hal_ethercat 进程死亡**(15:41 误杀, em 未复活) — 发布者不在, 话题自然无数据。
   日志末行为旧进程 shutdown 记录、SHM 无段。15:57 的 preflight 失败即此因。
2. **站立模式前置** — PASSIVE(刚开机)下 hal 注册发布者但不对外写数据; L2+X
   (JOINT_DEFAULT)后状态流恢复。16:43 干净开机 + 17:43 站立 → preflight PASS。
3. **MC 反复停/启 → SHM 传输系统性卡死**(SDK 分析子代理 FastDDS 日志取证:
   正常开机 OpenedWrite 1.5万~6万次, 异常开机 ~20 次 + 431 次 SHM 硬报错) —
   这解释了下午多轮"干净开机仍不可见"的窗口。纪律: 每开机周期只停一次 MC。

另: 官方 SDK(aimdk-sdk v1.0.0 + x2_rl_deploy v1.1)证实配方一致 — joint state
订阅 aimdk_msgs/JointStateArray + best_effort(SensorDataQoS), command 500Hz,
接管前 aim em stop-app mc, 开发侧永远在 SoC1/10.0.1.41。CycloneDDS 仅用于
官方 MuJoCo 仿真脚本, 实机栈是 FastDDS。

当日修复(均已入库): deploy_x2.sh em 停机超时竞态(5s→30s)、键盘工具
target_velocity 字段序((yaw,lateral,fwd,hip_h))、FIELD_CARD v3。
