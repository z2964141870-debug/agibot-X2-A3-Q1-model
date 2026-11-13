# X2 Morph 原生 Writer、故障与 MC 交接

记录日期：2026-11-13。阶段：TERMINAL23_NATIVE_R1。状态：软件接入、板端编译、隔离 SDK 验证与真实反馈静默推理完成；真实电机接管未执行。

## 范围与实现

用户在本侧会话明确要求假定 SMPL 稳定，复用已工作的 sonic_x2，补完第二、第三终端。实际修改仅独立 `script/x2_morph_backend/` 与本阶段记录；衣服/FGP、原 Sonic 安装和 Q1 暂停训练未改。本轮保持真实电机输出和 MC 修改为零。

新增 C++20 控制器直接链接设备原版 `AimdkIo`，50Hz 策略线程与500Hz writer 分开，writer 不等待 RPC；高频反馈读取采用 TrySnapshotState，保留快照仍按其原接收时间过期。四组为腿12/腰3/臂14/头2。Morph 原生目标、零目标速度和导出模型 PD 精确发送，不再叠加 Sonic 缩放、零位或腕映射。

正常切换协议：新鲜人体/反馈/候选与完整 MC 命令 -> 确认 STAND -> 冻结 MC 站立返回姿态 -> 确认 DAMPING -> EM stop/实际 Stopped/总线静默 -> 显式 arm/当前实测姿态渐入。正常返回为渐回所捕获 MC 姿态 -> HOLD -> EM start -> 必要时 PASSIVE 转 JOINT -> 实际外部总线接管/自身静默 -> STAND 确认。服务 ACK、旧 PID、控制器启动均不当成接管成功。

故障包括反馈/目标过期、RPC 超时或策略退出、非有限值、异常速度/侧倾、控制管道关闭及返回途中急停。锁定后 kp 为零，kd 使用原 Sonic `BuildPureDampingCommand` 的厂商阻尼，不能被新目标自动复位。发现外部 writer 即停止自身发送。launcher 不主动杀掉唯一阻尼 writer，也不在故障后自动站起。

## IMU 与目标语义

已核原 Sonic 实际读取 `/aima/hal/imu/torso/state` 的 wxyz/本体角速度；固定 XML 的 `imu_1` 位于 torso_link、安装旋转单位阵。独立 profile 保留该来源与 SHA，使用既有 FK 和腰关节速度重建 pelvis 姿态及角速度。标记为源码约定，不冒称已做新物理标定；`frame_id=base_link` 也不单独当成安装证明。

首轮真实反馈测试发现新 writer 误把 PD 虚拟平衡角按机械位置范围硬拒绝：295请求之前的 R1 共289回复均ready，仅127接受、162误拒。原上游 `PolicyRunner.step` 返回未按位置范围裁剪的 PD 目标，`SimRobot.dc_torque` 在电机力矩/速度曲线上饱和；已保存的不跌倒仿真也包含大量超范围 PD 目标。故最终实现保留该项为诊断，精确传递目标，仅对捕获的实际 MC 返回姿态核位置范围。没有为提高通过率裁剪、平滑或改动策略。R1原件完整保留；硬件执行器饱和/电流限制与训练 plant 的一致性仍待现场验证。

## 实际验证

| 检查 | 结果 |
|---|---|
| Python 边界/交接协议 | Mac、Orin 各23项通过 |
| C++ writer 状态机 | Mac、Orin 各19用例通过，Release测试显式保留assert |
| Orin ROS2/AimDK 构建 | R6成功，独立新二进制；原 Sonic 不重编译覆盖 |
| 隔离 SDK 正常返回/策略断连/反馈断流 | 三场景通过；四组原生目标与增益精确同，故障四组厂商阻尼、拒绝自动重arm、实际外部接管后停发 |
| 真实反馈到真实 Morph ONNX | 6秒观察，295请求/295 ready/294接受；合成人体311帧 |
| 接受请求 RPC | 中位8.971ms、p95 13.690ms、最大18.491ms |
| 全部请求长尾 | 1次22.570ms超过20ms，已拒绝；未放宽时限 |
| 实际电机发送 / MC 修改 / 新训练 / 物理仿真 | 全部0 |

真实MC通过类型化 GetMcAction 及只读EM查询，运行PID2251、模式PASSIVE_DEFAULT。真实机器人当前不满足 STAND 接管前提，本轮没有为测试改变模式。硬件模式另需显式 `--control --confirm-live X2_ULTRA_MORPH_CONTROL` 和当次run_id、独占锁、stdin授权；默认observe永不调用电机发送。

前两次编译分别暴露 ament 的 link signature 和 Sonic 全局关节名命名空间差异，已修复；随后补齐线程异常回收、非阻塞高频快照、返回前捕获冻结、故障厂商阻尼与中途急停。中间构建/测试根保留，最终证据使用R6二进制、ISOLATED_R3与LIVE_FEEDBACK_R2。测试仅清理自有进程，不重启共享服务。

## 路径与复现

设备：Mac有线原生SSH直连 `agi@10.0.1.41`。SoC1为Orin NX；项目根 `/home/agi/x2_morph_human_20261113`。原Sonic来源 `/home/agi/gear-sonic/ws/src/x2/agi_x2_deploy_onnx_ref`。原IO SHA `b1f31a59dc7d22e165452b532943a7de7b88287c6e4371bc69f294358bc5ce35`；原安全源码 SHA `c5f290cdb2c9448fb42a97e4e74dbcc5dd23f6e268875a1df3a29dcd7a844daa`。模型 SHA `cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512`。

```sh
bash script/x2_morph_backend/build_native.sh data/analysis/MY_BUILD
sh script/x2_morph_backend/run_orin.sh sensor-profile \
  --sonic-source /home/agi/gear-sonic/ws/src/x2/agi_x2_deploy_onnx_ref \
  --output data/analysis/MY_SENSOR.json
sh script/x2_morph_backend/run_orin.sh mc inspect
sh script/x2_morph_backend/run_orin.sh native-probe isolated \
  --binary data/analysis/MY_BUILD/x2_morph_native \
  --profile data/analysis/MY_SENSOR.json --output data/analysis/MY_ISOLATED
```

真实反馈静默检查将最后一条的 `isolated` 改为 `real-feedback`，使用新输出根。第三终端常规入口 `run_orin.sh control --binary <binary> --profile <profile> --output <新目录>` 默认也只读，第二终端仍是原 `x2_morph_human serve`。详细操作见[后端README](../script/x2_morph_backend/README.md)。

设备原件在 `data/analysis/X2_MORPH_NATIVE_BUILD_R6_20261113`、`X2_MORPH_NATIVE_ISOLATED_R3_20261113`、`X2_MORPH_NATIVE_LIVE_FEEDBACK_R2_20261113`；profile为 `X2_MORPH_NATIVE_SENSOR_V2_20261113.json`。日志为 `logs/x2_morph_native_*_20261113.log`。Mac原件与数值归档、逐件SHA及同步回执见[结果收据](../logs/session_records/x2_morph_native_writer_20261113.json)和后续publication。

## 结论与边界

第二、第三终端的 writer、故障锁定和 MC 交接软件已接通，真实机器人反馈能驱动真实 Morph 模型生成原生目标。隔离话题和假MC传输的成功不等于真实机械接管成功；源码约定也不等于物理外参重新标定。1次RPC超时仍保留，现有180ms人体窗口不变，不认证整链50Hz、自然遥操或硬件端到端延迟。尚需现场受控的实际输出、执行器/增益与接管退出验收；FGP和衣服稳定性仍由主线单独确认。

本阶段没有训练或新增物理/媒体，不新增训练六宫图或重复已有视频。源码、README、索引与小记录按Q1分支发布，三处同步以后续publication为准；数值原件为本地保存，网盘状态LOCAL_ONLY。
