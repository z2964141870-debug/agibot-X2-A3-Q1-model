# X2 Morph 后端适配：第二、第三终端

本目录把 Morph 策略服务的原生 31 槽建议目标接入 X2 后端。策略回复始终 `publish_allowed=false`；真实发送权限仅由第三终端显式取得。默认 `observe` 永不调用电机发送；`isolated` 只发送到独立测试话题；真实接管需 `--control --confirm-live X2_ULTRA_MORPH_CONTROL`，并通过 MC 交接前置检查。2026-11-13 已在 Orin 编译并验证独立 writer，现场电机输出和 MC 停启尚未执行。

## 数据路径

```text
衣服进程 ── HumanFrame ──> 第二终端：x2_morph_human serve
                                   ↑              │
                          X2Measurement      原生31关节建议目标
                                   │              ↓
                      第三终端：x2_morph_backend
                         ROS四组反馈+IMU → 校验 → 独立500Hz writer
```

第二终端负责人体窗口、观测历史、ONNX 和唯一一次 default/scale/sign/perm 映射；第三终端按名称整理反馈，校验返回模型、序号、原生顺序和有限性，再按腿12/腰3/臂14/头2分组。不会再做 SONIC 动作缩放、腕映射、默认姿态叠加或头部 bypass。正常 PD 来自 Morph 导出 JSON；故障采用已核 SHA 的 Sonic `BuildPureDampingCommand` 厂商阻尼，kp 为零。实机电流、增益及动作效果仍待现场测试。

拒绝陈旧/缺失反馈、重复序号、错误模型/关节顺序、非有限数值、缺失人体窗口和超过 RPC 预算的结果；拒绝时不产生零目标或沿用旧目标。RPC 通信失败后锁定，需操作者明确恢复与重置策略，后台不会自行 reset 另一个进程的人体流。

## SoC1 入口

Mac 已支持原生有线 `ssh agi@10.0.1.41`；SoC1 项目根为 `/home/agi/x2_morph_human_20261113`。2026-11-13 现场 SDK 为 `/home/agi/aimdk_msgs_pc2`，消息包含 `header/state/joints`。`run_orin.sh` 仅对自身进程组合项目 Python3.10、ROS Humble 和 SDK 路径，不安装或改写系统 Python。

第二终端在该项目根启动服务；先核现有进程，避免对已运行服务重复启动或 reset：

```sh
sh script/x2_morph_human/run_orin.sh serve \
  --endpoint tcp://127.0.0.1:5765 \
  --max-freshness-ms 120 --max-interp-gap-ms 100
```

第三终端可先只读采样，不请求策略。输出文件必须尚不存在，父目录需已准备：

```sh
sh script/x2_morph_backend/run_orin.sh ros --inspect \
  --profile script/x2_morph_human/config/robot_shadow.example.json \
  --seconds 3 --output data/analysis/my_feedback.jsonl
```

可使用实测 profile，或经 `sensor-profile` 核验的源码约定 profile。后者依据 Sonic 实际 torso IMU 接口和固定 XML 的 `torso_link/imu_1` 单位安装旋转，利用现有 FK/腰关节速度重建 pelvis 姿态及角速度。它没有被标成新物理标定；IMU `frame_id=base_link` 本身也不能证明实际安装。

```sh
bash script/x2_morph_backend/build_native.sh data/analysis/MY_NATIVE_BUILD
sh script/x2_morph_backend/run_orin.sh sensor-profile \
  --sonic-source /home/agi/gear-sonic/ws/src/x2/agi_x2_deploy_onnx_ref \
  --output data/analysis/MY_SENSOR.json
sh script/x2_morph_backend/run_orin.sh mc inspect
sh script/x2_morph_backend/run_orin.sh control \
  --binary data/analysis/MY_NATIVE_BUILD/x2_morph_native \
  --profile data/analysis/MY_SENSOR.json \
  --output data/analysis/MY_OBSERVE --seconds 10
```

最后一条默认只读。`mc inspect` 仅 HTTP GET 和 GetMcAction，不发 Set/POST。已验证构建为 `data/analysis/X2_MORPH_NATIVE_BUILD_R6_20261113`，设备 profile 为 `data/analysis/X2_MORPH_NATIVE_SENSOR_V2_20261113.json`。

## Writer 与控制权

独立 C++20 节点链接未修改的 Sonic `aimdk_io.cpp`，50Hz 推理线程通过本机 IPC 请求策略，500Hz writer 不等待 RPC，反馈读取使用 TrySnapshotState，陈旧快照按原接收时间过期。控制器默认无输出，硬件模式还有独占锁、当次 run_id 和显式 stdin 授权。

正常交接为：新鲜人体/反馈/建议目标 -> 确认 MC 在 STAND -> 冻结四组 MC 站立命令 -> 确认 DAMPING -> EM stop 及实际 Stopped -> 命令总线静默 -> Morph 从当前实测姿态渐入。正常退出为：渐回已捕获站姿 -> HOLD -> EM start -> 若 PASSIVE 则 JOINT -> 观察非自身 MC 命令后释放 writer -> 确认 STAND。不能用 HTTP/服务 ACK 或残留 PID 代替实际状态和总线证据。

目标过期、RPC 超时/断开、反馈过期、非有限数值、速度或侧倾异常会锁定故障；返回途中急停和控制管道关闭也进入阻尼。不会因新帧到来自动重启控制或站起。异常外部 writer 出现时立即停止自身发送。故障时 launcher 保持阻尼 writer 存活，要求操作者明确恢复 MC；不把杀进程当作交还控制权。

机械位置范围仅校验捕获的实际 MC 返回姿态，并记录 PD 目标的范围诊断。不能把虚拟 PD 平衡角当成实际位置命令硬拒绝或裁剪：Morph 上游 X2 是带力矩/速度饱和的 DC 电机 PD，已有不跌倒仿真中也存在超出机械范围的 PD 目标。writer 精确传递目标，实机执行器饱和/电流限制与模型 plant 的一致性仍属现场验收范围。

硬件入口是在上述 `control` 命令后增加 `--control --confirm-live X2_ULTRA_MORPH_CONTROL`。本轮没有执行该命令。现场 MC 当前只读查询为 PASSIVE，软件会拒绝直接接管；不自动把当前机器人切到站立。

## 离线联调

项目 Python 下运行；`X2_MORPH_HOME` 可指向已核验的模型资产根：

```sh
python -B -m unittest script.x2_morph_backend.test_contract -v
python -B -m script.x2_morph_backend.probe --output data/analysis/my_terminal23_probe
```

probe 自动创建独立策略服务与后端客户端、使用临时 loopback 端点及合成人体/固定合成反馈，检查断流/恢复/策略退出，并编译 SDK-free C++17 适配器逐目标核同。不会启动 ROS、既有启动仪式或真实 writer。仅清理自己创建的进程。probe 的通信超时为1000ms，候选接受预算仍为20ms；这不是真机时限配置验证。

2026-11-13 实测：Mac/Orin各11单测通过；独立联调各83请求、64/61个原生候选与C++逐值相同。Orin成功候选RPC p95 13.197ms，非整套遥操端到端延迟。另有3秒真实反馈149完整快照通过字段/名称/数值/接收年龄检查；其策略请求数0。详细结果见 `reports/README_X2_MORPH_BACKEND_TERMINAL23_20261113.md`。

## 验证与现场边界

23 项 Python 协议/边界测试与 19 项 C++ writer 用例已过；Orin 独立 ROS 话题完整验证正常交还、策略退出及反馈丢失，四组输出与原生目标/模型增益逐值同，故障为四组零 kp 加厂商阻尼，自身消息没有被误认成外部 MC。MC 协议状态切换的覆盖使用假传输，真实 ROS 服务只进行了读取。

可复现 SDK 测试：`run_orin.sh native-probe isolated --binary <已编译binary> --profile <profile> --output <新目录>`；真实反馈到真实 ONNX 的静默检查改用 `native-probe real-feedback`。后者仍使用假定稳定的合成人体输入，不代替衣服/FGP 验收，也不证明实际机械安装和硬件端到端延迟。

现场剩余为真实输出下的受控接管、关节动作/增益和故障恢复验收。原 Sonic 启动脚本、安装二进制和主线衣服代码保持原样。最新阶段证据见 `reports/README_X2_MORPH_NATIVE_WRITER_20261113.md`；之前 advisory 阶段结果保留为历史。

当前窗口仍需180ms真实人体支持；不在本目录缩短预览、不恢复Q1训练、不把上述只读测试称为真机遥操通过。
