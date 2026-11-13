# X2 Morph 后端适配：第二、第三终端

本目录把 Morph 策略服务的建议目标转换为 X2 后端可核查的原生 31 槽数据，始终 `publish_allowed=false`。没有电机 publisher、MC 服务调用或真实接管入口。`cpp/native_adapter.hpp` 给现有 C++ 后端提供原生数组适配，但尚未编译接入其 ROS 电机 writer。

## 数据路径

```text
衣服进程 ── HumanFrame ──> 第二终端：x2_morph_human serve
                                   ↑              │
                          X2Measurement      原生31关节建议目标
                                   │              ↓
                      第三终端：x2_morph_backend
                         ROS四组反馈+IMU → 校验 → 四组候选数据/日志
```

第二终端负责人体窗口、观测历史、ONNX 和唯一一次 default/scale/sign/perm 映射；第三终端按名称整理反馈，校验返回模型、序号、原生顺序和有限性，再按腿12/腰3/臂14/头2分组。不会再做 SONIC 动作缩放、腕映射、默认姿态叠加或头部 bypass。PD 数值只取导出 JSON 作模型元数据，尚非核准的真机增益。

拒绝陈旧/缺失反馈、重复序号、错误模型/关节顺序、非有限数值、缺失人体窗口和超过 RPC 预算的结果；拒绝时不产生零目标或沿用旧目标。RPC 通信失败后锁定，需操作者明确恢复与重置策略，后台不会自行 reset 另一个进程的人体流。

## SoC1 入口

Mac 已支持原生有线 `ssh agi@10.0.1.41`；SoC1 项目根为 `/home/agi/x2_morph_human_20261113`。2026-11-13 现场 SDK 为 `/home/agi/aimdk_msgs_pc2`，消息包含 `header/state/joints`。`run_orin.sh` 仅对自身进程组合项目 Python3.10、ROS Humble 和 SDK 路径，不安装或改写系统 Python。

第二终端在该项目根启动服务；先核现有进程，避免对已运行服务重复启动或 reset：

```sh
sh script/x2_morph_human/run_orin.sh serve \
  --endpoint tcp://127.0.0.1:5765 \
  --max-freshness-ms 120 --max-interp-gap-ms 100
```

第三终端可先只读采样，不请求策略，不使用示例外参进行推理。输出文件必须尚不存在，父目录需已准备：

```sh
sh script/x2_morph_backend/run_orin.sh ros --inspect \
  --profile script/x2_morph_human/config/robot_shadow.example.json \
  --seconds 3 --output data/analysis/my_feedback.jsonl
```

完成实测安装体/外参核查并保存独立 profile 后，去掉 `--inspect`、改用该 profile，可进行真实反馈到建议目标的只读联调。代码拒绝把 `REPLACE...` 示例标定当成实测。IMU `frame_id=base_link` 本身不能证明其安装体或外参。

## 离线联调

项目 Python 下运行；`X2_MORPH_HOME` 可指向已核验的模型资产根：

```sh
python -B -m unittest script.x2_morph_backend.test_contract -v
python -B -m script.x2_morph_backend.probe --output data/analysis/my_terminal23_probe
```

probe 自动创建独立策略服务与后端客户端、使用临时 loopback 端点及合成人体/固定合成反馈，检查断流/恢复/策略退出，并编译 SDK-free C++17 适配器逐目标核同。不会启动 ROS、既有启动仪式或真实 writer。仅清理自己创建的进程。probe 的通信超时为1000ms，候选接受预算仍为20ms；这不是真机时限配置验证。

2026-11-13 实测：Mac/Orin各11单测通过；独立联调各83请求、64/61个原生候选与C++逐值相同。Orin成功候选RPC p95 13.197ms，非整套遥操端到端延迟。另有3秒真实反馈149完整快照通过字段/名称/数值/接收年龄检查；其策略请求数0。详细结果见 `reports/README_X2_MORPH_BACKEND_TERMINAL23_20261113.md`。

## 真实控制剩余门

1. 实测 IMU 安装体、姿态和角速度外参，验证腰部状态参与的 pelvis 重建。
2. 在独立副本接入现有50Hz策略回调与独立writer；禁止在高频writer里等待RPC，单独验证阻塞/过期/进程退出时的行为。
3. 核对原生关节零位/限位、Morph PD 与真机电流及增益配方、接管初姿；不能直接使用 Sonic 配方。
4. 对照现场现存启动参数验证 MC 接管和退出。现存仪式的旧SIGSTOP注释与默认软退出开关存在漂移，不能因为历史成功就直接用于新策略。

当前窗口仍需180ms真实人体支持；不在本目录缩短预览、不恢复Q1训练、不把上述只读测试称为真机遥操通过。
