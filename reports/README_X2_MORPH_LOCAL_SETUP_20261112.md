# X2 MorphMimic human 本机系统搭建

> **2026-11-13 更新**：已完成旧版 EgoLocate FGP 与最新版两阶段 CLOTHO FGP（`U100h + GlobalPose/LIP`）的逐项对齐、无头推理接入、18/18 单测、实时 50 Hz ZeroMQ 丢包恢复及 8 组 X2 Ultra MuJoCo 闭环视频预测试，详见 [README_X2_LATEST_FGP_PRETEST_20261113.md](README_X2_LATEST_FGP_PRETEST_20261113.md)。


2026-11-12。状态：**本机开发链路已运行，真机后端绑定和Orin验证待做**。用户本轮从“只构思”推进到先在Mac搭系统；没有恢复应用goal、共享训练或真机测试。

## 给用户的done / todo

Done：独立Python环境、官方X2 human权重、SMPL标准骨架适配、十帧缓冲、真实ONNX策略、X2关节映射/IMU重建、MuJoCo开发回放、Python服务与C++客户端、可搬机代码资产包。已经有本地视频和六图诊断页，可以在没有机器人的情况下调试输入。

Todo：真实衣服标定/流、Orin软件ABI与时序、真机反馈外参、只读影子日志，最后把策略接进现有X2唯一writer和状态机。当前不是直接上电遥操成品；不会拿合成三秒回放代替这些验证。

## 目标与具体实现

代码为[script/x2_morph_human](../script/x2_morph_human/README.md)。现有固定源和暂停训练不改。链路：

```text
SMPL24 local axis-angle + root + capture clock
    → 标定后的标准骨架FK / HumanFrame v1
    → 十槽50Hz真实预览（当前到未来9槽，180ms跨度）
    → 官方PolicyRunner human / obs_history930 / goal_human2160
    → canonical action29 → 导出配置映射 → native target31
    → 本地advisory日志/开发仿真/只读影子接入组件
```

HumanFrame用24位置+24世界wxyz朝向，168值为逐关节xyz+wxyz交错。按捕获时钟重采样但按本机接收时间判断可见性，插值保留右端原始支持。未到齐、跨标定/时钟、错误骨架、不完整/非有限输入不输出建议目标。仍需180ms原预览，本轮没有降低未来帧数或恢复任何训练。

IMU适配解析导出FK，姿态与gyro同时从传感器体转pelvis，扣除腰部关节速度；用非零腰位/速度与非单位安装外参的独立有限差分验证。按名整理反馈；腕映射、动作尺度只应用一次，头保持发布默认，历史last_action保持原canonical定义。

模型、normalizer、human特征和策略前向直接使用固定官方实现，而非SONIC的840/990接口。源码/许可证/小型机器人JSON和真实ONNX的逐文件SHA在[资产清单](../script/x2_morph_human/assets_manifest.json)。官方[v0.1说明](https://github.com/lzyang2000/morphmimic/releases/tag/v0.1)只证明作者X2仿真，不证明X2硬件遥操。

## 实际运行结果

| 核验 | 实际结果 |
|---|---|
| 本机环境 | macOS arm64 / Python3.11.15，独立环境，无训练Torch/mjlab安装 |
| 真权重/接口 | policy SHA `cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512`，human输出29→native31有限 |
| 15项测试 | 0.255秒全过，含原PolicyRunner逐位同动作、坐标FK、右支持/可见性、ring eviction相位、反馈名排序与gyro差分 |
| 固定合成状态回放 | 150次真实策略调用，9包预览启动等待；完整runtime p50 1.164 / p95 1.398 / p99 1.612 / max 1.769ms |
| 开发场景回放 | 150控制帧/3秒，600物理步，15.660秒含渲染；无跌倒，runtime p50 1.378 / p95 2.083 / p99 6.057 / max 7.249ms |
| 视频 | 640×480 / 50fps / 150帧全解码；六时刻联系表已目视 |
| C++17客户端 | `-Wall -Wextra -Werror`编译成功；真正独立进程80 REQ/REP，31次ONNX目标，9次明确启动等待；0.757秒，全字段身份/序号/形状通过 |
| 实际训练/真机命令 | 0 / 0；原5036秒训练账本、共享暂停、final与采纳null不改 |

计时来自这台Mac的CPUExecutionProvider，包含本runtime预处理、特征和策略，**不包含衣服采集、跨机网络、HAL执行及硬件响应**。不推为Orin周期。视频是开发吞吐渲染、以50fps原速播放；15.660秒渲染时间不是控制延迟。

开发场景使用公开X2原始MJCF/mesh，施加发布X2 JSON的PD/effort和5ms×4步时序、独立floor；没有原训练环境`check_same_model`导出核同，solver/contact等差异尚未排除。合成人体仅标准stand加小幅双肘运动，无实际人体跟随数据；三秒无跌倒不当步态/自然遥操或真机合格。

## 可复现入口和工件

项目主根`/Users/yu/Documents/ChatGPT/元启`中运行：

```bash
bash script/x2_morph_human/run.sh doctor --meshes
bash script/x2_morph_human/run.sh replay --frames 150 --simulate --video --output data/analysis/x2_new_local_run
```

输出必须新目录。无需连接服务器或机器人。全部接口、输入NPZ约定、JSON消息、服务/只读ROS工具/C++编译和搬机步骤详见[开发包说明](../script/x2_morph_human/README.md)。

- 环境：`data/environments/x2-morph-human-20261112`；实际Mac锁在包内，Linux/Orin不使用Mac二进制环境。
- 原始资产：`data/runtime/x2_morph_human`；固定来源52个模型/源/mesh文件，加4个C++依赖/许可证，Git只存清单。
- 检查与原值：`data/analysis/X2_MORPH_LOCAL_SETUP_20261112/doctor.json`、`policy_replay/`、`development_simulation/`、`cpp_ipc_r2/`。
- [原速开发回放](../data/analysis/X2_MORPH_LOCAL_SETUP_20261112/development_simulation/preview.mp4) / [六图诊断页](../data/analysis/X2_MORPH_LOCAL_SETUP_20261112/development_simulation/report.html) / [视频联系表](../data/analysis/X2_MORPH_LOCAL_SETUP_20261112/development_simulation/contact_sheet.png)。这些是主根本地工件链接，Git不含媒体。
- 可搬机包：`data/deliveries/X2_MORPH_HUMAN_PORTABLE_20261112.tgz`，逐成员核SHA，大小/总SHA以本阶段[收据](../logs/session_records/x2_morph_local_setup_20261112.json)和包侧car清单为准。仅源码+资产，无Orin轮子、Mac环境、凭据或HAL writer。

## 故障和修正

C++首次实际启动因pyzmq wheel的Mach-O install name指向构建机`/DLC/…`失败，根`cpp_ipc`日志保留；不是网络/机器人故障。仅对本次生成的探针修改其依赖为`@rpath`并本地重签，未改系统配置或wheel。原样协议在新根`cpp_ipc_r2`已实际通过。

本地骨架/缓冲开发时核对了原始交错布局，并保持独立原PolicyRunner逐位同门；环形缓冲淘汰时的采样基准固定于首捕获时刻，新增测试防止长期流发生重采样相位漂移。没有改旧预览实验来源或补跑旧84等实验。

## Orin与后端剩余工作

首目标设备为X2 Orin SoC1：只部署策略，不在线跑训练WM。Python服务和C++客户端已经本地通信，ROS只读适配仅纯消息组装部分已Mac测试；`rclpy/aimdk_msgs`设备ABI尚未测。

实际JetPack/Ubuntu/Python ABI未知，因此Linux wheelhouse未伪造为可用。到现场先原生doctor/离线回放与只读反馈；确认真实衣服坐标、root地面、全24朝向和IMU安装外参。GPU/TensorRT在原模型动作等价与完整周期实测后选择，CPU模式作为确定基线。

将Morph接入现有x2后端仍需完成策略插槽及独立PD配方、MC交接/历史重置/单writer约束；本包没有擅自改现有SONIC写入链路。当前`x2`参考分支保持固定提交`79c4a1e7c019fe57130a2f530fcec78875eb7e4f`的只读来源；候选实现先存本Q1权威分支，不覆盖现场工作树。任何真机命令须后续明确任务和现场条件。

## 保存与恢复

本机代码、说明、原值和视频已保存；资产/环境/压缩包均LOCAL_ONLY，不声称网盘验证。重要代码/小记录按Q1分支发布，独立远端提交/Mac镜像核验见后续publication收据。服务器和Orin镜像本轮未尝试。原目标保持用户暂停；本次只完成本地搭建阶段，不结算低延迟自然遥操整体目标。
