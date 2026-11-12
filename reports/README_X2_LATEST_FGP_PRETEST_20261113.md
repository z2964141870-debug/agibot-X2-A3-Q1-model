# X2 MorphMimic human：最新 CLOTHO FGP 对齐与本机闭环预测试报告（2026-11-13）

本报告回答并落地两件事：
1. **明确对齐旧版 EgoLocate FGP 与最新版 CLOTHO FGP**：核实此前本地测试样例中的历史内嵌姿态流来自旧版单阶段 FGP（`EgoLocate/mocap/assets/weights.pt`，根平移固定为 `[0, 0, 0] m`），并已将用户指定的最新版两阶段 **CLOTHO FGP**（`CLOTHO-mocap-main (1).zip` + `checkpoints.zip`）完整无头接入 `script/x2_morph_human/`。
2. **在去真机现场前完成全套本机预测试**：完成 18/18 单元测试、C++17 零 SDK 异步 IPC 探针、便携发布包 V2 净室校验、8 组 X2 Ultra MuJoCo 闭环仿真 + 50 fps 原速视频、真实 30 Hz→50 Hz ZeroMQ 异步插值与 168 ms 蓝牙丢包恢复验证，以及三张六宫图/视频联系表交付。

> **安全边界保持**：本阶段仅限本机离线推理、开发级 MuJoCo 闭环仿真与本地 `127.0.0.1` IPC 验证；`publish_allowed=False`、电机控制写入 `hardware_commands=0`、训练服务器暂停状态保持不变（`training_updates=0`）。开发级 MuJoCo 场景不冒充官方未公开的训练动力学等价性（`original_training_plant_equivalence=NOT_VERIFIED`）。

---

## 一、旧版 FGP vs 最新版 CLOTHO FGP 逐项核对

| 维度 | 旧版 FGP（`EgoLocate.zip` / 历史内嵌流） | 最新版 CLOTHO FGP（`CLOTHO-mocap-main (1).zip` + `checkpoints.zip`） | 对 X2 MorphMimic human 遥操的影响 |
|---|---|---|---|
| **推理架构** | 单阶段网络直接回归姿态 | **两阶段流水线**：Stage-1 11-IMU 规范化去噪 → Stage-2 全身 SMPL + 3D 根位移 | 显著抑制穿戴传感器漂移与高频抖动 |
| **权重文件与大小** | `EgoLocate/mocap/assets/weights.pt`（`23,224,559 B`） | Stage-1：`U100hL7.07h_residual.pth`（`384,845,450 B`，SHA `b613eb82...`）<br>Stage-2 主选：`globalpose/weights.pt`（`134,051,421 B`，SHA `b63f62b8...`）<br>Stage-2 对照：`LIP_10_real_denoised_7.07_10imu_10.pth`（`26,748,498 B`，SHA `efc4bb85...`）<br>体模：`SMPL_male.pkl`（`39,056,404 B`，SHA `0e8c0bbb...`） | `U100h + GlobalPose` 在相同穿戴 11-IMU 实测流上较 `LIP` 降低 X2 支撑脚滑移 **>56%**（`p95` `22.45` vs `51.90 cm/s`），定为主选前端 |
| **骨盆 3D 根平移** | 历史录制 JSONL 固定 `translation_m = [0, 0, 0] m` | 输出真实米制 3D 根位移（实测穿戴流跨度 `[0.067, 0.171, 0.078] m`） | MorphMimic human 168D 输入同时依赖 24 关节世界位置与旋转，真实根位移消除“原地悬空踏步”冲突 |
| **标定与 Unity 显示 Hack** | 旧桥接混合录制世界系 | `ClothoCalibrator` 精确复现 `Socket/UDP.py` 标定（float32 误差 `0.0`），**剔除** `UDP.py` 仅供 Unity 渲染的 `tran[2] *= 0.7; tran[1] += 1.0` 失真，经 `config/clotho_fgp_yup.json` 转入 Z-up 地面接触系 | 保持物理真实步幅与骨盆高度，避免传入策略前引入 0.7 倍缩放与 1 m 悬空偏置 |
| **同源 11-IMU 姿态差异** | 基准对照 | 相对旧版 FGP：Active-18 测地角差均值 **`17.32°`**（`GlobalPose`：骨盆 `23.33°`，肩肘 `23.68°`）/ **`19.99°`**（`LIP`） | 证实旧版与最新版输出分布存在显著差异，接入真机前切换至最新 CLOTHO FGP 十分必要 |
| **4 线程 CPU 稳态耗时** | ~3–5 ms | `GlobalPose`：中位 **`30.24 ms`**（`p95=31.54 ms`；去噪 `27.91 ms` + 姿态 `2.25 ms`）<br>`LIP`：中位 **`28.03 ms`**（`p95=28.74 ms`） | 均低于 30 Hz 蓝牙衣服帧周期（`33.33 ms`），可在独立进程实时运行，不阻塞 50 Hz 控制环 |

---

## 二、本地工具链升级、单元测试与便携包净室验证

1. **新增/完善模块（`script/x2_morph_human/`）**：
   - `clotho_fgp_runner.py` + `config/clotho_fgp_yup.json`：无头（Headless，零 Unity 依赖）加载 `CLOTHO-mocap-main` 与 `checkpoints`，支持 `--backend {globalpose,lip}` 从原始 11-IMU 流直接推理导出标准 NPZ 与 `HumanFrame v1` JSONL。
   - `fgp_bridge.py`：支持旧版/录制 FGP JSONL 转换、30 Hz→50 Hz 严格因果插值，以及实时 ZeroMQ 异步桥接压测。
   - `runtime.py` & `__main__.py`：新增 `--max-freshness-ms`（与 `--max-interp-gap-ms` 解耦，允许按新鲜度保持短时安全窗口）和 `--human-points {full,five}`（24 全关节 vs 5 关键点稀疏输入对照），修复多行字符串转义瑕疵。
2. **自动化测试与 C++17 IPC 验证**：
   - `test_runtime.py`：**18/18 项单元测试全部通过（`0.797 s`）**，覆盖坐标轴映射、168D 交错编码、真实 ONNX 策略确定性、Torso→Pelvis 姿态/角速度转换、最新 CLOTHO FGP 标定一致性与异步丢包保持。
   - C++17 探针 `cpp/advisory_probe.cpp`：`-std=c++17 -O2 -Wall -Wextra -Werror` 零警告编译通过，零厂商 SDK 耦合完成 `CPP_LOCAL_IPC_PASSED`（80 次请求、31 次真实策略推理、启动等待 9 帧、二进制 SHA `a153f05bc791b3c1cf1b82143c170c0afcafb05547e365519f0bc0580d7d7131`）。
3. **便携部署包 V2 净室验证**：
   - 产物：`data/deliveries/X2_MORPH_HUMAN_PORTABLE_V2_20261112.tgz`（`99,500,160 B`，80 个成员，SHA `802a0f0a791520ef06f932369527578d1e0ada2542441c000b6ca8296aeb73c4`）。
   - 在全新临时目录解压执行 `doctor --meshes` 与 `18/18` 单元测试全部一次通过，不含 Mac 虚拟环境污染，可直接拷至机器人侧 Ubuntu/Orin 机器建环境启动。

---

## 三、8 组 X2 Ultra MuJoCo 闭环预测试结果汇总（全部 0 跌倒）

所有仿真均采用真实导出的 `human_x2_ultra_isaac` ONNX 策略（SHA `cf951947...`）在 50 Hz 控制频率下驱动 X2 Ultra 31 自由度模型，产物位于 `data/analysis/X2_LATEST_FGP_PRETEST_20261112/runs/`：

| 运行 ID | 输入源与场景 | 就绪帧 | 启动等待 / 中途 Hold | 跌倒 | 骨盆最低 / 中位高 (m) | 最大倾角 (°) | 支撑滑移 p95 (cm/s) | 策略耗时 p50 / p95 (ms) | 核心发现 |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| `real_worn_globalpose` | 实测穿戴 11-IMU → **最新 GlobalPose** | 991 | 11 / 0 | False | 0.499 / 0.544 | 23.28 | **22.45** | 1.92 / 2.87 | 全程稳定闭环，左右脚最大离地 `3.40 / 8.71 cm` |
| `real_worn_lip` | 实测穿戴 11-IMU → **最新 LIP** | 991 | 11 / 0 | False | 0.508 / 0.554 | 22.70 | 51.90 | 1.89 / 2.73 | 同源输入下支撑滑移是 `GlobalPose` 的 2.31 倍，淘汰为备选 |
| `real_manual_steady_gp` | 实测桌面/手持平稳段 → GlobalPose | 1205 | 10 / 0 | False | 0.359 / 0.376 | 11.57 | 54.15 | 1.95 / 2.90 | 大幅下蹲姿态（`z≈0.36 m`）下仍保持平衡不倒 |
| `real_manual_active_gp` | 实测相机并发蓝牙抖动流（`dt_max=107.5 ms`） | 1724 | 11 / **21** | False | 0.479 / 0.667 | 22.38 | **11.44** | 2.60 / 3.05 | 真实 BLE 卡顿超过 `100 ms` 门限时精准触发 21 帧 `NotReady` 保持，零外推乱跳并平滑恢复 |
| `synth_walk_start_stop` | 合成起走—前走—停走标准序列 | 491 | 9 / 0 | False | 0.648 / 0.667 | 4.81 | 43.60 | 2.56 / 3.39 | 水平净推进 **`1.248 m`**，验证根平移与步态起停链路 |
| `synth_turn_wave_full` | 合成转身挥手（`full` 全 24 关节） | 391 | 9 / 0 | False | 0.662 / 0.670 | 3.60 | **2.24** | 2.56 / 3.01 | 上肢肘部跟踪响应较稀疏模式显著更精细（左肘差 `>0.06 rad`） |
| `synth_turn_wave_five` | 合成转身挥手（`five` 5 关键点模式） | 391 | 9 / 0 | False | 0.662 / 0.670 | 3.20 | 2.58 | 2.48 / 2.93 | 证明 5 点退化模式亦可稳定站立，可作传感器局部失效降级手段 |
| `live_zmq_30to50_realtime` | **实时墙钟 50 Hz + ZeroMQ IPC + 注入 168 ms 丢包** | 280 | 10 / **10** | False | 0.558 / 0.668 | 19.86 | **4.51** | 2.12 / 2.41 | 真实墙钟周期 `p50=20.008 ms`（`p95=20.269 ms`），ZMQ 往返 `p99=2.650 ms << 20 ms`；连丢 4 包触发 10 帧（`200 ms`）安全保持后无缝恢复 |

> **工程反例留档（`runs/live_zmq_30to50_dropout`）**：当把 CPU 同步渲染 MP4 直接放进实时 50 Hz 控制线程时，渲染耗时拖慢循环导致人流新鲜度超限、连续 113 帧进入 stale hold。该对照直接给出两条真机铁律：**(1) 50 Hz 控制/IPC 线程严禁混入任何同步磁盘/视频重 I/O；(2) 真机后端遇到 `>300 ms` 长时间 `NotReady` 时，须从末帧关节保持平滑插值回双足站立名义姿态。**

---

## 四、可视化图表与交付压缩包清单

1. **三张六宫图与视频联系表**（`data/analysis/X2_LATEST_FGP_PRETEST_20261112/figures/`）：
   - `fig1_latest_fgp_frontend_6panel.{png,pdf}`：最新 CLOTHO FGP vs 旧版 FGP 测地角差、3D 根位移恢复、Stage-1/Stage-2 耗时分解与 BLE 帧间隔分布六宫图。
   - `fig2_x2_closed_loop_pretest_6panel.{png,pdf}`：8 组 X2 Ultra 闭环仿真的骨盆高度、最大倾角、支撑滑移 `p95`、ONNX 耗时、Full vs Five 肘部响应及实时 50 Hz ZMQ 丢包恢复时序六宫图。
   - `fig3_x2_pretest_videos_contact_sheet.{png,pdf}`：全部 8 组 50 fps 闭环仿真视频的多时刻抽帧联系表。
   - `pretest_metrics_summary.csv`：8 组闭环核心指标汇总表。
2. **证据交付包**：
   - `data/deliveries/X2_LATEST_FGP_PRETEST_EVIDENCE_20261113.zip`：打包本报告、`fig1/fig2/fig3`（PNG+PDF）、`pretest_metrics_summary.csv` 以及全部 8 组闭环仿真的 50 fps `preview.mp4` 与 `summary.json`（共 25 个文件，CRC 与 SHA-256 逐项校验通过）。
3. **机器人侧便携部署包 V2**：
   - `data/deliveries/X2_MORPH_HUMAN_PORTABLE_V2_20261112.tgz`：含最新 `script/x2_morph_human/` 源码、C++17 探针、配置及校验资产清单。

---

## 五、阶段总结（双版本 Done / Todo）

### 5.1 给用户看的简明版（Concise Summary for User）

**Done（已完成）**：
1. **核清并切换至最新版 CLOTHO FGP**：确认旧数据里嵌的是无根位移的旧版 EgoLocate 单阶段 FGP；现已把你给的最新版 `CLOTHO-mocap-main` + `checkpoints.zip`（`U100h` 11-IMU 去噪 + `GlobalPose`）完整无头接入本地链路，剔除了 Unity 显示缩放失真，CPU 单帧约 `30.2 ms`（满足 30 Hz 衣服实时预算），相比 `LIP` 将 X2 支撑滑移降低了 56% 以上。
2. **跑通全套本机 X2 闭环预测试与防呆验证**：完成 18 项单测、C++17 / ZeroMQ 实时 50 Hz 通信压测（IPC 往返仅 `2.1~2.6 ms`）、蓝牙抖动与 168 ms 丢包自动 Hold 恢复测试，以及 8 组 X2 Ultra MuJoCo 闭环仿真和 50 fps 视频（全部 **0 跌倒**），并打好了净室验证通过的 `V2` 部署包和证据 ZIP。

**Todo（去机器人旁后的下一步）**：
1. 把 `X2_MORPH_HUMAN_PORTABLE_V2_20261112.tgz` 和最新 FGP 权重拷到现场上位机/X2 Orin，先做只读接入：核对 X2 真机关节零位符号、IMU 安装体（torso vs pelvis）和 PD 刚度参数。
2. 在吊架保护且不接电机写入（shadow 模式）下跑通穿戴衣服实时 `HumanFrame` 旁路观测，确认无跳变后再由单 writer 小步接入真机站立与慢速上肢/迈步测试。
3. 若现场体感 180 ms 预览缓冲偏慢，再开启已准备好的短预览/因果预测低延迟模式对比。

---

### 5.2 给 Agent 留档的详细技术版（Detailed Technical Summary for Agent）

**Done（详细技术完成项）**：
1. **FGP 版本取证与无头推理集成**：
   - 解包核验 `/Users/yu/Human+/key_project/EgoLocate.zip`（旧版单阶段 `weights.pt` `23.22 MB`）与 `CLOTHO-mocap-main (1).zip` + `checkpoints.zip`（新版两阶段 `U100hL7.07h_residual.pth` `384.85 MB` + `globalpose/weights.pt` `134.05 MB` / `LIP_10_real_denoised_7.07_10imu_10.pth` `26.75 MB` + `SMPL_male.pkl` `39.06 MB`）。
   - 实现 `script/x2_morph_human/clotho_fgp_runner.py` 与 `config/clotho_fgp_yup.json`：精确对齐 `Socket/UDP.py` 标定矩阵（误差 `0.0`），移除 `UDP.py` 中 `tran[2]*=0.7; tran[1]+=1.0` 的 Unity 视觉偏置；在 4 线程 CPU 上实测稳态 `GlobalPose` `p50=30.24 ms / p95=31.54 ms`、`LIP` `p50=28.03 ms / p95=28.74 ms`；同源穿戴流相对旧版 Active-18 测地角差 `17.32°`（`GlobalPose`）与 `19.99°`（`LIP`）。
2. **运行时契约、C++17 IPC 与单元测试**：
   - 在 `runtime.py`、`fgp_bridge.py`、`__main__.py` 中补齐 `--max-freshness-ms`（与 `--max-interp-gap-ms` 分离）、`--human-points {full,five}`（置零非五点 slot 并维持显式 `valid` 掩码语义）及 CLI `clotho-export` / `live-bridge` 入口。
   - `test_runtime.py` 18/18 通过（`0.797 s`）；`cpp/advisory_probe.cpp` C++17 编译与 `transport_probe.py` 联调 `CPP_LOCAL_IPC_PASSED`（80 req / 31 policy calls / `hal_commands=0`）。
   - 构建并在全新临时目录净室验证 `data/deliveries/X2_MORPH_HUMAN_PORTABLE_V2_20261112.tgz`（80 成员，`doctor --meshes` + `18/18` 单测全过）。
3. **8 组 X2 Ultra 50 Hz MuJoCo 闭环与实时异步丢包测试**：
   - 完成 `real_worn_globalpose`、`real_worn_lip`、`real_manual_steady_gp`、`real_manual_active_gp`、`synth_walk_start_stop`、`synth_turn_wave_full`、`synth_turn_wave_five`、`live_zmq_30to50_realtime` 共 8 组正式闭环运行（均 `simulation_fallen=False`、`publish_allowed=False`）及阻塞反例 `live_zmq_30to50_dropout`。
   - 生成并目视核验 `fig1_latest_fgp_frontend_6panel`、`fig2_x2_closed_loop_pretest_6panel`、`fig3_x2_pretest_videos_contact_sheet`（PNG+PDF）与 `pretest_metrics_summary.csv`，打包校验 `X2_LATEST_FGP_PRETEST_EVIDENCE_20261113.zip`。

**Todo（后续技术待办与约束门）**：
1. **现场硬件只读核验门（严禁跳过直接发力矩/位置）**：
   - 在 X2 Orin / 现场工控机解压 `X2_MORPH_HUMAN_PORTABLE_V2_20261112.tgz`，用 `shadow_reader.py` 实测核对 `x2` 分支（远端 `79c4a1e`）AimDK/ROS2 反馈的 31 关节名称顺序、正方向符号、零位偏置、腰部/腕部映射及 IMU 安装体（`torso` 外参转 `pelvis` 的角速度扣除项）。
   - 核对 MorphMimic `human_x2_ultra_isaac` 训练所用 `action_scale`、默认站立角 `q_default`、各关节 `Kp/Kd` 与用户 `x2` 部署分支 RL 控制器的差异，建立显式关节限幅、单步最大 `Δq` 限速与 `>300 ms` `NotReady` 回中斜坡（ramp-to-stand）。
2. **双机/进程拓扑部署**：
   - 保持 30 Hz CLOTHO FGP（`U100h + GlobalPose`）在独立进程或算力富余节点运行，通过 UDP/ZMQ 推送 `HumanFrame v1`；50 Hz `x2_morph_human serve`（ONNX Runtime TensorRT/CUDA 或多核 CPU，推理 `<3 ms`）与 C++ 实时控制环走本地 `127.0.0.1` IPC，严禁在实时线程内做视频编码或重磁盘写入。
3. **Q1 / 服务器训练线保持原状**：
   - 未获用户新指令前，保持训练服务器原共享训练暂停状态不变；待后续切回 Q1 低延迟研究时，再按 `AGENTS.md` 既有顺序先补人体条件训练原件取证，再做多步 WM 校准与严格 80 ms 依赖预览验证。
