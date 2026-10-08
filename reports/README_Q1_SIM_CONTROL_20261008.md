# Q1 仿真关节控制与参考动作通路

记录日期：2026-10-08。阶段 / 实验编号：Q1-P1 / E001、E002。状态：22DOF 仿真控制通路、数值验证与渲染复查完成。

## 目标与范围

最终目标是 SMPL 驱动 Q1 全身跟随，包括腿部和移动。本阶段在 hp3090 的 `/media/yu/FAFF-E977/YuanQi_Q1` 建立可复现的仿真控制通路，为动作重定向及策略训练准备接口。没有连接或操作机器人；没有启动训练。Mac 只保存小型代码、报告、日志及预览图。

## 完成内容

- `script/q1/q1_sim.py`：按名称发现 22 个关节的 qpos/qvel/执行器对应关系，检查源 XML 身份，提供限位、限速、限幅 PD、观测、复位与倒地检测。
- `JointReference`：读取无 pickle 的 NPZ，校验时间戳与关节名称，按名称重排列并插值。当前输入是机器人空间关节角，尚不是 SMPL 输入。
- `script/q1/sim_config.json`：固定源模型 SHA-256、关节顺序、50 Hz 参考与 1000 Hz 物理/PD。`kp=50`、`kd=2`、力矩上限为 MJCF 声明值的 25%，仅用于仿真，不能作为真机参数。
- `script/q1/test_q1_sim.py`：9 项检查覆盖映射、传感器、异常输入拒绝、边界、时序/复位、重力/四元数、支撑区分和参考插值，实际全部通过。
- `script/q1/validate_control.py`：24 秒有人工骨盆支撑的逐关节小幅动作，以及 5 秒无支撑静态 PD 基线；保存轨迹、摘要与渲染拼图，拒绝覆盖已有实验目录。
- E001 渲染实际显示机器人网格，但支撑场景头部被裁切。相机高度改为随初始骨盆高度调整，在新目录 E002 重跑；四幅图机器人均完整入镜，网格正常。两次物理数值结果相同，E001 保留。

## 关键决定及依据

- 使用已验证的 MuJoCo 3.3.7 和 `q1_v3_mc.xml`，不等待完整 SDK 工具链迁移。
- 人工支撑只用于诊断关节通路，在内存中生成 weld，不修改厂家模型；该结果不能用于证明平衡。
- 无支撑静态基线实际没有倒地，因此不能预设 PD 必然失败。E001 的 `scope` 曾写为 `NEGATIVE_CONTROL`，属于不准确标签；源码改为 `STATIC_BASELINE`，保留原实验及哈希，用本报告解释其真实含义。
- 尚未确认动捕数据字段和坐标系，不把合成关节信号称为 SMPL 跟随。

## 版本、命令与证据

- 实验开始的 Git 基线：`61d2c68cfe0514c934aca17a2a39181005bb7705`；执行时本阶段四份源码尚未提交。配置 SHA-256：`ec34f22d18816acc5640f0405d7b92172b2e5320e9380b77359284066be209e7`。
- 源 XML SHA-256：`d9442f0de18071b100d927e310bc58b04e7be305fea8f126893b2700e8c4bd3b`；原始模型包身份及许可边界见[迁移报告](README_Q1_MIGRATION_20261008.md)。
- 环境：`/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python`，MuJoCo 3.3.7。
- 原始执行记录：服务器 `logs/q1_sim_tests_20261008.log` 与 `logs/q1_sim_20261008_E001.log`；检查时两次命令都已结束。

```bash
cd /media/yu/FAFF-E977/YuanQi_Q1
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/test_q1_sim.py -v
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/validate_control.py \
  --output-dir data/experiments/q1_sim_20261008_E001 --render
```

E001 已存在，复跑必须换新目录。原始工件：`data/experiments/q1_sim_20261008_E001/`，包含 `synthetic_joint_reference.npz`、`supported_trace.npz`、`free_base_trace.npz`、`summary.json`、`control_overview.png`。摘要登记各轨迹和图片的大小与 SHA-256；Git 另保存小型工件清单，原始数据不入 Git。

E002 使用同一配置，命令仅将输出目录改为 `data/experiments/q1_sim_20261008_E002`；其摘要另登记执行时 Python 源码 SHA-256（仍以旧提交为基线，源码身份由哈希确定）。原始日志为 `logs/q1_sim_20261008_E002.log`。摘要身份：

- E001：12,875 字节，SHA-256 `3a37b6a71e06bec022e870c3ff042996eb795a6b9fc1a8e3f8905df85b1bdd95`，[登记清单](../data/manifests/artifact_q1_sim_20261008_E001_3a37b6a71e06bec0_18657a9f.json)。
- E002：13,745 字节，SHA-256 `38ac3451a63f5647d82361cffccc0911463899916e9bbd8c4abe74bae6eb2050`，[登记清单](../data/manifests/artifact_q1_sim_20261008_E002_38ac3451a63f5647_72c807d5.json)。
- 审核后的重要执行摘要：[阶段日志](../logs/session_records/q1_sim_control_20261008.log)。

## 结果与边界

| 实际检查 | 结果 | 证据支持的范围 |
| --- | --- | --- |
| 单元检查 | 9/9 通过 | 仿真适配器的映射与输入边界 |
| 人工支撑逐关节脉冲 | 24 秒 / 1200 控制帧，22 个脉冲全部检测到 | 关节参考到力矩到仿真状态的通路 |
| 支撑跟踪误差 | 最大单关节 RMSE 0.00715881516 rad，最大绝对误差 0.02877914947 rad | 本次合成小幅输入下的数值结果 |
| 无支撑静态基线 | 5 秒未倒地；骨盆高度 0.419→0.4193914 m，末态倾角 0.0500847° | 模型与当前 PD 下的短时静态保持 |
| 力矩饱和 | 两项均为 0 | 当前实验输入下没有达到设定限幅 |

尚未验证扰动恢复、动态动作、行走、真实动捕重定向、动作跟踪 policy 或真机部署。无支撑静态保持不是全身遥操验收。E002 渲染实际检查通过。两次验证与单元检查均已结束，不存在本阶段遗留训练/控制作业。

## 保存与同步状态

阶段代码、README、重要日志摘要与工件清单按 `Q1` 规则审查并发布；本次提交身份由 Git 历史提供。发布核验以本地 HEAD 和远端 `refs/heads/Q1` 的实际比对为准，原始发布输出保留在 Mac `logs/q1_sim_publish_20261008.log`。本报告的物理验证不依赖推送是否成功。

工件保存在 hp3090；百度网盘未验证上传，状态 `LOCAL_ONLY`，没有删除原件。报告索引与当前状态同步更新，通用研究待同步内容独立保留。

## 下一步与恢复入口

1. 仿真通路已验证，不重复迁移或基础加载；从重定向阶段继续。
2. 核对已有 GMR 的实际路径、版本、许可及依赖，再建立人体骨架目标到 Q1 的重定向；人体模型参数不能直接作为机器人关节角。
3. 动捕样本仍缺字段、旋转表示、坐标轴、单位、帧率、传输方式。先推进离线重定向与仿真，接设备时再落实数据合同。
4. 恢复先读本报告、[当前状态](README_CURRENT_STATE.md)和[迁移报告](README_Q1_MIGRATION_20261008.md)，重新核实 Git 与实际作业。
