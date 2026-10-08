# Q1 骨架目标重定向与仿真通路

记录日期：2026-10-08。阶段 / 实验编号：Q1-P2 / E001、E002、CLI-E001。状态：离线接口、合成数据往返、校准/限速检查与支撑回放完成；真实人体和动力学跟踪未验证。

## 目标与范围

在 hp3090 的 `/media/yu/FAFF-E977/YuanQi_Q1` 建立人体骨架目标到 Q1 的离线重定向，为 SMPL 全身动作跟踪准备输入。最终验收包含腿部和移动，本阶段保留根部平移、朝向及全身关节目标；没有把根部运动作为外力施加给机器人。没有训练或真机控制。

## 完成内容

- `script/q1/q1_retarget.py`：显式世界坐标骨架 NPZ 合同、校准、尺度映射，调用已安装 Mink 求解器，并输出命名关节参考及含自由基座的运动学 qpos。
- `script/q1/retarget_config.json`：14 个骨架目标到 Q1 模型身体名称的映射，覆盖骨盆、躯干、双腿和双臂；使用 `quadprog`。
- `script/q1/test_q1_retarget.py`：输入拒绝、序列化、校准静止、整体平移/转向、尺度和初始朝向检查。
- `script/q1/validate_retarget.py`：Q1 正向运动学生成合成目标，反求 22 个关节并在人工支撑下回放；明确记录 `SYNTHETIC_Q1_FK_NOT_HUMAN_SMPL`。

## 关键决定及依据

- 只读核验服务器已有 `/home/yu/projects/humanoid-gpt/gmr`，仓库提交 `27851f496dcca2005af753362604bec6daf20700`，根目录 Apache-2.0。其现有配置仅支持 G1 的 BVH/FBX，不能当作 Q1 或 SMPL 接口直接使用；未复制或修改其代码。
- 使用同环境现有 Mink 1.2.0，其安装包许可证为 Apache-2.0，来源为 `https://github.com/kevinzakka/mink`；通过公开 API 调用，没有新增安装或 vendoring。
- 原 DAQP 路线实际失败：`qpsolvers` 调用 DAQP 时出现 `TypeError: solve() got an unexpected keyword argument 'primal_start'`。初轮 5 项检查中 3 项通过、2 项求解报错；E001 在 IK 前后接口处失败，不能当作动作跟踪失败。切换为已安装的 `quadprog`，不更改公共环境。原始日志保留。
- 使用 Mink 的关节配置限位、速度限制和骨盆硬约束；根部姿态来自校准后的参考，不让 IK 用根部偏移吸收关节误差。每控制帧分四次求解，总积分时间仍为 20 ms，避免多次迭代扩大 2 rad/s 的关节速度限制。
- 当前校准采用根部对齐、相对位置变化和身体旋转偏移，需真实人体样本验证形体尺度、骨段对应及偏移；14 个目标的字段名称不是已确认的设备协议。

## 输入合同与使用

原始 SMPL 的 `poses/betas/trans` 不能直接作为此接口输入。需合法可用的 SMPL 模型通过正向运动学转换出世界坐标关节位置及全局身体朝向，再按实际设备数据定义转换坐标与时间。本轮没有实现或验证该设备转换层，也没有下载/再分发 SMPL 模型。

NPZ 使用 `allow_pickle=False`，字段如下；`B=14`，`F>=2`。

| 字段 | 形状 / 要求 |
| --- | --- |
| `schema_version` | 标量整数 1 |
| `time_s` | `[F]`，从 0 开始，有限且严格增加，当前必须是 50 Hz |
| `body_names` | `[B]`，准确对应配置，可按名称重排，无重复/缺失 |
| `position_m_world` | `[F,B,3]`，世界位置，米 |
| `orientation_wxyz_world` | `[F,B,4]`，单位四元数，全局身体朝向 |
| `calibration_position_m_world` | `[B,3]`，与机器人基准姿势配对的人体校准姿势 |
| `calibration_orientation_wxyz_world` | `[B,4]`，校准姿势的全局身体朝向 |
| `source_kind` | 标量字符串，记录实际来源，不能把合成数据标为动捕 |
| `convention` | 精确字符串 `x_forward_y_left_z_up_m_wxyz` |

尺度通过 `--scale` 显式传入，不猜人体身高。只接受已转换的字段，不静默猜单位、坐标轴、四元数排列或帧率。

```bash
cd /media/yu/FAFF-E977/YuanQi_Q1
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/test_q1_retarget.py -v
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/validate_retarget.py \
  --output-dir data/experiments/q1_retarget_20261008_E002
# 已有目录不能复用。下面是读取本阶段合成样本的独立离线入口。
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/q1_retarget.py \
  --input data/experiments/q1_retarget_20261008_E002/synthetic_body_reference.npz \
  --scale 1 --output-dir data/experiments/q1_retarget_cli_20261008_E001
```

## 版本、命令与证据

- 开始时 Git 基线：`453e6ca47f0090ba5e65a6135b31ec48cdbe30fe`。执行时代码未提交，E002 摘要登记源码和配置 SHA-256，不能只用基线提交号代表执行版本。
- Python：`/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python`；模型同[仿真阶段](README_Q1_SIM_CONTROL_20261008.md)，SHA-256 固定。
- DAQP 失败日志：`logs/q1_retarget_tests_20261008.log`、`logs/q1_retarget_20261008_E001.log`。
- Quadprog 验证日志：`logs/q1_retarget_tests_quadprog_20261008.log`、`logs/q1_retarget_20261008_E002.log`。
- 最终 8 项检查：`logs/q1_retarget_tests_final_20261008.log`；独立 CLI 日志：`logs/q1_retarget_cli_20261008_E001.log`。实际版本：Mink 1.2.0、qpsolvers 4.12.0、quadprog 0.1.13、DAQP 0.7.2、NumPy 1.26.0、SciPy 1.15.3；本轮未更改环境。
- E001 保留失败产物；E002 保存 `synthetic_body_reference.npz`、`joint_reference.npz`、`kinematic_trace.npz`、`supported_replay.npz`、`summary.json`。没有覆盖既有实验。
- E002 摘要：6,364 字节，SHA-256 `c91f1ed279cff1ce5a992413880807962b94ee95b30309aed6e71068d253c260`，[工件清单](../data/manifests/artifact_q1_retarget_20261008_E002_c91f1ed279cff1ce_7a45eb0c.json)。扩展检查在 E002 后补充，E002 摘要中的测试源码哈希对应运行时的初版 5 项检查；求解/验证代码与配置未再改变。
- CLI-E001 摘要：1,463 字节，SHA-256 `9bc3d6f28b462f3790af875369633f6438d5219622682d8c4a3ac2c0dea54883`，[工件清单](../data/manifests/artifact_q1_retarget_cli_20261008_E001_9bc3d6f28b462f37_0a6480d0.json)。独立入口的关节参考 SHA-256 与 E002 相同，为 `34cc021947b53dc5ea10db262328b32de61a3405140821375c932af13e87872a`。
- 重要执行摘要：[阶段日志](../logs/session_records/q1_retarget_20261008.log)。本阶段验证/CLI 均已结束，无本阶段遗留训练或机器人控制作业。

## 结果与边界

| 实际检查 | 结果 | 支持的结论 |
| --- | --- | --- |
| 最终检查 | 8/8 通过，0.748 秒 | 输入、校准、根部保留、尺度与帧率合同 |
| 大幅单帧姿势变化 | 总关节变化不超过 0.04 rad / 20 ms；残差保持可见 | 多次 IK 求解未放大每帧速度预算，不将无法跟上的目标伪装成准确跟踪 |
| E002 往返 | 24 秒 / 1201 骨架帧，22 个关节脉冲均检测到 | 同一机器人几何下的 FK→IK 通路 |
| 往返误差 | 最大身体位置误差 1.1242e-9 m；最大单关节 RMSE 9.7253e-9 rad | 合成、几何完全一致目标下的数值结果，不是人体重定向精度 |
| 根部误差 | 位置 1.3903e-15 m；朝向 1.8310e-15 rad | 本次参考平移/转向在运动学输出中保留 |
| 支撑 PD 回放 | 最大单关节 RMSE 0.00715881515 rad，力矩饱和 0 | 输出关节参考可通过上一阶段仿真接口回放 |
| 独立 CLI | 正常退出；关节参考 SHA-256 与 E002 一致 | 文件合同与公开命令入口可用 |

E002 反求 1201 帧耗时 1.2191 秒，仅为本次离线 CPU 运行时间，不是实时传输/端到端延迟测试。

运动学根部平移/转向不等于动力学移动。人工支撑回放只检验关节目标通路；真实人体重定向误差、接触一致性、自平衡、抗扰动、行走、训练、部署均未验证。动捕字段/传输方式、合法人体模型和真实校准样本仍待确认。

## 保存与同步状态

代码、配置、README、执行摘要和小型清单按 `Q1` 规则审查并发布，完整提交身份由 Git 历史提供；实际发布输出保留在 Mac `logs/q1_retarget_publish_20261008.log`，发布核验以本地 HEAD 与远端 `refs/heads/Q1` 一致为准。数据留 hp3090，备份 `LOCAL_ONLY`；没有删除原件。

## 下一步与恢复入口

1. 离线重定向通路已验证，继续推进真实 SMPL 转换层；仍缺设备样本/字段、坐标与传输方式，以及合法可用的人体模型或设备提供的全局骨架输出。先核对已有资源，不重复迁移模型或基础仿真搭建。
2. 接入真实样本并落实校准合同，评估逐骨段尺度、位置偏移、姿态和脚接触；当前根部对齐的相对变化映射仅通过机器人合成数据测试，不能当作人体校准已完成。
3. 并行推进无支撑动力学跟踪任务设计与训练基线准备；数据和仿真任务经过验证后训练。不以合成往返或静态 PD 作为全身控制验收。
4. 恢复先读本报告、[仿真阶段](README_Q1_SIM_CONTROL_20261008.md)和[当前状态](README_CURRENT_STATE.md)，重查实际进程、Git 与日志。
