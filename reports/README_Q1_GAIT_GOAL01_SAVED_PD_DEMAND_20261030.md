# Q1保存轨迹的PD请求力矩与记录缺口

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SAVED_PD_DEMAND。状态：离线核查和小工件交付完成，普通自然步态目标未达。

## 目标与范围

继续在SuperMimic内区分位置目标、PD请求和实际施加力矩。读取底座V02与PPO原15份NPZ的三固定片，共45段、12015状态；复用前轮已经核验的参考相位和原生足力，不重算几何或媒体。

本次0新积分、0策略推理、0优化；没有构造GPU仿真环境。CPU只做135次位置/速度/执行器前向计算，时间未推进。两条训练仍分别301.1074845249823秒和600.0709074319457秒，共901.178391956928秒，均按原physical_regression结束。一次V02授权已用，采纳null、goal ACTIVE；原监督、缓存维护和其他自动任务保持。

## 实际控制契约

冻结场景编译出的22个执行器均为位置执行器，单位hinge传动比，无激活状态、控制延迟或ctrl范围限制；执行器力矩限制启用。startup/reset配置均关闭PD增益随机化。

| 项目 | 冻结模型值 |
| --- | --- |
| 22执行器KP | 15.791367041742973 Nm/rad |
| 22执行器KV | 1.0053096491487339 Nms/rad |
| 髋、膝、腰力矩范围 | -36至36 Nm |
| 踝、手臂力矩范围 | -22至22 Nm |
| 控制周期/物理步长 | 20ms / 5ms，每次控制4子步 |
| 积分器 | implicitfast |

当前安装源码的链路为：原生action项给出位置目标，Entity全宽写入ctrl，TorchEnv把同一ctrl交给后端执行4次物理子步。Warp执行器以固定gain和线性bias计算请求，随后按forcerange裁剪。模型没有额外关节总力矩限制或重力补偿。

在保存控制起点，公式为：

```text
requested = KP * (target - q_pre) - KV * qd_pre
model_force = clip(requested, force_min, force_max)
```

pre-state是initial或上一保存后态，不能使用本步末状态替代。targets产生随后20ms的states，原force/slip仅为该区间末5ms子步的原生样本。implicitfast影响积分中的速度处理，公式重建不能代替其后四个子步的动力学。

本轮公式的直接证据是冻结模型数组、当前安装的Warp源码和下述CPU执行器计算。没有引入电机速度曲线、电流或真实Q1硬件规格的结论。

## 结果：局部踝关节请求会触及模型力矩上限

264330个保存关节控制起点中，287个理论请求超过模型范围，均为踝pitch：左44、右243。没有髋、膝、腰或手臂同类超限。全体最大请求30.6775Nm，对应22Nm踝限制的1.3944倍；它出现在PPO初始起走，而非训练后才出现。

这与上一轮“最终动作没有触及±20裁剪、残差未触及半径”不矛盾：动作裁剪与PD输出力矩限制是两个位置。以下按每次重复先算右踝pitch请求绝对值p95，再取三重复中位数；超限帧数逐重复保留。

| 路线/阶段/动作 | 请求绝对值p95 | 超限帧数，各重复 | 总帧数，各重复 |
| --- | --- | --- | --- |
| 底座初始/停走 | 18.405Nm | 3 / 5 / 5 | 320 |
| 底座301秒/停走 | 18.922Nm | 10 / 4 / 5 | 320 |
| 底座初始/起走 | 21.825Nm | 11 / 7 / 12 | 210 |
| 底座301秒/起走 | 23.109Nm | 14 / 12 / 17 | 210 |
| PPO初始/停走 | 18.344Nm | 5 / 6 / 5 | 320 |
| PPO600秒/停走 | 18.461Nm | 6 / 3 / 6 | 320 |
| PPO初始/起走 | 21.377Nm | 11 / 8 / 10 | 210 |
| PPO600秒/起走 | 22.196Nm | 8 / 11 / 12 | 210 |

底座训练后起走右踝边界超限比例5.71%-8.10%，PPO为3.81%-5.71%。可列为局部执行层约束的线索，不能称整条腿一直被卡住，更不能由此直接提高模型力矩上限。

## 结果：多数参考抬脚窗口并没有边界超限

最终PPO右脚“声明摆脚且参考脚底>=10mm”窗口每重复为停走44帧、起走51帧。三重复合计右腿理论超限元素分别4和14，均对应右踝pitch；即该踝分别有128/132、139/153个控制起点未超限，比例96.97%/90.85%。这与前轮同窗口约96%-98%近地现象并列，说明不能用少数已重建触顶起点解释整个贴地过程。其余子步是否触顶仍未知。

保持每个已保存pre-state和当前教师目标不变，仅移除残差重算执行器公式，右腿6维的模型裁剪后力矩变化RMS三重复中位数约0.07720Nm/0.07551Nm。它是当前状态上的瞬时对比，不是新rollout，也不是未来抬脚收益。小残差带来的执行器修正仍小，不能把“半径没用满”或“局部踝触顶”各自当作PPO失败的单因。

## 验证与历史身份边界

5个针对性测试通过：PD正负号/非对称限幅、initial到pre-state时序、触顶后残差力矩增量消失、非单位gear/延迟/激活/错误bias等契约拒绝、非有限值与不合法力矩范围拒绝。

135次CPU交叉核对覆盖每段首/中/末保存起点，依次调用mj_fwdPosition、mj_fwdVelocity、mj_fwdActuation，没有mj_step或时间积分。代数裁剪值与CPU actuator_force及对应qfrc_actuator最大绝对差4.44e-15Nm。它验证公式与编译模型，不能替代历史Warp实际力矩的记录。

15份原NPZ及对应JSON与原候选历史SHA、上一轮saved-control来源一致，所用场景、配置、控制及原评测源码在读取前后与既有冻结来源一致。

另发现原两次plan没有冻结本次检查的unilab/unisim/Warp后端Python文件。当前版本为unilab1.5.0、mujoco3.11.0、mujoco-warp3.11.0；5个当前后端源文件已独立记录前后大小/SHA。**当前源码哈希不能证明训练当时这些文件相同。** 因而全文数值准确表述为“冻结场景/当前后端契约下的边界PD重建”，不是历史实际施加力矩审计。

## 缺少的通道与独立验证方案

原NPZ保存了20ms初末状态、目标、残差、末子步足力/滑速；未保存逐5ms的关节状态、ctrl、actuator_force、qfrc_actuator或逐环境实际gain/bias。因此整段饱和率、控制器是否先卸载再抬脚、接触约束如何阻止运动尚不能从这些文件确定。

下面仅为可审阅设计，未获新运行决定、未实现或启动新物理作业，也不是新训练授权：

| 路线 | 唯一比较变量 | 保持固定 | 新增观测回答的问题 |
| --- | --- | --- | --- |
| 底座 | 原initial与301秒actor检查点 | 原20ms/4子步、PD、force limits、参考、alpha.5、评测协议 | 训练后是否增加踝触顶持续时间，卸载/姿态/离地次序是否改变 |
| PPO | 同一冻结教师下，最终head开启与置零 | 同上及原残差半径、教师与head权重 | 当前小补偿实际改变了哪些子步力矩和承重转换 |

各路线仅两个固定普通走、两条件、三重复，上限3180个控制转移/12720个5ms子步，0optimizer；不跑全330库或重做旧媒体。记录每个子步的q/qd、同区间target/ctrl、实际执行器/广义力矩、实际gain/bias/限幅与同子步足力/滑移/脚底高度；积分前状态/该步力矩与积分后状态须分别标时，不把步后q/qd误配给步前力矩。实现应只添加观测，保持同一目标持续4子步的控制语义，并先核初始状态、历史、RNG与可用后端求解状态的匹配；不匹配就保留证据，不能称同起点因果对照。

逐子步观测需要适合当前后端的钩子，原Manager不直接提供多子步post-physics指标。任何实现后真实积分或推理仍需用户新的明确决定，原已消耗的V02/PPO预算不复用。若后续进入训练，底座须同时守住姿态/上肢，PPO须证明净补偿作用；仍以原物理门与用户原速视频审阅为准。

## 工件、命令与发布

独立工具：script/supermimic_q1_gait_goal_analysis/saved_pd_demand.py，SHA-256为ba0bbfab615e6cd8be0d8ea44b7ba76c731983d9950ee314f373083b9ecf44bf；分析基础提交3421079a20ea2c5272ad740eadf976ececba9da4。原训练代码、模型和奖励未改。

输出data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PD_DEMAND_20261030，6工件1164112字节，含990关节行、270足条件行、2120逐帧中位数行、PNG/PDF及来源summary。权威工作树/Mac主根/25386逐大小SHA已一致，PNG已目视；原权重和大NPZ仍只留服务器。图上的逐帧三重复中位数与表中先算每次p95再取中位数口径不同。

[边界请求力矩图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PD_DEMAND_20261030/boundary_pd_demand.png>)；正常退出0的日志为logs/session_records/q1_gait_goal_saved_pd_demand_20261030.log。下面为本次已执行命令；输出入口拒绝覆盖，无需重做：

```sh
pd_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 600s \
  "$pd_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$pd_root/script/supermimic_q1_hp/activate.py" -u \
  "$pd_root/script/supermimic_q1_gait_goal_analysis/saved_pd_demand.py" \
  --runtime "$pd_root/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027" \
  --base "$pd_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029" \
  --ppo "$pd_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029" \
  --phase "$pd_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030" \
  --control "$pd_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030" \
  --output "$pd_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PD_DEMAND_20261030" \
  --source-base-commit 3421079a20ea2c5272ad740eadf976ececba9da4
```

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_pd_demand_20261030.json) / [两线真实结果和本地视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md) / [原保存指令核查](README_Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030.md)。Git提交、独立远端和小记录镜像以publication旁收据为准；报告生成本身不代表已发布。本阶段完成不结算整体goal，也不启动或激活agent。
