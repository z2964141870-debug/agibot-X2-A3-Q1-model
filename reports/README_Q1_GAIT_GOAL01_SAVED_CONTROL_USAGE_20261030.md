# GAIT_GOAL01 保存指令幅度与关节响应

记录日期：2026-10-30。阶段：SAVED_CONTROL_USAGE。状态：离线分析与小工件交付完成，普通步态目标仍未达成。

## 目标与范围

继续在现有SuperMimic内部核查：PPO是否实际使用残差幅度、是否触及边界，以及参考抬脚时髋膝踝目标与真实关节/脚底是否同步。只读两次已结束试验的15份原始NPZ与JSON，覆盖45段固定动作、12015个状态；复用上一阶段已经核验的脚底/承重相位时序，没有重算脚底几何或重做媒体。

底座301.1074845249823秒、PPO600.0709074319457秒及原physical_regression结论保留，合计901.178391956928秒。本次0新积分、0策略推理、0优化、0训练；一次底座V02授权已经使用，不恢复、重试、扩时或补预算。采纳null、goal ACTIVE，原监督、缓存维护和其他任务保持。

## 结果：没有触及残差或动作边界

全部保存状态的264330个最终动作元素中，边界元素为0；动作绝对值最大6.3554，原裁剪范围是[-20,20]。168210个腿腰残差元素中，接近残差半径95%的元素为0，全部动作/阶段最大绝对残差0.021904rad，只到0.15rad半径的14.60%。本结论仅覆盖保存的三固定片，不外推整个330库或随机训练动作。

| PPO确定性评测腿部残差RMS | 停走 | 起走 |
| --- | ---: | ---: |
| 初始 | 0 | 0 |
| 301秒 | 0.003500rad | 0.004002rad |
| 600秒 | 0.005452rad | 0.005748rad |

每个重复先对12个腿关节及完整片段算RMS，再取三重复中位数；底座全阶段残差都是0。最终PPO约0.31-0.33度RMS，幅度有增长，但远未接近上限。不能据此把失败归因于0.15rad半径不够，也不能反推增大半径一定有效。

原训练日志最后一轮的随机腿腰14维残差RMS为0.015878rad；演员潜变量均值RMS为0.038928，原GaussianDistribution固定潜变量std=.1。随机采样残差和上述确定性评测腿部12维统计对象不同，不能把训练噪声幅度当成已经学会的稳定补偿，更不能仅凭这些日志认定探索是唯一原因。

## 结果：关节目标、实际运动与参考仍有差距

下表为完整普通走片，先在各重复内对12腿关节计算RMS，再取三重复中位数。

| 路线/片段 | 目标到参考 初始→训练后 | 实际到参考 初始→训练后 | 训练后目标到实际 |
| --- | ---: | ---: | ---: |
| 底座/停走 | .39773→.36894rad | .28211→.26739rad | .31468rad |
| 底座/起走 | .44677→.38060rad | .32401→.25578rad | .32833rad |
| PPO/停走 | .40184→.39706rad | .28533→.28320rad | .31387rad |
| PPO/起走 | .44412→.43417rad | .32334→.33043rad | .34070rad |

底座的平均腿部关节误差确有改善空间和局部收益，仍没有组成可靠离地/左右换脚；这与上一阶段的左右脚、姿态取舍相符。PPO均值补偿小，两片实际腿部误差也没有一致改善。原姿态、上肢、脚底和承重物理门不由本表替换。

右踝pitch的目标变化较大，但同一步末实际角度与目标差距明显：

| 训练后路线/片段 | 目标到实际RMSE | 目标峰峰值 | 实际峰峰值 | 参考峰峰值 |
| --- | ---: | ---: | ---: | ---: |
| 底座/停走 | .53355rad | 1.79044rad | .67186rad | .79304rad |
| 底座/起走 | .67570rad | 1.95370rad | .68666rad | .84684rad |
| PPO/停走 | .53478rad | 1.93325rad | .72132rad | .79304rad |
| PPO/起走 | .61454rad | 1.88888rad | .91688rad | .84684rad |

这些是PD位置目标，目标允许通过位置偏置产生控制力，不能要求其逐值等于参考或一步内达到。保存结果没有电机实际施加力矩/电流；仅凭此表不能断言电机饱和、PD太软、关节不能动或世界模型失真。

在相同样本数、无环绕的0-200ms相关性扫描中，右踝目标对后续实际角度的最佳偏移：底座停走80/80/80ms、起走100/100/100ms；PPO停走100/100/80ms、起走120/100/100ms。底座起走零偏移相关中位数-.121、最佳.701；PPO起走零偏移.014、最佳.800。这只描述保存闭环中的响应时序，不是独立测得的执行器延迟，不能据此平移参考、更改控制周期或认定记录时轴错位。

## 结果：参考抬脚窗口的补偿仍小

最终PPO右脚的“声明摆脚且参考脚底>=10mm”窗口：停走44帧、起走51帧，每重复相同。该腿6维残差RMS分别.004929/.004914rad；残差逐元素朝参考关节方向的比例中位数54.17%/50.65%。保持当前教师目标不变，仅扣回残差做几何比较，残差使目标关节平方误差分别增加.000255/.000399rad²。

这里的朝向比例与平方误差只是关节几何诊断。原sampler_only配置没有额外关节tracking奖励，PPO实际目标还含整身tracking、接触/脚底与未来回报，因此不能把50%左右朝向当作PPO奖励方向正确率。它支持“本次确定性修正没有形成明显的参考关节纠正”，不证明奖励无梯度或优化器没有更新。

上一阶段原始脚底结果仍是：PPO最终右脚参考抬脚窗口96%-98%处于<=5mm近地范围。将本次指令与该时序并列，只能确认这些已保存结果仍缺乏可靠抬脚，不构成新的因果干预。

## 实现、对齐与验证

新增独立工具`script/supermimic_q1_gait_goal_analysis/saved_control_usage.py`，未改训练或评测生产源码。源码基础提交`b6b4d2d3e766328c6a038cdeec4893280e9b941c`；计算工具SHA-256为`2e0699e751d166f069f8ec71f62e8603d003ad1722223a38d08fbf773dc434e2`。

关键口径：

- 保存targets和residual产生随后20ms的states；pre-state取initial或前一保存状态，不错用同一步后态。
- 保存残差在最终动作裁剪之前。仅在最终动作严格处于边界内部时，才以target-residual还原当前教师目标；到边界则保留不可辨识值。本次所有元素均在内部。
- AST核原eval_mode将关节偏移设为0；原q1_config的stand keyframe默认姿态、22关节顺序、20ms周期与control.json逐值匹配，再按原effort/kp还原动作。NPZ未直接保存context，此处依据冻结评测源码与配置，不伪称有逐帧context观测。
- 15份初始/阶段NPZ均核历史SHA、当前前后SHA及此前phase审计来源；JSON亦逐一对齐。45段只取有效pre-reset前缀，脚底CSV全局参考帧、原生力与NPZ逐值一致。
- 所需原控制、评测、配置及几何资产与两线各自冻结plan逐大小/SHA一致；分析依赖在计算前后未变。
- 5个反例测试通过：最终裁剪后残差不可辨识、非法控制重建、无环绕等样本滞后/常值无估计、前后态指标分离、eval偏移必须显式为零。

## 工件与复现记录

输出`data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030/`，9工件5514858字节已在权威工作树、Mac主根和用户25386逐大小/SHA一致，两个PNG均已目视；权重和大NPZ没有取回。表格包含990关节行、450足部条件行、8010逐帧三重复中位数行、113训练日志行。每个重复的统计与图上逐帧中位数不是同一种聚合。

[底座指令图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030/base_saved_control_usage.png>) / [PPO指令图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030/ppo_saved_control_usage.png>)；原执行日志`logs/session_records/q1_gait_goal_saved_control_usage_20261030.log`正常退出0。以下是已执行命令，现有输出入口拒绝覆盖，无需重做：

```sh
control_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 timeout --signal=TERM --kill-after=10s 600s \
  "$control_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$control_root/script/supermimic_q1_hp/activate.py" -u \
  "$control_root/script/supermimic_q1_gait_goal_analysis/saved_control_usage.py" \
  --runtime "$control_root/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027" \
  --base "$control_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029" \
  --ppo "$control_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029" \
  --phase "$control_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030" \
  --output "$control_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030" \
  --source-base-commit b6b4d2d3e766328c6a038cdeec4893280e9b941c
```

## 未解条件与下一步

两条路线仍各有空间：底座能改变整身/腿部指令，但当前奖励与模型更新没有共同保住姿态、换脚和脚底；本次PPO没有被半径卡住，却只形成很小的确定性修正。原[奖励方向核查](README_Q1_GAIT_GOAL01_SAVED_REWARD_ALIGNMENT_20261030.md)已说明不能简单称抬脚奖励被淹没，这次也不能简单称缺控制幅度。

允许继续的下一项是离线核现有PD/力矩上限实现与保存通道，明确哪些“卸载后抬脚”的动力学证据尚未记录，并形成SuperMimic底座/PPO各自可审阅的单变量验证方案。若需要真实施加力矩、相同起点的反事实或新训练，必须另获用户决定；不把终态数据重跑成新验证，不直接放大残差或延长训练。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_control_usage_20261030.json) / [双线原始结果与本地视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md) / [前轮脚底相位](README_Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030.md)。本轮Git正式发布与独立远端/三端记录核验见publication旁收据，不以报告生成预称发布完成。
