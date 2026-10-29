# Q1保存位置目标的静态脚底几何

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SAVED_TARGET_POSE。状态：保存数据静态计算与图表交付完成；普通自然步态目标未达，goal ACTIVE，采纳null。

## 范围与方法

本轮继续SuperMimic双线诊断，读取底座V02与PPO原15份NPZ的45段固定动作、12015状态。只做720批、72090个静态FK姿态计算，0新积分、策略推理、优化或训练；没有创建原生控制环境、接触求解或新的rollout。原两线301.1074845249823秒与600.0709074319457秒、合计901.178391956928秒及physical_regression停止保持。

位置执行器通过目标与当前角度的偏置产生力矩，**PD目标不等于下一帧姿态、可达姿态或脚部轨迹命令**。本轮只替换保存控制起点的22个关节角，保持根位置、根旋转和全部速度不变，计算以下静态配置：实际前态、实际后态、完整目标、目标扣回保存残差、同根姿态的参考关节、原参考根与参考关节。目标代入没有施加到仿真。

后两种参考配置使用下一保存参考样本的关节；同根配置使用控制起点根姿态，相差20ms的定义保留。带原参考根的FK与已发布参考脚底高度校对，实际后态也与已发布结果校对。最大绝对差分别1.36e-7m、1.62e-7m。未重做媒体、保存审计或原物理评测，大NPZ和权重仍仅留25386。

## 根姿态的几何影响

使用“右脚脚底最低点Z减左脚脚底最低点Z”，消去共同根平移。将同一份参考关节从参考根旋转换到保存控制起点的根旋转，二者高差的变化就是这个静态定义下的根旋转几何影响；它不包含新动力学或训练因果干预。

条件沿用声明摆脚且参考右脚脚底>=10mm，停走每重复44帧、起走51帧。先在每次重复内取窗口均值，再取三重复中位数；不能把这些窗口当作实测单支撑真值。

| 路线/动作 | 原参考右减左高差 | 换用保存根后的高差，初始→训练后 | 根旋转几何影响，初始→训练后 |
| --- | ---: | ---: | ---: |
| 底座/停走 | 9.66mm | 8.27→-2.02mm | -1.39→-11.67mm |
| 底座/起走 | 15.14mm | 9.48→-0.95mm | -5.66→-16.09mm |
| PPO/停走 | 9.66mm | 8.36→7.15mm | -1.30→-2.51mm |
| PPO/起走 | 15.14mm | 10.71→5.46mm | -4.43→-9.68mm |

底座训练后即使参考关节逐值相同，在保存根姿态下，两个窗口的平均右脚相对高度优势也会消失。它给原姿态退步增加了一项几何证据：只改善腿角误差不足以保住脚底几何。它不是“根倾斜是唯一原因”，也不能说明参考关节在当前接触/速度下能够实现。

本表高差不同于摆脚绝对离地高度；两脚同时抬高或降低不会改变高差。相对量仍保留根旋转、足底最低点选择及关节几何影响。

## PPO残差的静态几何增量

保持保存教师目标和控制起点根姿态固定，完整目标减去保存残差后再做FK。最终动作均未处于裁剪边界，扣回在本批数据可辨识；它仍不是“关闭head后重新运行策略”的物理对照。

最终PPO右脚参考抬脚窗口的结果，仍按先每重复计算、再取三重复中位数：

| 静态几何增量 | 停走 | 起走 |
| --- | ---: | ---: |
| 右脚脚底Z变化RMS | 0.726mm | 1.130mm |
| 右脚脚底Z变化均值 | -0.201mm | -0.338mm |
| 右脚脚底Z变化绝对值p95 | 1.769mm | 2.334mm |
| 右减左高差变化RMS | 2.005mm | 1.969mm |
| 右减左高差变化均值 | +0.299mm | -0.111mm |

脚底绝对Z与左右高差不是同一量：对侧脚也会被残差改变。此结果将前轮约0.0055rad的确定性腿补偿换算为毫米级静态几何变化；它没有证明相同大小的实际离地，也不能据均值符号判断PPO优化方向正确或错误。底座和所有initial的零残差配置逐值验证目标/扣回目标几何完全相同。

## 目标角与关节范围边界

完整位置目标的静态配置不保证在模型关节范围内。以下为右脚参考抬脚窗口中，该腿至少一个目标角超模型范围的帧比例，按repeat0/1/2列出：

| 路线/动作 | 初始 | 训练后 |
| --- | --- | --- |
| 底座/停走 | 25.0% / 34.1% / 36.4% | 29.5% / 20.5% / 29.5% |
| 底座/起走 | 31.4% / 31.4% / 33.3% | 19.6% / 17.6% / 21.6% |
| PPO/停走 | 36.4% / 31.8% / 34.1% | 38.6% / 38.6% / 38.6% |
| PPO/起走 | 27.5% / 19.6% / 21.6% | 31.4% / 35.3% / 37.3% |

初始即已有这种偏置，不能把它当作本次训练新引入的错误。PD允许用偏置产生控制力；这里不重新裁剪目标、不修改关节范围，也不将越界等同于真实关节越界或实际力矩饱和。图上较大的静态目标脚底尖峰尤其不能解释为真实抬脚命令幅度。

## 代码、验证与交付

独立工具script/project_tools/analyze_q1_saved_target_pose.py及对应test_q1_saved_target_pose.py，基础提交fa651609d1143206c3918616668aae3f4b82f6e4。原训练、奖励、控制、参考、模型和逐子步准备代码未改。FK及脚底几何依赖、场景资产逐一对原两线冻结plan核身份；输入CSV和summary与已发布清单核大小/SHA，历史NPZ/JSON与前轮保存指令来源一致，读取前后未变。

6项CPU测试通过：替换关节时根和速度保持、非法/错位姿态拒绝、左右符号及共同平移不变、残差差分恒等式、保留越界而不偷偷裁剪、空窗口保留null。执行与测试日志在logs/session_records/q1_gait_goal_saved_target_pose{,_tests}_20261030.log。

输出data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_TARGET_POSE_20261030，共7工件10072795字节，权威工作树/Mac主根/25386逐大小SHA一致。包含270条件行、24030足时序行、来源summary及两份PNG/PDF。两PNG已目视，无文本/图例重叠；图上为最终阶段，其他阶段及重复保留在CSV。

[底座静态目标图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_TARGET_POSE_20261030/base_saved_target_pose.png>) / [PPO静态目标图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_TARGET_POSE_20261030/ppo_saved_target_pose.png>)。

下面是已执行的一次静态分析命令；输出已存在，入口拒绝覆盖，无需重跑：

```sh
pose_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 600s \
  "$pose_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$pose_root/script/supermimic_q1_hp/activate.py" -u \
  "$pose_root/script/project_tools/analyze_q1_saved_target_pose.py" \
  --runtime "$pose_root/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027" \
  --base "$pose_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029" \
  --ppo "$pose_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029" \
  --phase "$pose_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030" \
  --control "$pose_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_CONTROL_USAGE_20261030" \
  --control-manifest "$pose_root/data/manifests/supermimic_q1_gait_goal_saved_control_usage_20261030.json" \
  --phase-manifest "$pose_root/data/manifests/supermimic_q1_gait_goal_saved_phase_20261030.json" \
  --output "$pose_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_TARGET_POSE_20261030" \
  --source-base-commit fa651609d1143206c3918616668aae3f4b82f6e4
```

根旋转表从新CSV配对左右脚同一line/label/repeat/clip/sample得到：原参考高差=右行reference_sole_z_m减左行同字段；再用右行reference_same_root_relative_other_m减该高差。每重复取窗口均值后再取三重复中位数，完整原值在本轮清单中。

## 未解条件与下一步

本次能够量化根旋转和目标几何各自的差异，仍未区分接触约束、实际执行器/控制时序、世界模型、critic或探索中的训练成因。实际卸载、求解力矩和离地先后关系仍须此前独立逐子步观测才能继续验证；现有V02精确plan282源/86121B/SHA c94cf9489c317ea6a2ca431af898bc4f8704e698067c5e927d17a389b1234942未变。

既有观测范围问题仍待用户答复，不重复询问或根据goal续行推定批准。原底座V02一次授权已用，不恢复、重试、扩时或补预算，不据静态几何直接改PD、残差半径、奖励或开新训练；goal ACTIVE/未达、采纳null、监督/缓存维护及其他任务保持，不启用agent。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_target_pose_20261030.json) / [发布旁收据](../data/manifests/supermimic_q1_gait_goal_saved_target_pose_publication_20261030.json) / [原两线结果与视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md)。发布只以实际push、独立40位远端和三端记录核验后的收据为准。
