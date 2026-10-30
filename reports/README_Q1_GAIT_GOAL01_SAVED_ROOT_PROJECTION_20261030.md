# Q1保存根姿态的脚底高度投影分解

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SAVED_ROOT_PROJECTION。状态：保存数据CPU分析与工件交付完成；自然步态目标未达，goal ACTIVE，用户采纳null。

## 目标与范围

上一轮静态几何发现：参考关节角相同时，底座最终保存根姿态使右脚相对左脚的高度损失约11.67/16.09mm（停走/起走）。本轮分解这个几何差异，区分根局部三轴投影与脚底最低材料点切换，给SuperMimic内下一步诊断缩小范围。

读取原底座V02和独立PPO的15份保存NPZ，覆盖两走与抬手共45段、12015状态。执行120批、12015姿态的静态参考关节FK，输出540条件行、12015时序行。0新控制环境、物理积分、接触动力学求解、策略推理、optimizer或训练；没有更改原奖励、控制、参考、权重或原逐子步观测plan。

原实训仍为底座301.1074845249823秒/PPO600.0709074319457秒，合计901.178391956928秒；两线physical_regression停止、原PPO媒体失败及独立补交付的历史均保留，不补预算或登记新采纳。

## 精确分解方法

先将原参考关节放在根位置0、根旋转单位阵下，用原Q1全树FK与原碰撞脚底几何得到局部材料点。令n为世界向上方向在根坐标系的表示，即根旋转矩阵的第三行；脚底高度是各材料点的n点积减球半径的最小值，再加共同根Z。

对于同一组参考关节，选出原参考根朝向下左右脚各自的最低材料点vL、vR。根朝向换成保存控制起点后，固定这些点的高度差变化精确分为：

```text
X项 = (n_saved.x - n_ref.x) * (vR.x - vL.x)
Y项 = (n_saved.y - n_ref.y) * (vR.y - vL.y)
Z项 = (n_saved.z - n_ref.z) * (vR.z - vL.z)
最低点切换项 = 新朝向重新选最低点的右减左高差 - 新朝向固定旧点的右减左高差
根旋转总效应 = X项 + Y项 + Z项 + 最低点切换项
```

球半径及切换到另一材料点的变化完整保留在最后一项。共同根平移在右减左高差中抵消；纯世界yaw左乘不改变n。三轴为Q1根局部轴：X/Y投影分别与前后倾/侧倾相关，但**不是独立改变pitch/roll的动力学干预或训练因果分摊**。

角度辅助量定义为局部Y倾斜atan2(n_y,n_z)，局部X倾斜-arcsin(n_x)，报告使用保存根相对参考根的有符号差。该角度差不同于原整段torso_tilt_rms，也不能以左右倾斜一项替换原姿态验收门。

时间定义沿用上一轮：参考关节和参考窗口是下一20ms保存样本，根朝向取控制起点。此时差未被静态计算消除；本轮不读取原生力来构造新的同时接触真值。

## 右脚参考抬脚窗口

条件为原声明摆脚且参考右脚脚底>=10mm；停走44帧/重复、起走51帧/重复。以下均先取每重复窗口均值，再取三重复中位数。负高差效应表示右脚相对左脚被压低。

| 路线/动作 | 根旋转总效应，初始→训练后（mm） | 根局部Y投影项，初始→训练后（mm） | 局部Y倾斜角差，初始→训练后（度） |
| --- | ---: | ---: | ---: |
| 底座/停走 | -1.39→-11.67 | -5.52→-14.34 | 1.59→4.23 |
| 底座/起走 | -5.66→-16.09 | -6.92→-16.02 | 2.23→4.96 |
| PPO/停走 | -1.30→-2.51 | -5.15→-5.88 | 1.47→1.70 |
| PPO/起走 | -4.43→-9.68 | -6.12→-9.90 | 1.96→3.17 |

各分量分别取中位数，不能把这些中位数相加替代总效应中位数；逐样本和逐重复均值的加法恒等式成立。完整三重复原值在本轮清单与条件CSV。

底座最终的Y投影在两走三重复均是主要负项。停走X项为+2.25至+3.48mm，抵消部分Y项；起走X项为-0.90至-0.21mm。最终两走最低点切换的窗口均值为-0.35至+0.94mm，未达到总效应的厘米量级。不能据此忽略全时序其他窗口的最低点切换；图中仍有较大瞬时值。

PPO同样有起走侧向投影退步，而停走变化较小；两线独立，不能把它们组合成新方案或把共同静态现象认定为相同训练成因。

## 片内时段

将保存片长均分四段，只在相同右脚参考抬脚窗口内计算Y项。底座起走四段中位数为-10.95/-19.15/-18.07/-15.59mm，均为负；PPO最终起走为+1.21/-11.36/-13.11/-13.86mm，负项主要出现在后三段。

每重复的起走窗口样本数依次11/10/15/15。停走四段为14/26/4/0；第三段只有4帧，第四段没有合格窗口，CSV保留null，不能外推为整段或完整步数。以上仅描述保存姿态的几何与时段，尚未回答侧倾、左右卸载、执行器输出和抬脚失败的先后关系。

## 代码、验证与工件

独立工具script/project_tools/analyze_q1_saved_root_projection.py及对应test_q1_saved_root_projection.py；基础提交588d86ce75677b60ef14fd707f13cf3e5025b8e2，执行工具SHA d1fe8d31fcbd1e3db22950a68d6c9021544c7e442cab99bcfaf9dcd4b40c7b54。原被检查的FK/脚底源码及场景资产逐项匹配两条原冻结plan；55项实际输入在前后保持大小/SHA不变，15份轨迹匹配上一轮已发布来源及原检查点收据。

最终8项CPU反例测试通过：纯yaw不变性、左右材料点侧向符号、前后与竖向投影区分、球半径及最低点切换、135行分块支持点与显式展开一致、同根零效应、非平凡足旋转坐标及非法输入拒绝。初版7测试虽通过，本机matmul路径出现浮点告警且结果与显式展开一致；原日志保留，改用等价einsum逐项点积后最终8测试无告警。该告警不用于解释训练结果。

实际保存参考脚高重建最大误差1.190060e-7m，同参考关节/保存根脚高重建最大误差2.323796e-7m；分解相加最大误差2.046974e-16m。不是新物理评测或轨迹等价验证。

输出data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_ROOT_PROJECTION_20261030，共7工件6319472字节，权威工作副本/Mac主根/25386逐大小SHA一致，两张PNG已目视无重叠遮挡。PNG/PDF各两张、root_projection_conditions.csv、root_projection_timeseries.csv及summary.json；图显示分量各自的逐帧中位数，其曲线相加也不保证等于总量中位数。大NPZ/权重仍SERVER_LOCAL_ONLY，未重传或重做原媒体/保存审计。

```sh
python3 -m unittest discover -s script/project_tools -p 'test_q1_saved_root_projection.py' -v
```

实际CPU分析命令（仅重现静态分析，输出目录必须不存在）：

```sh
projection_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 600s \
  "$projection_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$projection_root/script/supermimic_q1_hp/activate.py" -u \
  "$projection_root/script/project_tools/analyze_q1_saved_root_projection.py" \
  --runtime "$projection_root/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027" \
  --base "$projection_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029" \
  --ppo "$projection_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029" \
  --target "$projection_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_TARGET_POSE_20261030" \
  --target-manifest "$projection_root/data/manifests/supermimic_q1_gait_goal_saved_target_pose_20261030.json" \
  --output "$projection_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_ROOT_PROJECTION_20261030" \
  --source-base-commit 588d86ce75677b60ef14fd707f13cf3e5025b8e2
```

日志位于logs/session_records/q1_gait_goal_saved_root_projection_20261030.log及两个对应tests日志。代码、报告、清单、索引及日志经payload审计后推送Q1；实际独立40位远端和三端记录验收见publication旁收据，不能提前称发布完成。

## 未解条件与下一步

本轮将此前“根姿态影响脚高”缩小到两走右脚窗口的侧向投影与片内时序，下一物理诊断应优先核侧倾、左右真实载荷及抬脚先后关系。它仍不能证明世界模型、critic、探索、原奖励或控制中哪一项造成训练退化，也不直接支持提高脚高权重、修改PD或增加残差半径。

此前独立逐子步观测范围问题仍待用户答复，不重发或将goal续行当批准；V02 plan282源/86121字节/SHA c94cf9489c317ea6a2ca431af898bc4f8704e698067c5e927d17a389b1234942已三端核验未变，实际观测未开始。底座V02一次授权已用，训练仍停止；不恢复、重试、扩时、补预算或启动agent，监督/缓存维护与其他自动任务保持。goal ACTIVE/未达，用户采纳null。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_root_projection_20261030.json) / [发布旁收据](../data/manifests/supermimic_q1_gait_goal_saved_root_projection_publication_20261030.json) / [前轮目标与根几何](README_Q1_GAIT_GOAL01_SAVED_TARGET_POSE_20261030.md) / [双线结果与本地视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md)。
