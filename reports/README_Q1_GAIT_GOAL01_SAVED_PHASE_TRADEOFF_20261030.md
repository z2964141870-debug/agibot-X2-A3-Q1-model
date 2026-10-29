# Q1双线结果的左右脚与时间分段拆解

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01离线分析。**新增0积分、0策略推理、0优化；步态目标仍未达，目标ACTIVE，用户采纳null。** 上一轮完成了双线交付和发布，本轮继续分析已有证据，不补训练预算。

## 新证据

底座的局部改善没有组成稳定的左右脚交替。停走右脚在参考抬脚窗口内确实变高，但左脚同时变低；起走右脚变低且更常承重。PPO最终右脚在这些窗口内仍约96%-98%的帧处于脚底5mm以内。完整存活和整片滑速下降都不足以说明换脚已学会。

另一方面，底座起走在前后都承重的同一脚、同一参考帧上，三次滑速分别下降约8.5%、7.2%、39.0%。因此不能将已有滑速收益全部归为减少了承重帧。倾斜代价主要出现在中后段，不能由整片统计认定“越倾斜就越不滑”。

## 同一参考抬脚窗口

窗口定义为原不可变参考的声明摆脚相位，且参考整只碰撞脚底世界Z>=10mm。窗口对各阶段相同；停走左/右分别26/44个20ms足帧，起走19/51个。以下均为各重复窗口统计的三重复中位数；5mm与10mm是离线诊断条件，不替换原步态门。

| 试验/动作/脚 | 窗口脚底高度P95，初始→最终 | 脚底<=5mm帧比例，初始→最终 |
| --- | --- | --- |
| 底座停走左脚 | 21.0→7.7mm | 76.9%→84.6% |
| 底座停走右脚 | 3.0→15.8mm | 95.5%→84.1% |
| 底座起走左脚 | 2.9→4.9mm | 94.7%→94.7% |
| 底座起走右脚 | 8.6→3.2mm | 90.2%→98.0% |
| PPO停走左脚 | 14.6→15.0mm | 73.1%→73.1% |
| PPO停走右脚 | 2.9→2.9mm | 95.5%→97.7% |
| PPO起走左脚 | 3.0→8.9mm | 100.0%→73.7% |
| PPO起走右脚 | 5.1→3.0mm | 94.1%→96.1% |

底座起走右脚在该窗口内原生力>5N的帧比例中位数由62.7%增至74.5%；PPO最终停走/起走右脚分别为70.5%/68.6%。这是参考要求抬脚时仍常接触承重的证据，不把参考相位当作受力真值。P95偶有上升也不证明连续离地。

## 承重帧变化与滑速

原门分别在每个阶段自己的力>5N帧上求滑速。新增对照取两阶段都承重的同一脚、同一参考帧的交集，两边滑速都在同一个掩码上求均值；CSV同时保留共同承重数、失去/新增承重帧比例。交集仍是训练后选择的状态集合，不是因果干预。

| 起走共同承重滑速比，最终/初始 | repeat0 | repeat1 | repeat2 |
| --- | --- | --- | --- |
| 底座301秒 | 0.915 | 0.928 | 0.610 |
| PPO600秒 | 0.911 | 1.034 | 1.660 |

底座该条件下三次均下降；PPO仅一次下降，两次上升。底座起走约20.1%、PPO最终起走约29.3%的原承重足帧在新阶段不再承重，均为三重复中位数。这说明承重集合确实在变，新的同掩码对照有必要，但不能因此否定所有原滑速收益。

原结果的底座起走-19.4%、PPO起走-11.4%是“候选指标中位数/初始指标中位数”；本表是每次重复的同掩码比值。两者统计定义不同，原数字和原停止门均保留，不能混用。

## 起走收益与倾斜并非全程同步

将固定起走按参考帧顺序等分四段，以下为各重复候选/初始比值的中位数。

| 底座起走时间段 | 本阶段承重滑速比 | 与原门同定义的基座倾斜RMS比 |
| --- | --- | --- |
| 第1/4段 | 0.219 | 0.990 |
| 第2/4段 | 1.550 | 1.275 |
| 第3/4段 | 0.931 | 1.426 |
| 第4/4段 | 0.866 | 1.385 |

第一段滑速下降时倾斜并未增加；第二段两项反而一起变差。之后虽有滑速下降，倾斜代价仍在。这支持把起走按阶段定位，不支持认定已隔离“用倾斜换滑速”的唯一机制。

## 图表与可复现范围

[底座时序PNG](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030/base_saved_phase_tradeoff.png>) / [PPO时序PNG](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030/ppo_saved_phase_tradeoff.png>)。两图均已目视，展示repeat0的两普通走；全三重复和抬手保留在CSV。另有同名PDF供导出。图中的负Z保留为几何穿地诊断，不裁掉、不当作已离地。

输出目录：`data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030/`。45条保存片、360条件足行、225时间分段行、135配对行和24030足时序行；9工件5804571字节三端逐大小/SHA一致，大NPZ没有取回。`summary.json`记录21项依赖及15份历史NPZ/15份对应JSON身份，初始和阶段轨迹均核历史SHA及当前前后SHA。参考相位SHA一致，45片原有脚底RMSE/承重滑速/倾斜指标重建通过。

实际碰撞表面点/半径指纹与先前冻结base4096收据完全相同，左右各13637点；这里只读取该收据和当前几何，没有重做预检。原PPO缺ffprobe的FAILED、独立postdelivery COMPLETED及两线901.178392实训秒保持。

分析入口：`script/supermimic_q1_gait_goal_analysis/saved_phase_tradeoff.py`，SHA256 `18abec51a567322a158b7e634c0aa9165546d7345b079be1f1cd9d20a6352bcd`。5个本地反例检查通过，覆盖3/5相位边界、空条件保留null、5N严格阈值、不同长度四分段的加权统计、同掩码配对/参考一致性。训练来源仍为底座155260ab与PPOcccefe08，分析父提交a7995b73。

服务器实际执行命令（600秒外限，正常退出0）：

```sh
phase_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 timeout --signal=TERM --kill-after=10s 600s \
  "$phase_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$phase_root/script/supermimic_q1_hp/activate.py" -u \
  "$phase_root/script/supermimic_q1_gait_goal_analysis/saved_phase_tradeoff.py" \
  --base "$phase_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029" \
  --ppo "$phase_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029" \
  --scene "$phase_root/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027/q1_assets/scene.xml" \
  --output "$phase_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030" \
  --source-base-commit a7995b73a62a53fd6e63dc09aa44566e887416f8
```

输出已存在，工具会拒绝覆盖；此命令记录实际执行，不要求重复分析。读取原生力的是末5ms子步样本，并非20ms冲量。碰撞几何只做mj_kinematics，未创建控制环境或积分。

## 下一步与未解条件

下一项离线检查集中在SuperMimic内部：参考抬脚时真实卸载与脚底离地是否能由同一组奖励/观测一致表达，以及起走中后段姿态代价是否受到原目标足够约束。需要同训练量的旧奖励对照或另行登记的独立干预，才能归因奖励、模型、critic、探索或残差权限；本轮保存数据无法隔离单因。

原两条训练授权均已执行并按门停止。整体目标ACTIVE、采纳null；不追加积分、推理预检、优化、恢复/重试或补预算，不启动agent或改变其他自动任务。新的正式训练待用户审阅具体数据和视频后的明确决定。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_phase_20261030.json) / [发布收据](../data/manifests/supermimic_q1_gait_goal_saved_phase_publication_20261030.json) / [双线完整结果与四本地视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md)。
