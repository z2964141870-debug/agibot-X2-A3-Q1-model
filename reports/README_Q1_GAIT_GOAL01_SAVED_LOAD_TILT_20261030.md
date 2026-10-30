# Q1保存低力样本、侧倾与抬脚时序

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SAVED_LOAD_TILT。状态：保存CSV的CPU分析与工件交付完成；自然步态目标未达，goal ACTIVE，用户采纳null。

## 目标与范围

上一轮静态几何将两走右脚高度损失缩小到侧向根姿态投影。本轮拼接已发布的foot_timeseries.csv和root_projection_timeseries.csv，检查右脚是否始终承重，或者已有低力样本但脚底仍贴地，并核对力阈值与样本配对的敏感性。

只读两份已发布清单、两份CSV及各自summary，覆盖45段、12015状态、24030足样本；生成810条件行与2538片段行。片段行包含双脚、三种阈值和三种配对偏移的重复统计，不是2538次实际步行事件。没有读取原NPZ，没有新FK、接触动力学、控制环境、积分、策略推理、optimizer或训练。

底座V02仍301.1074845249823秒、独立PPO仍600.0709074319457秒，合计901.178391956928秒；两线原physical_regression结果、PPO原媒体失败与独立补交付均保留。原一次V02授权已用，不补预算。

## 定义与时间限制

参考窗口固定为原声明摆脚且原参考全碰撞脚底Z>=10mm。停走右脚44样本、起走51样本/重复；声明相位不是实际承重真值。

默认力阈值5N。窗口内分三类：本脚>5N；本脚<=5N且对侧>5N；双脚均<=5N。第二类简称“低力/对侧有力”，只是保存瞬时样本。力值取自NativeContacts.install对脚地接触world-Z力绝对值的求和，不能视为体重占比、净竖直力、20ms冲量或连续卸载证据。图上右脚力占比还要求双脚和>10N。

同一索引的三个时间点不同：

| 记录 | 相对控制起点 |
| --- | ---: |
| 根姿态 | 0ms |
| 最后子步接触求解的力 | +15ms |
| 保存脚底姿态与参考 | +20ms |

保持力及参考窗口索引固定，配对前一/同一/后一份几何样本，脚底相对力偏移分别为-15/+5/+25ms，根姿态相对力偏移为-35/-15/+5ms。越界不回绕；窗口不随配对移动。此敏感性检查不能恢复缺失的5ms逐子步记录。

低力片段遇到任何条件不满足便断开，不补洞。至少3个连续样本只表示端点相隔40ms，不能称持续60ms卸载。脚底<=5mm称近地、>10mm称抬高；下面近地比例的分母是窗口内全部低力/对侧有力样本，不只>=3样本的片段。

## 右脚三重复结果

以下均默认5N/同索引，数组依次repeat0/1/2。最后一列要求同一低力片段内含>=3个连续>10mm脚高样本，仍不是已验证的一步。

| 路线/动作 | 初始低力片段>=3样本 | 最终低力片段>=3样本 | 最终低力样本近地比例 | 最终含连续抬高的低力片段 |
| --- | --- | --- | --- | --- |
| 底座/停走 | 3/3/2 | 3/3/1 | 58.8%/61.1%/53.3% | 1/1/1 |
| 底座/起走 | 4/4/4 | 2/2/1 | 91.7%/78.6%/90.9% | 0/0/0 |
| PPO/停走 | 3/2/3 | 2/3/2 | 92.3%/88.2%/100% | 0/0/0 |
| PPO/起走 | 5/4/4 | 3/3/4 | 87.5%/93.3%/88.9% | 0/0/0 |

PPO最终起走右脚在51个参考窗口样本中，低力/对侧有力16/15/18个；其中脚底<=5mm为14/14/16个，最长低力片段5/4/4样本。双脚均低仅0/1/0个，不能把这些低力样本主要解释成双脚同时没力。此前全片样本分析已确认同三片右脚均没有>10mm保存帧，本轮没有重跑或新增物理评测。

底座起走最终低力样本12/14/11个，低力片段由4/4/4降到2/2/1，最长3/5/3样本。它既有低力片段减少，也有低力时仍贴地；不能只用“从未卸载”解释。底座停走则有部分真实保存离地样本，不应把两动作或两路线混成同一结果；此前已记录其左右交替仍不匹配参考。

## 阈值与配对敏感性

组合力阈值2.5/5/10N与配对偏移-1/0/+1，共9种设置、每种三重复。PPO最终起走的低力样本数在9种设置中均为16/15/18，>=3样本片段均3/3/4；低力时近地比例为87.5%-100%，含连续>=3个>10mm脚高样本的片段始终0。结果没有因轻微阈值或配对移动而变成可靠抬脚。

底座最终起走所有设置仍各有1-2个低力片段，近地比例78.6%-100%，连续抬高片段0；PPO停走为2-3个、82.4%-100%、0。底座停走每重复在全部9种设置下都保留1个含连续抬高的低力片段，近地比例53.3%-72.2%。完整逐重复、阈值和时间偏移均在条件CSV及清单，不改原训练奖励或验收阈值。

## 与侧倾的关系

同索引、5N下，底座最终起走本脚有力样本的局部Y倾斜角差均值5.46/5.28/5.63度；低力/对侧有力样本仍为3.32/2.89/3.96度，初始低力样本为0.54/0.90/1.66度。它表明保存低力时刻仍伴有根姿态差，不能将姿态损失完全归入本脚有力样本。

PPO最终起走对应有力均值3.63/3.37/3.48度、低力均值0.86/2.60/3.13度；重复差异保留。分组条件不同、根比力早15ms，相关均值不能确定谁先发生，更不能隔离姿态、目标、执行器、探索、critic、奖励或世界模型中的唯一原因。也不据这些条件统计提高PD、抬脚奖励或残差半径。

## 版本、验证与工件

基础提交1fac4e26f831dfdbdaa7d446df68aa28c3071328。工具script/project_tools/analyze_q1_saved_load_tilt.py，SHA256 2de9365d787fa57784c0aa58f856f8d94ca1a7758fe8cf14bbdf0fbc6930fa02；对应test_q1_saved_load_tilt.py的9项CPU测试全部通过、无告警。覆盖低力贴地反例、阈值边界、窗口/对侧条件缺口、同一片段连续抬高、偏移不回绕和窗口锚定、空条件null、非法参数、根范围以及帧/时钟/预算/参考标志一致性拒绝。

正式分析仅执行一次，120秒外限内正常结束SAVED_LOAD_TILT_READY，退出0；9项输入代码/清单/CSV/summary在前后大小SHA一致，CSV及summary与既有发布清单一致，原45片及时间轴逐一匹配。已通过的测试没有无因重跑。

```sh
python3 -m unittest discover -s script/project_tools -p 'test_q1_saved_load_tilt.py' -v
```

仅复现保存CSV分析的服务器命令，输出目录必须不存在：

```sh
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 timeout \
  --signal=TERM \
  --kill-after=10s 120s \
  /opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python \
  /opt/data/private/ZHY/script/supermimic_q1_hp/activate.py -u \
  /opt/data/private/ZHY/script/project_tools/analyze_q1_saved_load_tilt.py \
  --phase-manifest \
  /opt/data/private/ZHY/data/manifests/supermimic_q1_gait_goal_saved_phase_20261030.json \
  --phase \
  /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030 \
  --root-manifest \
  /opt/data/private/ZHY/data/manifests/supermimic_q1_gait_goal_saved_root_projection_20261030.json \
  --root \
  /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_ROOT_PROJECTION_20261030 \
  --output \
  /opt/data/private/ZHY/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_LOAD_TILT_20261030 \
  --source-base-commit 1fac4e26f831dfdbdaa7d446df68aa28c3071328
```

输出data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_LOAD_TILT_20261030：两张PNG、两张PDF、load_tilt_conditions.csv、reference_low_force_runs.csv和summary.json，共7工件2158816字节；权威工作副本/Mac主根/25386逐大小SHA一致，两张PNG已目视，无文字遮挡或重叠。图展示repeat0初末，时间点分别绘制并保留窗口阴影。大权重/NPZ仍SERVER_LOCAL_ONLY，没有重传、重做旧媒体或保存审计。

日志为logs/session_records/q1_gait_goal_saved_load_tilt_tests_20261030.log及q1_gait_goal_saved_load_tilt_20261030.log。代码、报告、清单、索引及日志经显式暂存与payload审计发布Q1；独立40位远端及记录三端验收以publication旁收据为准，不在推送前宣称发布完成。

## 未解条件与下一步

本轮支持的更精确描述是：右脚存在低力样本段，但低力、根姿态和抬脚尚未形成稳定配合；不是仅缺一次卸载信号。保存20ms样本仍不足以回答实际子步力矩、连续卸载时长、离地与姿态的先后关系。下一项有区分力的物理测量仍是已准备的独立逐5ms观测。

既有两线观测范围问题仍待用户答复，不重发或将goal续行视为批准。V02 plan282源/86121字节/SHA c94cf9489c317ea6a2ca431af898bc4f8704e698067c5e927d17a389b1234942保持，实际观测未开始。goal ACTIVE、用户采纳null；不恢复、重试、扩时、补训练预算或启动agent，监督/缓存维护/其他自动任务不改。上一goal回合与本回合均有具体分析进展，不把报告交付或等待当整个步态目标完成。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_load_tilt_20261030.json) / [发布旁收据](../data/manifests/supermimic_q1_gait_goal_saved_load_tilt_publication_20261030.json) / [根投影分解](README_Q1_GAIT_GOAL01_SAVED_ROOT_PROJECTION_20261030.md) / [原双线结果与本地视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md)。
