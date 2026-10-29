# Q1保存样本的离地片段与左右换脚时序

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SAVED_SAMPLE_RUNS。状态：保存数据分析与两图交付完成，**0新增积分、策略推理、运动学/动力学求解及优化**。普通步态目标未达，goal ACTIVE，用户采纳null。

## 新证据

PPO最终起走三重复各210个保存帧中，右脚**没有一个脚底高于10mm的帧**。将条件放宽到高于5mm，也没有连续三个高于该高度的保存样本；同时满足低载与对侧承重的片段自然也是0/0/0。左脚仍反复出现离地片段，说明这轮PPO结果的左右不对称在全三重复中一致，不是整片平均误差隐藏的一次偶然。

底座301秒停走则有另一种问题：右脚合格片段从初始1/1/2段增加到5/5/5段，但顺序均为先连续五段右脚，再出现左脚，只有一个参考右脚窗口被对齐。多几段离地没有构成按参考时序的稳定左右交替。

这是对已交付轨迹的进一步诊断，未隔离奖励、世界模型、critic、探索或执行器中的单一原因。它不替代原姿态/上肢/滑移门，也不说明已有候选合格。

## 样本定义与时间边界

输入仅为[已发布左右脚分析](README_Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030.md)的foot_timeseries.csv和summary.json，分别按原清单的大小/SHA核验；前者24030足样本、45条固定保存片。未重新读取或传输大NPZ，未重做FK、原保存审计、媒体或仿真。

默认描述条件是整只碰撞脚底世界Z>10mm、该脚原生力样本<=5N、对侧>5N，连续至少三个20ms保存样本。不满足其中任一项就分段，不填补短空隙。**三个样本端点跨40ms，不是已验证连续60ms离地**。受力是最后一个5ms子步的求解样本，比对应保存姿态早5ms；组合条件不能证明严格同时的持续接触、冲量或完整卸载。

参考窗口沿用原声明摆脚条件且参考脚底Z>=阈值。窗口内也须至少三个连续合格样本才算对齐，跨窗口只擦到一两个样本不算。参考声明不当作实测承重真值，阈值穿越也可能将一次摆脚分成多个窗口，所以本报告的片段/窗口**不是物理步数**。

另保留5mm/10mm与最少1/3样本的四种组合。均为离线描述，不新增训练准入或验收硬门。左右标签相邻交替只描述事件顺序，不约束间隔、步频、真实前进、姿态或稳定性。

## 两走原值

以下片段数均用默认10mm/三样本/低载/对侧承重条件，按repeat0/1/2列出。

| 路线与动作 | 初始左脚→最终左脚 | 初始右脚→最终右脚 | 最终对齐参考窗口，三重复相同 |
| --- | --- | --- | --- |
| 底座停走 | 3/5/4→3/4/5 | 1/1/2→5/5/5 | 左0/4，右1/7 |
| 底座起走 | 4/3/6→3/3/5 | 4/1/0→3/2/2 | 左0/3，右0/7 |
| PPO停走 | 4/4/3→4/2/4 | 3/2/1→2/3/1 | 左1/4，右0/7 |
| PPO起走 | 5/5/4→4/5/6 | 3/2/3→0/0/0 | 左0/3，右0/7 |

底座最终停走标签顺序为RRRRRLLL、RRRRRLLLL、RRRRRLLLLL。PPO最终起走为LLLL、LLLLL、LLLLLL；其301秒阶段已经是LLLLLLL、LLLL、LLLLLL。这里每个L/R只是一段满足条件的保存样本，不代表一步。

PPO起走的5mm/三样本条件下，初始右脚5/3/4段，最终0/0/0段；最终左脚均7段。允许单样本时，最终右脚5mm条件仍有1/1/2段，最长只2/1/2个样本；10mm条件连单样本也为0。因而不能把“缺连续片段”写成“从未有任何微小离地”。

底座最终起走右脚在10mm/单样本条件下仍各对齐一个参考窗口，但三样本条件变为0；5mm/三样本则各对齐一个。阈值及连续性条件确实影响结论，完整敏感性已保留，不能只选最宽松条件称步态成功。

## 图表与验证

[底座时序图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_SAMPLE_RUNS_20261030/base_saved_sample_runs.png>) / [PPO时序图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_SAMPLE_RUNS_20261030/ppo_saved_sample_runs.png>)，另有同名PDF。两图已目视检查，展示repeat0；其余重复与抬手都保留在CSV。

初版图的图例与页脚重叠，原PNG/PDF移入layout_v01/保留，原代码逐字快照analysis_source_layout_v01.py与summary中的原源码SHA一致。仅改plot布局再画两图；AST核实除plot外的代码不变，四张CSV与summary前后身份相同。原分析未重跑，未把布局修正当作新物理结果。

7项最终CPU测试通过：严格高度/5N边界、无对侧承重、短空隙与截断边界、三样本40ms时轴、窗口内连续重叠、同侧重复不计交替链、缺失/重复时间/非有限CSV。脚本只使用标准库和Matplotlib3.10.3，不导入或构造控制环境、策略、MuJoCo或Warp。

输出根data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_SAMPLE_RUNS_20261030，共14小工件1248986字节，权威工作树/Mac主根/25386逐大小SHA一致。其中9份当前数据/图表，另5份为初版图和源码快照。360足汇总行、1845参考窗口行、3932片段行、180整片汇总行，覆盖45保存片和全部四档描述条件。

## 可复现记录

权威Q1工作副本仍为/Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028；分析父提交357e0b1bf8ffa6405b7b71ff690a742b8a118bc1。新代码script/project_tools/analyze_q1_saved_sample_runs.py与test_q1_saved_sample_runs.py，原训练/观测源码未动。以下是已执行的一次纯CSV分析命令，输出已存在，不要求重复运行：

```sh
sample_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 120s \
  "$sample_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$sample_root/script/project_tools/analyze_q1_saved_sample_runs.py" \
  --source-manifest "$sample_root/data/manifests/supermimic_q1_gait_goal_saved_phase_20261030.json" \
  --input "$sample_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030" \
  --output "$sample_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_SAMPLE_RUNS_20261030" \
  --source-base-commit 357e0b1bf8ffa6405b7b71ff690a742b8a118bc1
```

实际分析、最终测试和仅重画图的记录均在logs/session_records/q1_gait_goal_saved_sample_runs_{analysis,tests,layout_v02}_20261030.log。原901.178391956928实训秒、两线physical_regression停止、PPO原交付失败/独立交付完成和采纳null保持。

## 下一步

下一项可区分的缺口仍是执行层：右脚要求抬起时，实际求解力矩、承重卸载和脚底高度的先后关系。现有[逐子步观测V02准备](README_Q1_GAIT_GOAL01_SUBSTEP_PREPARATION_20261030.md)的精确plan SHA c94cf9489c317ea6a2ca431af898bc4f8704e698067c5e927d17a389b1234942未变，新工具不是其冻结依赖，没有更换检查点、测量范围或预算。此前两线观测范围问题仍待用户答复，不重复询问或据goal自动续行启动。

整体步态目标ACTIVE且未达；原底座V02授权已使用，不恢复、重试、扩时或补预算，不启用agent，不修改监督、缓存维护或其他自动任务。本轮不能据保存样本直接决定新奖励、控制参数或加训。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_sample_runs_20261030.json) / [发布旁收据](../data/manifests/supermimic_q1_gait_goal_saved_sample_runs_publication_20261030.json)。发布状态只以实际push、独立40位远端和三端文件核验后的旁收据为准。
