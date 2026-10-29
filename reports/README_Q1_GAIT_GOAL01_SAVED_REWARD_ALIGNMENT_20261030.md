# Q1保存姿态的奖励方向与站稳覆盖

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01离线研究。45条保存片段、12015状态；新增0积分、0策略推理、0优化。**两条训练仍为原901.178392秒，均已按physical_regression停止；步态目标未达，goal ACTIVE，用户采纳null。**

## 本轮结论

当前证据不能将Q1走路问题简化为“没有姿态奖励”或“脚高奖励一直太弱”。原全身与基座姿态跟踪一直存在；新增PPO站稳项在起走只覆盖1.9%的参考帧。抬脚和原跟踪的局部腿关节梯度确有相反方向，但PPO抬脚项的梯度范数常更大，不能据此认定原奖励把它抵消了，也不能将局部冲突定为训练失败的唯一原因。

下一步问题因此更具体：抬脚信号是否通过实际残差动作、卸载换脚和长期回报发挥作用，以及整身平均奖励为何容许局部姿态代价。仅继续加大脚高系数，目前缺少依据。本轮没有修改奖励、策略、原训练代码或停止门。

## 实际奖励分支与覆盖

读取冻结运行时`candidate_options.json`、`mdp.py`、`mdp_base.py`和构造入口：当前为`sampler_only`，`joint_tracking_reward=false`、`joint_error_observation=false`，使用原始RewardCfg。关闭额外关节误差观测不等于策略没有参考关节信息，原未来goal仍有参考关节目标。

原tracking包含身体位置、方向、线速度、角速度，以及基座位置/方向六项。身体/基座方向尺度均0.4rad；原动作变化和关节限位代价保留。原方向项没有新增PPO项的`still`门控。

新增PPO站稳项要求参考根XY速度<0.05m/s、角速度<0.25rad/s，且两脚在不可变声明相位中均为支撑。各保存阶段覆盖相同：

| 固定片 | 门控帧/全片帧 | 覆盖率 |
| --- | --- | --- |
| 普通停走A095 | 62/320 | 19.375% |
| 普通起走A071 | 4/210 | 1.905% |
| 双手抬起A058 | 159/271 | 58.672% |

这只说明新增站稳项的覆盖范围，不能说走路其余帧没有姿态监督。底座试验本身不使用这项PPO额外站稳奖励。

## 同一组腿关节上的局部方向

只对12个腿关节角求解析偏导，均以rad为坐标；根位置/方向、根与关节速度、控制目标、记录的力/滑速固定。原速度跟踪仍可能经FK依赖关节姿态，其偏导照常保留。没有混合米和rad的范数。

条件为“至少一脚处在声明摆脚相位且参考整只碰撞脚底Z>=10mm”；停走60个控制帧，起走70个。主对照抬脚项仍包含该帧所有声明摆脚脚，不只取达到10mm的脚。每次重复先在有效帧上统计，表中再取三重复中位数。相反方向指点积<0，不表示180度完全反向。

| 试验/阶段 | 动作 | 与原tracking方向相反的帧比例 | cos中位数 | 抬脚/原tracking梯度范数比 |
| --- | --- | --- | --- | --- |
| 底座initial | 停走 | 61.7% | -0.164 | 2.64 |
| 底座301秒 | 停走 | 66.7% | -0.156 | 1.01 |
| 底座initial | 起走 | 54.3% | -0.033 | 2.87 |
| 底座301秒 | 起走 | 50.0% | 0.005 | 0.87 |
| PPO initial | 停走 | 56.7% | -0.138 | 5.48 |
| PPO 600秒 | 停走 | 55.0% | -0.202 | 5.77 |
| PPO initial | 起走 | 57.1% | -0.121 | 4.65 |
| PPO 600秒 | 起走 | 67.1% | -0.145 | 5.74 |

上述范数采用实际权重：底座合并摆脚高度项每20ms最大0.005，PPO高度半项最大0.01。PPO的局部脚高偏导并不普遍小于原tracking。相反方向不能证明加权总奖励阻止抬脚；这里也未测量总奖励梯度或实际策略梯度。

左右脚另有180行对照，在各自6个腿关节上比较。例如底座最终停走左脚cos中位数约-0.529、相反帧约76.9%；PPO最终起走右脚约-0.422、68.6%。逐脚CSV统一使用0.01高度权重作方向诊断；其中底座逐脚范数换算成实际0.005系数须除2，夹角和比例不变。不能推广为全库结论。

6090个保存状态的PPO高度偏导<=1e-8，其中包括没有声明摆脚的帧。空条件、零梯度均保留缺失夹角与独立计数，不写成cos=0；抬手参考抬脚10mm条件为空，不编造该条件的比例。

## 整身平均得分与局部代价

底座起走三重复中位数：原身体方向项均分0.003637→0.010861，原tracking总均分0.063118→0.070954；基座方向项却由0.009453→0.008968。它与前轮真实倾斜恶化并存，各项均按原20ms奖励定义计算，不是新rollout回报。

这表明完整奖励在这些保存结果上确实可以提高，同时容许局部物理指标退步；不能以整身平均分代替姿态、抬脚和承重换脚验收。尚不能证明优化器“有意用倾斜换得分”，也未隔离上肢、腿、模型误差或critic单因。

基座方向与PPO站稳项在本次“固定根，只变腿关节”比较中偏导均为0。这来自自变量选择，不代表真实控制中腿与基座姿态不耦合。

## 产物与验证

输出：`data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_REWARD_ALIGNMENT_20261030/`。10工件7734017字节已在权威工作树、Mac主根、用户25386逐大小/SHA一致；大NPZ未取回。包含1890条件梯度行、180逐脚行、45站稳覆盖行、12015状态时序，两组PNG/PDF、原计算summary和绘图修订收据。

[底座图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_REWARD_ALIGNMENT_20261030/base_saved_reward_alignment.png>) / [PPO图](</Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_REWARD_ALIGNMENT_20261030/ppo_saved_reward_alignment.png>)均已目视。每列共用完整动作时间轴，停走后段无摆脚偏导的空白保留。曲线先在同一参考帧跨三重复取中位数；表格先在各重复内统计再取中位数，两种聚合不混用。

15份初始/阶段NPZ均核历史SHA及当前前后SHA，对应JSON亦核前后身份；三固定参考库与保存根位置、方向和关节轨迹逐值对齐。135个状态的解析拆分与真实base/PPO候选入口一致。5个本地反例测试通过，覆盖方向/尺度、零梯度、条件分母、非有限及异维坐标拒绝、聚合保留空白。

工具：`script/supermimic_q1_gait_goal_analysis/saved_reward_alignment.py`。计算时SHA为`3d8342f092c9b07d9269d49d75d1ee048d7aa528f9163732697c9e6c8ec1f80c`；目视后仅修各行共用时间轴，当前SHA为`02ad4e89a15cad75b78fae4adc48c3918bd30a65744961861594c87bdde47c79`。原summary保留计算时来源，`plot_revision.json`保存两版代码身份、修图前后身份和5份未变数字文件；修图只读CSV，未重读NPZ或重算梯度。

服务器实际计算命令，600秒外限，正常退出0；输出已存在，入口拒绝覆盖，不要求重做：

```sh
alignment_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 timeout --signal=TERM --kill-after=10s 600s \
  "$alignment_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$alignment_root/script/supermimic_q1_hp/activate.py" -u \
  "$alignment_root/script/supermimic_q1_gait_goal_analysis/saved_reward_alignment.py" \
  --runtime "$alignment_root/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027" \
  --scene "$alignment_root/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027/q1_assets/scene.xml" \
  --base "$alignment_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_BASE_PHASE_SOLE_E4096_MAX20M_V02_20261029" \
  --ppo "$alignment_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_PPO_SOLE_Z_E4096_MAX20M_20261029" \
  --output "$alignment_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SAVED_REWARD_ALIGNMENT_20261030" \
  --source-base-commit 27bb6dca29d8ba606db17538f1db4fb8b78460be
```

## 未解条件与后续范围

这些是瞬时姿态奖励偏导，不是actor/action/H32或模型动力学梯度；PPO不直接使用这些解析姿态偏导训练策略。固定根和速度的局部增分方向未必能通过有限力矩、接触卸载与连续换脚实现。碰撞最低点切换处存在分段可微边界。原生力/滑速是末5ms子步样本，不是20ms冲量。

下一项可离线核已保存残差的髋/膝/踝幅度、限幅和相位时序，结合既有训练日志分清“奖励有信号”与“控制实际使用”。真实动力学、旧奖励对照或新的因果干预仍需另行登记和用户新决定。不据本轮自动改权重、新预检/推理/训练或补预算；不启用agent、结算goal或修改原监督/其他任务。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_saved_reward_alignment_20261030.json) / [双线结果与本地视频](README_Q1_GAIT_GOAL01_RESULTS_20261030.md) / [前轮逐脚与承重交集](README_Q1_GAIT_GOAL01_SAVED_PHASE_TRADEOFF_20261030.md)。正式发布以本轮publication旁收据及独立远端核验为准。
