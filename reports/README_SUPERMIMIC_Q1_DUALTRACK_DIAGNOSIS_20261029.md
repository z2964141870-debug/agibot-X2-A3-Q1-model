# SuperMimic底座与PPO适配：脚底时序诊断

记录日期：2026-10-29，北京时间。阶段：DUALTRACK_DIAGNOSIS。状态：已存证据分析完成，服务器连接与原生验证待恢复。

## 用户决定与本次范围

用户在确认“SuperMimic底座先训练、PPO再额外适配，两手抓”后明确要求“可以，继续推进”。本次已从方向说明进入工具实现、真实保存数据配对和独立候选设计；没有沿用已结束的夜间窗口追加预算。候选采纳仍为null。

实际工作在Mac权威Git的Q1分支进行，父提交为64ebb2e988f48b554afee8a073d6e223b29a54c6。本次没有新物理积分、策略推理或optimizer更新，没有恢复旧训练、启动agent、调用旧goal或修改自动任务。原监督仍保留ACTIVE/每10分钟，不能把本阶段分析完成登记为整体目标完成。

## 已完成的实现与实测

- `script/project_tools/analyze_q1_lift_windows.py`对齐实际碰撞脚底高度、保存的原生足底力和参考几何抬脚时段，生成分相位、逐窗口、左右受力快照及相对未训练起点的配对变化。
- 9项单元测试通过，覆盖步后帧对齐、贴地但有力、共同前缀、缺脚/缺对照、缺整组重复、相位偏移、空窗口、CSV缺帧/重复/非有限值，以及同内容异路径和来源不一致。
- 已对RF16本地CSV实际运行：288行分相位脚底、156行参考抬脚窗口、36行双脚受力快照、216行配对变化；5产物共128283字节。所有比较的共同前缀均等于各自完整保存时长，本次没有裁掉固定片失败后缀。
- 绘图辅助脚本`plot_q1_lift_windows.py`已实现并通过CLI help；本地系统及捆绑Python均无matplotlib，实际绘图在导入时失败，未生成PNG/PDF，也未为此安装环境。不能把脚本准备称为图片验收。

实际/参考脚底CSV首帧为0.02秒、接触标签包含0秒。工具按20ms整数时间戳匹配并跳过标签初始帧，拒绝缺帧；不按CSV行号直接拼接。三份旧收据的场景SHA和动作库内容SHA一致，动作库的不同目录作为路径别名保留，不因此判成不同数据。

本次基线标签`initial_static075`是历史名称，实际静态上肢alpha为0.50，不是0.75；该起点没有学到的新残差head。`stage_02`约601秒，`stage_03`约1201秒。三次是相同checkpoint评测重复，不是独立训练。

## 对齐后发现了什么

只取“参考最低碰撞脚底高于20mm”的对应帧，下表列RF16右脚三次评测的范围。它是描述性几何窗口，不是新增训练相位、数据准入门或自然步态认证。

| 固定片/版本 | 每次窗口帧数 | 同窗口参考脚底p95 | 同窗口实际脚底p95 | 实际脚底不高于5mm的帧占比 |
| --- | ---: | ---: | ---: | ---: |
| 停走/未训练静态.50 | 18 | 26.21mm | 1.56-2.28mm | 94.4%-100% |
| 停走/601秒 | 18 | 26.21mm | 0.50-3.65mm | 94.4%-100% |
| 停走/1201秒 | 18 | 26.21mm | -0.07-1.59mm | 100% |
| 起走/未训练静态.50 | 19 | 30.25mm | 0.02-1.97mm | 100% |
| 起走/601秒 | 19 | 30.25mm | 0.07-7.77mm | 89.5%-100% |
| 起走/1201秒 | 19 | 30.25mm | 1.34-4.46mm | 94.7%-100% |

停走整片320帧，起走210帧；参考右脚高于20mm分别只覆盖约5.6%和9.0%的帧。左脚相同条件只有4和8帧，原值保留，不把这几个窗口外推成全库结论。抬手参考没有此类窗口，指标为缺省而不是零误差。

这补充了整片p95的局限：最终停走右脚整片p95曾升至约19-29mm，但在参考实际抬高的对应时段仍贴近地面。高度和时序需要同时看。601秒停走滑移下降也没有对应的右脚可靠离地，不能用防滑均值代替学会换脚。

601秒停走上述窗口中，66.7%-94.4%的保存帧仍有右脚力大于5N；起走为36.8%-89.5%。这些力仅为每20ms控制步最后一个5ms物理子步的快照，不是20ms积分冲量，5N也不等于承担体重；零快照不能证明整步完成卸载。原始左右单脚受力变化次数只是诊断，不能称正确步数。

当前实际轨迹是RF16上肢混合/残差体系，不包含纯SuperMimic教师的同口径脚底对照，不能据此单独归因底座。G1与Q1也没有完成同控制、数据和预算的配对实验，不能从现状推导机器人或算法本身的优劣。

## 源码依据与两条候选

现有`script/supermimic_q1_gait_comp/rewards.py`的`FootReference`跟踪踝roll link。摆脚奖励一半为局部踝XY误差、一半为世界踝Z误差，没有直接度量碰撞脚底高度或原生卸载。RF16更换了参考接触相位，但继承这套踝位置奖励。这是任务表达上的待检验差异，尚不证明它导致全部训练退化。

| 研究线 | 下一项具体工作 | 独立候选主变量 | 配对时保持 |
| --- | --- | --- | --- |
| SuperMimic底座 | 先在原生后端核H32闭环actor/critic的想象改进方向与真实物理方向 | 在两个足端位置tracking槽位中表达碰撞脚底Z误差，保留足端XY、其他跟踪体及其他奖励分项和总尺度 | 原300/30、Q1控制/PD/effort、动作尺度、WM结构、H32、采样器和更新配方 |
| PPO/残差适配 | 实现独立奖励候选并完成张量/原生契约比较 | 只把现有摆脚评分的踝Z替换为同几何定义的脚底Z，保留XY分支及RF16原接触意图相位 | 同教师、静态上肢.50、14维腿腰残差、.15rad tanh范围、std.1、奖励系数及PPO更新设置 |

两项目前都是明确设计，尚未实现成可训练候选或得到物理验证。完整HP MDP/learner runtime只在服务器，本地不能假装已核其精确表达或直接修改旧冻结runtime。底座候选的精确奖励等价关系须在恢复后读实际源码再落独立副本；不从外围脚本猜源码细节。

H32验证应让未来动作随模型/真实状态闭环反馈，包含实际actor目标和critic自举，并在克隆参数上做有界方向扰动、恢复归一化/状态/RNG。旧H1/H4/H8固定未来动作检查不代替这一项。通过也只证明该检查范围内方向，没有证明训练会改善；原生足底力先作为物理评测量，不直接假设世界模型有可用的力梯度。

适配线后续保留观测归一化、残差权限/探索和PPO更新稳定性作为分别的对照。本次不同时改变这些设置，以免无法区分收益来源；也不把ACTIVITY动态上肢与新head拼成未经评测的“改进”。

比较次序为：B0/B1都不适配以看底座；同一底座上A0/A1以看适配；再实测对应组合与其未适配底座。比较矩阵是归因设计，不是自动授权批量开四轮训练。原30验证已参与搜索，不称独立最终测试。

主要看真实前进、脚底离地幅度及时序、负重切换/承重滑移、姿态、上肢幅度时序及全程失败，根XY仍为诊断。后续独立短试验沿用长期单轮最多3600实训秒的上限，不自动扩时或故障重试；本次尚无可启动配方，没有发出训练命令。

## 复现与本地工件

Mac根为`/Users/yu/Documents/ChatGPT/元启`，代码权威根为其`data/repositories/q1-rf12-20261028`。以下命令写入新的复现目录，已有目录会被拒绝，不能覆盖本次原值。

```sh
Q1_ROOT=/Users/yu/Documents/ChatGPT/元启
Q1_REPO="$Q1_ROOT/data/repositories/q1-rf12-20261028"
python3 "$Q1_REPO/script/project_tools/analyze_q1_lift_windows.py" \
  --actual "$Q1_ROOT/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/review/sole_v01/fixed_actual_sole_timeseries.csv" \
  --reference "$Q1_ROOT/data/supermimic_for_q1/Q1_SAVED_SOLE_GEOMETRY_20261029/fixed_sole_timeseries.csv" \
  --support "$Q1_ROOT/data/supermimic_for_q1/Q1_REFERENCE_SUPPORT_AUDIT_20261029/fixed_support_timeseries.csv" \
  --baseline-label initial_static075 \
  --output "$Q1_ROOT/data/supermimic_for_q1/Q1_DUALTRACK_LIFT_WINDOWS_20261029_REPRO_NEW"
```

- [分相位脚底原值CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_DUALTRACK_LIFT_WINDOWS_20261029/phase_feet.csv)
- [逐抬脚窗口原值CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_DUALTRACK_LIFT_WINDOWS_20261029/reference_lift_windows.csv)
- [左右受力快照CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_DUALTRACK_LIFT_WINDOWS_20261029/paired_support.csv)
- [配对变化CSV](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_DUALTRACK_LIFT_WINDOWS_20261029/paired_changes.csv)
- [输入前后SHA/来源及限制](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/Q1_DUALTRACK_LIFT_WINDOWS_20261029/summary.json)
- [本阶段小型结果及候选设计清单](../data/manifests/supermimic_q1_dualtrack_diagnosis_20261029.json)

工具记录本次输入CSV/旧收据的前后SHA及内容兼容性，未重复历史trace/媒体/保存审计，不把当前文件SHA补写为初始历史SHA。小负脚底高度允许来自软接触穿入，不强行截零。

## 阻塞、保存与下一步

本轮唯一允许的25386入口通过本机127.0.0.1:18873连接时返回Connection refused；一次有界本地核查显示无监听、控制socket缺失。服务器状态为UNKNOWN，不能据此认定容器停机或训练状态。已停止继续探测，没有访问旧入口/师兄容器、读取认证或重启远端/Windows桥。

代码、分析、候选及异常已存Mac，Git发布/独立40位远端核验和主根镜像按实际[发布收据](../data/manifests/supermimic_q1_dualtrack_publication_20261029.json)登记。25386小记录与5分析产物镜像待连接恢复，不能预称三端一致；权重和大NPZ仍SERVER_LOCAL_ONLY。

入口恢复后，先核唯一根与既有环境/来源身份，再补原生H32方向验证和两个独立候选的契约实现，完成CPU/16/4096及冻结发布后才进入短试验。继续依据本次两线推进决定工作，不恢复旧RF15/RF16、不补夜间剩余预算；材料未变时原监督保持静默。[方向](README_SUPERMIMIC_Q1_FRAMEWORK_DIRECTION_20261029.md)/[夜间完整结果](README_Q1_OVERNIGHT_RESULTS_20261029.md)保留。

2026-10-29后续[SOLE01](README_SUPERMIMIC_Q1_SOLE_CANDIDATE_20261029.md)已实现共享碰撞脚底几何和PPO摆脚Z独立候选，7本地几何测试通过，真实Torch/原生契约与底座H32待。Windows桥已存在并保留，但单次容器密码认证仍被拒；当前阻塞已从无本机转发变为容器认证，未新物理/推理/优化/训练。本页此前“候选尚未实现”与“本机无监听”是本阶段历史，最新状态以SOLE01为准。
