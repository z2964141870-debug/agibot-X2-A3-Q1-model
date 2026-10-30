# Q1双线逐子步观测

2026-10-30 18:11:07北京时间更新：按用户更正改用[原始Windows两段式SSH](README_Q1_NATIVE_WINDOWS_SSH_20261030.md)，36daafb的7个待镜像小记录366473B已在原25386补齐并核大小SHA。严格公钥pin/原boot与GPU根一致；此前UNKNOWN/服务器镜像待仅历史，观测和CPU分析未重跑。恢复证据见原发布旁收据的native恢复字段及新连接清单。

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SUBSTEP_OBSERVATION。两线各一次观测与后续保存数据CPU分析均已完成，合计6360控制/25440子步/7950策略批，0optimizer。PPO起走右脚在低力时仍多数贴地，所核贴地样本踝关节未限矩；底座本次单环境右脚抬高伴随左脚降低和起走倾斜代价。普通自然步态目标仍未达，goal ACTIVE，采纳null。

## 实际观测结果

底座北京时间09:51:39至09:56:19、controller249353/start23874444与worker249360/start23874588；PPO10:34:33至10:39:23、controller253131与worker253138/start24131963。两guard均WORKER_COMPLETED/reap0；尝试/返回/核验完成数各12720，12片各完整/无提前跌倒，每路线3180控制未超界。底座3180策略批，PPO4770策略批，0优化更新。原四进程均已现场核不存在，10:43:46核原boot/GPU、15MiB/0%/无计算/OOM0。

两路线各6组配对已记录初态、首个策略输入、系数与RNG全部逐值一致，仍非完整求解状态等价或历史330环境重放。底座完成后临时本机18873及双master消失导致UNKNOWN，原启动会话返回0；按用户要求加载既有hp_han助手恢复Windows路由/原25386，取得终态后才启动PPO。没有恢复或重做底座，没有改变认证配置/服务器环境。缓存原工具一次正常低于8GiB阈值跳过，未删文件或修改维护任务。

14小工件654236字节已取回Mac并核三端大小SHA；每路线36份原始NPZ与检查点仍留服务器。8项CPU测试检查原生时轴/状态连续性、子步计数、执行器ID映射/PD力重建、空条件与同刻低力mask。随后唯一一次只读CPU分析正常退出0：24片/25440子步，144足条件、1440关节条件、3480采样片段行；分析不创建策略或仿真，不新增积分、推理或优化。72份NPZ与两份results JSON的74项身份在读取前后不变。9份派生产物6069077字节已核权威工作树/Mac主根/25386大小SHA，两张PNG已目视检查。

执行、原始与派生产物身份见[结果清单](../data/manifests/supermimic_q1_gait_goal_substep_observation_results_20261030.json)。观测源码冻结9c71405ddf216dac3b30326a6302de71c8e7d7f7，CPU分析源码来自cf57f80ac33ca775f52179b8b63c0463dfc1b7b9；后续文档提交不改变两者。下文授权时的启动快照仅历史，两条一次授权均已用。

## 同时刻低力与贴地

低力定义为同一solve/pre时刻本脚世界Z力幅值<=5N、另一脚>5N；贴地为同刻碰撞脚底最低点<=5mm，包含负高度。没有插值到参考摆脚相位，也没有把积分后的姿态与之前的力拼接。PPO的trained600仅打开原600秒head，zero_head为同一冻结教师/同一head权重的零残差对照。

PPO起走右脚三重复：低力样本227/215/216，其中贴地185/211/199，即81.50%/98.14%/92.13%；整片脚底p95为5.463/2.848/3.722mm。>10mm样本仅10/0/4，低力且>10mm、至少12个相邻采样点的片段均为0。2.5/5/10N阈值下此处计数与结论相同。左脚同期p95为17.414/19.640/18.218mm，不能称双脚都完全不抬，也不能用左脚遮盖右脚缺失。

右脚低力贴地样本中的踝pitch力矩绝对p95为4.308/4.367/4.194Nm、最大7.052/7.177/7.235Nm，近22Nm边界样本均0；整段仍有34/33/28个限矩采样点，约4.05%/3.93%/3.33%，均在本脚>5N的承重条件。近限定义是与记录的任一力矩边界相差<=1e-4Nm。这说明“右脚贴地全因踝持续限矩”的解释不足，不说明承重期限矩无影响，也不隔离PD、接触、奖励、模型或学习的单因。

全部24片的记录实际力矩与按记录系数重建的直接PD/限矩最大差2.229e-6Nm。对齐和重建是记录一致性核查，不是实机扭矩测量。约束广义力含多类约束，不能直接当足部接触力矩。

## 底座收益与代价

下表均为本次单环境三重复的中位数。脚底p95取整片高度，不是参考摆脚误差；滑速按各脚本身>5N的样本统计，不是“共同承重滑移”。原始逐重复值保存在foot_conditions.csv。

| 路线/动作 | 条件 | 左/右脚底p95 mm | 左/右承重滑速 m/s | 根倾斜RMS rad |
| --- | --- | --- | --- | --- |
| 底座/停走 | initial | 12.708 / 4.674 | .1002 / .1147 | .0900 |
| 底座/停走 | trained301 | 9.925 / 11.643 | .1156 / .1078 | .0908 |
| 底座/起走 | initial | 20.765 / 4.365 | .1067 / .1845 | .0919 |
| 底座/起走 | trained301 | 12.802 / 14.086 | .0785 / .1413 | .1057 |
| PPO/停走 | zero_head | 12.708 / 4.674 | .0999 / .1143 | .0901 |
| PPO/停走 | trained600 | 11.569 / 4.631 | .1030 / .1083 | .0876 |
| PPO/起走 | zero_head | 20.765 / 2.897 | .0963 / .1845 | .0919 |
| PPO/起走 | trained600 | 18.218 / 3.722 | .1106 / .1640 | .0882 |

底座训练后右脚p95停走三次均11.643mm、起走13.973-14.086mm；低力且>10mm的>=12样本片段分别4/4/4、3/3/3。左脚p95中位数同时降低，停走左脚承重滑速升高。用同次原gait控制帧指标比较，起走倾斜比为1.149/1.164/1.087，增加约9%-16%。因此存在右脚离地样本增加，也存在左右脚和姿态的代价，尚非稳定自然步态。

这些是新建单环境钩子观测，与历史330环境保存结果不同；12组配对首状态/输入/系数/RNG相同，不等于完整求解状态、完整轨迹或原评测重放。不能拿这次右脚抬高覆盖历史训练的physical_regression，不能据此通过目标、采纳checkpoint或组合两线。

## 产物与时轴边界

派生根为data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_ANALYSIS_20261030，含foot_conditions.csv、joint_conditions.csv、sample_runs.csv、native_solve_series.csv、summary.json及两路线各PNG/PDF。summary保存74项原始身份与24片元数据；结果清单保存9项派生大小/SHA。原始72个NPZ、权重保持SERVER_LOCAL_ONLY，未为发布而重跑分析或搬运原始存档。

- [底座同刻脚高、受力、踝力矩](../data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_ANALYSIS_20261030/base_native_substeps.png)
- [PPO同刻脚高、受力、踝力矩](../data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_ANALYSIS_20261030/ppo_native_substeps.png)
- [CPU分析执行日志](../logs/session_records/q1_substep_observation_analysis_20261030.log)与[8项测试](../logs/session_records/q1_substep_observation_analysis_tests_20261030.log)

两图是同刻三重复逐帧中位数，可读性已检查；中位数曲线不是任一实际完整轨迹。原生每5ms记录pre/solve/post；solve力/脚高/执行器力矩属于积分前，post qpos晚5ms。12个相邻点曝光合计60ms、首尾仅55ms，采样间不可见，不等于连续卸载或已验证步数。负脚高和重复差异保留，不能将曲线峰值当参考相位正确的摆脚。

保存分析使用CUDA_VISIBLE_DEVICES=-1、单CPU线程及300秒外限，现存输出拒绝覆盖。以下为已执行命令，仅供来源复核，不应再次运行：

```sh
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=10s 300s \
  data/environments/supermimic_q1_hp_20261026/bin/python -u \
  script/project_tools/analyze_q1_substep_observation.py \
  --base data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_BASE_20261030 \
  --ppo data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_PPO_20261030 \
  --output data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_ANALYSIS_20261030 \
  --source-base-commit cf57f80ac33ca775f52179b8b63c0463dfc1b7b9
```

## 授权与固定范围

[授权记录](../data/manifests/supermimic_q1_gait_goal_substep_authorization_20261030.json)对应原问题中底座与PPO各一次零优化观测。执行[已发布V02 plan](../data/manifests/supermimic_q1_gait_goal_substep_observation_plan_v02_20261030.json)，86121字节、282来源、SHA-256 c94cf9489c317ea6a2ca431af898bc4f8704e698067c5e927d17a389b1234942，文件不重新生成。原问题13测试为V01；V02只修正异步返回/核验完成和失败重复累计，17测试已通过，case、检查点、种子与全部上限相同。

底座比较原initial与301秒actor；PPO在同一原冻结教师上比较同600秒head的零/开状态，alpha.5、PD、限矩、参考均用plan冻结值。原固定停走320控制与起走210控制，各条件3次配对seed32000/32001/32002；每线12个单环境case，最多3180控制、12720子步、6360策略批调用；两线串行，合计最多6360控制、25440子步。每线1800秒停止阈值加10秒回收，不重试、不训练。图捕获、环境初始化/reset forward与rollout转移分列，不能当作训练。

唯一新根：data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_BASE_20261030及Q1_GAIT_GOAL01_SUBSTEP_PPO_20261030。只运行冻结入口probe_substeps.py supervise，不能修改plan或换名再做。若失败，保留异常、原始部分轨迹、尝试/返回/已核验数与实际资源回收证据。

## 发布与现场

分析结果提交e69fb305ee650a101f17a57fd63fe757dd15a099已push origin Q1，独立ls-remote返回相同40位值；6小记录356430字节已核Git blob/权威工作树/Mac主根大小SHA。最终小记录镜像时本机18873拒绝连接；按用户指示加载既有hp_han助手，Windows曾返回DESKTOP-3C1DPPR，随后对端reset/Broken pipe，原25386登录随Windows转发消失中断，未执行容器命令。不把这次断线判为容器密码错误或服务器重启。一次有界核查后停止连接尝试，当前远端UNKNOWN。

观测14文件及分析9工件的三端核验均发生在断线前且已完成；本次6小记录与随后连接状态/发布收据的25386镜像待补，不能称最终三端发布完成。Mac和Git先保留本次完整结论及连接异常，待连接可用只补这些小记录；原观测、CPU分析、NPZ和训练不重做。最后GPU/OOM/原进程退出只引用10:43:46历史现场，不冒充当前在线状态。

授权已按README、索引、payload audit、origin Q1独立40位远端及三端大小SHA流程发布，见data/manifests/supermimic_q1_gait_goal_substep_authorization_publication_20261030.json。观测完成和分析源码已发布于cf57f80ac33ca775f52179b8b63c0463dfc1b7b9；本次分析结论及执行日志沿用相同流程。最终发布状态以data/manifests/supermimic_q1_gait_goal_substep_observation_publication_20261030.json为准，收据只登记实际完成的核验。

2026-10-30 09:45:08北京时间只读快照：原boot4d859008-4c83-4a6d-814d-cbb9dfe885c9、GPU-26028785-9fb9-45af-e614-4cc53af15d08，GPU15MiB/0%/无计算进程/OOM0；容器总5.18GB、cache4.31GB、RSS20.4MB，两个唯一观测根均不存在，V02 plan大小SHA一致。正式启动仍由原24GiB准入/30GiB守卫和独立GPU锁检查；快照不替代现场进程收据。

## 命令与时轴

```sh
substep_root=/opt/data/private/ZHY
"$substep_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$substep_root/script/supermimic_q1_hp/activate.py" \
  "$substep_root/script/supermimic_q1_gait_goal_analysis/probe_substeps.py" supervise \
  --plan "$substep_root/data/manifests/supermimic_q1_gait_goal_substep_observation_plan_v02_20261030.json" \
  --line base \
  --output "$substep_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_BASE_20261030" \
  --authorization-record "$substep_root/data/manifests/supermimic_q1_gait_goal_substep_authorization_20261030.json" \
  --code-commit PUBLISHED_40_HEX
```

以上是已完成的历史观测命令，PPO当时仅将line改为ppo并用预登记PPO根，在底座退出并确认GPU释放后执行。实际代码提交、PID/start_ticks、guard返回和原始输出已在结果清单补齐；两条一次授权均已用，不得据本命令重跑。

每5ms保存pre/solve/post；当前后端先forward/solve再积分，实际力矩与接触、脚姿态属于积分前，post qpos/time属于后态。不能将不同时间点拼成因果关系，也不能将采样近限时长当连续饱和。每片保存首输入、模型系数、可用原生状态/RNG并比较；相等不证明完整求解状态等价，更不是历史330环境重放。

## 边界与下一步

本阶段观测和保存分析已完成；大NPZ与权重留25386，诊断范围已从“指令是否存在/是否始终限矩”缩小到低力时的姿态与抬脚协调，但尚未隔离训练失败单因。SuperMimic底座与PPO仍各有改善空间，新的配方或训练须用户看具体数据后另作决定；不直接扩大残差或提高力矩/PD，不将本阶段诊断当干预收益。

原底座301.1074845249823秒、PPO600.0709074319457秒与physical_regression结果保留，总901.178391956928秒不追加。训练V02和两条独立观测授权均已用，不恢复、重试、扩时或新建agent；采纳null、goal ACTIVE且未达，不结算或暂停，监督/缓存维护/其他自动任务及认证环境均保持。
