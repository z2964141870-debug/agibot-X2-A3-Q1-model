# 2048环境与两小时训练

记录日期：2026-10-25，北京时间。阶段：RF04-E2048-UP01。状态：23:21:31已正式启动2048环境训练；23:35实读300轮/614400真实转移，WM69组、actor19次，critic及梯度有限，未有本轮跟踪效果结论。

用户随后追加[4096环境论文规模](README_SUPERMIMIC_Q1_E4096_20261025.md)：本轮2048保持至少两小时与既有评测，完成并释放资源后顺序4096完整批量实测及独立下一轮。共用当前监督任务，不把下一轮授权解释为本轮热改或提前停止。

## 目标与范围

用户要求停止64环境路线，分析后采用2048环境实际训练至少2小时，并只开一个子agent分析位移；主代理负责2048配置、训练和监控。新实验独立创建，旧RF03已停止不恢复。

保留上一已确认的365帧双臂抬起加移动参考`train_3419f09ba93c91d75fc1`，先检查单动作可学习性。全局目标仍是SuperMimic for Q1，世界模型、H32 BPTT与critic，不增加新的数据硬门或直接改奖励/观测。环境数扩大提高独立状态、参考时刻和随机化上下文的并行覆盖，不等于每环境噪声幅度变大。旧配置已有domain randomization、观测噪声和动作N(0,0.1²)，不能写成没有噪声；低并行覆盖是否是主要根因仍需新结果。

## 配置依据与计划

原论文arXiv2610.28480 Table A.5记每GPU4096环境、世界模型每组10批×2048个8步窗口、策略4096个H32想象起点、UTD0.4、每环境1000行replay、动作噪声N(0,0.1²)。本次选择2048环境、WM batch1024×10、策略batch2048，保留UTD0.4，使每真实转移的训练采样量按论文同比例缩放；不只修改环境数而保留原64/128小batch。原64环境设置对应WM/真实转移与想象起点/真实转移均为该比例约2倍，但并行覆盖小。

先做有限2048实例、真实MuJoCo采集及一次WM/actor/critic更新的profile，检查数值与实际CPU/GPU内存，记录吞吐；这是运行可行性检查，不是数据或策略性能门槛。原后端为CPU MuJoCo/复用模型槽，2048指独立状态和随机化上下文，模型槽数不等于环境数。现场CPU cgroup额度24核，GPU3090/24GiB，停止后共享内存约22.1GB/OOM14；线程/槽数按实际资源选择，保留30GiB内存、100GiB余盘、12GiB单实验保护。

正式训练记录实际起止与活跃训练秒数，目标至少7200秒，正常达到目标后在完整采样/更新边界停止；另有明确最大采集轮安全预算，不靠等待凑时长。保护停止或异常导致不足2小时必须如实记失败/未完成，不能算满足要求或自动重启旧实验。

## 产物与验证

统一根`huanyu:/opt/data/private/ZHY`，沿四目录。预检和正式训练分别用独立产物目录，正式根为`data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025/`。每轮照常保存六宫格训练图；固定本轮初始/最终MuJoCo和上肢/位移分析，视频保存本地MP4，不建立网站。Mac不拉权重。

唯一子agent只读分析位移，不改训练代码、输入和共享入口，不跑新仿真/训练，不再分派；结果由主代理整合。训练期间保持源码和配置冻结。

已实现`collection_budget.py`与训练入口的可选限时预算：在完成的25轮报告边界且采集阶段实耗至少7200秒后正常结束；初始化、绘图和评测不计入，最大采集轮12000若先耗尽则报错，不能凑时长。原按步数训练保持原行为。模型槽/线程和batch均显式记录。4项预算边界控制和4项实际终态/配对控制在Mac及服务器均通过；首次预算测试因Mac Python3.9不支持立即求值的联合类型失败，已用延迟注解修正，再跑4/4通过。

新增`script/project_tools/profile_q1_2048.py`：2048独立状态、8线程/8槽、16采集轮，完整WM1024×10及actor2048/H32一次更新；仅profile加载既有RF02网络以避免把尚未预热的模型数值问题冒充资源不足，正式训练仍从零。输出真实吞吐、上下文/参考时刻覆盖、GPU峰值、共享内存和数值结果；无权重副本，不做跟踪效果评测。外部guard保留资源及进程回收保护。原训练/绘图入口复用，未改WM/BPTT/奖励/控制公式。

```sh
PY=data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
$PY script/project_tools/profile_q1_2048.py --root /opt/data/private/ZHY --output data/supermimic_for_q1/E2048_PROFILE_P01_20261025 --threads 8 --model-slots 8 --expected-oom-kill 14
$PY script/project_tools/run_q1_single_motion.py --root /opt/data/private/ZHY --output data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025 --num-envs 2048 --threads 8 --model-slots 8 --wm-batch 1024 --batch 2048 --steps 12000 --training-seconds 7200 --dashboard-label RF04-E2048 --code-commit f03db7566fcd590912b4cc9b24621d32b19319a2 --expected-oom-kill 14 --detach
```

## 2048实际预检结果

冻结代码`f03db7566fcd590912b4cc9b24621d32b19319a2`已推送Q1并独立核远端，服务器预算4控制和终态/配对4控制全部通过。profile于23:16:58至23:17:54运行并正常退出/worker回收，16采集轮32768真实转移；初始覆盖364个参考帧、2048个不同随机化上下文，摩擦系数样本标准差0.26217。

2048环境/8槽/8线程真实采集耗35.08秒，轮中位2.076秒，采集约934转移/秒；一组WM1024×10耗3.413秒，actor2048/H32及critic更新耗8.537秒，全部指标有限。replay实际1112113152字节，CUDA峰值分配8493581824字节/保留9080668160字节，共享容器最后25.374GB/OOM14/GPU37摄氏度。profile终后共享内存21.977GB，盘余926605905920字节；本轮profile仅16KiB分配，没有权重副本。该结果只证明运行可行，不证明训练效果。

原样证据：[profile](../data/manifests/supermimic_q1_E2048_profile_20261025.json)、[guard与回收](../data/manifests/supermimic_q1_E2048_profile_guard_20261025.json)。日志`logs/E2048_PROFILE_P01_20261025.log`。按实测正式训练可使用计划批量和8槽/8线程；该profile不计入本轮两小时。

## 正式训练启动与监督

2026-10-25 23:21:31，北京时间，正式新流程启动。controller为885444/start_ticks3623369662，worker为885445/start_ticks3623369674，boot为`d994e71f-bae0-46ba-a5f7-6c2163771c6c`。23:30:39已重新读取`/proc`身份及命令行，两者均存活且与启动记录一致。来源代码冻结为`f03db7566fcd590912b4cc9b24621d32b19319a2`，后续文档提交不代表改动运行中源码。

实际admission记录2048环境、8线程、WM1024×10、actor2048/H32、8模型槽、UTD0.4，从零初始化`RANDOM_FROM_SCRATCH_NO_A3_WEIGHTS`；唯一参考SHA为`f9bce4c3452720cda88a5adc4111fe2354440dd04e8c8ecb50cbf94d7181b8c3`。保留domain randomization、观测噪声和0.1动作噪声。配置/阶段命令见原样[启动](../data/manifests/supermimic_q1_RF04_E2048_launch_20261025.json)与[计划](../data/manifests/supermimic_q1_RF04_E2048_plan_20261025.json)。这些是历史启动证据，恢复时须重新核现场。

23:30实读225轮、460800转移、WM39组、actor0次，采集/训练已实耗约536秒；仍在模型预热，不能将其称为策略已学会。现场GPU39摄氏度/2128MiB，共享内存25098952704字节、under_oom0/OOM14，盘余861810061312字节。共享盘余的变化不等于本项目新增量；guard的最近本实验分配约69.3MB，未触发保护。

23:35:34正式确认策略已进入实际更新：300轮/614400转移、WM69组、actor19次，已记录训练822.86秒；actor梯度范数0.064625、critic loss0.162053及全部日志指标有限。GPU40摄氏度/9150MiB，共享内存23366381568字节、under_oom0/OOM14、盘余835628650496字节；本实验分配仍约69.3MB。78项冻结来源/流程入口身份全部匹配，controller/worker身份保持。原样生成证据见[启动后实际更新观测](../data/manifests/supermimic_q1_RF04_E2048_startup_observation_20261025.json)。这只能证明训练链路正在更新，尚不能证明已学会移动或抬手。共享盘余持续变化，监督继续按真实余量检查，不能把共享变化计成本实验增量。

本聊天监督任务`q1-2048`已启用，每10分钟检查现场进度、身份、GPU、内存、OOM与磁盘；普通未变化状态静默，仅异常、需要用户处理或最终结果通知。它不改冻结源码、不自动重启、不扩大预算、不创建或重新激活子agent。训练正常结束后自动接保存审计、同片初始/最终MuJoCo、上肢/位移分解、六宫格和两段MP4。最早约2026-10-26 01:22后满足两小时，实际完成以`training_seconds>=7200`及正常终态为准，评测另需时间。

唯一子agent已完成[位移只读诊断](README_SUPERMIMIC_Q1_DISPLACEMENT_REVIEW_20261025.md)，没有GPU/新仿真/训练作业。最支持旧RF02处于站立附近局部最优：目标速度未丢失，实际速度约目标5.7%、腿部幅度约26.5%、双支撑约97.86%；尚不能归因于单一奖励、接触或重定向问题。本轮先检验整套2048/同比例batch设置是否改善，不能提前宣称环境扩容成功，也不能把旧64环境667轮当等预算对照。

2026-10-26 00:31:02首个1000轮保存后的只读观测：`run/step_001000.pt`为205591335字节，SHA`3ef010ce23c170fe91a1389901930d2bcb2b0d977a1d82004addeb5956b6bbf1`。实际训练已到1050轮/2150400转移、WM369组/actor319次，已记录4085.71秒，仍不足7200秒；pipeline/guard/train均RUNNING，controller和worker的boot/start_ticks保持。GPU40摄氏度/9150MiB，共享内存19718049792字节、under_oom0/OOM14，盘余507.86GiB；最近guard本实验分配274960384字节。没有重复加载checkpoint张量、仿真或optimizer更新；保存存在不证明效果改善。共享盘余继续下降，不能计成本项目新增量。原样收据[首保存观测](../data/manifests/supermimic_q1_RF04_E2048_checkpoint_1000_observation_20261026.json)为4183字节/SHA`f90d2b1afbd6d6209667401c390e4d1ca652ea36a9101a0f0bae587e6e8c744e`，服务器源为本实验`checkpoint_1000_observation.json`，小收据原样取回；权重未复制到Mac。

恢复入口：本实验`status.json`、`train_guard.json`、`run/status.json`及`logs/BONES_RF04_UP01_E2048_T02H_20261025_{controller,train}.log`。终后仅把小图表和MP4复制到Mac的`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025/`，核大小/SHA；权重和大轨迹留服务器，仍按LOCAL_ONLY登记。
