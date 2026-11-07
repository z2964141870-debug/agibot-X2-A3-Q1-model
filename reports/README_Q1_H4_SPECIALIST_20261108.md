# H4专用适配：复用R1保存数据与验证选点

2026-11-08，北京时间。目标仍是减少人体未来依赖并保持闭环质量；本阶段只考察H4/80ms专用适配，尚非f0、真机或整机低延迟验收。原共享训练保持暂停，用户采纳null。

## done / todo

done：H4_SPECIALIST01已真实完成3000次新增Adam，验证选1500，915保护来源相同、守卫清洁回收、77秒GPU占用已结算。六宫PNG/PDF、CSV和17成员训练证据ZIP已本地核验。等来源H4验证误差仅下降0.923%；[后续七例闭环](README_Q1_H4_SPECIALIST_REVIEW_20261108.md#实测结论与交付)现已完成，未呈现一致改善：三名义片上肢均值+0.55%、四事件上肢均回退，右脚滑移等部分指标改善。旧R1已混合五窗口，旧f0和COVERAGE失败保留。

todo：对固定selected1500新增三名义片及四事件，共七个独立闭环。严格复用旧R1/H10/H4保存对照，不用last或事件结果重新挑权重；先独立方案、来源与资源审计发布，再有界执行。拟合收益不算遥操质量收益。

## 假设与边界

旧R1把H10/H5/H4/H2/H1等权混合训练；本阶段只把已验证选中的R1编码器继续适配请求H4，检验去除其他窗口的竞争目标是否有用。初始权重、原12训练/6验证来源组、真实状态与H10教师标签、冻结decoder/normalizers/世界模型、参考锚点和动力学保持。不是从零重训共享策略，也不把追加训练自动归因为某种GigaBrain机制。

仅使用原1404训练/711验证条记录；H4首目标不可用的行不参与拟合，实际可用前缀可为1至4。先根据真实到达与120→50Hz插值右支持确定prefix，mask之后未知人体点及heading均为零，全部24点仍保留。不压缩轨迹时间、不把H4称作f0；原始人体预处理的因果性仍未验证。

单一主要变量为从固定R1选点进行H4-only继续适配。Adam学习率保持原0.0001，batch128，最多4000新增更新、每250验证、六次无改善早停。按来源均衡采样，验证按六来源等权的H4动作MSE选点，含第0步；其他H仅同验证集诊断，不影响选点。旧R1五H等权风险与本阶段H4风险口径不同，另报样本加权H4风险用于核旧0.056171基线，不能直接混比两种风险。

训练教师仍可见H10作为离线标签；部署输入只允许请求H4。没有新的原始人体/test读取、物理采集或渲染。本阶段的拟合改善只能决定是否值得闭环，不能用于宣称自然遥操合格。

## 成本与安全

实核唯一成本账本累计2492/14400秒，r1=549、r2=572、r3=1371，全部SETTLED，无RESERVED。R1原2700秒上限余2151秒；新阶段最多600秒，不重置账本或扩大阶段配额。资源准入总8036737024B/限34292629504B、OOM0、GPU1MiB25度、磁盘余183749042176B，private266931113984B；这是准备时快照，派发必须重新准入。

保留22GiB准入/24GiB退出、GPU18GiB/80度、磁盘余100GiB、private<500GB、原非阻塞GPU锁和自有子树回收。输出128MiB/总600秒，零新物理和媒体。复用已完成的缓存边界，不清缓存、不改原冻结来源。未知派发保留成本预留并先查同任务，不重复启动。

## 实现与证据

新增独立`student_h4_training.py`、`student_h4_worker.py`、`student_h4_fit.py`、固定plan及六项聚焦CPU合同。真实原运行时检查、部署与训练以后续收据为准，不把随机CPU夹具Adam计为32k训练。模型保存为CPU有限张量；selected保留验证选点，last及optimizer只作诊断。原教师、decoder及normalizers哈希前后核同。

唯一服务器输出：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_H4_SPECIALIST01_20261108`。成本仍为原`logs/session_records/q1_low_latency_gpu_cost_20261107.json`。操作前核原root/boot/GPU、旧五PID退出与STOPPED_SIGNAL、任务guard/锁、OOM、内存及磁盘；第一跳仍原样source zshrc再ssh -tt hp_han。

下一闭环需独立准入，三名义片加stop/reverse/dropout/late_burst四事件，保留完整率、脚底离地/换脚、双脚承重滑移、侧倾、上肢时序/首秒与动作突变，以及完整软件长尾。先比较原R1选点，不用旧均值或loss挑test；未通过闭环时不采纳checkpoint，也不恢复原160k训练或改变final门。

参考既有一手来源与时间语义：[SONIC/TWIST/GigaBrain固定源码审计](research/2026-11-07/LOW_LATENCY_SUPERMIMIC_REBASE.md)。SONIC的独立四帧专用权重只说明该路线值得验证，不代表本模型已经获得相同效果。

## 原运行时与真实准入

源码`8a4bf496c7547359a60d8736bcd4b4d9ff202bcb`已scoped audit/显式push origin Q1、独立40位远端和Mac12文件839101B核同。11276B六成员源码包SHA `2bed0ca07a0fcdfd3fbf770698ba45890b9b46ecbb675d16cca72c792e2b50c3`沿45014原SSH四片精确ACK部署，906保护输入前后同。

原策略运行时六合同全过，本体1.354秒/进程3.971秒；CPU夹具不算真实策略训练。915保护输入、原R1实际1404/711状态与动作trace、原设备/暂停与GPU锁准入12.083秒全部通过。准入总8049188864B/限34292629504B、OOM0、GPU1MiB25度、余盘183748939776B；尚未新GPU训练或预留。

三个服务器原始回执5385B已Mac逐字节SHA核同，2140B四成员传输包SHA `7b98d1bdf358cdf3d565a0ec5193df06ad88ed17e7fb0802a7892451de759446`保留。[部署](../logs/session_records/q1_h4_specialist_deployed_20261108.json) / [原运行时合同](../logs/session_records/q1_h4_specialist_contracts_20261108.json) / [准入](../logs/session_records/q1_h4_specialist_admitted_20261108.json)。发布本准入后，先核唯一output/log/dispatch/精确进程全无才派一次；不重新部署或重复六合同，原训练仍暂停。

## 真实训练与交付

源码8a4bf49及准入8e7b24a发布后，唯一371344/start39324721于北京时间2026-11-08 03:38派发；03:42:18确认控制器、timeout371343和worker371418均已退出，守卫WORKER_COMPLETED/reaped0，来源915项前后核同。真实3000 Adam更新、十个参数计数均3000，六次验证无改善早停；固定selected1500，last仅作诊断。

| H4验证口径 | 初始R1选点 | 新selected1500 | 变化 |
| --- | ---: | ---: | ---: |
| 六来源等权，唯一选点指标 | 0.0558613691 | 0.0553457197 | -0.923% |
| 样本加权，保持旧口径 | 0.0561711051 | 0.0557506119 | -0.749% |

六来源逐项变化为+0.18%、-1.10%、-6.18%、-0.60%、-5.93%、+7.33%，收益不一致。请求H4时实际prefix1/2/3/4曝光为1610/127522/229669/25199，缺失未来没有补成真实观测。纯fit50.620秒、GPU占用77秒；累计2569/14400秒全SETTLED，r1=626、r2=572、r3=1371，无RESERVED，r1余2074秒。

03:42快照总8151814144B/限34292629504B/OOM0，GPU1MiB26度，磁盘余183630491648B、private267049664512B，原锁和计算空。原共享人工暂停。selected29529221B/SHA `62c5d97403912284b1dd55a949149eac922939954627bb3f25c52bff58eaeb15`留服务器，不传大型权重。

[派发原件](../logs/session_records/q1_h4_specialist_dispatch_20261108.json) / [完成原件](../logs/session_records/q1_h4_specialist_completed_20261108.json) / [图表与交付审计](../logs/session_records/q1_h4_specialist_training_delivery_20261108.json)。10原文件62313B归档逐件SHA核同。Mac训练六宫图已目视，PNG/PDF/CSV及README与原件压缩包：`data/deliveries/Q1_H4_SPECIALIST_TRAINING_EVIDENCE_20261108.zip`，307711B/SHA `914f47b8ed26291bfc9de22cd36419ccc8cf0efe7915d68041d9c4aa31414f58`，17成员CRC/逐件SHA及8链接通过。本包没有新视频，因为本阶段只训练编码器；下一七例才产新原速媒体。

训练六宫图：

![H4训练六宫图](../data/analysis/Q1_H4_SPECIALIST01_FIGURES_20261108/training_six_panel.png)

结论：同一小数据上的H4-only继续拟合只有小幅收益，不值得直接延长同配方。先用七个闭环检验动作累积、滑移和事件响应；尚不能认定自然遥操或硬件端到端合格。最新小文档服务器镜像见publication收据；旧大包与全索引仍PENDING。
