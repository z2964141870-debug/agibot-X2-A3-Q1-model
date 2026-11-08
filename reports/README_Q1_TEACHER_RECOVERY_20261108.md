# Q1：教师恢复能力与物理状态恢复门

2026-11-08，北京时间。独立实验TEACHER_RECOVERY01。状态：首派导入失败保留，独立RESUME1已通过原CPU运行时和来源资源准入，待唯一防重派发；真实新推理、物理和训练均0。原共享训练保持人工暂停，32k候选与原final门不变，用户采纳null。

## 问题与依据

[纯学生标签配对训练](README_Q1_F0_ONPOLICY_PAIR_20261108.md)已经真实完成，两组所有非零选点均未守住旧锚点，最终都选update0。继续增加同一单步拟合的更新数没有证据支持。下一步先问：在学生已经偏离的同一真实物理状态上，原H10教师是否能改善短段支撑与跟随？教师只能作为训练期特权上界，不作为可部署零预览策略。

原Q1Simulator源码19173B/SHA b941655b4c527fb92c498b213ca2c6e035fd4ced3626d0bb7a0502aa4f1627a8已只读核实。其state79保存qpos29、世界线/角速度与关节速度共28、上一动作22；CPU内部另保留MuJoCo double状态、接触求解热启动、时间与控制器张量。reset会清上一动作并重新初始化物理状态。因此任意把保存state传入reset，不能未经验证就称原现场恢复。这里先以实际下一步等价验证，不由静态推断宣布实际偏差大小。

## 独立方案

仅使用已登记采集源0/2/6/8：训练A095/A075，验证A267/A033；固定锚点64/96，共8个，不参考新结果选择。缺失、已终止或当时不可用的锚点明确记缺，不替换。不会打开test、新留出事件或重跑旧七条和整84例。

每个源只重建必要的第0至97状态前缀，执行原保存动作，无新模型调用，逐状态容差2e-6。锚点复制完整MjData及Q1控制张量、CPU随机状态；恢复后相同动作的下一状态必须逐位相等，才允许分叉。另外记录reset-only与补回动作记忆后的下一步误差，均不作为分叉恢复方式。

每个锚点的学生f0和特权H10教师都从同一全状态副本和相同机器人历史开始。历史只从原保存状态前缀重建，不做新模型前向；首个学生动作及教师动作分别复现原学生动作和原独立标签，容差2e-5。学生继续调用原ProprioReviewSession/CausalReviewSession路径，不接CPU加速或教师未来；教师分支才读取f+1至f+10。保留原事件可见性与离线120→50Hz依赖，合成时钟不冒充硬件延迟。

每分支最多80步/1.6秒，遇原终止条件立即结束；若同片十帧特权未来支持不足，明确缩短支持范围，不伪称80步完整。保存终止、接触承重、脚底离地/换脚、滑移、侧倾、上肢时序和动作突变。逐段诊断标明是否包含终止步，并保留排除终止步的有效前缀；不对失败片报虚构全片均值。确定性分叉不是独立随机化试验，也不能直接证明跌倒恢复或自然遥操。

## 来源、预算与执行

五份独立[controller](../script/supermimic_teleop/student_recovery.py)、[CUDA worker](../script/supermimic_teleop/student_recovery_worker.py)、[整状态恢复](../script/supermimic_teleop/student_recovery_state.py)、[固定计划](../script/supermimic_teleop/student_recovery_plan_20261108.json)、[合同检查](../script/supermimic_teleop/test_student_recovery.py)，不改任何旧执行或训练源码。

新根`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_TEACHER_RECOVERY01_20261108`。最多1800新物理步/1280新模型前向/900秒/128MiB，0Adam/0渲染。CPU物理保持已验证Primebot固定运行时，策略使用原NVIDIA CUDA。原r2台账4503=1614+1518+1371秒全SETTLED/无预留，总14400、r2余2982不重置；本轮计实际GPU持锁墙钟。22GiB准入/24GiB退出、GPU18GiB80度、100GiB余盘/private500GB/原非阻塞GPU锁和自有子树回收保持，不清缓存。

先源码与精确清单发布，再在原运行时跑新合同、全部既有保护输入/前一实验清洁回收/原暂停/实际资源与台账准入，发布准入回执后，唯一output/log/dispatch/精确进程全无门通过才单派。失去回执先查现场，不重发。

复用原Linux45014；必要第一跳仍原样source /Users/yu/.zshrc后ssh -tt hp_han。用户批准的ConnectTimeout90/ServerAliveInterval60/ServerAliveCountMax5已再次自主核实，未改变认证或其他配置，不要求用户代测。认证不进入记录。

## 当前验证与下一步

本地六项合同：五通过/.003秒，真实MjData复制绑定检查因Mac无MuJoCo明确待原服务器运行时；没有用替代物理或跳过该门开始实验。CLI通过。源码检查中修正了本地测试搜索路径，不涉及已冻结源码或真实新运行。服务器源码部署、原运行时检查、来源资源准入与真实分叉尚未执行。

下一先完成原运行时整状态绑定与来源门，再执行本轮短分叉。若教师也无法稳定恢复，不能直接把其同状态标签当可行控制目标；若教师明显改善，则用恢复状态/序列目标设计有旧能力约束的后续适配。不会因本轮完成或文档已写完而结算低延迟遥操整体目标。Mac证据和服务器小记录以实际publication核同；旧大索引镜像仍PENDING。

## 原运行时与真实准入

源码e94ede9bc67d27ee42f368b6048dd6026def8fba及五执行文件清单bc4fed161893d605fc63c72a3312270abe77bb39均显式push origin Q1并独立40位远端核同；Mac主根十一文件887631B与权威副本逐字节相同。16706B八成员包SHA14990bd01d2ab4713c976460514ff1b6cf9a6352fd11c8c0babc7e768a589655经14逐片ACK仅新增精确文件，1152旧保护来源前后同。[部署原件](../logs/session_records/q1_teacher_recovery_deployed_20261108.json)。传输延迟时等待对应ACK，没有重复提交分片或把延迟称断联。

原Morph运行时六项合同全部通过，本体.017秒/进程2.020343秒，包含真实MjData复制与控制器/RNG恢复后的同动作下一步完全相同；本项为简单模型绑定夹具，不替代后续真实Q1锚点门。[原运行时记录](../logs/session_records/q1_teacher_recovery_contracts_20261108.json)。

1162保护输入、前一配对实验终门和清洁回收、实际原学生选点、固定来源/锚点、原4503成本与暂停/资源锁准入11.324664秒通过。[准入原件](../logs/session_records/q1_teacher_recovery_admitted_20261108.json)。总8255102976B/实际限34292629504B，GPU1MiB24度/OOM0/余盘183539904512B；三原回执4675B已Mac逐字节SHA核同。本轮真实Q1物理和模型前向仍0，尚无预算预留。下一先发布本准入，再唯一新output/log/dispatch/精确进程全无门后单派900秒短分叉；不重复六合同或部署。旧大索引服务器镜像PENDING。

## 首派CPU导入故障与独立续行

49a612aeb92385146d5160323665f378677673ef准入已显式推送/独立远端同后，防重门单派timeout388155/controller388156/start40991605。[原始派发](../logs/session_records/q1_teacher_recovery_dispatch_20261108.json)。CPU child388230在导入morphmimic.constants时触发包__init__所需mjlab缺失；该错误发生在Q1Simulator或CUDA worker创建之前，真实Q1物理/模型调用/Adam均0。新入口没有在CPU固定运行时隔离常量导入，这是实现遗漏，不能归因网络、教师或步态质量。

原guard WORKER_FAILED/return1已清洁回收，三个PID全退、原锁和GPU计算空；FAILED根及六原工件保留，1162输入SHA全同。实际4秒SETTLED，累计4507=1614+1522+1371秒，无预留；原900秒实验按起止时间保守扣14秒，仅余886秒。总8247029760B/限34292629504B、GPU1MiB25度/OOM0/余盘183539552256B。[失败及完整来源原件](../logs/session_records/q1_teacher_recovery_import_failed_20261108.json)7476B/SHA e96d55bd64ec3446fc6c26a577552a5e1c73b71ec589fd20770f40e5fb893977已Mac核同。

仅新增[独立续行controller](../script/supermimic_teleop/student_recovery_resume.py)和[四项回归](../script/supermimic_teleop/test_student_recovery_resume.py)。CPU通过AST读取冻结常量的字面量关节stems并验证JOINT_NAMES表达式，不导入/执行Morph包；CUDA原worker和物理恢复/分叉函数其余AST逐项相等，五原执行文件逐字节保留。四本地检查.028秒全过。新根Q1_LOW_LATENCY_CKPT32000_TEACHER_RECOVERY01_RESUME1_20261108；原8锚点/1800步/1280前向/128MiB/0Adam不变，仅886秒和原4507账本，不重新发放900秒。

续行仍待独立源码清单发布、原CPU运行时新四检查/真实常量读取及包含FAILED根的来源/资源准入；通过并发布后唯一防重单派。没有重复旧六项合同或任何旧物理，整体低延迟目标未结算。45014原Linux继续可用，桥接没有被误报中断。

## 续行真实准入

独立源码99c0d047f6b6322e905a329775ca0cd525132cc1及七执行清单98d7f52a7e5d81f1008cda2a7d6e3c0b83a3e715已显式push origin Q1、独立40位远端核同。9213B三成员新增包经八个精确ACK部署，1162原保护输入及FAILED工件未变。[部署原件](../logs/session_records/q1_teacher_recovery_resume_deployed_20261108.json)。

原Primebot CPU运行时四项回归全过，本体.044秒/进程.460116秒；真实冻结29关节声明已AST读取，morphmimic包未初始化。[CPU合同原件](../logs/session_records/q1_teacher_recovery_resume_contracts_20261108.json)。包含原失败工件的1172保护输入、原暂停/成本/资源锁准入11.085109秒通过，总8271482880B/限34292629504B，GPU1MiB24度/OOM0/余盘183539441664B。[准入原件](../logs/session_records/q1_teacher_recovery_resume_admitted_20261108.json)。三原回执4335B已Mac逐字节SHA核同。

真实新Q1物理/前向/Adam仍0，没有新预留。发布本准入后，检查独立RESUME1输出、日志、派发收据与精确进程均无，才单派剩余886秒；保持原900秒失败消耗及4507秒台账，不重复原运行时检查。旧大索引服务器镜像仍PENDING。
