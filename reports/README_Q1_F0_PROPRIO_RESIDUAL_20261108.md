# Q1 f0机器人历史残差适配配对实验

2026-11-08，北京时间。独立F0_PROPRIO_PAIR01；原共享训练保持人工暂停，32k教师与原final门不变，用户采纳仍为null。独立源码已发布，原运行时及准入已真实通过；实际训练与闭环结果尚待。

## 问题与假设

旧f0学生只在人体编码器中处理10×226当前/历史人体特征，930维机器人观测历史在冻结decoder处融合。原R2及扩展覆盖学生均0/7完整；扩展验证误差降低36.689%后反而更早失败。H4专项续训也没有一致改善，故不延长这些相同配方。

这次保留原R2 selected500人体编码器，在latent上增加930→128→latent小型适配器，仅学习已有因果机器人历史对latent的修正。公式为`z+(1-abs(z))*tanh(adapter(history))`，末层精确零初始化，初始动作与旧f0精确相同；latent仍在tanh范围内。

GigaBrain固定源码在decoder融合本体状态和参考token。这次编码器残差是本项目独立低成本假设，不宣称直接移植GigaBrain，也不意味着其整机或环境交互能力已被复现。既有[一手研究对照](research/2026-11-07/LOW_LATENCY_SUPERMIMIC_REBASE.md)及所有负结果保持。

## 配对与选点

两组均为相同适配器、相同初始化、seed、样本顺序、曝光和3000新增Adam更新。`zero_history`将适配器输入置零，`causal_history`读取原有归一化930维历史；共享decoder仍读取两组各自的真实机器人历史。只训练适配器，教师、WM、人体编码器、decoder、归一化、物理和参考时钟全部冻结。

严格复用R1、AGGREGATE01、COVERAGE01六个既有train/validation数组，仅加载`f0/history/teacher_action/clip_id`；不读取R1保存的full_goal或未来prefix。合并后train6725行/36片/22来源组，validation2358行/12片/6组。来源组、片和状态origin分层均衡采样。

验证每500更新一次，包含update0。按扩展验证风险最低选点，同时要求原R1和mixed两项固定验证锚点不劣于update0。两组不采用不对称早停；任一组未完成相同曝光，则整体不认证配对结果。新的stop/reverse/dropout/late_burst事件参数已登记在[不可变计划](../script/supermimic_teleop/student_proprio_plan_20261108.json)，在拟合和选点中不读取；先探索性七例完整率，再决定是否开展留出事件。

## 范围与预算

服务器独立输出为`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_PROPRIO_PAIR01_20261108`。总600秒/128MiB、0新物理/渲染，记原r2成本账本；进入前累计3557/14400秒，r1/r2/r3为1614/572/1371，无RESERVED，不重置预算。22GiB准入/24GiB退出、GPU18GiB/80度、100GiB余盘、private500GB、原GPU独占锁及自有子树回收保持，不重复清缓存。

六项本地CPU检查通过：初始精确动作/相同容量、因果历史差异、实际Adam只更新适配器及latent界限、错误维度/未来窗口拒绝、三更新配对曝光/选点/CPU有限保存、test来源拒绝。模型与控制器编译、CLI通过；原服务器运行时、来源资源准入及实际训练尚待。

主要入口为[controller](../script/supermimic_teleop/student_proprio_training.py)、[worker](../script/supermimic_teleop/student_proprio_worker.py)、[model](../script/supermimic_teleop/student_proprio_models.py)、[fit](../script/supermimic_teleop/student_proprio_fit.py)。发布后先原运行时测试及来源完整性/资源准入，再唯一输出/log/dispatch/进程防重单派；缺少回执时先核实际状态。

## Done与Todo

已完成独立实现、本地合同及训练前登记；真实新GPU训练/物理/媒体均0。下一步部署和实测两组训练，交付两张训练六宫图/CSV/原始收据。选中零适配器时先证与旧f0策略精确相同并复用旧失败；非零才在新独立worker中做同条件闭环，不能仅凭loss选点宣称自然遥操成立。

软件耗时、人体参考等待和硬件端到端延迟分别记录；本方案f0不使用尚未发生的人体帧，但仍不等于整机零延迟。最新发布、部署和服务器小记录镜像以独立publication收据为准，大包/全局索引旧镜像PENDING不混称完成。

## 原运行时与准入

源码提交`0549ea64677b4fbf2102c813c3d4b387014deec5`和后续不可变清单`e2e3cedab7558582eca5ad768c1475b3241a0f5c`均显式push origin Q1，并独立核对40位远端提交。十成员18512B源包沿既有45014会话五片精确ACK部署；1015原保护来源前后未变，只新增清单外文件。

真实Morph运行时六检查全过，本体1.681秒/进程约4秒；1031来源、六原数组/原R2保存点、原暂停及资源锁准入12.320935秒通过。总内存8196124672B/限34292629504B、OOM0、GPU1MiB25度、余盘183602642944B，GPU计算与原锁空闲。三原始收据5802B已Mac二进制大小SHA核同：[部署](../logs/session_records/q1_f0_proprio_deployed_20261108.json)、[原运行时](../logs/session_records/q1_f0_proprio_contracts_20261108.json)、[准入](../logs/session_records/q1_f0_proprio_admitted_20261108.json)。

真实新增GPU/Adam/物理仍0；发布本准入后再唯一输出/log/dispatch/精确进程防重单派。此前所有已完成物理、512配对和六检查不为等待重复。新增只读[训练六宫图入口](../script/supermimic_teleop/deliver_student_proprio.py)仅CLI通过，待真实数值使用；缺失的RL回合指标和逐更新loss不虚构。
