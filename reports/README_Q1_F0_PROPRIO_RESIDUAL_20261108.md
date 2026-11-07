# Q1 f0机器人历史残差适配配对实验

2026-11-08，北京时间。独立F0_PROPRIO_PAIR01已完成配对训练，source_integrity PASSED；两组各3000次新增Adam更新。[后续七例闭环已完成](README_Q1_F0_PROPRIO_REVIEW_20261108.md#实测结论与交付)，完整率仍0/7，不采纳或延长相同拟合。原共享训练保持人工暂停，32k教师与原final门不变，用户采纳仍为null。

## 真实训练结果

| 同容量配对组 | 初始验证风险 | 选中风险 | 选中更新 | 新增更新 |
| --- | ---: | ---: | ---: | ---: |
| zero_history | 0.08454723 | 0.08454723 | 0 | 3000 |
| causal_history | 0.08454723 | 0.05385383 | 3000 | 3000 |

真实历史组的分组均衡验证误差下降36.303%，六个来源组均下降，两个固定旧验证锚点分别从0.08218014降至0.04787220、0.10754757降至0.08536243。置零历史组选中原始零适配器。各组123296个可训练参数、384000次样本曝光、四个实际Adam计数均3000；没有不对称早停，没有读取留出事件。

这只支持机器人历史对保存状态上的动作拟合有帮助。此前扩展覆盖也曾降低验证误差但0/7且更早摔倒，故不能从本次loss下降推断闭环改善。下一步先CPU证明置零组选点与旧f0策略相同并复用旧0/7；只为非零causal_history selected3000新增七例探索性闭环，再决定是否打开训练前登记的新事件。

训练持锁114秒已SETTLED；累计3671/14400秒，r1/r2/r3为1614/686/1371，无RESERVED。timeout380284、controller380285/start40114360和worker380359已退，守卫WORKER_COMPLETED/reapedtrue/return0；原五训练PID仍退，GPU锁与计算空。峰总内存10410434560B、GPU480MiB、子树RSS2884673536B、OOM0。未发原训练信号、未回收缓存，新增物理/渲染0。[完成原始证据](../logs/session_records/q1_f0_proprio_completed_20261108.json)。

两张训练六宫图PNG/PDF、两CSV及15份服务器原始记录已本机核同并目视。记录间隔500更新，逐更新raw loss和RL回合指标未记录，图中明确缺失，不虚构平滑曲线。原值包91068B/SHA `82e624e6dd2c2ae709b56f7701c0a2aff73d174cce3e8ee60d311f12498b0135`；[训练证据ZIP](../data/deliveries/Q1_F0_PROPRIO_TRAINING_EVIDENCE_20261108.zip)550450B/SHA `6510e5a2ec00d30d6410eac7c75c218bace5a795da41db5d2151788f3bb3389a`，23成员CRC/逐文件原值及22相对链接通过。此训练包无新增视频。[交付审计](../logs/session_records/q1_f0_proprio_training_delivery_20261108.json)。

![causal_history训练六宫图](../data/analysis/Q1_F0_PROPRIO_FIGURES_20261108/causal_history_six_panel.png)

![zero_history训练六宫图](../data/analysis/Q1_F0_PROPRIO_FIGURES_20261108/zero_history_six_panel.png)

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

已完成独立实现、原运行时及来源资源准入、两组真实训练与训练六宫图/CSV/原始收据交付。后续置零组CPU身份核同并复用旧失败，非零组七例物理和GPU原速媒体全部完成，仍0/7完整；两例前缀延长、五例缩短。完整[新证据包](../data/deliveries/Q1_F0_PROPRIO_REVIEW_EVIDENCE_20261108.zip)包含两组训练图与七例闭环图，不据loss称自然遥操成立。下一实际学生状态/序列目标方案先独立审计，不继续重复单步标签拟合。

软件耗时、人体参考等待和硬件端到端延迟分别记录；本方案f0不使用尚未发生的人体帧，但仍不等于整机零延迟。最新发布、部署和服务器小记录镜像以独立publication收据为准，大包/全局索引旧镜像PENDING不混称完成。

## 原运行时与准入

源码提交`0549ea64677b4fbf2102c813c3d4b387014deec5`和后续不可变清单`e2e3cedab7558582eca5ad768c1475b3241a0f5c`均显式push origin Q1，并独立核对40位远端提交。十成员18512B源包沿既有45014会话五片精确ACK部署；1015原保护来源前后未变，只新增清单外文件。

真实Morph运行时六检查全过，本体1.681秒/进程约4秒；1031来源、六原数组/原R2保存点、原暂停及资源锁准入12.320935秒通过。总内存8196124672B/限34292629504B、OOM0、GPU1MiB25度、余盘183602642944B，GPU计算与原锁空闲。三原始收据5802B已Mac二进制大小SHA核同：[部署](../logs/session_records/q1_f0_proprio_deployed_20261108.json)、[原运行时](../logs/session_records/q1_f0_proprio_contracts_20261108.json)、[准入](../logs/session_records/q1_f0_proprio_admitted_20261108.json)。

上述为训练前准入现场。后续唯一训练已完成，原输出与收据保留，不再派发同训练。[训练六宫图入口](../script/supermimic_teleop/deliver_student_proprio.py)已仅读本轮真实保存数值生成两组图表；所有已完成物理、512配对和六检查不重复。
