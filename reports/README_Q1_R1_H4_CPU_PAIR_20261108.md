# R1 H4的CPU转换：保存状态对照

2026-11-08，北京时间。H4-only权重验证已经完成且没有一致质量收益，本阶段保持原R1 H4权重，只检查独立CPU转换对动作与完整软件周期的影响。

## done / todo

done：新增独立控制器、worker和六项聚焦合同，六检查本地全过/0.038秒，CLI和编译通过。原R1 H4名义前走与丢包的完整保存轨迹、时钟、selected5000及32k冻结教师作为输入；每条件从frame0连续128帧。

todo：先发布代码/清单并核远端与Mac，再通过现有45014部署、原运行时六检查和来源资源准入。实际新CUDA、物理、媒体和Adam目前均为0。真实执行只单派唯一CPU_PAIR01根，不能把准备结果当作动作或时限通过。

## 有界设计

四条件顺序固定：名义原版、名义加速、丢包加速、丢包原版，各128次请求，共512次上限。四个worker串行，每次仅一套策略使用原GPU锁；相反顺序用于减小固定先后偏差，但不是同时基线、随机重复实验或硬件端到端计时。

唯一变量为`q1_native_review.native_state`在新worker单次请求内的绑定。原R1的StudentReviewSession.action、人体窗口/接收时钟、历史、归一化、shared decoder和选定权重均复用；退出或异常恢复函数引用，不编辑原worker或已执行H4/CPU来源。独立KinematicState此前已过234状态输入与64次f0动作门，本轮针对不同的R1 H4策略及事件，不重做旧f0检查。

每帧先与实际保存动作按atol=rtol=2e-5比较，缺失canonical槽须精确零。原版和加速版各自的事件可见性须过原validate_timing，再核used_sequences/ready/fallback/依赖支持/预览长度与原时钟一致；末尾再逐帧比较两者动作。全部128帧和原始首两次保留，另报去前两次以及capture时钟50–60区间，不能通过截取稳定片段隐藏长尾。

保留原同步目标scope、原worker scope、新请求分派在内worker与本机pipe RPC。嵌套区间不相加，不据单项CPU速度外推完整链路；没有物理回放，不能据此声称滑移、侧倾或闭环质量改善。H4仍请求80ms未来预览，未减少其未来帧；原始人体预处理offline_unverified，丢包为合成事件。

禁止物理积分；新worker对mj_step/step1/step2全程拒绝，请求期间另拒绝forward/inverse，仅使用已审计运动学原语。最多512次CUDA前向、0新物理/渲染/训练。总300秒/阶段300秒/输出64MiB，22GiB准入/24GiB退出、GPU18GiB80度、100GiB余盘、private<500GB、原非阻塞锁及自有子树回收保持，不清缓存。

## 来源、成本与恢复

先核H4保存分析真实PASSED、七例完整、24比较、summary原SHA `f2e1e7e95784f4304740d2c7e15227cfacff7e69a0c0d4063887de97fa1e6eee`和全部1009保存来源。原R1 RESUME2中的名义/丢包H4 case、trace和stream必须精确属于该来源；不能误用新H4专用权重或旧失败f0轨迹。

账本原累计3403/14400秒全SETTLED，r1已1460/2700、剩1240秒。本阶段最多300秒，在持原锁且GPU空闲后reserve，四guard确认清洁回收后settle，不重置原预算；无法确认回收就保留RESERVED。原共享五PID保持退出/STOPPED_SIGNAL，采纳null，goal工具和新agent不用。

唯一输出：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_R1_H4_CPU_PAIR01_20261108`。日志`logs/q1_r1_h4_cpu_pair_20261108.log`；dispatch为`logs/session_records/q1_r1_h4_cpu_pair_dispatch_20261108.json`。派发前同时查output/log/receipt/精确进程，未收回执先查同任务，不重派。

现有45014原Linux仍可用；必要新第一跳固定原样source /Users/yu/.zshrc再ssh -tt hp_han，原生第二跳密码，获批SSH三值90/60/5保持。Agent自主尝试，不要求用户代测。H4结果小镜像五文件28230B已精确ACK/SHA同，旧大包/全索引镜像仍PENDING。

本阶段实现、来源清单与后续真实派发以对应小记录和publication为准，准备不构成时限通过。下一根据完整软件结果判断是否仍需处理调度/传输长尾，同时继续序列/闭环目标主线；不延长已无一致改善的H4-only拟合。

## 原运行时和准入

源码`8af31885325333f217b47d7df1617d8e38e3ac6f`及208项清单提交`4cd8ff7387e47a830b013e486b0b4f09976ebab5`已显式push origin Q1，均独立40位远端核同；原203源码保持，新增五项包括三新文件和两个既有转换依赖。25611B七成员包/SHA `f78353274c97c7f33bfee24ca9bac844b2c4dd6a040b2c4785c1961b08a6493b`经45014九个精确ACK部署，七目标大小SHA同，1009保护来源前后相同。

北京时间05:06:19，原Morph策略运行时六项检查全过，本体0.047秒、进程0.433299秒。05:06:32真实准入13.115925秒通过，1015保护来源、原R1名义/丢包轨迹与32k审计、冻结训练暂停、原锁空闲和3403秒账本均通过；新GPU调用仍0。总8201555968B/限34292629504B、OOM0、GPU1MiB24度、余盘183604289536B。三份原始回执5089B已本机逐件大小SHA核同。

[部署原件](../logs/session_records/q1_r1_h4_cpu_pair_deployed_20261108.json) / [原运行时检查](../logs/session_records/q1_r1_h4_cpu_pair_contracts_20261108.json) / [真实准入](../logs/session_records/q1_r1_h4_cpu_pair_admitted_20261108.json)。发布本准入后先核唯一output/log/dispatch/精确进程全无，再单派300秒实验；当前没有RESERVED或后台策略任务，不能把准入当完成。
