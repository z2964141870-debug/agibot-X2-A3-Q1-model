# R1 H4的CPU转换：保存状态对照

2026-11-08，北京时间。H4-only权重验证已经完成且没有一致质量收益，本阶段保持原R1 H4权重，只检查独立CPU转换对动作与完整软件周期的影响。

## done / todo

done：四个条件各128次保存状态请求已经完成，`CPU_H4_PAIRED_ACTIONS_PASSED/source_integrity PASSED`；实际492次模型前向、20次明确回退，四守卫清洁回收，0新物理/渲染/Adam。原R1 selected5000和32k教师不变，只替换请求内CPU状态转换。六宫图、PDF、CSV与原始数值已经本机生成并核验。

todo：保存并发布本阶段结果，后续转向序列一致性/闭环目标及未来帧依赖；不延长没有一致质量收益的H4-only拟合，也不重复本轮512请求。当前H4仍需要80ms预览，完整软件长尾和真机端到端尚未达标。

## 真实结果

北京时间2026-11-08 05:09:25单派，timeout378150、controller378151/start39868374均已退出；四个guard均`WORKER_COMPLETED/reaped/return0`。05:16:21完成核验：原五训练PID仍退出/训练`STOPPED_SIGNAL`，原GPU锁释放、计算进程空，1015来源终门通过。名义原版/加速各128次前向；丢包加速/原版各118次前向和10次回退。请求数不等同于CUDA前向数。

原版精确复现保存动作；加速版名义/丢包最大动作差分别`1.013279e-6/1.102686e-6`，原atol=rtol=2e-5通过，缺失槽精确零、接收可见性和原时钟同。此结果仅证明这些保存前缀上的动作保持，不能外推为新闭环质量改善。

下表均排除每个worker的最初两次调用，仍含全部126个后续请求和明确回退；全部128调用及初始化另在六宫图、原始JSONL和CSV中保留。完整worker包含独立请求分派，RPC另含本机pipe交互，嵌套计时不相加。

| 条件 | 完整worker p50/p95/max(ms) | worker >20ms | RPC p50/p95/max(ms) | RPC >20ms |
| --- | --- | --- | --- | --- |
| 名义原版 | 16.347 / 21.200 / 34.810 | 8/126 | 17.086 / 22.418 / 87.320 | 9/126 |
| 名义加速 | 10.406 / 21.500 / 62.895 | 10/126 | 11.270 / 25.292 / 91.455 | 13/126 |
| 丢包加速 | 10.859 / 15.034 / 18.850 | 0/126 | 11.789 / 17.556 / 59.818 | 2/126 |
| 丢包原版 | 16.439 / 32.479 / 57.536 | 25/126 | 17.309 / 33.903 / 100.832 | 27/126 |

CPU替换使两类完整worker中位数约下降36.3%和33.9%，但名义p95和超20ms次数没有改善，不能称稳定满足50Hz。全部调用的完整worker最大1461.9–1560.0ms，RPC最大1598.1ms；第二调用仍约494–584ms，不能通过去首两次统计声称初始化已解决。丢包窗口的持有/回退与正常前向分开计数，不能把少算前向作为预测成功。

ABBA串行只作同保存状态描述性比较；不是随机重复或同时基线，不能把各次长尾变化归因于一个确定的CUDA/调度原因。实际人体输入预处理仍`offline_unverified`，丢包为合成事件，80ms是请求的未来预览等待，不是测得的硬件端到端延迟。

本阶段154秒已SETTLED，累计3557/14400秒；r1=1614/2700、r2=572、r3=1371，无RESERVED，累计剩10843秒。峰总10650349568B、GPU460MiB，训练信号0/未回收缓存；核验总8181190656B/限34292629504B/OOM0、GPU1MiB25度、余盘183603040256B。资源和计费只按实际记录，不重置预算。

[完成原件](../logs/session_records/q1_r1_h4_cpu_pair_completed_20261108.json) / [派发原件](../logs/session_records/q1_r1_h4_cpu_pair_dispatch_20261108.json) / [六宫图](../data/analysis/Q1_R1_H4_CPU_PAIR_FIGURES_20261108/r1_h4_cpu_pair.png) / [PDF](../data/analysis/Q1_R1_H4_CPU_PAIR_FIGURES_20261108/r1_h4_cpu_pair.pdf) / [CSV](../data/analysis/Q1_R1_H4_CPU_PAIR_FIGURES_20261108/r1_h4_cpu_pair.csv) / [证据README](../data/deliveries/Q1_R1_H4_CPU_PAIR_EVIDENCE_20261108/README.md)。本阶段没有新视频，因为没有物理回放；H4七例原速媒体见上一阶段证据包，余六视频仍服务器。

[证据ZIP](../data/deliveries/Q1_R1_H4_CPU_PAIR_EVIDENCE_20261108.zip)为463701B，SHA256 `3824e2ac7b210a1f247988e498e97b60747c64d1abbd5d8d1ac500e7c4ddb917`，26成员CRC与逐件大小SHA、11个README相对链接均过，六宫图已目视。原始数值包169798B/SHA `c5d02ec26f4a9c90b001a67bc2916690eab6e5a2dc2cba637e65fb8c12bfa0ec`经原45014十九块精确ACK传回，无重算策略；[交付核验](../logs/session_records/q1_r1_h4_cpu_pair_delivery_20261108.json)。

以下保留原有设计与准入历史；其中“尚未派发”描述已由上述真实结果覆盖。原三份执行源码与208清单不再修改，原共享训练继续暂停，采纳null；本机与Git/服务器同步范围以最新publication为准，旧大包/全索引镜像仍PENDING。

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
