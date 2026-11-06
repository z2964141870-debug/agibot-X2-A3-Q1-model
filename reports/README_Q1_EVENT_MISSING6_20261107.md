# Q1事件缺少六例补验与保存计时定位

日期：2026-11-07。阶段：独立方案和本地实现已就绪，尚未部署、分析或派发新物理。

## 目标与边界

原事件R2在7200秒总时限退出，78/84例完整。保留原FAILED根与全部记录，独立补late_burst下damped_cv H5/H4各前走、起走、停走六例。复用已通过的21组动作门、32k保存审计和78例证据；不重跑原84、不恢复用户暂停的共享训练。补验完整也不自动通过原160k final或自然遥操验收。

新入口为`script/supermimic_teleop/event_missing_six.py`，只调用未变的`event_review.py --child replay/media`。CUDA策略、共同Primebot CPU物理、NVIDIA EGL原速媒体和事件几何/时钟不变。新增控制器核原FAILED与部分分析精确SHA、原进程退出、21组动作门和全部复用来源。精确78+6分区有重复、缺片或已完成项被改排时拒绝。

新独立输出：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_EVENTS_MISSING6_20261107`。新预算总1800秒、单阶段1800秒、输出1GiB；在原守卫外增加更严格总时限/存储检查，原22GiB准入/24GiB退出、GPU18GiB80C、100GiB磁盘余量和原非阻塞锁/自有子树清理保持。中途失败保留，不自动重试或扩时。结束后仍需独立完整84工件比较，不能把六例任务完成写成原R2完成。

## 保存计时定位

独立`script/q1_checkpoint_review/event_timing_review.py`仅读已完成部分分析和原78份stream_timing。逐份与已保存SHA及既有样本/超20ms/最大值核同；按统一capture时钟分事件前、事件窗口、事件后，另保留一秒窗、fallback与恢复前后两帧的分母及重叠关系。所有统计都去首调用，首调用单独记录。300秒上限，只新增JSON与超20ms逐调用CSV，不加载策略或积分。

这些记录只有整个目标调用的同步墙钟时间，无法据此分开Python调度、目标构建、传输与GPU前向。相关性不作为事件导致长尾或GPU算力不足的因果结论；合成时钟不当真机端到端延迟。

## 已验证

Mac仅新增5检查全过、0跳过：精确缺六例/78复用、拒绝重复或换事件、拒绝重标原失败、capture时钟与恢复边界、超时阈值与分母。原未变45、7、5检查及动作/物理没有重复。

01:11:28原会话89657成功返回原根/boot/GPU、旧训练与实验PID全退、STOPPED_SIGNAL、锁空/计算空；内存7751073792B/限34292629504B/OOM0，GPU1MiB26C，磁盘余187631718400B。两个新输出均不存在，原FAILED结果SHA不变。此为本次准入现场，不借旧传输会话失败泛化桥接不可达。

[方案与现场收据](../logs/session_records/q1_event_missing6_preparation_20261107.json)。代码、README、索引和当前状态先按范围审计、commit/push origin Q1、独立40位远端及Mac大小SHA核同，再生成匹配新提交的代码清单，沿既有原生密码会话部署、仅做新增运行时检查与资源重核后单次派发。此时服务器新增训练仍0/4GPU小时未用，应用goal清除/用户采纳null，无新agent或其他任务变化。
