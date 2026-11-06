# Q1事件缺少六例补验与保存计时定位

日期：2026-11-07。最新阶段：独立缺六例补验、保存计时定位和84例合并比较均已完成；12张六宫图和六原速视频已Mac核同。02:10:52独立JOINED84_ANALYSIS_PASSED/84完整，原FAILED不改，原训练保持暂停。[完整结果与证据包](README_Q1_LOW_LATENCY_EVENTS_JOINED84_20261107.md)。下方准备与未回执部署记录保留为历史，不能再当当前未完成状态。

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

## 源码发布与部署包

准备源码`5875596f1b37076fee8f2340ba0e064ff2998c3b`已显式push origin Q1并独立40位远端核同，9份673059B已与Mac主根逐份大小SHA匹配。原221执行文件本地逐份SHA同，新四文件均在原清单外。新[223项代码清单](../data/manifests/q1_event_missing6_code_20261107.json)只在旧221基础增加缺六例控制器及其检查，指向该已发布源码提交，41686B/SHA `e17c17444f46fc3a11a2ca3175fdba7b28605bbc7631d0d09a117aaac4813b99`。

独立五成员代码包20082B/SHA `6c576620dcea58a7c3ee7c3b27148c017af4d32fcb9f4bf4b5e4e8d016872aaf`，包含四新Python文件与新清单；每成员普通文件、相对路径、大小SHA均已与工作副本匹配，不含认证或模型。服务器部署、运行时五检查与实际计时/物理尚待，不能据此称任务已运行。[发布与包收据](../logs/session_records/q1_event_missing6_source_publication_20261107.json)。

## 已部署与运行时检查

清单发布`2bc14b48eef03a7869e9b184fd9467b16dc9a9b7`已push/独立远端核同，3份49828B已Mac镜像匹配。01:17:38经原生密码会话89657仅新增五个目标文件，先确认不属于原执行或冻结来源，再核每成员及目标大小SHA。原运行时新增3+2项检查全部通过/0跳过/分别0.001秒；部署前后原221执行、200冻结、4控制和46名义证据SHA同。没有重复旧动作门、旧分析或物理。[服务器部署回执](../logs/session_records/q1_event_missing6_code_deployed_20261107.json)2492B传回内容与SHA `f19c323770ad1be5939d183ff3f3e410d484847cf75b4e48b4a5437fe7784611`核同。

## 六例实际完成

01:21:28一次派发330705/start29860648，准入223执行/200冻结/721复用工件核同，精确复用78并只排六例；GPU1MiB25C，总7772119040B/限34292629504B/OOM0，原锁探测空。01:21:29.486890开始，01:30:13.094936全部完成，523.608046秒低1800秒；12个守卫阶段均完成，六物理和六NVIDIA EGL原速媒体完整，新增任务source_integrity PASSED。原R2的FAILED状态及SHA不变，合并84只读比较仍待，不据六例单阶段结束结算探索。

01:33:00原设备/唯一根和旧五PID退出/人工暂停核同，330705已退出，末worker已回收0，原锁和GPU计算空；GPU即时1MiB25C。资源记录峰10795425792B/GPU400MiB，最后总7784710144B/限34292629504B/OOM0；训练信号0/回收未尝试。[真实派发](../logs/session_records/q1_event_missing6_dispatch_20261107.json) / [完成快照](../logs/session_records/q1_event_missing6_completed_20261107.json)。

观察89657在后续只读取回时peer reset255结束；没有重派六例。此次独立观察故障唯一有界恢复2703经原Windows与原生25362逐跳立即交互密码成功到原Linux，并返回上述完成状态。旧会话退出不概括为Linux不可达，认证未入记录。

## 保存计时的真实定位

独立只读定位1.526679秒通过，原78份计时SHA和先前统计核同，无新推理、物理或训练。去首调用41106次，中位7.658ms/p95 12.829ms，336次>20ms；78次>100ms全部在每段frame1（第二次调用），最大565.169166ms。再去掉第二次调用仍258次>20ms、最大49.823569ms，不能把预热当作实时性已解决。

| 统一capture子集 | 调用数 | >20ms | >100ms | 最大ms |
| --- | ---: | ---: | ---: | ---: |
| 事件之前 | 3405 | 104 | 78 | 565.169 |
| 事件窗口 | 780 | 4 | 0 | 24.503 |
| 事件之后 | 36921 | 228 | 0 | 49.824 |
| fallback | 291 | 1 | 0 | 24.704 |
| 恢复前后两帧 | 180 | 0 | 0 | 18.180 |

前三行互斥且合计全部调用，后两行是可重叠的描述子集。这个样本不支持把>100ms尖峰归为合成事件或恢复本身；尚未分解Python调度、目标构建、传输与GPU前向，也不能给真机延迟或预热收益承诺。[完整定位记录](../logs/session_records/q1_event_timing_review_completed_20261107.json)。

Mac独立交付根`data/deliveries/Q1_EVENT_R2_TIMING_REVIEW_20261107`的summary71850B/SHA `fd5433c7552e7e5820b3e6f18376e5d624bb96062f6d000790e785d7eb688a72`及超20ms逐调用CSV35620B/SHA `3885d07325c0aade56521259eaa7dffeeb2b69c53a413954fb51bd2df3656738`已沿原密码桥接按二进制base64传回，含原CSV换行的完整字节核同，没有改写旧图表ZIP。六代表视频仍服务器待补，新文档镜像仍待补。

## 独立84例合并准备

新增`script/q1_checkpoint_review/event_joined_summary.py`及对应检查，仅在新的独立根读取保存78例与六例补验。固定原FAILED结果及部分分析SHA，要求补验COMPLETED/source_integrity PASSED、全部12守卫已正常回收，并重核721复用工件、223执行来源、200冻结及控制输入；六份新增trace、事件可见性、目标一致性与原速视频身份重新读取。精确84例不重叠，所有组均需三片完整才给全组均值。

Mac新增3项检查全部通过/0跳过/0.003秒。入口尚未部署或实际执行，原84 campaign完成门不改；成功时只认证独立保存工件合并分析，不把原FAILED改为COMPLETED。新输出预定`Q1_LOW_LATENCY_CKPT32000_EVENTS_JOINED84_ANALYSIS_20261107`，内外各300秒保护，不申请GPU、不调用策略或积分、不生成新媒体。[准备记录](../logs/session_records/q1_event_joined84_preparation_20261107.json)。

## 发布与未回执部署恢复点

合并入口与六例实际结果已发布`332bc47e01ff8aa63a1e05adfacaad52a225c424`，显式push origin Q1/独立40位远端核同，11份708899B与Mac主根大小SHA一致。两新源码包4736B/SHA `aa66d47ad34a76bd424b55a6861941e6f2530ac956df11c37f44b5318d614c3f`仅含合并入口及三项检查，均在223/200清单外。

向2703提交的部署命令未收回执，随后该客户端报告100.65.183.91 peer reset/Broken pipe255，故部署及服务器新增检查均UNKNOWN，不能当已成功或确定未执行。唯一有界新恢复20344等待90秒后明确在Mac到Windows原22端口TCP超时，未Windows shell、未尝试Linux；不把旧会话退出或这次一跳失败说成Linux已停。最后完整现场仍01:33，六例已完成的事实与本地收据不受后续观察失败影响。

恢复先查`logs/session_records/q1_event_joined84_deployed_20261107.json`与两目标SHA，未查之前不重复部署。合并分析尚未下发，先核新输出、日志、收据与进程，再一次独立300秒只读分析。原FAILED与所有已完成78+6例保持，不再发物理任务。新增本地`deliver_event_joined.py`复用原绘图器，已AST解析，待真实84例数值后生成新的12PNG/12PDF/总览/表格ZIP；尚未运行、没有虚构图表。六代表视频仍服务器待补，新文档服务器镜像待补。[发布、连接证据与恢复顺序](../logs/session_records/q1_event_joined84_deployment_unacknowledged_20261107.json)。

源码静态边界：`event_gpu_worker.py`在状态转换、obs和历史更新之后才开始`perf_counter_ns`，所记耗时包括事件目标构建、pack、数据到GPU、策略前向与结果回CPU同步；不包括全部CPU预处理、RPC及物理积分。因此这里不能把软件目标调用称为完整推理链或硬件端到端。下一性能实验应在独立入口分段测量并保留原计时，先核相同动作输出，区分第二调用初始化与稳定阶段49.824ms长尾；尚未执行新的剖析推理。

q1只更新prompt为六例实际完成、保存计时已定位、2703已退出/20344原一跳超时及合并恢复顺序。工具回执ACTIVE，独立文件核heartbeat/十分钟/同聊天不变，保存SHA `21a665672f9c9c1446da1cd23c4170d11ffa9c5605fb7f789a13f56d05adbf8d`；其他自动任务、goal、agent与原训练未动。[续行记录](../logs/session_records/q1_missing6_timing_continuation_20261107.json)。
