# X2 R2 初始化协议隔离与独立补测

2026-11-14。状态 DISPATCH_ACKNOWLEDGED_RESULT_UNVERIFIED。仅修复自有测试入口的启动传输，保持原模型、30Hz人体物理、50Hz策略请求、20ms实际门及180ms预览；不改旧控制器、R2四执行源码、作者代码、终端2/3或系统服务。

## 当前恢复点

新入口源码已发布 `fa667901bc53b7581baced44a2d42dcbf8448929`，显式push origin Q1并独立核到同一40位远端。五项新板端合同、原35依赖、247上游源码/24资产及失败根保护门通过后，唯一补测已收到派发回执：外层PID35179/starttick247845，监测PID35181。执行根`data/analysis/X2_JOINT_ALLOC_20261113/joint_r2_startupfix`，同步监测根`monitor_startupfix`，原100秒控制器/120秒外层和900录制帧/2显式graph初始化上限保持。

设备原始派发收据`logs/session_records/x2_joint_r2_startupfix_dispatch_20261114.json`、准入`x2_joint_startup_admitted_20261114.json`及严格协议日志`x2_joint_r2_startupfix_protocol_20261114.log`尚未回Mac。此处PID来自之前实际收到的回执，是恢复定位信息，不是当前存活证明。

23:53北京时间续行时，90475返回第二跳Connection aborted及第一跳Connection reset/Broken pipe后退出255；不能据此认定补测失败或机器人停机。唯一原生恢复`ssh sshuser@100.92.227.62`在第一跳TCP连接阶段Operation timed out/255，未到密码提示，未尝试第二跳；未修改网络或认证。当前补测终态、真实帧数、角色内核掩码、来源终门及自有进程/锁回收均未核验，不能标为仍在跑或已完成。详见[续行记录](../logs/session_records/x2_joint_startup_dispatch_observation_20261114.json)。

恢复后先取原收据与现存输出/日志/进程，禁止重派：核身份boot、原PID/starttick及monitor，再读run_receipt、fgp_summary、affinity、native/status、三份kernel_binding和startup_config；比较终前/终后247+24、原35依赖、新五源码和失败根SHA。只有实际完成及来源/回收/全部角色掩码门通过，才用已保存工件做独立analysis_startupfix及20/25/30/33.333ms事后分析。R1完整日志的多门槛统计也只读一次，不重跑模型。准备的`script/x2_joint_alloc/deliver.py`只通过CLI检查，真实R2绘图尚未执行；它要求完成数值、相同历史基线、实际掩码审查及同boot已退出监测，拒绝拿不完整结果出提速图。

## 实际前序与失败证据

原生Windows→run桥接已恢复，新boot115b4d3f-aaff-47dc-8f1f-5ac42d20a106。背景20秒CPU平均25.05%、GPU中位11%最大22%、最高64.968度、最低可用11.64GiB；原始监测包已Mac SHA核同，见资源监测README。

R2六项板端合同、35个项目依赖SHA及原247上游源码/24资产通过，原R1已清洁结束。唯一prefix_r2实际7.556145秒完成、64输出及64GPU帧、worker退出0，GPU/CPU/发送真实TID23872/23918/23919、核2,3/4,5/2,3均过；与原串行保存输出的2e-5门通过，不能当实际联合性能。

唯一joint_r2在4.653688秒初始化失败，保留FAILED根：read_ready抛JSONDecodeError，两个子进程25872/25879返回0/-2，尚无native/status或fgp_outputs。FGP错误栈停在TrtSession.deserialize_cuda_engine，尚未进入CudaGraph的两个显式初始化enqueue和录制循环；本次可审计录制GPU帧0、显式graph初始化0，不等于没有CUDA初始化活动。两监测器均TARGET_EXITED/自有tegrastats回收，原247/24及历史证据终后JSON与终前逐位同。

原read_ready没有保存导致JSON错误的首行，不能确认它的具体内容。源码只做Python redirect_stdout，未隔离原生库的fd1输出；原生诊断混入协议是待核假设。新入口同时解决隔离与原始报文留存，继续严格解析一个JSON对象，不跳过未知行或放宽就绪门。

## 独立实现与本地验证

新增script/x2_joint_startup：把原生fd1诊断路由到已有stderr日志；独立dup的协议流保留FGP显式__stdout__及policy原Python就绪报文。异常时恢复文件描述符；不改模型推理和数值。启动记录原样保存后再严格JSON解析，拒绝非对象、空、超长或不完整行。

另外记录每次原bind调用后实际内核读回的角色/TID/核掩码。native仅允许原observe命令，在原taskset之后记录实际0,1核再exec原二进制；原ROS/SDK环境和命令参数保留，不允许armed。

五本地合同0.059秒及CLI通过，包含原生os.write/C printf与Python诊断隔离、policy协议、异常恢复、限定worker/observe命令、错误JSON原样保存后拒绝。均为标准库合成子进程，不加载模型或GPU，不当板端真实修复结果。

## 复用与独立预算

保留原joint_r2失败根，唯一新joint_r2_startupfix及独立协议日志。必须新源码发布/部署SHA和五板端合同、原35依赖及失败原件/已成prefix/247+24来源同、资源/锁准入后再派发。

复用真实R2 64输出门：四原执行源码、权重、算法、affinity计划均保持逐字节同；新增改变只有输出通道和启动审计，不是改推理后绕过前缀门。新入口再验证原allocation prefix，记录其parity与失败收据SHA。前一失败尚未进入录制，因此补测仍最多900录制GPU帧/2graph初始化；加原64/2，原总964帧/4初始化上限不扩。若此轮再失败，保留结果并据新原始报文诊断，不能循环重派。

保持100秒原控制器/120秒外层、4GiB可用/80度/10GiB磁盘、原非阻塞锁、自有进程回收和0电机/MC/BLE/训练。同步监测只跟踪实际PID/starttick子树，GPU是整机利用率，不归因到单进程。

## 下一步与边界

真实完成后检查三worker退出、全部角色核掩码、源前后同、监测回收，再只读分析吞吐/间隔、全部ready RPC、20/25/30/33.333ms事后门和源年龄，生成六宫图。原串行与R1来自旧boot，比较必须标历史对照及背景负载变化，不将短测称稳定30Hz或硬件端到端/真人质量验收。

原Q1暂停、用户采纳null保持。新入口已派发，终态与来源终门尚未核验；设备派发原件、R2数值、六宫图和本次文档设备镜像待补。不能据当前连接退出重派或宣称提速，后续以真实收据为准。
