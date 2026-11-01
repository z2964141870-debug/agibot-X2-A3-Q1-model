# MorphMimic共享通信适配准备

2026-11-01。MORPHMIMIC_FAITHFUL01后续单卡unified通信契约准备及CPU验证完成：9项本地边界检查、三rank各12类真实Gloo检查通过，共72次微型网络Adam更新，与集中式均值损失oracle的最大参数差3.725290298461914e-9。原生Trainer、CUDA张量和三本体训练仍未执行，不能当unified训练完成。原G1未发送停止、恢复或重启命令，执行来源71859ca8abe238477aa18245a31d81204da9197f保持。

14:13:38至14:13:53北京时间，独立CPU V02正常COMPLETED，三个子进程均退出0/无残留/未创建GPU上下文；四份固定上游源码SHA前后相同。5小工件12135B Mac/25386逐大小SHA一致，首次导入失败根和完整trace继续保留。14:16:22核原G1两PID启动tick/boot/GPU根未变，69381/160000轮/五optimizer持续更新/非有限0/OOM0/GPU8779MiB，训练始终继续。[实测结果](../data/manifests/morphmimic_faithful_collective_results_20261101.json)。

13:47:55北京时间最新：用户在询问服务器后再次明确要求ssh hp_han，已另记人工新授权尝试，20962成功经DESKTOP-3C1DPPR进入原25386。原boot/GPU/唯一根与两个原PID启动tick一致，66750/160000轮、五optimizer持续更新、非有限0/OOM0/GPU8779MiB。服务器和原训练均在，下面UNKNOWN为恢复前历史；自动恢复1次计数未重置。现在可先完成本准备发布及小记录三端核验，再执行真实CPU验证，不改现训。

## 目标和实现

固定MorphMimic aa40fe46a7fa995ba7a7354ad065a2c8ad64a302的四个通信模块，准备单卡三本体共用cuda:0时的同步CPU传输层。原版一机器人一rank、NCCL、cuda:local_rank不能直接当作单卡三本体可运行证据。

新增script/morphmimic_cross_robot/collective_transport.py，仅在明确的distributed、multi_robot、normalizers、trainer模块内临时替换dist别名，退出或异常时还原；不修改torch.distributed全局对象，不修改上游文件。仅接受Gloo和同步broadcast、SUM/MIN/MAX all_reduce、all_gather、all_gather_object，拒绝未审计操作、异步请求和非CPU/CUDA稠密张量。张量先复制到CPU，原Gloo执行集合操作，再写回原存储。

保留上游每次小批更新的梯度平均、裁剪、Adam顺序，以及各机器人形态/限位/normalizer等buffer。policy输入normalizer按原pool_norms配置池化。latent仍以同stem、最短帧长、相同迭代种子配对，目标为detach后的跨机器人latent均值；不把三个独立训练模型当共享模型。CPU和GPU集合运算的浮点归约顺序可能不同，未承诺逐位等价。

## 已验证与未验证

本机不含Torch，未为本步骤安装新环境。test_collective_guards.py的9项标准库测试检查：已初始化Gloo条件、未审计接口、非支持存储、空/重复/异源模块集、别名退出还原及全局对象不变、固定源码哈希、哈希以外的调用表拒绝、允许调用表、既有Git导入环境的复用。该测试使用显式替身对象，只验证拒绝和生命周期，不验证通信数值、autograd或分布式运行；实际数值由另列服务器CPU测试验证。

三个脚本py_compile通过。四份本地上游快照的SHA256均与既定值相同，AST核其直接dist调用只包含审计接口。检查点、权重和NPZ均未加载。

verify_collectives_cpu.py真实三进程CPU Gloo测试已完成：原distributed/multi_robot/normalizers配合微型替身网络，每rank12类检查、24次替身Adam更新，三rank共72次。检查包括五条优化路径逐次平均后clip-Adam及moment与集中式均值损失对照、clip顺序反例、独立buffer、池化统计含空rank、原latent梯度和共同帧采样、非连续张量/对象状态、别名正常和异常退出还原、单rank零对齐。每rank记录broadcast11/all_reduce29/all_gather3/object1；staged_input_bytes1908仅输入张量计数，不是总网络字节或峰值内存。

CPU通过只证明小型数据的通信契约，仍不证明原生Trainer、CUDA张量、三机器人24GiB同时驻留或端到端训练等价。微型fixture使用普通范数裁剪，不是原生policy/value裁剪和完整损失执行。未来正式共享入口和资源profile均未启动。

## 运行方式和保护

14:03:49至14:03:59首次真实CPU验证在morphmimic包导入时因GitPython找不到Git退出，尚未init_process_group，三rank替身Adam均0；三个测试子进程已回收，原G1继续到14:06:27的68453轮/非有限0。[完整失败收据](../logs/session_records/morphmimic_faithful_gloo_cpu_import_failure_20261101.json)保留。修复仅在测试进程复用现有supervise.py所用data/environments/morphmimic_git_tools/usr/bin/git及MUJOCO_GL设置，启动前检查可执行性，父环境传给spawn子进程；未安装Git、未改原G1运行环境或上游。新增反例后共9项本地边界测试通过。修复发布并核三端后仅一次新CPU根MORPHMIMIC_FAITHFUL01_GLOO_CPU_V02_20261101，原失败根保留，不循环重试或新增物理训练。

```sh
python3 -m unittest discover -s script/morphmimic_cross_robot -p test_collective_guards.py -v
python3 -m py_compile script/morphmimic_cross_robot/collective_transport.py script/morphmimic_cross_robot/verify_collectives_cpu.py script/morphmimic_cross_robot/test_collective_guards.py
```

服务器真实CPU V02已在修复519c35d24911d1046f21899769f600a16d6d405b独立40位远端及6文件60508B Mac/25386大小SHA核验后执行；实际命令如下，已完成的根不重跑。

```sh
CUDA_VISIBLE_DEVICES='' data/environments/morphmimic_original_20261029/bin/python script/morphmimic_cross_robot/verify_collectives_cpu.py --upstream /opt/data/private/ZHY/script/third_party/morphmimic_aa40fe46 --output /opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_FAITHFUL01_GLOO_CPU_V02_20261101 > logs/morphmimic_collectives_cpu_v02_20261101.log 2>&1
```

限定原boot/GPU/唯一根，cgroup准入<=24GiB，CPU测试仅对子进程在>28GiB或180秒时停止并回收，磁盘保留100GiB；各进程1线程/nice10，CUDA隐藏，不创建仿真或原生Trainer。中断同样回收测试子进程，失败输出根保留，不自动循环重试；不会终止原G1训练。

## 连接与训练状态

本轮既有原生观察会话15595返回Windows100.65.183.91超时/退出255。05:17:45Z先记录独立故障，再按已授权方式source ~/.zshrc、ssh -tt hp_han，仅一次新会话67932仍在Windows22端口连接超时，未进入容器。随后一次nc -vz -G8 100.65.183.91 22成功，仅证明TCP可达，不能证明SSH认证或25386、训练状态。未创建转发或控制socket，未探测其他服务器入口，未再登录同一故障。

恢复前一度实时训练状态UNKNOWN；交接11:38:11的54826轮和10:50:12已发布50376轮均为历史。用户再次明确登录后20962已核原controller362416/start38298223、worker362421/start38298276、boot/GPU根，14:16:22达69381轮。连接超时没有被当作训练停止，不曾自行重启。步态是否改善仍待官方物理评测，采纳null。

应用get_goal本轮实际返回blocked；继续授权有效，本轮继续可执行的本地准备，不创建替代goal、不结算、不暂停或伪报应用ACTIVE。原监督及其他自动任务、训练资源守卫保持。

## 保存与恢复

[准备清单](../data/manifests/morphmimic_faithful_collective_preparation_20261101.json)登记源码SHA、测试边界、独立故障一次计数和待办；操作性恢复状态保留logs/q1_connection_recovery_state.json，不纳入Git认证或主机配置。

准备e2c1044和导入修复519c35d已独立远端及Mac/25386大小SHA；真实CPU结果5小工件12135B亦已Mac/25386逐SHA。最新结果说明和索引的发布看morphmimic_faithful_collective_results_publication_20261101.json，不预称尚未核验的发布完成。远端共同559条BVH、权重和NPZ继续SERVER_LOCAL_ONLY。后续G1按160000轮继续，GPU释放后串行做官方评估、X2/T2重定向和原生单卡共享资源验证，不并行开启GPU任务、不重复CPU验证或原提取。
