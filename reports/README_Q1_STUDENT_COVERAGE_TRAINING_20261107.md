# Q1 因果学生扩大覆盖实验

记录日期：2026-11-07。实验：COVERAGE01。训练和后续闭环已完成：30新片、9585物理步、3000真实Adam，575秒成本已结算；验证选500。11月8日七例真实闭环0/7完整且均早于旧f0失败，详见[闭环结果与证据](README_Q1_COVERAGE_CLOSED_LOOP_20261108.md)。训练验证改善不能当控制改善。

## 给用户的简报

done：Agent自行固定source/ssh入口85082进入原Linux，确认上次已实际启动并于23:02:14完成。30新片的9585步及3000Adam全成，六计数各3000、source PASSED、575秒已结算；累计2181/14400秒且无未结算预留。选500的扩展验证动作误差下降36.689%，两个不变旧验证集分别下降40.069%和34.396%。原始12成员80172B包已Mac SHA核同，训练六宫图PNG/PDF/CSV已生成并目视。原共享保持人工暂停。

to do：仅对已选500的新候选做独立三名义片加四事件闭环、GPU原速媒体和完整软件计时；复用十四H4对照及原f0七例失败，不重做采集训练或原对照。验证改善尚不能当闭环完整或自然遥操达标，不能用last挑test。

## 依据与变量

[INPUT_AUDIT01](README_Q1_STUDENT_INPUT_COVERAGE_20261107.md#实际审计结果)已核训练仅12/22来源组和12/167片，启动200ms样本稀少，局部来源截断明显。扩大覆盖是可检验假设，不预设它就是失败原因。此前增加辅助下一状态、教师latent和FP32物理loss均未胜原R2，不延长相同数据拟合。

采用已发布的固定哈希候选，train22组36片、validation6组12片；原18片前缀严格复用，仅30新片。新片最多384步且保留最早因果可用的偶数帧。物理中提前终止如实保存，不把前缀称全片完成。来源分组及原test划分不变，不读test人体数组。

采集行为为有十未来帧的冻结32k教师，属于特权离线采集，不是因果遥操。标签必须与同一实际状态的执行动作一致；学生输入仍是可见当前人体加十槽历史，120→50Hz插值右支持和包到达时钟纳入可见性检查。新采集不混入学生动作，也不重跑旧混合状态采集。

冻结原R2归一化、教师、decoder和WM，复制原R2已选编码器，只做动作蒸馏。训练按来源组、片段、可用状态原点三级均衡。因同时扩来源和启动覆盖，本轮不能单独区分这两项的因果贡献。

## 固定训练和选点

seed2026110704，Adam学习率0.0001、batch128、最多3000更新、梯度范数上限1，每500更新验证且包含0。连续六次没有合格改善则停。记录六个Adam真实计数、每来源/原点曝光、有限CPU selected/last保存与复读，教师和decoder前后SHA不变。

扩展验证风险按来源组→片段→可用状态原点均衡计算；另外保持原R1验证集和原混合验证集为两个固定锚点，逐点单独报告。只在两锚点均不高于update0时，才按扩展验证风险选更优点。不能把扩展集loss直接与历史不同集合的loss作改善比较，也不使用test或未选last挑模型。

## 实现与验证

独立模块：`student_coverage_training.py`、`student_coverage_worker.py`、`student_coverage_data.py`、`student_coverage_fit.py`和`deliver_student_coverage.py`。计划：[student_coverage_experiment_20261107.json](../script/supermimic_teleop/student_coverage_experiment_20261107.json)，候选：[原覆盖清单](../data/manifests/q1_student_coverage_candidate_20261107.json)。原184执行来源和200冻结来源不修改。

本地九项检查通过：旧两原点精确连接且未来标签数组不进学生、未知来源/test/跨组拒绝、三级均衡、物理状态/动作对应和首个可用启动样本、未来依赖/捕获拒绝、三次夹具Adam冻结统计/decoder/teacher/CPU保存和锚点选点，另三项原候选元数据合同。首次测试缺少测试文件自己的import已修正，真实服务器CUDA未触发。

## 资源与执行

固定独立根：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_COVERAGE01_20261107`。策略和拟合CUDA，物理继续共同Primebot CPU；无媒体重渲染或旧物理重跑。原非阻塞GPU锁、22GiB准入/24GiB退出、GPU18GiB/80度、磁盘100GiB和private500GB边界保留。

控制器最多1740秒、原阶段1800秒和输出1GiB，采集预留120秒给拟合/保存，训练预留30秒退出；同一累计账本R3记账。入场前完整核原身份、原暂停和五PID退出、账本及全部来源，退出后核来源、回收与结算。没有新的缓存清理。现场内存8048500736/34292629504B，GPU1MiB/25度/OOM0、磁盘184009240576B；这是准备时快照，不替代正式准入。

调用独立入口沿用原`train_preflight`参数，并增加`--initial-r2-root`、`--aggregate-root`、`--input-audit-root`、`--coverage-plan`及其SHA、`--candidate-plan`；先核发布清单和原运行时九项，再唯一新根派发。SSH由Agent按用户固定source/ssh入口自行尝试，不请求用户代测。

## 结果与恢复

源码8432242d412bcbe50e52d2bf6be94363aab93c2a、清单8991ca35c84ded05d1c1af1e2ff226e4594a1109均显式push origin Q1，独立远端完整40位相同；Mac15文件863217B与Git逐字节相同。193项清单35639B/SHA`d9fc8182b5b5a648b7fed837e2f0007500734993d7228226f958ab7b0140cec8`，12成员代码包42163B/SHA`589e21d7d9dbebc3ca9cda149eeebc40b935733cff29f6cf23c5367d026e8b70`已准备。

53893旧会话超时后，Agent自主按固定source/ssh入口新55892已进Windows，再原生第二跳密码成功到原Linux，并确认`/opt/data/private/ZHY`。提交部署接收器后尚无READY/部署回执即该SSH明确Timeout退出255，代码包正文发送0。独立一次54567仍原样固定入口，仅第一跳Operation timed out，未到Windows认证或Linux。这个新重连失败不等于Linux不可达；没有请求用户代测。详见[发布与连接实录](../logs/session_records/q1_student_coverage_publication_and_connection_20261107.json)。

下次自主沿固定入口恢复，先查`q1_student_coverage_deployed_20261107.json`、12目标大小SHA、接收器实际进程及唯一COVERAGE01输出/派发/日志。接收器无正文不可当已部署；若旧接收器仍在，先核其确切命令/启动tick和退出状态，不重复启动接收器。不因SSH退出推断后台状态，也不重派未知任务。补缺部署后以固定原运行时做九检查，再完整`coverage_preflight`和原锁/账本准入，保存回执；实际训练Popen尚未提交，任何待发命令不得当已开跑。服务器镜像和原运行时均未完成。

上述无回执部署状态已由下面的真实服务器回执覆盖；当前真实新物理/训练仍0，只有准入完成情况待核。若选点仍0，须CPU核实际编码器及策略元数据身份，再复用原0/7历史结果；不能为产生视频重跑同策略。若选出不同策略，后续在独立来源/资源门下做三名义和四事件闭环、原速视频、脚底换脚/滑移/侧倾/上肢时序/动作突变及完整软件时钟。六宫图是编码器训练记录，不编造RL回报或500iteration滑动均值。

## 服务器部署与原运行时已完成

北京时间2026-11-07 22:30:30，Agent经固定`source /Users/yu/.zshrc`、`ssh -tt hp_han`及原生第二跳交互密码，在88029收到原Linux现场。原root/boot/GPU同，原五训练PID退、guard仍STOPPED_SIGNAL，原锁与计算空；总8027848704B/限34292629504B、OOM0、GPU1MiB/25度、磁盘184008663040B。COVERAGE01收据、包、输出、派发、日志和接收器均不存在，才继续部署；没有让用户代测。

22:32:16收到真实`COVERAGE_CODE_DEPLOYED`：接收器加300秒外层保护，先READY再传正文；42163B包及12成员精确SHA通过，新增11文件、既有相同input_audit复用，779保护来源前后同。22:33:20收到九项原Primebot运行时CPU检查全部通过，测试1.722秒/整个进程4.708503秒。检查包装器首次因重复环境关键字在子进程启动前拒绝，已修正环境组装，未改执行源码或放宽测试。

随后提交独立`coverage_preflight`及临时原锁/预算准入，外层300秒/kill-after5；此命令不预留成本、不创建训练输出、没有训练Popen。该SSH收到明确Timeout255前未返回准入回执，所以只把准入记为UNKNOWN。唯一有界自主恢复97733仍执行用户固定入口，第一跳banner exchange超时，未Windows认证或Linux；这不能证明Linux不可达。92124/88029/97733均已结束，不再探旧句柄或向用户要求代测。

下一周期自行沿固定入口恢复，先查服务器`logs/session_records/q1_student_coverage_admitted_20261107.json`及精确preflight进程，再核原暂停/资源/锁/账本与唯一输出/dispatch/log。已成功准入则复用原回执；仍在运行则等待，确已退出且无成功收据才补足。部署和九检查不重做。三个服务器原收据和本阶段说明尚未传回/镜像；本地[观察记录](../logs/session_records/q1_student_coverage_server_deployed_observed_20261107.json)来自已收到的SSH输出，不冒充原始文件大小SHA同步。实际训练命令从未提交。

## 准入回执恢复并精确传回

22:44:18，Agent自行固定入口新38409进入原Linux。上次准入实际在22:34:17完成，耗时13.339043秒，`COVERAGE_SOURCE_AND_RESOURCE_ADMISSION_PASSED`；193执行来源、200冻结来源、736控制保护输入以及原GPU锁和累计预算均通过。准入总8391303168B/限34292629504B、OOM0、GPU1MiB/25度，原锁获取后释放；账本1606秒、R3已485秒、无未结算预留。直接复用成功回执，未重复部署、九检查或准入。

恢复时原五PID退/STOPPED_SIGNAL，GPU计算空/原身份同，实际准入进程已退，唯一新输出/派发/日志仍不存在；总8027820032B、磁盘184006594560B。说明上次SSH观察超时不等于准入未执行或服务器停止。

[部署](../logs/session_records/q1_student_coverage_deployed_20261107.json)、[九检查](../logs/session_records/q1_student_coverage_native_checks_20261107.json)、[准入](../logs/session_records/q1_student_coverage_admitted_20261107.json)三份原始文件9230B已通过同一原生SSH取回，逐文件大小SHA均同。四成员原包3035B/SHA`bbe43b3de13588441c6556a92e91ceaf1e469aabceac4173dd7e4f86611f8caa`。实际训练Popen仍未提交；先发布本准入再单派原登记命令，继续完整源门/自有进程回收/成本结算/训练六宫图，不因已有准入称已开跑。后续实际状态以派发及结果回执为准。

## 派发提交与未回执恢复点

准入已以`238938688b53735984b5c70f9972e464375c9080`显式push origin Q1并独立核远端完整40位，Mac八文件781359B与Git核同。38409最后一次服务器时钟回执为22:48:18。随后向同一原Linux提交派发脚本：严格读取真实准入参数，核原暂停、当前资源/原锁/账本和唯一输出/dispatch/log不存在，再以DEVNULL输入、独占日志和新session Popen启动原登记命令，并保存PID/starttick派发收据。

该SSH在PID或派发回执返回前明确Timeout255，因此只记“派发脚本已提交、Popen执行和实际训练UNKNOWN”，不记已启动或确定未启动。独立唯一有界恢复2864仍为用户固定source/ssh入口，第一跳连接超时，未Windows认证或Linux；两句柄均退。当前训练、物理步数和预算预留须重新核实际账本，不沿用派发前0或全SETTLED当当前值。

下一周期Agent自行恢复后先读`logs/session_records/q1_student_coverage_dispatch_20261107.json`、`logs/q1_student_coverage_20261107.log`及新根result/fit.guard/resources和相关PID/starttick，再核原身份/暂停/锁/OOM/磁盘及同一累计账本。若已下发则仅观察同任务；若已终态则保留结果并核源门、回收与结算；只有收据、输出、日志和实际进程全部明确不存在时才允许补派。部署、九检查和已成功准入不重复。详见[精确恢复记录](../logs/session_records/q1_student_coverage_dispatch_unacknowledged_20261107.json)。服务器最新项目说明镜像仍待补，实际任务未核不生成训练图或结果。

原共享训练人工暂停、原final门、用户采纳null保持；本轮不调用goal工具、不新agent、不改其他自动任务。阶段提交/远端40位和Mac/服务器大小SHA核验以后续publication为准。此前两份输入审计publication收据随本次实质实现归档，不为收据单独循环提交。

## 真实完成结果

2026-11-07 23:03:25，Agent自主执行固定第一跳及原生第二跳交互密码，85082返回原root/boot/GPU。此前未知派发实际已于22:52:12启动360043/start37605132；23:02:14完成并退出，360115已reaped/return0。不是再次训练或补派。原五训练PID均退，STOPPED_SIGNAL不变；原GPU锁及计算空，OOM0，总8146440192/限34292629504B，磁盘183504244736B。

全部30新片达到各自登记前缀长度，总9585步。新增train3865行/24片、validation910行/6片，首200ms分别104/26行；物理状态、标签动作及重采样可见性检查均过，教师显式打包最大差3.5762786865234375e-7。特权H10教师用于采集，不能把采集行为称因果遥操；学生输入仍f0加历史。合并后train6725行/36片/22来源组，validation2358行/12片/6组。

真实3000次Adam、六计数各3000，拟合45.376263秒。扩展验证update0为0.08454722753522126，选500为0.053527909478476736，下降36.6887466%；末3000为0.057445158782414796，未选末轮。两个旧验证集不变，0.08218013929824035→0.049251144751906395、0.10754757064084212→0.07055584558596213，分别下降40.0692854%、34.3956863%。归一化、teacher、decoder及WM不变；这是覆盖与启动样本共同变化的结果，不能单独归因为其中一项。

成本575秒SETTLED，累计2181/14400秒、R3为1060/7200秒，无RESERVED。峰总12087549952B/显存478MiB/OOM0、输出约89.5MB，训练信号0、无缓存清理。193执行、200冻结、736控制输入原结束源门通过。

原始包[data/deliveries/Q1_STUDENT_COVERAGE_RAW_20261107.tgz](../data/deliveries/Q1_STUDENT_COVERAGE_RAW_20261107.tgz)为80172B/SHA`ced82c96d72b85a3d58b24306dee79e9b1855769328a4f08f06001babcdeb11c`，12成员385247原字节。十一原值已解包与包内大小SHA全同；动态账本原字节保留在包内，未覆盖Mac旧账本。新[训练六宫图](../data/diagnostics/q1_student_coverage_training_20261107/training_six_panel.png)、PDF、七点CSV已生成并目视。首次系统Python无matplotlib，在生成图前退出，保留该事实；随后使用已有q1-teleop-mps环境生成，无安装。没有新推理物理或训练用于绘图。

下一独立闭环只用selected500，不使用last；前走、起走、停走及四事件共七新例，保留旧H4和原f0结果。还没有本候选的闭环结果、真机人体数据或硬件端到端延迟，故自然遥操仍未验收。当前说明服务器镜像待后续publication。
