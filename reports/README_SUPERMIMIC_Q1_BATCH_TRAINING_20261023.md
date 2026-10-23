# Q1修正摩擦后的批量训练对照

记录日期：2026-10-23，北京时间。阶段：SM-Q1-R02。状态：页缓存修正后的独立新登记已启动，A臂运行中；首轮停止终态保留。

## 目标与范围

用户明确指示“开始训练”。执行S01已保存方案：修正足地接触摩擦后，从随机初始化依次训练actor batch32/128；每臂16环境、3000采集轮，即48000新训练控制转移。唯一组间设置改变是actor batch，保持世界模型k8、actor H32、WM64×10、论文观测、seed20261023、原奖励和初始学习率2e-4、PD50/2及模型25%力矩。走路/浅蹲训练，出拳是已复用开发片段。

不修改R01冻结核心。新产物使用独立R02目录；A3停止安排和旧实验终态保持。本阶段只做仿真训练及评测，没有机器人操作。

## 完成内容与关键决定

新增`script/supermimic_for_q1/run_batch_pair.py`，发布源码清单逐文件绑定Git内容和SHA，再在服务器登记输入/模型/运行时/boot/资源。一次性执行记录拒绝覆盖，文件锁阻止重复成对运行；不自动重启。每15秒监测GPU86摄氏度、可用显存2GiB、容器30GiB和OOM计数；每阶段核对冻结文件和包版本。子进程退出/收据/审计异常即停止，保留原生退出码、stdout和进程身份。

顺序为A训练、保存审计、step0及final各自三片完整动力学评测/独立轨迹审计/分肢位移统计/奖励拆分，然后B执行相同流程，最后检验组间配置并生成比较。效果门未通过保留为科学结果，不阻断B；执行或审计失败会停止。未扩展训练预算。

## 版本、命令与证据

服务器：`huanyu:/opt/data/private/YuanQi_Q1`。Python：`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。Q1代码工作副本：`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。先发布并独立核对远端提交，再生成release清单和部署；实际提交与运行编号在启动后补记。

```bash
# 本地Q1副本，发布后生成小型清单；该命令不训练。
python3 script/supermimic_for_q1/run_batch_pair.py release --root "$PWD" --output logs/session_records/supermimic_q1_R02_release_20261023.json
# 服务器，先登记再启动；已有目录/started记录时不得重跑。
$PY script/supermimic_for_q1/run_batch_pair.py register --root "$PWD" --release logs/session_records/supermimic_q1_R02_release_20261023.json --output data/supermimic_for_q1/R02_batch_pair_20261023
$PY -u script/supermimic_for_q1/run_batch_pair.py execute --plan data/supermimic_for_q1/R02_batch_pair_20261023/plan.json
```

实际启动采用脱离SSH会话的进程，stdout在`logs/supermimic_q1_R02_batch_pair_20261023/controller.log`；各子阶段独立日志在同目录。训练权重、评测数组和计划在`data/supermimic_for_q1/R02_batch_pair_20261023/`，工件`LOCAL_ONLY`。

## 验证与边界

控制器5项检查实际通过：原生退出7留证并停止后续臂、异常收据拒绝、效果失败不误作执行失败、信号阻止后续阶段、输出拒绝覆盖；全部0物理/optimizer。命令`python3 -m unittest discover -s script/supermimic_for_q1 -p test_batch_pair.py -v`。Git diff检查通过，R01的train/evaluate/学习核心与cc3ecb6差异为零。真实训练进度以随后检查点为准。P07和B01证据支持启动，但不证明完整训练峰值或跟随效果；0.15rad门保持。比较另报水平根RMSE、腿部RMSE、实际/参考位移及完整率，不能按总奖励上升认定成功。

## 保存与恢复

运行时查看`plan.json`、`started.json`、`status.json`、`events.jsonl`和活动臂的`status.json`。用boot/PID/start_ticks核对实际进程后再判断是否仍运行，不以旧PID自动恢复。若失败，读取controller和最后阶段日志并记录原因；不重用现有输出目录。顺序任务结束后读取`receipt.json`和`comparison.json`，再决定独立长预算试验。

本阶段README与索引/当前状态随启动证据更新，先审查diff和payload audit，再显式推送Q1并核对远端。启动准备、首轮停止和缓存修正均已先保存再进入下一阶段；当前运行证据见末节，记录发布状态以实际Git核验为准。

## 首轮实际启动与保护停止

发布版本`873c13408caaf2d8533641e2f3ae2e414bf7daa0`已推送并独立核对远端Q1一致。服务器逐文件核对部署SHA，冻结计划45401字节，SHA`0bef71276121958d6f91004234e0954d1cd464983c1c94aff7af2e0e1c9f972a`；小副本`data/manifests/supermimic_q1_R02_plan_20261023.json`，release清单`logs/session_records/supermimic_q1_R02_release_20261023.json`。

北京时间19:29:56启动R02A，controller PID751224/start_ticks3604700222、训练PID751227/start_ticks3604700252、boot仍d994e71f-bae0-46ba-a5f7-6c2163771c6c。19:30:28控制器确认终止：监测到容器32225853440字节，超过30GiB保护线32212254720字节；GPU41摄氏度，可用17205MiB，OOM仍14。控制器发SIGTERM，训练保存final并原生退出0，收据为`STOPPED_SIGNAL`，不是完成。累计140轮/2240新交互、5组WM更新、0actor更新、训练22.0679秒。第二臂和评测均未启动，没有自动重启。

首轮终态证据`data/manifests/supermimic_q1_R02_pair_stopped_20261023.json`与`supermimic_q1_R02A_stopped_20261023.json`，原始日志/工件保留原R02目录，不能用保存的final当学成策略。step0为68432619字节/SHA`0edd78ab9297131c0d493a641fa1b8ffdb81146224b26beb78d5fd5c7d2169ca`，final138272469字节/SHA`ca7bdcfa01d625f74a8a03b15976604946877a48c9077a3ed0ed467a25d3791f`，LOCAL_ONLY。

停止后读取容器内存约28184068096字节，其中rss15405191168、cache11918667776（含shmem6202675200）。存在六个约3GiB RSS的其他Python进程，本阶段不停止它们。R01入场26671681536字节，本轮驱动入场28921434112字节；GPU资源不能替代容器内存余量。额外一次只构造/释放预检asset的0积分/0optimizer诊断：进程RSS529140→1174092→1051416KiB，说明构造与内存分配存在显著CPU开销，但尚不能单独归因为唯一根因。

下一步先检查本项目历史权重文件的可回收页缓存，不改权重字节、30GiB门或旧停止终态。若有可核验修复，使用独立新登记，不恢复旧R02；资源不足时保持停止并明确所缺内存。

## 页缓存修正与独立新登记准备

首轮停止记录已提交`1e32e00`。一次仅对本项目10份已有PT发出`POSIX_FADV_DONTNEED`，包括R01的8份与已停R02的2份；不删除/改写文件，不清全局cache，不操作其他进程。容器观察值28176887808→26465144832字节，下降1711742976字节；这是共享容器的前后观察值，不当成独占内存测量或磁盘释放量。

控制器新增有界页缓存处理：在子阶段之前、每15秒资源检查之前和完成之后，仅建议释放项目目录内已完成`.pt`的读缓存；排除symlink与写入中的`.tmp`，保留所有模型字节。原驱动每25轮和控制器每15秒的30GiB/86摄氏度/OOM保护不变。此为I/O资源修正，不改变网络/采样/奖励/优化器。

修正后6项控制器测试实际通过，新增检查只触碰项目内完成PT、排除外部symlink/其他文件且前后字节一致。没有新物理/optimizer测试；实际训练继续由新登记后的运行验证。

使用独立新目录`data/supermimic_for_q1/R02_cachefixed_pair_20261023/`登记同一预算的两臂，从零初始化，不读取旧停止权重作为初始化；原`R02_batch_pair_20261023`的FAILED/STOPPED_SIGNAL账本不复用、不解锁。新目录中的两臂仍名为R02A_batch32/R02B_batch128。新发布清单使用`supermimic_q1_R02_cachefixed_release_20261023.json`，实际登记与启动证据见下节。没有循环重试或自动恢复机制。

## 新登记的运行检查点

冻结代码`6b43cc0fb4e42482a570fe74e08f19ec3c43c9c0`已推送并独立核对Q1远端；部署全源码SHA通过。新计划45902字节，SHA`e27a1a45d02293a26c60ee8034d2539c6c74a0956e32f07f9db40b6fb878da9f`，Git小副本`data/manifests/supermimic_q1_R02_cachefixed_plan_20261023.json`。新controller PID751815/start_ticks3604769075、A训练PID751818/start_ticks3604769102，同boot，均由脱离SSH会话的独立session运行。

新A臂已生成step0/admission，实际协议`Q1_FOOT_PRIORITY_V2`、batch32、16环境、3000轮。350轮时5600交互/89组WM/39次actor更新，当前WM/actor/critic损失和梯度指标有限，GPU42摄氏度、容器30983843840字节、OOM仍14。启动核验另存`launch_verification.json`，重新核对源码/输入、boot/PID/start_ticks及命令匹配；不把在线奖励当完整策略评测成绩。

北京时间19:47:39独立核验实际通过：训练快照375轮/6000交互/99组WM/49次actor、326.004秒，所有已报告数值有限；controller与child的boot/PID/start_ticks及命令匹配，冻结源码/输入逐文件一致。收据4517字节，SHA`ba777cdf1db99031ff3fb32a1e49c14eb7c2c0f1c542bba8310ea0bd2f939841`，Git小副本`data/manifests/supermimic_q1_R02_cachefixed_launch_20261023.json`。当时实时容器30998523904字节、GPU42摄氏度/空闲16981MiB，OOM仍14；该操作本身不做物理/optimizer更新，不代表训练已完成。源码冻结提交与后续仅记录提交分开保留。

新运行入口为`$PY -u script/supermimic_for_q1/run_batch_pair.py execute --plan data/supermimic_for_q1/R02_cachefixed_pair_20261023/plan.json`，已执行，不重复运行。实际使用`subprocess.Popen(..., stdin=DEVNULL, stdout=controller.log, stderr=STDOUT, start_new_session=True)`；控制器不依赖Mac/SSH会话。主日志`logs/supermimic_q1_R02_cachefixed_pair_20261023/controller.log`，A日志同目录`R02A_batch32_train.log`；源代码保持冻结，后续只更新记录。

从新目录的`status.json`/`R02A_batch32/status.json`查看最新进度，按`started.json`中的boot/PID/start_ticks核对；不要用此静态快照当实时证明。预计完成1149组WM/1099次actor，之后保存审计和step0/final三片评测，再B；失败即停止，未授权无限追加预算。训练、仿真评测和真机结果分开，当前没有新最终策略效果。

## 20:29中途检查点

本轮继续目标时先以实时进程核对运行，未重复启动：controller/child的boot、PID、start_ticks与原登记完全一致，均为非僵尸进程；实际轮数从先前675继续增加。北京时间20:29:07的新独立收据`midpoint_verification.json`为4437字节/SHA`0ef828aeab679e1e9460682a7d45682541ccb20c933b3cc43b3b209fcde3e7c2`，小副本`data/manifests/supermimic_q1_R02A_midpoint_20261023.json`。

当时A训练1625轮/26000新交互、599组WM/549次actor、2847.178秒；全部已记录标量有限，冻结源码/输入逐文件核验通过。step1500已实际保存，205230055字节，SHA`5099f0e8c828a866705a732202c0b2d938bbf1850bb8fc74a3f7e6b9f3e1f2a4`。核验仅统计已有日志、进程和文件SHA，0新物理/策略推理/optimizer，不载入检查点张量。

GPU40–41摄氏度、实时可用16981MiB；容器30995947520字节低于30GiB保护，OOM仍14。无训练或pair终态收据，B尚未启动，不能记为完成。继续等待该活跃训练句柄结束，由原控制器执行保存审计/独立整片评测；不另跑中期动力学或变更训练设置。成对结果完成后才根据腿部/水平根跟踪、位移和完整率登记独立长预算或方案调整，现有pair不会自动扩展预算。

本轮web工具未返回可用原文内容；复核使用服务器已保存的原论文PDF（SHA仍为`fe213017ab062f4acc3965da89abd3c29969163e144a2628a662b8ce46691009`）第13–16页，确认Algorithm A.1/Table A.5的采集、k8模型拟合、H32冻结模型内策略梯度、GAE critic与0.4更新比例。该静态复核不证明Q1学习成功或达到原文训练规模，当前小批量和数据差异仍见S01记录。
