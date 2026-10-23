# Q1修正摩擦后的批量训练对照

记录日期：2026-10-23，北京时间。阶段：SM-Q1-R02。状态：首轮已启动后触发容器内存保护并停止；第二臂未启动，正在诊断资源余量。

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

本阶段README与索引/当前状态随启动证据更新，先审查diff和payload audit，再显式推送Q1并核对远端。新控制器准备记录保存后再执行登记和训练；此处尚未宣称已同步或策略改善。

## 首轮实际启动与保护停止

发布版本`873c13408caaf2d8533641e2f3ae2e414bf7daa0`已推送并独立核对远端Q1一致。服务器逐文件核对部署SHA，冻结计划45401字节，SHA`0bef71276121958d6f91004234e0954d1cd464983c1c94aff7af2e0e1c9f972a`；小副本`data/manifests/supermimic_q1_R02_plan_20261023.json`，release清单`logs/session_records/supermimic_q1_R02_release_20261023.json`。

北京时间19:29:56启动R02A，controller PID751224/start_ticks3604700222、训练PID751227/start_ticks3604700252、boot仍d994e71f-bae0-46ba-a5f7-6c2163771c6c。19:30:28控制器确认终止：监测到容器32225853440字节，超过30GiB保护线32212254720字节；GPU41摄氏度，可用17205MiB，OOM仍14。控制器发SIGTERM，训练保存final并原生退出0，收据为`STOPPED_SIGNAL`，不是完成。累计140轮/2240新交互、5组WM更新、0actor更新、训练22.0679秒。第二臂和评测均未启动，没有自动重启。

首轮终态证据`data/manifests/supermimic_q1_R02_pair_stopped_20261023.json`与`supermimic_q1_R02A_stopped_20261023.json`，原始日志/工件保留原R02目录，不能用保存的final当学成策略。step0为68432619字节/SHA`0edd78ab9297131c0d493a641fa1b8ffdb81146224b26beb78d5fd5c7d2169ca`，final138272469字节/SHA`ca7bdcfa01d625f74a8a03b15976604946877a48c9077a3ed0ed467a25d3791f`，LOCAL_ONLY。

停止后读取容器内存约28184068096字节，其中rss15405191168、cache11918667776（含shmem6202675200）。存在六个约3GiB RSS的其他Python进程，本阶段不停止它们。R01入场26671681536字节，本轮驱动入场28921434112字节；GPU资源不能替代容器内存余量。额外一次只构造/释放预检asset的0积分/0optimizer诊断：进程RSS529140→1174092→1051416KiB，说明构造与内存分配存在显著CPU开销，但尚不能单独归因为唯一根因。

下一步先检查本项目历史权重文件的可回收页缓存，不改权重字节、30GiB门或旧停止终态。若有可核验修复，使用独立新登记，不恢复旧R02；资源不足时保持停止并明确所缺内存。
