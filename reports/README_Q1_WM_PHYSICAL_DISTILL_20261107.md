# Q1当前与历史学生：冻结FP32世界模型物理蒸馏

2026-11-07北京时间19:57:38，独立物理蒸馏真实完成，3000次Adam、来源门和回收通过。验证仍选update0，20:02 CPU逐张量审计确认与原R2相同；本次没有产生更好的因果策略。原共享训练继续暂停，采纳null。

## 用户版done与todo

done：固定FP32世界模型的物理损失实验完成。末validation动作风险0.110133，较初始0.094864退16.10%；比严格同曝光action-only末值低4.03%，仍未超过初始策略。物理validation MSE也升16.05%，不能把训练loss下降称为遥操改善。

done：3000更新/六Adam各3000、3119次WM.step，拟合113.654740秒、持锁成本143秒已SETTLED；累计1606/14400秒，r3已485/7200秒。Mac原值逐件SHA同，训练六宫PNG/PDF和七点CSV已生成并目视。选中8个encoder张量与原R2最大差0，仅复用历史0/7失败，新闭环和新视频0。

todo：停止延长本配方或继续叠加辅助loss。先核查已保存因果输入、按来源的训练/验证覆盖和失败时段，并比较短预览H4可保留的信息；再选择单项数据/表示改进。软件完整周期另查CPU状态转换和目标处理，不把单独GPU前向当端到端延迟。下一真实实验仍须独立方案、来源/资源门及同预算控制。

## 实际结果

| 指标 | update0 | update3000 |
| --- | ---: | ---: |
| validation动作风险，两origin后各source等权 | 0.094863855 | 0.110133066 |
| 原action-only控制动作风险 | 0.094863855 | 0.114754134 |
| validation物理特征MSE | 2.789744531 | 3.237624357 |
| 原教师状态validation动作风险 | 0.082180139 | 0.104542360 |
| 混合状态validation动作风险 | 0.107547571 | 0.115723773 |

七个验证点的动作风险均未优于update0。最末训练batch的action MSE为0.000875504、physical MSE为0.016286185；它们不是来源等权的validation指标，不直接求比值或混作同分布平均。物理损失相对严格匹配控制仅减轻本轮验证退步，尚无采纳收益；单次数据与权重0.1的负结果不否定完整GigaBrain，也不证明未来信息是失败的唯一原因。

派发控制器358968/start36541060与worker359040均已退出，守卫WORKER_COMPLETED/return0、worker_reaped，source_integrity PASSED。原184执行/200冻结/773保护来源由控制器在退出前核同。批次SHA为`862a2d90afcf5d9eec560de409bd5f445a7f8b7d0ca25690b64db33eb711a6a5`，与原PAIR02每个验证点及24个origin/source各16000曝光精确匹配；没有重训控制或采集新物理。

选中策略CPU审计比较全部8项策略元数据、教师来源及8个encoder张量，均精确相同，CUDA未初始化。新selected.pt的外层方法/训练元数据不同，文件SHA不同不表示部署策略变化。仅复用原R2七个失败例的历史证据，本轮不新跑相同策略，也不以未选中的last权重挑test。[原始身份收据](../logs/session_records/q1_wm_physical_selected_identity_20261107.json)。

峰cgroup10581987328B/GPU518MiB/OOM0。20:02现场总8323018752B/实际限34292629504B，GPU1MiB26度、计算进程和原锁记录空；五旧训练PID均退出、原guard仍STOPPED_SIGNAL。成本143秒已结算，首轮1606/14400秒、剩12794秒，r3已485/7200秒。没有回收缓存或向原训练发信号。[完成现场](../logs/session_records/q1_wm_physical_completed_20261107.json)。

新[训练六宫图](../data/analysis/Q1_WM_PHYSICAL01_FIGURES_20261107/wm_physical_training_six_panel.png)、[PDF](../data/analysis/Q1_WM_PHYSICAL01_FIGURES_20261107/wm_physical_training_six_panel.pdf)、[七点CSV](../data/analysis/Q1_WM_PHYSICAL01_FIGURES_20261107/training_points.csv)已经生成。原始包80182B/SHA `23913f6da5c59cf407186d1a153c5fac389ab9a1758801675f025ab9461e9547`，16成员及15份原始记录大小SHA全同。证据包包含这次训练图和明确标注复用的两段R2失败原速视频；不是新策略视频。[完整证据ZIP](../data/deliveries/Q1_WM_PHYSICAL01_EVIDENCE_20261107.zip)，实际包身份见随后figures_delivered收据。

CPU身份审计首条一次性命令因字符串路径未转换Path退出，修正后完成；未改训练代码或结果。Mac归档初次使用本地Python3.9不支持的extractall(filter=...)，只建立空输出目录；按已核验的纯普通文件清单提取后通过，真实训练与来源门均未受影响。原始日志和训练来源保持。

## 假设与边界

本项目独立假设：冻结一步世界模型在同状态下比较教师与学生动作的物理结果，可以改变动作蒸馏的优化几何。它不提供缺少的未来信息，不是GigaBrain损失的忠实复现，也未证明此前f0倒地由BF16造成。[精度结果](README_Q1_WM_PRECISION_20261107.md#真实完成结果)只认证保存样本上的数值一致性。

冻结原32k teacher、部署共享decoder、WM全部权重与buffers、原R2当前目标统计。独立同权重影子仅关闭BF16，TF32关闭。学生仅f0/history，部署不增加WM或辅助头。

只加`0.1 * physical_loss`：在相同保存实际状态和名义context下，教师目标为WM(state, saved_teacher_action)的detach物理特征；候选为WM(state, student_action)。二者都使用同一个detach当前状态坐标系、原std和feature_presence。末29命令缓存严格排除，缺失关节输入及导数贡献为零。不能把不同混合行为动作产生的日志下一状态当作学生或教师动作的反事实标签。

## 公平控制与来源

复用完成PAIR02的action-only，不重训控制。原R1与AGGREGATE01 train/validation、终止mask、clip/frame、f0/history/teacher_action原行保持，初始化encoder SHA、全部验证点batch SHA与24源曝光须精确相同，初始动作风险容差2e-7。只用validation动作风险选点，含update0；不读test或新增采样。

原控制设置：3000次Adam、batch128、lr0.0001、seed2026110703、clip1、每500次验证。最终每个origin:clip训练曝光16000，共384000条抽样；六个Adam计数均须3000。原始source与fit/trace/provenance按大小SHA核同，运行时另外核实际encoder初始SHA、冻结teacher/decoder/WM前后SHA。

## 独立实现

四个新执行文件`wm_physical_training.py`、`wm_physical_worker.py`、`wm_physical_fit.py`、`wm_physical_loss.py`，不改原178执行或200冻结来源。控制器沿原非阻塞GPU锁/guard/持久成本账本，只计入r3；worker复用原控制的转移校验和归一化，按source/frame精确连接保存实际状态，0新物理。

五项聚焦CPU检查已全过/0.675秒，包括：草案学习率错配拒绝、命令缓存与缺失关节排除、冻结与detach门、保存行连接/无未来输入变化，以及随机CPU夹具3次真实Adam与历史控制批次/曝光/选点一致。夹具不是32k GPU训练。新训练六宫图入口已实现，实际生成待真实记录。

唯一新训练根：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_WM_PHYSICAL01_20261107`。启动前核output/log/dispatch/receipt/实际进程，不能因观察无回显重复派发。

最多3000反向/4096次WM.step、1740秒拟合、1800秒阶段保护、1GiB输出。22GiB准入/24GiB退出、GPU18GiB80度、100GiB磁盘余量、private500GB及原设备/锁/自有子树回收保持；R2缓存边界复用，不再次advice/drop或删文件。累计1463/14400秒、r3已342/7200秒继续，不重置。真实训练前须发布源码/新清单、原运行时检查、原身份暂停/资源及来源准入。

## 后续验收

若选中策略发生实质变化，另行有界闭环复核名义与stop/reverse/dropout/late_burst，比较完整率、脚底离地/换脚/支撑滑移、侧倾、上肢时序、动作突变和恢复。验证动作MSE、WM物理MSE或loss下降都不代替自然遥操或硬件端到端验收；相同选中策略不重跑历史失败。

本阶段发布/部署/实际准入和结果分别以收据为准；不调用goal工具、不新agent。33391原Linux自主复用，断开后按用户原样source及ssh入口自行尝试，不询问用户代测连接。

## 真实部署与准入

源码907683e9c447beceee27b58a406d44d15164095c与184项清单7941bdc6b43dd507d723e3ac5125cf653a9b1952已push并独立40位远端核同，Mac分别14份811350B和两份36795B大小SHA同。八成员代码包26155B/SHA `7a7b8028e83399406fcc1421bfbe9cd032cd914977c402f4fb8acb4cc6ad1cc1`经33391部署，仅新增清单外源码/计划/新清单；757份原保护来源前后同。

原运行时五项检查全部通过，测试本体1.790秒、进程4.732982秒，原PyTorch标量转换提醒保留，实际32k GPU训练0。外层准入命令曾因字符串引用语法错误、随后参数手抄成65位SHA被原门拒绝；两次均在GPU调用与训练前退出。按代码包精确身份纠正调度参数后准入通过，没有修改执行源码或放宽门槛。

北京时间19:51:31，184执行/200冻结/773保护输入全部核同，原共享暂停与五PID退出保持，原GPU锁临时获取后释放/计算空。总8383143936B/限34292629504B，GPU1MiB26度/OOM0，成本1463/r3已342秒且全部SETTLED。真实输出/派发/日志均未创建；[完整准入与精确参数](../logs/session_records/q1_wm_physical_admitted_20261107.json)、[部署](../logs/session_records/q1_wm_physical_deployed_20261107.json)、[原运行时检查](../logs/session_records/q1_wm_physical_native_checks_20261107.json)三原文件7284B已Mac逐字节大小SHA核同。下一发布本准入再唯一WM_PHYSICAL01，实际结果与服务器说明镜像以后续回执为准。
