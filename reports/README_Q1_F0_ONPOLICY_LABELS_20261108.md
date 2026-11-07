# Q1：纯学生访问状态与教师纠偏标签

2026-11-08，北京时间。阶段PURE_STUDENT_LABELS01：先采集与审计，暂不训练。原共享训练保持用户暂停；32k教师、原final门与用户采纳null保持。

## 问题与决定

历史残差的保存验证误差下降36.303%，真实七例却仍0/7完整；前走和迟到前缀延长，其余五例变短。[完整负结果](README_Q1_F0_PROPRIO_REVIEW_20261108.md)已交付36成员证据ZIP，不能再据单步loss继续相同拟合。

旧AGGREGATE01执行`.5*teacher+.5*student`，并非纯学生访问的状态。此轮单一主要变化是实际行为分布：原causal_history selected3000完全驱动物理，教师H10仅在同一个实际状态上每两帧提供独立特权标签，绝不混入动作、学生特征或下一步机器人历史。参考时钟、事件可见性、原CPU转换、原CausalReviewSession动作路径均保持。

这是本项目的学生状态分布诊断，不能称已复现TWIST的联合BC/RL或GigaBrain整套BWM。[既有一手源码与迁移边界](research/2026-11-07/LOW_LATENCY_SUPERMIMIC_REBASE.md)继续适用。教师标签是否具有恢复能力仍未验证；标签存在或动作差较大都不能单独证明可以恢复。

## 独立数据与预算

从既有候选计划的original_jobs按原顺序取前六train组和前三validation组，九组互斥，不根据新结果挑片。不读取test载荷，不采集A022/A023/A024比较片，不打开已登记的新留出事件。动作与物理任务是原名义参考，输入可见性保留对应nominal/dropout/late_burst。

每片最多192步，短片维持原边界；总1699个新物理步、最多1699次学生前向加850次教师查询、850条样本。遇原终止条件立即停止该片，保留失败前缀，不用补齐/教师接管伪造完整性。当前与下一实际状态按同一trace严格配对，保存terminal_after_action，不跨终止拼接序列。

学生输入仍f0与历史。120→50Hz插值依赖、到达时钟、历史flags、缺失与fallback沿用已审计合同；教师未来依赖单列并标特权，训练数据不把它作为f0输入。离线人体预处理仍offline_unverified，真实人体或硬件端到端未测。

独立输出`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_LABELS01_20261108`，600秒/128MiB、0新增Adam、0渲染，成本记原r2。进入前原账本4256=1614+1271+1371秒，全SETTLED；总14400、r2上限4500均不重置。22GiB准入/24GiB退出、GPU18GiB80度、100GiB余盘、private500GB、原非阻塞GPU锁与自有子树回收保持，不清缓存。

## 实现与验证

[controller](../script/supermimic_teleop/student_onpolicy.py)、[worker](../script/supermimic_teleop/student_onpolicy_worker.py)、[数据门](../script/supermimic_teleop/student_onpolicy_data.py)、[不可变计划](../script/supermimic_teleop/student_onpolicy_plan_20261108.json)。只新增清单外文件，不改已执行216来源或六训练文件。

六本地检查全过/.002秒，CLI与导入通过：来源互斥/固定选择；教师变化不能改变执行动作；状态/动作/标签及下一状态配对；split/终止/启动样本门；教师未来与学生可见性分离；未采样步精确继承原学生回应。真实采样和教师查询尚未派发。

先发布源码/五文件清单，再服务器原运行时合同、1111既有来源及新输入、选点、资源、暂停、锁和账本准入。唯一输出/log/dispatch/精确进程全无后单派；缺回执先查实际结果，禁止重发。

## Done与Todo

已完成独立实现、本地检查、原运行时准入、下述真实采集和保存数据配对审计。没有新增Adam或渲染视频；后续训练尚未派发。

只有来源/状态/标签及成本回收门通过后，才设计同容量/曝光的保存数据与on-policy数据对照，验证选点与旧锚点保持，随后独立真实闭环。序列目标需明确冻结WM有效范围和终止mask，不在本采集入口内隐式开训。新证据不足时可否定后续训练方向；不能仅凭此诊断完成结算整个探索。

## 原运行时与准入

2026-11-08 06:59:30北京时间，复用45014原Linux会话收到真实回执。获批的hp_han三值90/60/5已由本机`ssh -G hp_han`再次核验，无需再修改配置或询问用户代测。原root/boot/GPU一致，五个旧训练PID均退出，训练保持STOPPED_SIGNAL。

源码33d973e4885010f161086589d5473d04cab8493b和五文件清单34c53b72e671c137c070560bfa019447559fba2f已显式push origin Q1并独立40位远端核同，Mac十一文件与权威副本大小SHA同。八成员15484B部署包SHA d5a836335f008b06232644eca890bcbe75f4c0059399bc63a8cdce78ca73a34d逐片13ACK完成；只新增八文件，1111旧来源前后同，未改216执行或六训练文件。[部署原件](../logs/session_records/q1_f0_onpolicy_deployed_20261108.json)。

原Morph运行时六项检查本体0.004秒/进程0.259070秒全过；1119来源和选点、原4256秒账本、暂停及实际资源准入10.700250秒通过，原GPU锁可独占测试后释放。[检查原件](../logs/session_records/q1_f0_onpolicy_contracts_20261108.json)、[准入原件](../logs/session_records/q1_f0_onpolicy_admitted_20261108.json)。三原件5671B已Mac逐字节大小SHA核同。末总8204079104B/实际限34292629504B，OOM0/GPU1MiB24度/余盘183574794240B；真实新采集和Adam仍0。下一先唯一output/log/dispatch/精确进程防重，再单派600秒采集。大索引服务器镜像仍PENDING，不宣称全三端同步。

## 真实采集与结论

准入36c5dbd5a6855df33ae5d6c8c550f1eca015ef55显式推送及独立40位远端核同后，唯一防重门确认output/log/dispatch/精确进程不存在，07:02:49北京时间单派timeout386422/controller386423/start40547654。[派发原件](../logs/session_records/q1_f0_onpolicy_dispatch_20261108.json)1936B已Mac大小SHA同。07:04:52真实COMPLETED/source_integrity PASSED，guard WORKER_COMPLETED/worker386433回收且返回0；三自有进程已退出、原锁释放、GPU计算空，原共享仍暂停。

实际1259个物理步、1250次学生前向、625次H10标签查询，少于1699/2549/850上限。406条训练、219条验证记录全部通过状态/学生动作/标签与物理trace的精确匹配，next_state和terminal_after_action已再核。教师混合动作始终0，test和新留出事件未读取。保存数据诊断1.808604秒/0新推理物理Adam，1119保护输入及生成工件前后SHA同。[服务器完成与逐例原件](../logs/session_records/q1_f0_onpolicy_completed_20261108.json)、[可复现诊断命令](../logs/session_records/q1_f0_onpolicy_saved_diagnostic_20261108.txt)。

| 来源 | 分组 | 实际/请求物理步 | 标签 | 提前终止 |
| --- | --- | ---: | ---: | --- |
| A095 | train | 148/192 | 74 | 是 |
| A040 | train | 190/192 | 94 | 是 |
| A075 | train | 112/192 | 55 | 是 |
| A145 | train | 84/192 | 42 | 是 |
| A084 | train | 121/192 | 60 | 是 |
| A065 | train | 163/163 | 81 | 否，达到原短片边界 |
| A267 | validation | 187/192 | 94 | 是 |
| A031 | validation | 152/192 | 75 | 是 |
| A033 | validation | 102/192 | 50 | 是 |

八个前缀提前终止，不能把采集入口成功当学生闭环成功；这些是训练/验证来源，不能与旧七条比较片合并成质量提升。九来源教师/学生动作RMS差0.3722至0.5656，单位为原canonical policy action而非关节弧度；七来源最后25步差异高于自身整前缀。该现象支持进一步检查偏离状态的纠偏目标，不能证明教师能从失稳状态恢复。

累计计费新增124秒，原账本4380=1614+1395+1371秒，全SETTLED/无RESERVED，余10020秒；未重置预算。[资源原件](../logs/session_records/q1_f0_onpolicy_resource_final_20261108.json)：总量峰11499769856B/显存峰444MiB/OOM0/训练信号0/回收未尝试；退出后总8239566848B/限34292629504B/GPU1MiB25度/磁盘余183554822144B。

## 六宫图与下一步

Mac[六宫PNG](../data/analysis/Q1_F0_ONPOLICY_LABELS_FIGURES_20261108/collection_six_panel.png)、[PDF](../data/analysis/Q1_F0_ONPOLICY_LABELS_FIGURES_20261108/collection_six_panel.pdf)、[逐来源表](../data/analysis/Q1_F0_ONPOLICY_LABELS_FIGURES_20261108/per_source.csv)仅读原始完成和资源记录，已目视，未生成新视频。新[证据ZIP](../data/deliveries/Q1_F0_ONPOLICY_LABELS_EVIDENCE_20261108.zip)213391B/SHA 78969df3a84ad4ec273cf2438c490c83ee66c5fb3dc9608a92904e14ad299044，七成员CRC/逐文件大小SHA/五相对链接通过。[交付审计](../logs/session_records/q1_f0_onpolicy_delivery_20261108.json)。原训练、旧七失败和既有对照均未重跑；原始数据数组仍服务器，数值审计和图表已Mac。

下一把同容量、同初始化、同曝光的“仅原保存数据”与“加入纯学生访问状态标签”对照具体化。固定教师/WM/decoder/归一化和验证选点规则，先保留旧验证锚点，非零新选点才做独立闭环；选零且策略精确同则复用旧失败。同时必须区分教师动作标签与真正恢复能力，序列训练不能跨terminal拼接，也不能用未验证WM的多步回报代替真实物理。任何新训练先独立方案/来源/资源及剩余成本审计和发布，不在本采集入口追加训练。10帧降至H4的80ms候选与零未来帧学生仍分别报告，当前没有硬件端到端或自然遥操达标结论。
