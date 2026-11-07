# Q1：纯学生状态标签的同来源配对实验

2026-11-08，北京时间。实验F0_ONPOLICY_PAIR01。原共享训练保持人工暂停；本实验单独拟合小型历史残差，不恢复160k训练。最终验收和用户采纳null保持。

## 问题与实验

[上一轮纯学生采集](README_Q1_F0_ONPOLICY_LABELS_20261108.md)真实完成1259物理步、625条教师标签，其中406训练、219验证。八个前缀提前终止；教师仅提供同状态H10标签，从未接管动作。教师从失稳状态的恢复能力仍未证实，不能把标签存在当恢复保证。

本次检验访问状态分布的学习价值。两组都从实际causal_history selected3000继续，保留非零适配器、人体编码器、decoder和归一化；不能重新置零适配器或换回较早selected500。相同容量、初始化、种子、每组3000次Adam、每批128条：共同64条来自原三来源的均衡保存数据；另64条按相同来源组和片段配对，对照取旧状态，实验取纯学生实际访问状态。总曝光相同，行内容按实验设计不同，不把差异归于来源覆盖增多。

旧6725训练/2358验证记录直接复用；新增标签仍是九个预先登记来源组，不读test或新留出事件。不新增推理、物理或渲染。模型仅接收已审计f0人体历史和930维机器人历史，teacher_action仅是标签；actual_state、next_state、terminal及教师未来依赖均不得输入模型。这仍是动作蒸馏，不称联合BC/RL、序列世界模型或GigaBrain整套复现。

## 选点与停止

每500更新验证，包含update0。主风险为同来源旧状态验证与纯学生状态验证各半，来源组/片段/来源均衡；同时原全验证、原R1、原混合行为验证三个锚点均不得比update0退步。两组均完成相同更新，不做不对称早停。实际四个Adam计数必须全等于3000，teacher/WM、编码器、decoder与统计冻结。分别保存有限CPU selected/last，checkpoint_update明确该工件实际更新位置。

若选择update0且参数/统计/decoder精确相同，则复用原七条失败，不重复相同策略物理。非零变化选点需新增独立审查schema及匹配闭环/原速媒体后才能评价，不能凭验证loss提升称自然遥操合格。

## 来源与预算

新[计划](../script/supermimic_teleop/student_onpolicy_pair_plan_20261108.json)、[controller](../script/supermimic_teleop/student_onpolicy_pair_training.py)、[worker](../script/supermimic_teleop/student_onpolicy_pair_worker.py)、[来源配对](../script/supermimic_teleop/student_onpolicy_pair_data.py)、[拟合](../script/supermimic_teleop/student_onpolicy_pair_fit.py)、[合同检查](../script/supermimic_teleop/test_student_onpolicy_pair.py)，仅六新执行文件，不改216冻结执行或原六训练来源。

独立输出`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_F0_ONPOLICY_PAIR01_20261108`。单次600秒/128MiB，原r2成本台账；起点4380=1614+1395+1371秒，全SETTLED/无RESERVED，总14400和r2的4500上限不重置。22GiB准入/24GiB退出、GPU18GiB80度、100GiB余盘、private500GB与原非阻塞GPU锁保持，不清缓存。

先发布精确源码清单，原Morph运行时检查及1119原保护输入、采集终门与回收、标签和选点、资源成本准入，再发布准入回执；唯一output/log/dispatch/精确进程全无后才单派。失去回执先查实际结果，不重发。

## 已完成与待完成

六本地CPU合同0.880秒全过：来源/分组互斥；未来与终止元数据不进入输入；同anchor及来源slot调度；非零warm start及冻结模块；两组三次实际合成Adam及保存/选点；update0精确保留。该合成夹具不是新的服务器训练。CLI通过，现有45014原Linux会话可用，原暂停/boot/GPU、4380台账及锁空已核；系统python缺NumPy的只读探查已换回原项目运行时，未开始训练。

真实部署、原运行时检查、准入与训练均待执行。完成后交付两组训练六宫图、原始曲线和实际结算成本，再据独立闭环决定标签是否产生效果。硬件端到端延迟、实测人体遥操与自然性仍未验收。大索引服务器镜像仍PENDING，具体同步以publication原件为准。

## 服务器原运行时准入

2026-11-08 07:37:05北京时间，现有45014会话收到真实准入。源码66ff17d2b11713679f19f20b9315e9b9c22f95e5及六文件清单d6cd6bded5c030463b3524de177d1201387e55ee均显式push origin Q1并独立40位远端核同，Mac十二文件892766B与权威副本SHA同。

18486B九成员包SHA d6c4f0e749b099b17947b0138b55051e9bdfd84ae6cc851aa30371916553728e经15次精确ACK传输，只新增精确成员；1119原来源前后同。[部署原件](../logs/session_records/q1_f0_onpolicy_pair_deployed_20261108.json)。原Morph六CPU合同本体2.046秒、进程5.660673秒全过；[运行时原件](../logs/session_records/q1_f0_onpolicy_pair_contracts_20261108.json)。

1152来源、625纯学生标签与实际selected3000、采集终门和回收、原4380账本、暂停与资源锁准入11.882499秒通过。[准入原件](../logs/session_records/q1_f0_onpolicy_pair_admitted_20261108.json)。三原件4472B已Mac逐字节大小SHA核同；内存8235298816B/实际限34292629504B，GPU1MiB25度/OOM0/余盘183554568192B。真实新增服务器训练仍0。下一先发布本准入，再唯一输出/log/dispatch/精确进程全无门后单派600秒配对训练。旧大索引服务器镜像PENDING。

## 真实训练结果

准入ef6ed04399766e71ae3934db41e58bbb8ed7b7cb显式推送及独立40位远端核同后，唯一防重门通过，2026-11-08 07:39:02北京时间派发timeout387308/controller387309/start40766102。[派发原件](../logs/session_records/q1_f0_onpolicy_pair_dispatch_20261108.json)。07:41:26真实ONPOLICY_PAIR_TRAINED_PENDING_CLOSED_LOOP/source_integrity PASSED，worker387383由guard正常回收/return0；原五PID及本任务三PID全退，原GPU锁和计算空。

两组各实际3000次Adam、四个优化器计数全3000、各384000样本曝光，初始化/容量/来源slot与调度SHA一致。训练前后的1152保护输入全同；教师/WM/人体编码器/decoder/归一化均未更新。所有500/1000/1500/2000/2500/3000选点都未守住三旧锚点，两组最终均选择update0。保存last供审计，不能改选last或放宽门。[完成原件及数值](../logs/session_records/q1_f0_onpolicy_pair_completed_20261108.json)。

下表均为各组**未采纳的最后3000步**相对本组update0的验证动作MSE变化，负值表示误差下降。不是关节弧度、真实物理误差或遥操成功率。

| 验证分布 | 只旧状态 | 半纯学生状态 |
| --- | ---: | ---: |
| 旧/纯同来源各半主风险 | +0.047% | -7.625% |
| 纯学生状态 | -0.499% | -17.147% |
| 同来源旧状态 | +2.271% | +31.130% |
| 全旧状态锚点 | +3.611% | +31.572% |
| 原R1锚点 | +6.932% | +40.497% |
| 原混合状态锚点 | +1.295% | +21.080% |

新标签确实改变了该模型在保存验证分布上的拟合取舍，但没有产生同时保留旧能力的合格选点。不能据此说纯学生标签普遍无效，也不能称闭环已改善；只否定当前初始化/容量/采样比例/损失下的继续单步蒸馏。没有对该配方延长训练。

独立CPU保存审计0.959327秒：每组选中模型的8个base encoder张量、4个adapter张量及统计、teacher/decoder身份与实际初始selected3000完全相等；last适配器实际变化且保存四Adam计数3000。新schema未送入旧review入口，也未改旧合同；相同ProprioResidualStudent结构和原因果action路径保持，因此直接复用原七例0/7完整的失败结果，0新增物理/媒体/模型调用，不伪造新视频。[精确可复现审计命令](../logs/session_records/q1_f0_onpolicy_pair_saved_audit_20261108.txt)。

新增123秒SETTLED，累计4503=1614+1518+1371/14400秒，无预留，余9897秒、r2余2982秒。总内存峰10704166912B/显存峰520MiB/OOM0/回收未尝试/原训练信号0；审计后总8522579968B/实际限34292629504B、GPU1MiB25度、余盘183540178944B。[资源原件](../logs/session_records/q1_f0_onpolicy_pair_resource_final_20261108.json)。

## 交付与后续

Mac两份[旧状态训练六宫PNG](../data/analysis/Q1_F0_ONPOLICY_PAIR_FIGURES_REVIEWED_20261108/saved_only_six_panel.png)、[纯学生标签训练六宫PNG](../data/analysis/Q1_F0_ONPOLICY_PAIR_FIGURES_REVIEWED_20261108/onpolicy_mix_six_panel.png)，同目录两PDF和两CSV均已生成并目视。原数值包31494B/SHA 0fadbeb499e64724375cc751fa724df9d548fd65b2d78ff9e2efa607bb582555已Mac逐字节核同；两图同时展示最后未采纳状态，避免只展示选0的零柱图。

新[完整训练证据ZIP](../data/deliveries/Q1_F0_ONPOLICY_PAIR_REVIEWED_EVIDENCE_20261108.zip)523919B/SHA e997f65fc28317c186f83671acc84dda17916777cd1275790c86f8f8d5fc129b，17成员CRC/逐成员大小SHA/16相对链接通过；本包没有新增视频，因为策略相同未重复物理。原视频与旧失败包保留。[交付记录](../logs/session_records/q1_f0_onpolicy_pair_training_delivery_20261108.json)。首版图包是生成过程的草稿，不作为本阶段主交付。

**下一步先核教师恢复能力，再决定训练目标。** 在独立新方案中，从已保存训练/验证学生状态分叉短真实物理前缀，对比原学生与特权H10教师，保留终止和接触/滑移/侧倾；先明确模拟器状态恢复是否包含必要隐状态并用零分叉动作/下一步等价门证明。没有此门，不把任意设置位置速度视为同一物理状态，不用未验证WM的多步预测替代物理。仅选择预先登记的train/validation锚点，不打开test/新留出事件；不得重跑旧七条或整84例。先方案/来源/资源与剩余账本发布，再有界GPU执行。该探查尚未实现或派发，新GPU预算未预留。

本阶段拟合、保存审计和证据交付已结束，低延迟遥操整体目标未达到；H4的80ms预览与f0零预览仍需分别评估。硬件端到端、实测人体与自然性均未验收；原共享仍暂停、用户采纳null。具体Mac/服务器同步见随后publication，大索引服务器镜像仍PENDING。
