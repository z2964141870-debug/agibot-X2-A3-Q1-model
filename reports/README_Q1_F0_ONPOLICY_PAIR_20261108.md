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
