# YUANQI 当前状态与下一步

更新：2026-10-14 03:35。E51仍未登记/0worker；新增宿主内核OOM已证实来自另一个容器，ZHY自身OOM0/同boot。原宿主级内核门被触发，暂停本工程自动执行，入口[内核归属与启动阻塞](README_ZHY_KERNEL_BLOCK_20261014.md)。官方035仍baseline，暂无效果提升。

## 当前A3工作与恢复

10:12–10:13只读定位占用：root PID1183477，`/opt/data/private/hechengzhu/hxr/taco_multiview/replay20.py`，显存3822MiB/GPU45%，与ZHY同memory cgroup、项目目录不同。只确认任务路径，不能确认具体个人；没有停止进程或改变自动任务。见[归属记录](README_ZHY_GPU_OCCUPANCY_20261014.md)。

09:57按用户重申的缓存授权复查：原可用9.97GiB，高于8GiB；GPU45%/3879MiB且存在其他计算，因此本次无需且不符合空闲清理条件，0维护/0释放。以后条件满足直接清理不再询问。本轮没有修改内核保护、基准、E51计划或暂停状态，见[内存记录](README_ZHY_MEMORY_20261014.md)。

最新状态为BLOCKED_NEW_HOST_KERNEL_EVENT，覆盖下方较早WAIT_EXTERNAL_COMPUTE恢复动作。E51登记门不区分cgroup，原基准/门槛保持，未实际执行register或创建STOPPED账本。其他计算仍占GPU46%/3879MiB；不维护缓存、不干预。按原新增内核异常停分支规则暂停zhy-a3，保留旧监控/每日研究设置。恢复先读[证据与范围](README_ZHY_KERNEL_BLOCK_20261014.md)，再明确处理宿主级门；不因ZHY OOM0自行过滤外部事件。

- 23:01内存核验/定向释放完成，原可用估计约8.087→8.509GiB，16GiB容器RSS仅约0.34GiB，主要文件cache和文件系统kmem；同boot/OOM0，无正在运行训练残留。[记录](README_ZHY_MEMORY_20261013.md)。

- E46已单次完成/禁止复跑：2状态、12forward、10grad、0optimizer/rollout，参数buffer/输入/保存optimizer未变，原概率门和step4保存teacher指标通过；6健康同boot/OOM0、CPU42/GPU31°C，最低可用6.60GiB。E45首batch在step4为局部反事实，非历史第19步或效果。证据见[累计清单](../data/manifests/a3_zhy_retention_direction_20261013_E46.json)。

- 最新入口：[E45/R19停止](README_A3_CLOTH_FAILURE_20261013.md)。4整更新+第5次3batch、19optimizer/1920交互，step0/2/4仅存证，未到8/16/32；0新独立训练核验/效果回放。首次概率通过、冻结未变、实际已加载含zzx的4key；语料改变未避免原teacher早停，不能据此断言效果或相反梯度。
- 累计20健康同boot/OOM0、CPU41/GPU35°C、最低可用4.72GiB，无确认机器故障；阶段间314.80秒无连续覆盖。停止后GPU空闲，实际state/console/合同为准。登记源码e3156ce；训练HEAD0ffe4ce只增记录。[停止清单](../data/manifests/a3_zhy_cloth_failure_stop_20261013_E45.json)及[配置差异](../data/manifests/a3_zhy_cloth_failure_config_20261013_E45.json)保留原raw比较，仅存储目录不同。
- 历史：[E44有限曝光停止](README_A3_OUTCOME_CURVE_20261013.md)。状态STOPPED，4次完整更新加第5次3个batch，19optimizer/1920交互。锚点RMSE0.050075超过0.05；不恢复、不追加预算、不重复register/prepare/train。只存官方step0，8/16/32均未达到，0新增闭环评测。具体异常见logs/a3_zhy_cloth_curve_20261013_E44/training/console.log及cloth_contract.json；state.reason为null不代表无异常。
- E44准备18控制/900帧复用通过；训练首概率门通过、冻结摘要未变。末次累计21健康同boot/OOM0/CPU43/GPU35°C，最低可用6.19GiB，无确认机器故障；阶段间441.19秒无连续覆盖。恢复必须重新核验活跃作业/boot/GPU/资源，不以旧PID判断。
- [停止小清单](../data/manifests/a3_zhy_cloth_curve_stop_20261013_E44.json)与[累计清单](../data/manifests/a3_zhy_cloth_curve_20261013_E44.json)已保存。登记源码2708fff，训练时HEAD a78e597只增记录；旧注册代码/输入SHA未改。
- [E43保存轨迹审计](README_A3_CLOTH_EFFECT_AUDIT_20261013.md)已COMPLETE，10动作/8769步初始输入与参考相同。共同跌倒前衣服均误差仅-0.257%、官方+5.875%；原效果门仍拒绝，禁止复跑。
- [E42/R17](README_A3_CLOTH_CACHED_20261013.md)：2更新/8optimizer合同通过，衣服全程RMSE .1413798→.1549925（+9.63%）、仍5/6不倒；3官方超退化预算，REJECTED。zzx共同前缀腿误差增、官方单腿手臂误差增，见E43。旧R12及所有拒绝结果保留。
- 基线为[E26](README_A3_SMPL_FROZEN_VALIDATED_20261013.md)六衣服、E02四官方。六衣服中zzx5.66s跌倒；七段衣服反复用于开发，不能当真正未见最终测试。E23运动学6/7通过不等于动力学平衡，下蹲仍不合格。

## 下一条具体工作

先处理上述已留证的外部容器OOM造成的宿主级内核门阻塞；自动执行暂停。下述E51流程仅为阻塞解除后的原方案，当前不得直接启动。

最新[E50单次候选保留](README_A3_CANDIDATE_RETENTION_20261014.md)于2026-10-14 01:07 COMPLETE，候选LOCAL_PROJECTION_REJECTED。用户已明确同意纠错接续；E49仍STOPPED，E50的JSON类型修正验证通过，原E48全部11标量/方向对照差0。16Actor/decoder+1Critic、3grad、1原候选+1投影，0新rollout/optimizer/checkpoint，禁止复跑。

teacher RMSE原.046055388→投影.046205418，MSE增加1.3841811e-5；实际舍入一阶变化1.24524e-9过门，有限teacher非增门失败。固定Actor目标下降.04469024、原.05/.5保护通过。只证明切平面约束不保证该有限步保留，不是机器故障或baseline提升。小清单[a3_candidate_retention_result_E50](../data/manifests/a3_candidate_retention_result_20261014_E50.json)。参数/输入/保存optimizer未变，退出前源SHA核验及progress/result齐全。

6健康同boot/OOM0，CPU42/GPU30°C，最低可用7.008GiB；最大采样间隔61.57秒包含登记至启动间隙。当前0worker。此投影方法分支结束，不进入短训练，不缩步/换batch/扫LR，不自动追加同类零更新诊断。E46–E50方向链收束，E44/E45不恢复/不verify/evaluation。

下一入口[有限更新保留研究与E51](README_A3_FINITE_RETENTION_20261014.md)。已优先复核本地持续学习资料、补读D2R§3.1–3.4及CPO固定作者源码；不重做MimicX/PriMo。D2R实际软KL、持续学习依赖任务分支，均不能直接保证共享A3 decoder；本次借鉴CPO实际候选接受机制，不宣称理论迁移或复现。

E51/R20实现完成但尚未登记：从官方035/空Adam开始、E45相同4段，唯一变量为隔离原生AdamW提案→固定8比例实际teacher/同batch目标接受→参数和历史矩一次提交。原.05/.5保护与六衣服四官方效果门不变，不将E50改成通过。预算8更新/32提案/3072交互，保存0/2/4/8，仅最终8做10完整评测。先4项ZHY合成事务控制；任一STOPPED不续跑/不评测早期点、不延伸同类诊断。计划[a3_finite_retention_plan](../data/manifests/a3_finite_retention_plan_20261014.json)。

北京时间01:29预检同boot/OOM0/GPU空闲，但原available约6.68GiB低于8GiB。01:46定向维护因GPU忙在空闲检查拒绝，0释放/删除，未重试。01:47新见其他Python计算/GPU19%/5286MiB，原available约8.63GiB但仍不得并行。E51 WAIT_EXTERNAL_COMPUTE，未登记/0本实验worker；恢复低频核验，无需用户处理。发布后保护新计划输入，维护可信基准E50登记；只定向终结工件缓存、32文件/2GiB/30分钟、不删数据或全局drop。再三次5秒原预检，未通过WAIT、不建账本；通过才各阶段一次执行。维护由shell直接运行，避免Python父包装器被判其他计算。旧E44/E45和E46–E50禁止重跑。 实现d0b1ae3已推送并同步ZHY，源码SHA全部一致；01:56复核仍有其他Python/GPU43%/4284MiB，available约9.80GiB也不能并行，E51 OUT仍不存在；[发布收据](../data/manifests/a3_finite_retention_publish_20261014.json)。


## 固定执行与存储约定

- 模型/张量/IK/仿真/训练仅在经ssh hp3090连接root@59.77.15.15:25115的/opt/data/private/hechengzhu/ZHY，既有data/environments/a3-sonic环境；原hp3090的/media/yu/FAFF-E9771/YUANQI_A3只Git/代码/标准库传输，绝不恢复原机训练。Mac只小型代码/README/清单。
- 无真机/驱动/功率/BIOS授权；不并行计算、不自动开机恢复。新计算三次5秒预检可用估计>=8GiB、无其他RSS>1GiB/计算、GPU空闲、可靠遥测；不改记账/门槛，不终止他人进程。用户最新要求缓存增长时回收，现有heartbeat空闲期调用zhy_file_cache：cache>4GiB时只对已结束项目闲置文件限32个/2GiB释放，至少间隔30分钟；保护当前/下一阶段输入、打开/映射文件。两次共575.59MiB，0文件删除，不作全局drop_caches；见[内存核验](README_ZHY_MEMORY_20261013.md)。CPU90/GPU85两次5秒保护、4GiB保留、900秒/阶段；首真实机器故障或计算异常停分支，不自动复跑。SSH失败记UNKNOWN。
- 每重要阶段立即模板README、索引、current state、小SHA清单；审diff/暂存、audit_git_payload.py后scoped提交推送指定GitHub main并核对远端。保留并行修改/旧账本，不强推。大型工件LOCAL_ONLY，网盘未核验，不写凭据或代发消息。
- 本工程zhy-a3因新增宿主内核事件暂停（保留每30分钟配置），恢复以本次阻塞README与真实state为准，旧yuanqi-a3与hp3090保持PAUSED；每日研究yuanqi不变。只通知新结论/阶段/异常/需用户处理的新阻塞，状态不变静默。机器首故障或可验收目标达成后保存入口并暂停工程heartbeat。

## 其他工作入口

- Q1在原机/media/yu/FAFF-E977/YuanQi_Q1、origin/Q1；恢复须读该副本状态，不以main旧记录启动工作。见[Q1入口](README_Q1_SMPL_20261008.md)。
- 学习Agent继续本人必要知识训练；控制链待答项见[学习会话](README_LEARNING_SESSION_20261012.md)、[交接](README_LEARNING_AGENT_TASK_20261011.md)。10/14用户决定至少跟一遍课程，采用Agent+平板笔记+上课；视频配套10章PDF已检查，见[资料与学习安排](README_LEARNING_COURSE_20261014.md)。尚无新增掌握证据，课程不调度工程计算。
- 每日研究已保存[10/16简报](research/2026-10-16/README.md)：TRACE跨本体适配、C-GMT参考修正、CoWear共同目标/对时；原文相关段落及TRACE部署文档已核验，未复现，训练资产/历史公告/HF仍有缺口。去重与Git状态见[研究阶段](README_RESEARCH_CHECK_20261016.md)及[回执](../data/manifests/research_watch_sync_20261016.json)。工程与教学主线不由新闻调度。
- 原机故障/空闲/负载试验及迁移历史见[报告索引](README.md)，旧细节保留在各阶段README与Git历史；本入口只保留当前恢复所需信息。

独立侧支：[头环与衣服参考研究](README_HEADBAND_CURRENT_STATE.md)：HB06离线计分工具9项控制通过；公开评测包两端连接超时、WAIT_DATA，独立heartbeat已暂停并保存获取/恢复入口；不接手主线实验，尚未训练或证明效果提升。
