# YUANQI 当前状态与下一步

最新：2026-10-18北京时间11:05，[HP06剩余PhysX验收](README_A3_HUMANPLUS_PHYSX_20261018.md)已STOPPED：STAGE_DEADLINE_900_SECONDS，实际896.670秒含采样截止余量。EULA修复有效、CUDA PhysX上下文已创建，但无十步完成收据，实际步数未知；不能记为完整环境通过。181健康同boot、CPU最高78/GPU44°C、可用最小20,334,821,376字节、0FATAL/新增OOM，停止后无自有进程组残留，代码/环境源码SHA未变。

输入迁移63/63、关键10/10已完成；HP05已通过CPU/CUDA四事务及MuJoCo10步不重跑，HP05/HP06终态不改。当前0策略模型/rollout/训练、035无新提升。zhy-a3保持PAUSED，不自动追加PhysX或基线/训练。恢复先读HP06 README及data/manifests/a3_humanplus_physx_20261018_HP06_{plan,result,audit}.json；优先只读审查STOP回调/退出合同，卡点尚未确认。原始日志LOCAL_ONLY，GitHub写权限未恢复，scoped本地提交尚未同步。下方10:40及更早内容均为历史。

最新：2026-10-18北京时间10:40，[HP05运行验收](README_A3_HUMANPLUS_RUNTIME_20261018.md)STOPPED_ENVIRONMENT_BOOTSTRAP_EULA_EOF。新监督器13项标准库两端通过；CPU/fused CUDA4事务与固定MJCF无策略10步通过，PhysX在EULA提示EOF处退出、未进入仿真。原因是新入口漏配非交互启动变量，非已证实硬件故障。12健康同boot、CPU最高81/GPU42°C、可用最小22,350,381,056字节、0FATAL/新增OOM；worker全部退出。已存在小张量/合成optimizer及MuJoCo仿真，0策略模型/rollout/训练，不再沿用下方历史0张量描述。

当前完整环境验收未完成，035仍baseline、无新效果；输入63/63、必需10/10已齐。HP05原账本不恢复/覆盖，不重跑已通过阶段，zhy-a3按异常停止规则暂停。恢复方案：先明确解除此次STOPPED约束，仅将遗漏的Isaac Sim启动变量纳入独立PhysX验收登记；通过后才新035匹配基线与原8更新训练。计划/结果/终态独立审计data/manifests/a3_humanplus_runtime_20261018_HP05_{plan,result,audit}.json；原始logs同前缀目录，LOCAL_ONLY。代码与证据scoped提交，GitHub写权限未恢复、尚未同步。下方10:20及更早文字为历史现场。

更新：2026-10-18北京时间10:20。用户明确要求代理直接迁移、允许慢传后，[HP04迁移完成](README_A3_INPUT_MIGRATION_20261018.md)：11大文件613,119,744字节单次112.032秒迁完；与HP03合计63文件628,994,284字节独立大小/SHA通过、训练必需输入齐10/10。当前INPUTS_VERIFIED_WAIT_RUNTIME_VALIDATION。完整阶段监督器、模型环境与科学运行未验收，0本工程张量/训练，035仍baseline，无新效果，GitHub未同步。

035 PT 402,050,931字节及10历史轨迹现均在目标路径、0覆盖；证据为data/manifests/a3_humanplus_large_transfer_{plan,result,verification}_20261018_HP04.json。旧交接清单保留为固定来源证据，HP02超时部分归档未复用；HP03不复跑。新根/home/humanplus/YUANQI_A3与96个已验MJCF网格保留，源E51旧记录不存在、E45 STOPPED/E50 COMPLETE不变。10:19只读复核同boot、可用22,611,034,112字节、GPU1396MiB/0%/43°C，无GPU/可见计算；journal1410条、0FATAL/新增OOM。此快照不等于三次启动门，启动前重查。

## 当前A3工作与恢复

源连接：此前临时socket失效，已使用用户在当前会话提供的认证恢复humanplus的/tmp/yuanqi-zhy-source.sock，12小时闲置寿命；凭据未持久化，使用前仍须核验。迁入文件大小/SHA已通过，不把旧ZHY基准当新机通过证据。下一步独立登记新机监督器集成与小运行验证，再测新机035六衣服/四官方完整匹配基线，最后原8更新短训练。原遥测计划/结果及只读收据保留、已完成探针不重跑。当前内存充足不清缓存，不直接调用旧ZHY登记器。

最新授权目标为humanplus，新机只读盘点/内核日志完成，现有FATAL匹配0但不是连续运行验收。候选x2-sonic-isaaclab环境仅METADATA/关键4源码SHA核验，不修改共享环境；新机需独立环境/资源门、输入SHA及匹配035基线，不沿用ZHY boot/cgroup v1/历史效果数值。旧E51未在新机登记，旧STOPPED账本不动。长期任务仍用zhy-a3 id转向新根，旧巡检/每日研究不变；恢复先读新机README与小清单，再处理数据/认证和资源，不直接运行旧finite_retention_trial。下方旧现场只作来源。

10/17 22:21用户要求先查上游：GitHub main仍10/9 c9df753，仅比固定vendor多1个下载统计/元数据提交；旧14项发布条目包括035 PT SHA全部未变，只有main分支、无tags/releases。HF在线API超时，未全面核验模型站；不升级vendor，不把此更新当策略修复。见[检查](README_A3_UPSTREAM_UPDATE_20261017.md)与[a3_upstream_update小清单](../data/manifests/a3_upstream_update_20261017.json)。本次合并到humanplus阶段保存；先前transport清单的文档SHA是20:31历史快照，恢复同步时保留后续增补并核对实际diff。

## ZHY历史入口

以下为换机前的现场与旧计划，不能用于humanplus启动。当前迁移/计算规则以本页顶部和新机README为准。

最新SSH_UNKNOWN覆盖19:58及下方旧资源快照；19:58成功同步48449c4至GitHub/ZHY、52项source SHA通过且E51 OUT不存在，当前仍须重查。新计划`data/manifests/a3_finite_retention_plan_20261017.json`保留原计划/SHA、训练预算和效果门；只增加五条历史外部OOM的逐条审查，无未来异常豁免。E51登记前及实际捕获基准再次检查，登记后原监控不改。10标准库审查+3协议控制通过，ZHY实际5条匹配；最后确认的4项张量事务/训练/效果均0次。本次审计/提交/推送因Git工作副本所在入口不可达尚未执行；连接恢复后先同步这四个已保存文件并核验，保留并行修改。

19:41 root PID1451734/父1367698在`/opt/data/private/hechengzhu/hxr/GraspGenX`执行`batch/run_batch100_worker.py`，worker约5.47GiB RSS；GPU瞬时0%但非空闲，原available约3.18GiB/同boot/OOM0。未清缓存或干预。待无其他计算，若仍不足8GiB则按新计划+E50登记自主有限回收，再各3次5秒原预检，OUT不存在才register一次；任一STOPPED不续跑/扩预算/评测早期点。恢复工程heartbeat ACTIVE，变化才通知；下一直接prepare控制→原8更新→最终六衣服四官方效果，不重复旧方向诊断。

以下10/14占用与暂停为历史证据，不能当实时作业或当前恢复指令。

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

先按10/17恢复入口等待真实资源门；历史外部OOM精确审查已接入，新计划固定、E51未登记。下述原训练流程的预算和效果门保持，资源未满足不得启动。

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
- 本工程zhy-a3按10/17继续指示恢复ACTIVE/每30分钟，恢复以10/17 README和实际state为准，旧yuanqi-a3与hp3090保持PAUSED；每日研究yuanqi不变。只通知新结论/阶段/异常/需用户处理的新阻塞，状态不变静默。机器首故障或可验收目标达成后保存入口并暂停工程heartbeat。

## 其他工作入口

- Q1在原机/media/yu/FAFF-E977/YuanQi_Q1、origin/Q1；恢复须读该副本状态，不以main旧记录启动工作。见[Q1入口](README_Q1_SMPL_20261008.md)。
- 学习Agent继续本人必要知识训练；控制链待答项见[学习会话](README_LEARNING_SESSION_20261012.md)、[交接](README_LEARNING_AGENT_TASK_20261011.md)。10/14用户决定至少跟一遍课程，采用Agent+平板笔记+上课；视频配套10章PDF已检查，见[资料与学习安排](README_LEARNING_COURSE_20261014.md)。尚无新增掌握证据，课程不调度工程计算。
- 每日研究已保存[10/17简报](research/2026-10-17/README.md)：PredActor补出采集/标注/离线BC训练入口，固定16文件静态核验；仍限G1、DAgger/部署待发布、HF资产未核验。三类公告与其余关注源码无实质变化。昨日回执旧补丁未落盘已识别，补同步以[研究阶段](README_RESEARCH_CHECK_20261017.md)及[回执](../data/manifests/research_watch_sync_20261017.json)为准；无复现、无新假设，工程与教学主线不由新闻调度。
- 原机故障/空闲/负载试验及迁移历史见[报告索引](README.md)，旧细节保留在各阶段README与Git历史；本入口只保留当前恢复所需信息。

独立侧支：[头环与衣服参考研究](README_HEADBAND_CURRENT_STATE.md)：HB06离线计分工具9项控制通过；公开评测包两端连接超时、WAIT_DATA，独立heartbeat已暂停并保存获取/恢复入口；不接手主线实验，尚未训练或证明效果提升。
