# YUANQI 当前状态与下一步

更新：2026-10-14 00:02。E48单次COMPLETE：新batch上原始总Actor下降方向减小teacher偏差，但保存Adam候选一阶和有限步均增大偏差，RMSE .046055→.046877（仍过原门）。这支持研究实际位移保留约束，不是历史第19步复现或闭环提升；官方035仍baseline。

## 当前A3工作与恢复

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

E46/E47/E48均禁止复跑。最新入口：[E48新batch结果](README_A3_FRESH_BATCH_20261013.md)，[小清单](../data/manifests/a3_fresh_batch_result_20261014.json)。34Actor+2Critic/3grad/384交互/1候选/0optimizer；完整映射、概率门、保存teacher控制和状态不变通过。新KL .000290394使原规则LR升至6.75e-6；原始teacher方向-.613754，Adam一阶+6.25635e-5、有限MSE+7.63463e-5。

11健康同boot/OOM0、CPU43/GPU33°C，最低可用4.513GiB，登记至启动49.86秒缺口。app.close快速退出导致progress.json缺失，result已原子保存且独立标准库退出后SHA核验通过；不重跑补文件。

下一固定E48 batch/同step4/moments，独立E49研究实际Adam位移的完整teacher一阶非增约束；单次数学投影加有限teacher/同batch目标检查，0optimizer/新rollout。草案入口[实际位移保留E49](README_A3_CANDIDATE_RETENTION_20261014.md)，预算/来源见对应计划；已实现/5标准库控制通过/容差固定，尚未登记，不能当已启动。当前0worker。不恢复E44/E45或其verify/evaluation，语料/曝光分支关闭。

用户新增明确要求：文件内容缓存压低8GiB启动余量时自行定向回收。E48后一次维护因未辨明进程可见性拒绝，0释放/删除；随后只读核验无大进程、GPU空闲、可用8.225GiB。已补维护失败PID留证和3控制，随后按原保护单次释放2闲置文件的129.77MiB页缓存，0删除；原可用估计8.324→8.450GiB，同boot/OOM0。见[10/14缓存记录](README_ZHY_MEMORY_20261014.md)，维护保护E49计划输入，不放宽门槛或全局drop_caches。
官方仍baseline；开发结果通过后也必须独立seed和真正未见会话，不用代码/检查/checkpoint替代效果。工程heartbeat保持ACTIVE，待下一明确分支；不是机器故障停机。恢复必须查实际进程/boot/GPU/资源及原账本。

## 固定执行与存储约定

- 模型/张量/IK/仿真/训练仅在经ssh hp3090连接root@59.77.15.15:25115的/opt/data/private/hechengzhu/ZHY，既有data/environments/a3-sonic环境；原hp3090的/media/yu/FAFF-E9771/YUANQI_A3只Git/代码/标准库传输，绝不恢复原机训练。Mac只小型代码/README/清单。
- 无真机/驱动/功率/BIOS授权；不并行计算、不自动开机恢复。新计算三次5秒预检可用估计>=8GiB、无其他RSS>1GiB/计算、GPU空闲、可靠遥测；不改记账/门槛，不终止他人进程。用户最新要求缓存增长时回收，现有heartbeat空闲期调用zhy_file_cache：cache>4GiB时只对已结束项目闲置文件限32个/2GiB释放，至少间隔30分钟；保护当前/下一阶段输入、打开/映射文件。两次共575.59MiB，0文件删除，不作全局drop_caches；见[内存核验](README_ZHY_MEMORY_20261013.md)。CPU90/GPU85两次5秒保护、4GiB保留、900秒/阶段；首真实机器故障或计算异常停分支，不自动复跑。SSH失败记UNKNOWN。
- 每重要阶段立即模板README、索引、current state、小SHA清单；审diff/暂存、audit_git_payload.py后scoped提交推送指定GitHub main并核对远端。保留并行修改/旧账本，不强推。大型工件LOCAL_ONLY，网盘未核验，不写凭据或代发消息。
- 本工程zhy-a3 ACTIVE/每30分钟，恢复以真实README/state为准，旧yuanqi-a3与hp3090保持PAUSED；每日研究yuanqi不变。只通知新结论/阶段/异常/需用户处理的新阻塞，状态不变静默。机器首故障或可验收目标达成后保存入口并暂停工程heartbeat。

## 其他工作入口（本轮未修改）

- Q1在原机/media/yu/FAFF-E977/YuanQi_Q1、origin/Q1；恢复须读该副本状态，不以main旧记录启动工作。见[Q1入口](README_Q1_SMPL_20261008.md)。
- 学习Agent继续本人必要知识训练；下一检查Actor输出→目标角→PD→反馈，见[学习会话](README_LEARNING_SESSION_20261012.md)、[交接](README_LEARNING_AGENT_TASK_20261011.md)。工程不要求先完成理论推导。
- 每日研究已有[10/13简报](research/2026-10-13/README.md)，去重与原文覆盖见[研究阶段](README_RESEARCH_CHECK_20261013.md)。本聊天针对性论文研究不新增每日任务。
- 原机故障/空闲/负载试验及迁移历史见[报告索引](README.md)，旧细节保留在各阶段README与Git历史；本入口只保留当前恢复所需信息。
