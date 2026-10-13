# YUANQI 当前状态与下一步

更新：2026-10-13晚，北京时间；E44停止时点20:09。官方035仍是基线，尚无同条件验证提升。E44/R18因原官方动作锚点漂移保护停止，未取得可评测的新候选；已完成[针对性论文/固定源码研究](README_A3_OUTCOME_RESEARCH_20261013.md)，下一实现语料3→4加入已知失败zzx的独立试验；其他配方/原门不变，明确改作开发训练而非未见测试。

## 当前A3工作与恢复

- 最新入口：[E44有限曝光停止](README_A3_OUTCOME_CURVE_20261013.md)。状态STOPPED，4次完整更新加第5次3个batch，19optimizer/1920交互。锚点RMSE0.050075超过0.05；不恢复、不追加预算、不重复register/prepare/train。只存官方step0，8/16/32均未达到，0新增闭环评测。具体异常见logs/a3_zhy_cloth_curve_20261013_E44/training/console.log及cloth_contract.json；state.reason为null不代表无异常。
- E44准备18控制/900帧复用通过；训练首概率门通过、冻结摘要未变。末次累计21健康同boot/OOM0/CPU43/GPU35°C，最低可用6.19GiB，无确认机器故障；阶段间441.19秒无连续覆盖。恢复必须重新核验活跃作业/boot/GPU/资源，不以旧PID判断。
- [停止小清单](../data/manifests/a3_zhy_cloth_curve_stop_20261013_E44.json)与[累计清单](../data/manifests/a3_zhy_cloth_curve_20261013_E44.json)已保存。登记源码2708fff，训练时HEAD a78e597只增记录；旧注册代码/输入SHA未改。
- [E43保存轨迹审计](README_A3_CLOTH_EFFECT_AUDIT_20261013.md)已COMPLETE，10动作/8769步初始输入与参考相同。共同跌倒前衣服均误差仅-0.257%、官方+5.875%；原效果门仍拒绝，禁止复跑。
- [E42/R17](README_A3_CLOTH_CACHED_20261013.md)：2更新/8optimizer合同通过，衣服全程RMSE .1413798→.1549925（+9.63%）、仍5/6不倒；3官方超退化预算，REJECTED。zzx共同前缀腿误差增、官方单腿手臂误差增，见E43。旧R12及所有拒绝结果保留。
- 基线为[E26](README_A3_SMPL_FROZEN_VALIDATED_20261013.md)六衣服、E02四官方。六衣服中zzx5.66s跌倒；七段衣服反复用于开发，不能当真正未见最终测试。E23运动学6/7通过不等于动力学平衡，下蹲仍不合格。

## 下一条具体工作

阅读[研究结论与固定草案](README_A3_OUTCOME_RESEARCH_20261013.md)。MimicX公开源码已固定78a1ab7并静态核验8文件，未安装/执行；PriMo相关方法、D2R摘要均仅候选。下一独立编号只将zzx完整参考加入原三段训练，不同时改奖励/路由/teacher/重置采样，不复跑E44。旧E23划分不动，新角色DEVELOPMENT_TRAIN；原六衣服四官方效果门仍完整保留。

草案最多32更新/128optimizer/12288交互，新增早期2/4完整保存点（其余8/16/32），训练干净完成后固定整点最多50完整回放，首停止不自动评测/复跑。先固定新源码/输入SHA并验新增MotionLib及数据角色；现尚无新账本/worker。若再次teacher早停，停语料分支，下一单独处理优化冲突，不加码。详情与SHA见[小清单](../data/manifests/a3_outcome_research_20261013.json)。完整开发改善后才独立seed/真正未见会话；没有论文/代码/训练次数可代替效果。

## 固定执行与存储约定

- 模型/张量/IK/仿真/训练仅在经ssh hp3090连接root@59.77.15.15:25115的/opt/data/private/hechengzhu/ZHY，既有data/environments/a3-sonic环境；原hp3090的/media/yu/FAFF-E9771/YUANQI_A3只Git/代码/标准库传输，绝不恢复原机训练。Mac只小型代码/README/清单。
- 无真机/驱动/功率/BIOS授权；不并行计算、不自动开机恢复。新计算三次5秒预检可用估计>=8GiB、无其他RSS>1GiB/计算、GPU空闲、可靠遥测；不改记账/门槛、不丢缓存、不终止他人进程。CPU90/GPU85两次5秒保护、4GiB保留、900秒/阶段；首真实机器故障或计算异常停分支，不自动复跑。SSH失败记UNKNOWN。
- 每重要阶段立即模板README、索引、current state、小SHA清单；审diff/暂存、audit_git_payload.py后scoped提交推送指定GitHub main并核对远端。保留并行修改/旧账本，不强推。大型工件LOCAL_ONLY，网盘未核验，不写凭据或代发消息。
- 本工程zhy-a3 ACTIVE/每30分钟，恢复以真实README/state为准，旧yuanqi-a3与hp3090保持PAUSED；每日研究yuanqi不变。只通知新结论/阶段/异常/需用户处理的新阻塞，状态不变静默。机器首故障或可验收目标达成后保存入口并暂停工程heartbeat。

## 其他工作入口（本轮未修改）

- Q1在原机/media/yu/FAFF-E977/YuanQi_Q1、origin/Q1；恢复须读该副本状态，不以main旧记录启动工作。见[Q1入口](README_Q1_SMPL_20261008.md)。
- 学习Agent继续本人必要知识训练；下一检查Actor输出→目标角→PD→反馈，见[学习会话](README_LEARNING_SESSION_20261012.md)、[交接](README_LEARNING_AGENT_TASK_20261011.md)。工程不要求先完成理论推导。
- 每日研究已有[10/13简报](research/2026-10-13/README.md)，去重与原文覆盖见[研究阶段](README_RESEARCH_CHECK_20261013.md)。本聊天针对性论文研究不新增每日任务。
- 原机故障/空闲/负载试验及迁移历史见[报告索引](README.md)，旧细节保留在各阶段README与Git历史；本入口只保留当前恢复所需信息。
