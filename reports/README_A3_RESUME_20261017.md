# A3继续测试：历史外部OOM审查与E51恢复

22:39后用户已指定切换humanplus工作机；当前恢复入口为[新机HP01](README_A3_HUMANPLUS_20261017.md)，覆盖下文旧ZHY启动动作。本报告保留换机前证据，本次与HP01一起保存，发布状态以新机回执为准。

记录：2026-10-17，北京时间19:31起；20:31更新。当前状态：SSH_UNKNOWN，入口hp3090不可达，远端资源/作业/E51账本未能重新核验。19:58前E51未登记/0worker是最后成功快照，不当作当前现场。官方035仍baseline，尚无新效果提升。

## 20:28巡检：入口不可达

本次仅做只读连接检查。两条独立查询（远端资源/E51状态、Git与远端main）均在连接hp3090的100.70.125.121:22时超时，退出码255；尚未连接ZHY。Mac路由仍指向utun5，Tailscale本机BackendState=Running/self_online=true；目标yu-MS-7D31显示online=false，LastSeen=2026-10-17T12:10:00.1Z（北京时间20:10:00）。ping两包均丢失。目标active=true不代替online判断。

这些证据只支持入口当前不可达，原因未知；不能据此认定重启、OOM、温度保护或机器首故障。当前boot/GPU/容器内存/GraspGenX是否仍运行及E51 OUT/state均UNKNOWN，不能沿用19:58数值。本次未清缓存、未登记或执行E51、无模型/张量/训练/评测。工程heartbeat保持ACTIVE，未修改旧巡检及每日研究。

最后成功同步为48449c427cd37f247f7918a624b0eb5b99d212f3：GitHub main与ZHY HEAD核验一致、52项source SHA通过；19:58发布回执位于ZHY logs/zhy_a3_resume/20261017_publication.jsonl。该时点E51 OUT不存在。本次连接证据及待同步清单为data/manifests/a3_transport_wait_20261017.json；本段、索引、current state及该清单只保存Mac，尚未提交/推送/同步，不能称GitHub已保存本次阻塞。

下一次低频重查hp3090。连接恢复后先读取实际OUT/state、boot/容器OOM/完整内核及资源，禁止仅凭网络恢复启动；按原计划处理缓存和三次5秒预检。若账本仍不存在且所有门通过才register一次，否则按实际state继续；新真实故障依原停止规则留证，不自动复跑。本次没有新增实验或修改计划、预算、门槛。

## 目标与本轮范围

用户要求继续测试，并指出相对baseline没有进步。本轮接续原E51/R20有限候选接受训练；先解决10/14起导致它始终未登记的恢复问题，保留同条件六衣服/四官方完整效果出口，不再追加E46–E50方向诊断。

用户这次明确继续，工程恢复资源巡检。E51未有登记或执行，允许在登记前另存版本修订；旧20261014计划、E50基准和全部历史账本不动。没有训练失败后换编号重跑。

## 实测结果与当前阻塞

19:31同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`、本容器OOM0，原available约3.22GiB；19:41复查约3.18GiB，低于8GiB。CPU44°C/GPU38°C，GPU瞬时0%但仍分配约1.13GiB显存。GPU利用率为0不能替代空闲核验。

其他计算为root PID1451734（父1367698），工作目录`/opt/data/private/hechengzhu/hxr/GraspGenX`，入口`batch/run_batch100_worker.py`及父入口`batch/batch100.py`；worker约5.47GiB RSS，带cache-only等参数。只查/proc任务路径和参数名，不读取其代码/环境/数据，不确认具体个人，不干预进程。

文件cache约10.02GiB，其中inactive约3.11GiB已经计入原available；不能把所有cache再次重复加成“可用”。原维护要求无其他计算，当前不满足；本轮0次维护、0释放/删除。待其他任务退出后，若仍不足8GiB，按新计划保护输入自主调用原32文件/2GiB/1800秒回收，再三次5秒复验。

## 历史外部OOM的有限审查

现有宿主dmesg相对E50含5条OOM FATAL；每条相邻oom-kill均为CONSTRAINT_MEMCG，PID/单调时间匹配，oom_memcg与task_memcg相同，且与ZHY memory cgroup互不包含。没有本容器OOM、boot变化、全局宿主OOM或硬件故障证据。

新增`script/a3/kernel_review.py`只处理此次保存SHA的五条历史记录：绑定boot、ZHY cgroup、E50完整kernel_baseline摘要、原始上下文SHA和逐条FATAL SHA。任何缺证/改证、PID/时间错配、本容器及其祖先/后代cgroup、全局OOM、硬件错误、新的未知FATAL均拒绝。新出现的外部OOM也不会自动放行，必须另行审查。本次不是删除日志、重置E50或把未知事件纳入基准。

E51在register前检查，并再次检查底层register实际捕获的kernel_baseline；两次之间如出现未审查FATAL，保存STOPPED，禁止启动/重试。注册后的监控沿原规则，所有后续新增FATAL、OOM/boot/温度/4GiB余量/900秒停止保护保持。

缓存维护入口复用同一精确历史审查；没有相应计划引用时保持原默认拒绝。空闲/boot/OOM/温度/打开映射文件/回收预算等其他维护条件不变。

## 版本与固定计划

- 新计划：`data/manifests/a3_finite_retention_plan_20261017.json`，原20261014计划不改，并以SHA作为新计划输入留证。
- 精简历史审查：`data/manifests/a3_kernel_review_20261017.json`；原始上下文仅ZHY/Mac小型日志`logs/zhy_a3_resume/20261017_reviewed_external_oom.json`。
- 恢复状态与输入/源码SHA：`data/manifests/a3_resume_readiness_20261017.json`，明确0实际训练及资源等待，不当作实验运行账本。
- 代码：`script/a3/kernel_review.py`、`test_kernel_review.py`；改动只接入E51登记与`script/project_tools/zhy_file_cache.py`。
- E51/R20编号不变，OUT仍不存在。模型/四段开发参考、奖励/reset/路由、实际token、teacher系数1、原KL名义LR、全局clip.1、8更新/32提交/3072交互、保存0/2/4/8、仅最终8的10完整效果回放全部保持。
- 原teacher RMSE<=.05/maxabs<=.5及六衣服无跌倒/均RMSE改善>=1%、旧不倒/官方关节<=+2%、世界根腕<=+5%、官方无新增跌倒效果门不改。通过仍须独立seed和真正未见会话。

## 实际验证与未执行项

本机10项标准库历史审查反例控制及3项原有限更新协议控制通过，4文件AST通过。ZHY部署后13项标准库测试及轻量plan入口也通过，只读真实核验5条历史FATAL精确通过；新计划全部source SHA匹配，E51 OUT不存在。底层注册快照第二次检查为源码修正，实际登记竞态未触发运行。ZHY验证原始输出保存在`logs/zhy_a3_resume/20261017_standard_library_validation.json`。

E51原4项CPU/CUDA合成Adam事务控制仍0次，训练0optimizer/rollout、效果回放0次。当前不具备运行这些测试的资源条件；不能把标准库检查或误拦截修正称为相对035效果进步。

## 命令与恢复

全部模型/张量/仿真/训练仅ZHY既有环境。原hp3090只Git/代码/标准库传输；Mac只小文件。无真机/驱动/功率/BIOS操作，无并行计算、无开机恢复。先确认没有其他任务，再按需维护：

```bash
/usr/bin/python3 -m script.project_tools.zhy_file_cache --trim --protect-plan data/manifests/a3_finite_retention_plan_20261017.json --baseline data/experiments/a3_zhy_candidate_retention_followup_20261014_E50/registration.json
```

三次5秒原启动门通过、OUT仍不存在时，仅一次register，顺序执行prepare/train/verify-training/evaluate-cloth/evaluate-official/verify-effect；具体原命令见`README_A3_FINITE_RETENTION_20261014.md`。已有账本则按实际state继续，不重置；首次STOPPED不评测提前点、不自动纠错复跑、不扩预算。

## 保存与同步

本报告/索引/current state、新计划、审查小清单和代码经diff/暂存区检查及audit_git_payload.py后scoped推送main，远端与ZHY HEAD以发布回执验证。保留每日研究/头环修改、ZHY未跟踪E48/E49/E50完整receipt。原始上下文与资源快照LOCAL_ONLY，不声称网盘备份。

恢复本工程heartbeat为ACTIVE，只对资源/新结果变化通知；旧巡检和每日研究不变。当前先低频等GraspGenX释放资源；内存不足时按授权自主回收。原暂停记录保留为历史，本报告覆盖其恢复步骤。
