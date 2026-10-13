# ZHY 外部容器 OOM 与 E51 启动阻塞

记录日期：2026-10-14，北京时间03:35。阶段：E51登记前只读巡检。状态：BLOCKED_NEW_HOST_KERNEL_EVENT；E51仍未登记、未运行。

## 目标与范围

恢复已实现的E51有限更新接受短训练，先检查真实Git、账本、进程、GPU、容器及内核证据。本次只经hp3090访问ZHY做标准库只读核验和保存记录；未加载张量、训练、评测、维护缓存或干预其他任务。

## 完成内容与实际结果

- ZHY HEAD为`da57c84e265be3f53cc21c22bd7621d376ec81ff`；恢复README/AGENTS的SHA与上次一致；E51 OUT/state/registration不存在，本实验0worker。
- 同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`。16GiB容器原口径可用约10.08GiB；GPU46%/3879MiB/55°C、CPU41°C，其他Python约4.81GiB RSS，仍不满足空闲门。
- 本次另外逐行对照E50注册的kernel_baseline，发现1条新增FATAL匹配：`Memory cgroup out of memory: Killed process`。此前03:02及02:32只读收据未保存内核差异核验，不能称这些时段没有事件。
- 内核相邻`oom-kill`记录明确`constraint=CONSTRAINT_MEMCG`。`oom_memcg`与`task_memcg`相同，但均不同于本进程`/proc/self/cgroup`的memory路径。因此确认是同宿主另一容器的额度内OOM，不是ZHY容器OOM，也不是已证明宿主全局内存耗尽。
- ZHY `memory.oom_control`仍为`under_oom=0/oom_kill=0`，没有重启证据。`memory.failcnt`本次100119960，高于旧99539871；该计数不能当作OOM kill或硬件故障证明，不改写旧计数记录。
- 内核事件单调时间35206788.378907秒；以当前epoch/uptime换算，约北京时间2026-10-14 02:06:22，属于近似墙钟时间，非独立历史时钟证据。

## 关键决定及依据

原`script/a3/finite_retention_trial.py:97-100`对宿主所有新增FATAL行拦截，没有按cgroup过滤；`zhy_file_cache.require_idle`也保留同类内核门。此次确认该原门存在新增匹配，但没有为了演示失败去执行register，也没有创建STOPPED实验账本。

按用户既定“首次新增内核异常停分支留证、不自动重跑”规则暂停本工程zhy-a3 heartbeat。明确是**原宿主级门被外部容器事件阻塞**，不能写成A3训练崩溃或显卡硬件故障。没有清空dmesg、重建基准、改FATAL规则、过滤新行或更改资源/效果阈值；其他任务不干预，旧巡检和每日研究不变。

## 版本、命令与证据

ZHY根`/opt/data/private/hechengzhu/ZHY`，控制器`/usr/bin/python3`。原判据源码`script/diagnostics/host_loadcheck.py`的FATAL与`script/a3/finite_retention_trial.py`登记门；实际只读取：

```bash
dmesg --color=never
cat /proc/self/cgroup
cat /proc/uptime
cat /sys/fs/cgroup/memory/memory.oom_control
cat /sys/fs/cgroup/memory/memory.failcnt
```

并用既有`zhy_trial.health()`及`stage_resources.processes()`核对资源，以原E50 registration的`kernel_baseline`、boot和OOM计数比较。未读取他人代码或环境变量；原始内核邻近行和cgroup身份仅留LOCAL_ONLY。

- 小清单：`data/manifests/a3_finite_retention_kernel_block_20261014.json`，记录原始证据SHA、归属结论和未执行项目。
- 原始资源：`logs/zhy_a3_heartbeat/20261014T033220_health.json`。
- 原始归属证据：`logs/zhy_a3_heartbeat/20261014T033220_kernel_review.json`。
- E51方案继续保留：`reports/README_A3_FINITE_RETENTION_20261014.md`及原计划；不改输入/源码SHA，不新增实验编号。

## 结果与边界

本轮没有任何baseline效果进展。官方035仍baseline，E42/E44/E45/E50拒绝与停止结果不变。E51张量控制0次、训练0步、效果回放0次、缓存维护0次、文件删除0次。

新事件归属已查明，但当前原门是宿主级，不能在未记录的情况下自行改为容器级或“清除旧错误”。其他计算仍占GPU，即使以后处理内核门，也必须重新满足全部三次5秒资源门。

## 保存与同步状态

本报告、索引/current state和小SHA清单本轮scoped审计/提交/推送；实际提交与远端一致性以Git命令收据为准，不预称成功。原始证据在Mac小文件及ZHY日志目录保存并比对SHA，均LOCAL_ONLY，未核验网盘。既有未跟踪E48/E49/E50完整receipt与并行工作保留。

## 下一步与恢复入口

工程heartbeat暂停，E51保持未登记。恢复时先读本报告和E51 README，明确处理已证实外部容器OOM的宿主级启动阻塞，再按新指示决定是否需要独立、可审查的保护范围修订；不可把一次“继续”解释成可隐去内核证据。若不修改原门则保持不计算。

任何后续修订必须保留本次证据、旧E50 baseline及旧账本，另固定版本与计划；未通过原资源门前不登记/启动。不得恢复旧E44/E45，也不重复E46–E50。
