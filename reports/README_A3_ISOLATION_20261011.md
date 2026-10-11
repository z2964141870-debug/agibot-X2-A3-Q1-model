# A3重启链路分离排查

记录日期：2026-10-11。E23日志异常停止；E24普通GPU对照确认新boot，STOPPED_REBOOT，全部后续计算停止。A3细日志版本已实现但未启动。

## 目标与授权

用户“开始排查”“继续”授权分离普通PyTorch GPU、Isaac无policy步进、A3仅前向和完整更新。E22第一更新保存后重启只能建立运行关联，具体调用点UNKNOWN。本阶段先执行普通GPU对照；首故障停止整个链路，不重复旧E01/E22，不恢复历史A3作业，不修改驱动、功率或BIOS，不操作机器人。

## 固定实验与依据

E23单独登记，现有a3-sonic环境运行，无Isaac导入、无A3权重。随机seed0、float32、TF32关闭、CPU线程1。矩阵256×256占空比目标0.1，然后1024×1024目标0.25；小CNN合成8×3×32×32输入，两个卷积/ReLU/池化/Linear，分离前向、反向无更新、Adam更新lr1e-4三档，占空比目标0.25。占空比按同步操作耗时安排sleep，不是实测GPU利用率；矩阵乘单位矩阵校验数值，小网络检查输出/梯度/参数有限性，检查仅optimizer档改变参数。

每档30/120/600秒，档内三个时长连续测试，档间300秒无计算冷却；5档共15项，计划约82.5分钟加初始化与采样开销。不是3090全负载验收，有限小型普通CUDA对照不能排除大规模/混合负载硬件故障。YOLO没有采用；合成小CNN避免安装和下载。

外层标准库独立监控，每5秒采样CPU/GPU、内存、boot与关键内核事件。CPU90°C/GPU85°C连续2次、可用RAM不足4GiB、遥测不可读、计算/数值错误、阶段超时立即停止。每阶段超时为目标时长加90秒。临时systemd unit Restart=no/RuntimeMaxSec=100min/CPUQuota=100%/MemoryMax=12G，无开机恢复；固定登记SHA、一次READY启动与原boot检查。阶段标记fsync涵盖torch导入、CUDA初始化、分配、首次计算和每5秒完成点，不能将采样空窗误作温度安全证明。

## 代码、命令、证据

主机`ssh hp3090`，根`/media/yu/FAFF-E9771/YUANQI_A3`。新脚本`script/diagnostics/gpu_isolation.py`，保护契约`test_gpu_isolation.py`。

```bash
/usr/bin/python3 -m unittest script.diagnostics.test_gpu_isolation
/usr/bin/python3 -m script.diagnostics.gpu_isolation register
# register只允许新登记一次；run只允许原boot READY启动一次。
systemd-run --user --unit=yuanqi-a3-isolation-E23 \
  --property=Restart=no --property=CPUQuota=100% --property=MemoryMax=12G \
  --property=RuntimeMaxSec=100min \
  --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI_A3 \
  --setenv=PYTHONPATH=/media/yu/FAFF-E9771/YUANQI_A3 \
  /usr/bin/python3 -m script.diagnostics.gpu_isolation run
/usr/bin/python3 -m script.diagnostics.gpu_isolation summarize
```

job/state/每项result：`data/experiments/a3_isolation_20261011_E23/`；健康/phase/console：`logs/a3_isolation_20261011_E23/`。登记与汇总小清单`data/manifests/a3_isolation_{registration,result}_20261011_E23.json`。原始工件LOCAL_ONLY，Mac只保存小型代码/报告/清单。

## 实际验证与结论边界

15:11:28只读核验boot740972e7-c88e-4588-8fa3-a86e8ed6bc64，CPU49°C/GPU30°C、GPU0%且无计算进程，RAM可用约29.6GiB。3项保护契约通过（一次启动/boot隔离、连续温度阈值与复位、缺失温度/内存保护），无CUDA测试。15:14:33登记READY，源码SHA固定在登记清单；尚不是GPU计算通过证据。后续需记录实际启动与每项结果。

15:16:59实际启动unit，Restart=no。15:17:07确认state RUNNING/0_matrix_30s，同boot，4健康记录CPU最高51°C/GPU33°C；CUDA初始化和首次同步矩阵操作已完成，第一30秒测试当时未结束。已更新原hp3090 heartbeat为每5分钟只读巡检本次链路，ACTIVE；旧A3自动恢复不启用。不能将unit的Result=success默认值当作已完成，实际以state/每项result为准。

## 保存、同步与恢复

登记记录6文件64,488字节通过载荷审计，提交839c29d346efc48e75a850dbc813d8bb080d94d8推送main且远端SHA一致。启动记录另行审计同步，完整SHA以Git历史为准。保留并行修改，不上传原始日志/模型。暂停的yuanqi-a3保持暂停，日研yuanqi保持不变。E23监测仅观察本次作业，状态未变静默，重启/故障立刻停止并记录；SSH失联记UNKNOWN，恢复后核验boot，不自动续跑。

普通GPU分支全部通过后先保存结果，再独立登记Isaac无policy分支，随后无优化器policy前向，最后完整更新；本登记没有自动启动后续分支的代码。主机或计算首故障则剩余全部搁置。恢复先读本README、job/state及日志，不能重置E23或将旧E22重新训练。

## E23记录代码异常与E24修复

E23于15:17:32已完成256矩阵30.0003435秒、2757次同步操作，数值检查未报错，result已原子落盘PASSED。随后WORKER_COMPLETE日志把result内的at_utc再次作为显式dict参数传入，触发TypeError重复关键字，worker退出1，外层15:17:35 STOPPED/WORKER_FAILED_1，没有继续120秒。10采样CPU最高51/GPU34°C、原boot未变，无GPU残留计算。这是自编诊断日志代码缺陷，不是CUDA/数值错误或机器重启；E23整套未通过，14项未做，保留其job/state/结果，不重跑。

原执行版本与登记SHA保留在Git提交839c29d，启动记录提交1615c6e。修复marker以合并字典记录，增加带at_utc的完成结果回归检查，不修改历史E23账本。新E24沿用相同负载参数，仅修复记录逻辑且使用独立路径。允许继续来自用户持续排查授权；不是在确认硬件/计算故障后自动重试。

当前代码命令面向E24：`/usr/bin/python3 -m script.diagnostics.gpu_isolation register/run/summarize`（CLI每次只传一个action，不用斜杠），unit改为`yuanqi-a3-isolation-E24.service`。job/state/results在`data/experiments/a3_isolation_20261011_E24/`，日志同名logs目录，登记/汇总清单末尾E24。上面的E23旧命令仅说明历史，不可再启动；新E24仍一次启动、原boot、同保护与RuntimeMaxSec100min。E24实际登记/启动/结果需按下次采样确认，不能把准备中记为完成。

15:21:07 E24原boot登记，4项保护/记录契约通过；修复与登记提交56891122e704e3f89280abfe6f13df32f9e06c3c已推main并核验远端。15:22:07实际启动E24临时unit，15:22:38第一30秒完成2828次/30.006794秒，有限性与单位矩阵校验通过，WORKER_COMPLETE日志正常；15:22:58 state RUNNING/0_matrix_120s、15健康采样最高CPU48/GPU36°C、原boot不变。原hp3090监测已更新为E24并ACTIVE每5分钟，旧yuanqi-a3仍暂停。当前仅1/15项通过，剩余未完成；不是普通GPU全套通过或排除硬件故障。

## E24最终：普通GPU也发生重启

15:42:19只读核验确认新boot `c7805e85-7467-4691-9037-b805ccae131f`，与初始`740972e7-c88e-4588-8fa3-a86e8ed6bc64`不同。256矩阵30/120/600秒全部完成（2828/11497/57821次），300秒冷却后15:39:52启动1024矩阵30秒档。该档日志确认PyTorch2.8.0+cu128导入、CUDA初始化、分配及同步完成，15:39:53.797792最后记录`MATRIX_FIRST_OPERATION_BEGIN`，没有对应`SYNCHRONIZED_OPERATION_END`或result。故障区间在首轮矩阵运算、allclose/isfinite校验与CUDA同步内，不能由该粗粒度标记单独认定某个API为根因。

新boot开机时间由墙钟减uptime推算约15:40:20，非故障精确时间；确认观察区间为15:39:53.797792最后阶段至15:42:19.894540首次确认新boot。221健康记录总体CPU最高50°C/GPU46°C、GPU功率采样最高120.62W、可用RAM最低28.63GiB，最后15:39:52.933 CPU44/GPU34°C、25.2W。未记录热保护或关键内核事件；首次GPU操作后的空窗不能排除瞬时温度/功率变化。256档实际active_seconds/elapsed约2.2–2.4%，受最短10ms sleep影响低于占空比目标，不是完整GPU高负载验收。

| 分支 | 结果 |
| --- | --- |
| 256矩阵30/120/600秒 | 3通过，结果文件SHA核验 |
| 1024矩阵30秒 | 重启中断，无完成结果 |
| 1024矩阵120/600秒、小CNN9项 | 11项未启动，按首故障停止 |
| Isaac、A3仅前向、A3细日志短训练 | 未启动，停止后续计算 |

该实验没有导入Isaac、A3或RL，没有加载官方权重。因此当前重启不需要A3专有代码即可发生，优先线索转为普通GPU执行链路；具体是PyTorch/CUDA/驱动还是该负载触发硬件问题仍UNKNOWN。单次重启没有定位物理元件，也不保证历史所有重启都同因。

E24在新boot无service计算残留/GPU进程，旧autoresume/longtrain仍disabled。hp3090 heartbeat已PAUSED，yuanqi-a3保持PAUSED、日研不变。重启后只做读取日志/元数据/源代码及小清单，未再次运行任何GPU、加载模型或训练。所查前boot内核时间段输出`-- No entries --`，没有故障附近内核覆盖，不能用无匹配排除驱动/硬件错误。停止证据`data/manifests/a3_isolation_stop_20261011_E24.json`，包含最后阶段、时间区间、健康及校验；原始日志与内核查询留logs/LOCAL_ONLY。用户最新要求的A3细日志实现见[实现与暂停原因](README_A3_PHASE_TRACE_20261011.md)。
