# X2 同步资源监测与后续性能测试

2026-11-14。阶段 MONITOR01，状态 PASSIVE_OBSERVATION_COMPLETED。用户要求通过 Windows 桥接继续工作；机器人与同学共用，先被动观察，不用空闲内存或项目锁代替整机负载判断。

## 实际结果

原生两跳密码会话90475已进入同一SoC1。新源码发布 `b2d3050ea68b8128b2c77d4bf8fb06d0ede59d05`，显式push origin Q1后独立40位远端核同；四文件部署大小SHA一致，板端七项合同也全部通过。

唯一passive01真实完成，20个一秒CPU区间、外层返回0，boot前后同。结果如下，全部为背景负载，没有启动模型：

| 指标 | 结果 |
| --- | ---: |
| 整机CPU平均利用率 | 25.05% |
| CPU0平均 / 其余七核范围 | 45.85% / 18.72%至24.34% |
| GPU利用率中位 / 最大 | 11% / 22% |
| 最高温度 | 64.968°C |
| 最低可用内存 | 12502802432B，约11.64GiB |
| 采样器自身CPU时间 | 0.394008秒 / 约20秒，不含tegrastats子进程 |
| 单次采集读取墙钟p95 | 19.488520ms，每秒一次；含低优先级调度等待 |

Top16中20次全出现的进程按单核100%定义：nvargus-daemon平均41.36%、soc1_hal.._orin40.85%、component_conta37.56%、soc1_drp15.63%。登录时ps累计CPU值440%不能当成这段稳态占用。整机仍有系统负载，不称空闲或推断此前模型瓶颈。1Hz采样不能定位单个20ms请求的精确因果。

自有tegrastats以-15退出，监测器已等待回收，未调用全局停止。新增模型调用、反馈订阅、物理、训练、BLE、MC及电机命令均0。完整四成员原件包11259B，SHA `5db87a0376dbb9b4d7ab2027cd6678ce9dc49de35e365cacd9fd79af0ab97f64`，已回Mac核同：`data/deliveries/X2_RESOURCE_MONITOR_PASSIVE01_20261114.tgz`。源数值位于 `data/analysis/X2_RESOURCE_MONITOR_20261114/passive01/`。

提交采样命令后原会话有数分钟无回执，没有重发。独立只读观察连接3236在第一跳明确Operation timed out/255，随后原90475仍返回启动与完成回执；该失败不代表整条桥接或Linux不可达。

## 已完成与范围

原生密码两跳 `ssh sshuser@100.92.227.62`，Windows 内 `ssh run@10.0.1.41` 已进入 X2 Ultra SoC1。2026-11-14 23:04:19 北京时间核到项目 `/home/agi/x2_morph_human_20261113`、aarch64、run uid1000、8 个允许 CPU；本次 boot 为 `115b4d3f-aaff-47dc-8f1f-5ac42d20a106`，不同于此前现场。系统刚启动约200秒。项目 R2 两个输出根均不存在；本次未派发推理或修改服务。

新增独立 `script/x2_resource_monitor`，只依赖 Python 标准库。每秒读取 `/proc/stat`、`/proc/*/stat`、内存以及一个自己启动的 `tegrastats`。单次1至120秒，默认20秒；仅调整监测进程自身 nice 值，不改系统时钟、功耗、服务或其他任务亲和性。不启动模型、ROS反馈、BLE、MC或电机命令。

记录逐核区间利用率、按单核100%定义的进程 CPU 占用、GPU整机利用率、EMC、温度、可用内存、原始 tegrastats 和单调时钟。可选原PID/starttick限定自有任务子树与线程，PID重用拒绝继承统计。GPU数值不能归属到单个进程；采样时间是主机读取时间，不是硬件精确同步或机械端到端测量。

七项本地计数合同和 CLI 已通过：包含guest计数不重复、计数器重置、带空格括号进程名、PID重用、跨核百分比和子树身份。板端七项合同与本轮实际监测均已完成，不证明模型负载或稳定30Hz。

## 独立执行

先核当前主机/boot/任务和目标目录不存在，再只执行一次：

```bash
cd /home/agi/x2_morph_human_20261113
timeout --signal=INT --kill-after=5s 35s python3 -B -m script.x2_resource_monitor.monitor --seconds 20 --interval 1 --output data/analysis/X2_RESOURCE_MONITOR_20261114/passive01
```

输出 `samples.jsonl` 与 `summary.json`，监测器清理自己启动的 tegrastats，保留退出码及采样开销；不得调用全局停止命令。本阶段先监测背景负载。之后若资源允许，R2仍须独立来源、64输出一致性、实际TID分配和资源门，再运行有界联合测试；现有四串行组、R1和S1加速探针不重跑。

## 后续判断

将利用率与CPU阶段、GPU提交、Morph RPC、排队年龄对齐，之后选择资源分配或进程隔离。当前180ms预览、衣服默认、终端2/3和原Q1暂停保持。没有声称新的速度收益或真机遥操通过。

## 保存状态

源码与计划已发布；板端部署、实际采样、原件回传均已完成。最新报告与索引的Git发布及设备文档镜像以后续publication为准，旧部署收据中的报告SHA对应部署时原计划版本。下一R2仍先独立源码/资产及资源准入，再唯一64前缀门；尚无R2实际GPU调用或提速结论。
