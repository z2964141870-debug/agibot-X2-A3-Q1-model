# 既有GPU运行环境对照 E30/E31

记录日期：2026-10-11。状态：E30通过；E31首故障为确认新boot，整条普通GPU循环对照已停止。

## 目标与范围

用户在E28首故障停止后明确“继续测试”，授权新的独立普通GPU对照。E28前2轮完成、第3轮校验后日志中断并确认重启，具体API/硬件根因UNKNOWN。此次不恢复E28/E29或A3，使用已装twist环境，不安装/下载、不改驱动/功率/BIOS，无A3/Isaac/RL/模型加载/机器人操作。

| 新编号 | 固定测试 | 前置条件 |
| --- | --- | --- |
| E30 | 1024 float32随机矩阵×单位矩阵一次，seed0，TF32关闭，blocking=0，额外同步保留 | 新授权、原boot、GPU空闲、安全遥测/实时内核可读 |
| E31 | 同计算与日志，30秒循环、占空比目标25%、最少10ms sleep | E30完成并发布、同boot、无异常/未完成调用 |

每项最长worker120秒、结束后60秒停载观察；第一异常/保护/新boot停止整链，SSH不通记UNKNOWN，不自动重试/开机恢复。新的独立READY只消费一次，不修改历史账本。所有阶段保存发布后才进入下一项。

## 完成内容与关键决定

`gpu_runtime_compare.py`调用原`gpu_operator_compare.worker`本身，只在新进程内绑定新输出目录，沿用完全相同计算/细日志函数；原源码不改、其SHA也登记。环境选择`/home/yu/miniconda3/envs/twist/bin/python`，静态METADATA登记torch1.13.1+cu117，预期实际CUDA11.7；环境尚未实际导入，实际版本在worker记录及监督结果核验。对照当前2.8.0+cu128是环境整体变化，包含Python/torch/CUDA相关库，不能单独归因CUDA版本。

新增独立标准库journalctl实时跟随进程，先保存3条本boot内核日志证明可读，再启动GPU；新事件实时JSONL/fsync，关键事件/跟随进程异常也停止。仍每5秒采样CPU/GPU/可用RAM，CPU90/GPU85°C连续两次、RAM保留4GiB；监督unit Restart=no、CPUQuota100%、MemoryMax4G、RuntimeMaxSec240、KillMode=control-group。并行仅标准库监测/只读日志，无其他GPU计算。

Mac只读镜像每0.1秒优先读取内核日志，再读阶段/健康：前1MiB逐条fsync，最后500条按收到块fsync；修复前缀达到上限后不再插入后续小行。完整日志只留服务器，Mac只留小前缀/末窗，覆盖据实核验。持续订阅和更快镜像有额外监测开销，会改变时序，故不是只有环境变量变化的严格因果试验；也不保证内核停机前必然打印或收到崩溃信息。

## 版本、命令与证据

- ssh hp3090，`/media/yu/FAFF-E9771/YUANQI_A3`；初始预检boot34cc7c88、CPU50/GPU30°C、GPU无计算进程。
- 新入口：`script/diagnostics/gpu_runtime_compare.py`、`collect_gpu_runtime_compare.py`、`record_gpu_runtime_compare.py`；计算复用原E28函数。6源码SHA、torch METADATA SHA与原boot绑定登记。
- 登记命令：`/usr/bin/python3 -m script.diagnostics.gpu_runtime_compare E30 register`；E31替换编号，必须E30通过，不可重置。
- 运行：同模块`<case> run`，仅一次临时unit；只读`<case> watch`/`<case> summarize`。
- job/state/results：`data/experiments/gpu_runtime_compare_20261011_<case>/`；原始日志`logs/gpu_runtime_compare_20261011_<case>/`。
- 小清单：`data/manifests/gpu_runtime_compare_{registration,result,evidence}_20261011_<case>.json`。
- 15无CUDA保护/日志检查Mac通过；服务器检查及Python3.8静态编译待核验，实际运行待后文登记。

## 实际结果与边界

开始前服务器通过15项检查，提交4f93327cb5d3f7a5b94f3b9a323360c9ab5d16f4已推送并核验远端。E30于18:29:30.311启动；worker确认`torch1.13.1+cu117`/CUDA11.7，18:29:33.501完成结果保存，退出0；18:29:35开始停载观察，18:30:35.987结束，原boot同boot。15健康采样CPU最高49/GPU33°C、可用RAM最低31,280,164,864字节；无保护/计算异常，unit inactive/dead/Restart=no。56阶段记录无ERROR/未完成，worker结果SHA和Mac镜像小记录核验；实时kernel follow写入3条历史/启动段记录，运行期间未见FATAL。E30只做一次操作，不能证明旧环境循环稳定。

E30与E27的主要差异是运行库整体：E30用既有Python3.8/twist的PyTorch1.13.1+cu117，E27用2.8.0+cu128；两次都保留额外同步和持久化日志，E30单次耗时约0.94秒、E27约0.08秒。E30单次通过说明这次旧运行库在本次操作中能完成，不能把原因归为某个单独CUDA组件，也不能排除循环时长、驱动/硬件或监测扰动。

E31在E30发布后按前置条件登记并启动，固定同一 `/home/yu/miniconda3/envs/twist` 环境（实际 `torch1.13.1+cu117` / CUDA 11.7）、同一1024矩阵操作、blocking=0、额外同步与持久化阶段日志，唯一主要变化是30秒循环。登记时boot为 `34cc7c88-6bb0-491d-800a-3714e7db9f37`，北京时间约18:36:17.727（UTC `2026-10-11T10:36:17.727151Z`）启动。

E31在第一轮的 `MATMUL_SYNCHRONIZE END`（UTC `10:36:19.326968Z`）之后写入 `ALLCLOSE BEGIN`（`10:36:19.327466Z`），没有对应完成记录；监督状态随后确认新boot `c99ddfa9-8478-441a-ab73-a6495ef0feb3`，首次读到新boot为UTC `10:38:00.216198Z`，当时uptime 75.16秒，按此只能估算重启约在UTC `10:36:45`附近，不能当作故障精确时刻。state为`STOPPED_REBOOT/CONFIRMED_NEW_BOOT_NO_RESUME`，未生成worker结果，也没有恢复、重试或启动E32/A3。

E31结果清单记录27条phase、0条ERROR，最后完成边界为`MATMUL_SYNCHRONIZE`，唯一未闭合区间为`ALLCLOSE`。重启前可见CPU最高48°C、GPU最高30°C、可用RAM最低约31.3GB；实时kernel跟随记录未见FATAL，重启后`kernel_previous.txt`为`-- No entries --`。这些是观测边界，不能据此证明ALLCLOSE本身是根因；停机前可能存在未采到的GPU/驱动/硬件事件。重启后只做只读快照，GPU为空闲，未加载模型或继续测试。

E31与E28（当前2.8.0+cu128）都表现为：单次1024操作可完成，但30秒循环在第一轮/早期边界发生新boot。E30的旧环境单次操作通过，E31说明单次通过不代表循环稳定；两次跨运行库的循环故障共同指向“重复GPU调用/驱动或硬件响应”这一更宽范围，仍不能把原因归结为A3代码、某一CUDA版本或某个API。

## 保存与恢复

E30/E31结果、停止与证据清单和README/索引/current state在发布前审计，原始日志LOCAL_ONLY，两个旧heartbeat不启用，日研不变。E31首故障后不再增加GPU负载，不恢复A3。后续只保留只读内核/硬件证据收集，若要做新的计算对照需单独重新授权且不能重复E31触发路径。恢复先读本README、对应state与phases/live kernel；确认首故障即停，不绕过前置门禁。
