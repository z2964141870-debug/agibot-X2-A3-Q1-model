# 既有GPU运行环境对照 E30/E31

记录日期：2026-10-11。状态：代码检查通过，新实验尚未登记/启动。

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

当前仅代码准备，不声称旧环境能正常导入或稳定运行。单次通过不证明循环稳定，旧环境循环通过也不能排除共有驱动/硬件或不同内核负载；旧环境失败也不直接定位显卡、PCIe或供电。温度/功率5秒采样可能漏瞬时峰值；外部日志尾不能代替完整服务器证据。

## 保存与恢复

代码、README、索引/current state与小清单审计后scoped提交推main并核验远端，保留并行修改。原始日志LOCAL_ONLY，两个旧heartbeat不启用，日研不变。不恢复A3，当前任务直接监测短测。恢复先读本README、对应state与phases/live kernel；确认首故障即停，不绕过前置门禁。
