# GPU首轮调用拆分 E26

记录日期：2026-10-11。状态：E26已完成，单次同步拆分诊断未复现重启。

## 目标与范围

用户“继续拆”授权针对E24普通1024矩阵首操作重启，进一步定位未完成调用。仅一次1024×1024 float32随机矩阵乘单位矩阵，不导入A3、Isaac或RL，不加载模型。E24及历史账本保持停止，E25未启动。

## 完成内容与关键决定

独立E26标准库监督进程与PyTorch子进程，登记绑定原boot、源码SHA与READY一次消费。把导入、CUDA初始化、两次分配、矩阵乘、allclose、isfinite、all归约、item标量读回、各显式同步及结果保存单独记录BEGIN/CALL_RETURNED/END，每条fsync。allclose本身返回Python布尔值，内部设备读回无法由外部标记继续分解。

固定seed0、TF32关闭、现有PyTorch环境，CUDA_LAUNCH_BLOCKING=1在导入前设置；同步诊断改变运行时序，成功不能排除原异步路径问题。只运行一轮，子进程完成后独立监测60秒，不继续加负载。CPU90/GPU85摄氏度连续两次5秒采样、RAM保留4GiB，遥测/内核不可读、关键内核错误、计算异常/120秒超时或新boot均停止。非开机恢复unit、Restart=no，不改驱动/功率/BIOS。

每5秒保存健康和本boot启动后内核查询，Mac只读watch每秒镜像阶段/健康/内核小记录并逐行fsync；不修改journald。即使持久写入，也不能保证突然断电时所有缓存落盘；外部镜像仅覆盖收到的记录。短时调用可能在5秒采样间完成，峰值仍可能漏采。

## 版本、命令与证据

- 主机：ssh hp3090，`/media/yu/FAFF-E9771/YUANQI_A3`。
- 代码：`script/diagnostics/gpu_operator_split.py`，保护与持久日志复用登记中5个源码SHA。
- 环境：`data/environments/a3-sonic/bin/python`；当前已知PyTorch2.8.0+cu128/CUDA12.8，实际版本在worker日志核验。
- 新登记命令：`/usr/bin/python3 -m script.diagnostics.gpu_operator_split register`，登记后不可重复。
- 单次unit运行命令：`/usr/bin/python3 -m script.diagnostics.gpu_operator_split run`，只允许READY且原boot，不可恢复。
- 只读：同模块`watch`或`summarize`；后者只读GPU遥测并对确认新boot保存停止状态。
- job/state/小结果：`data/experiments/gpu_operator_split_20261011_E26/`；原始日志：`logs/gpu_operator_split_20261011_E26/`。
- 登记/结果小清单：`data/manifests/gpu_operator_split_*_20261011_E26.json`。
- 代码/登记提交4455a2c78abe80552d2dbb13a4d41c37c747e68b已推main，远端SHA一致。本机与服务器11项无CUDA安全/日志检查通过。

## 结果与边界

预检确认boot c7805e85-7467-4691-9037-b805ccae131f、GPU无计算进程、CPU47/GPU30摄氏度，可用RAM约29.6GiB。17:28:48登记、17:29:32.785启动、17:29:35.194全部操作和结果保存完成；worker退出0后17:29:37开始额外停载观察，17:30:38.347完成超过60秒观察。17:31:04再次只读核验原boot未变、GPU无计算进程、unit inactive/dead/Restart=no。

| 已拆开的边界 | 本轮实际结果 |
| --- | --- |
| torch导入、CUDA初始化、随机/单位矩阵分配、分配同步 | 全部完成 |
| 1024矩阵乘、矩阵乘同步 | 完成 |
| allclose、对应同步 | 完成，返回true |
| isfinite、对应同步 | 完成 |
| all归约、归约同步 | 完成 |
| item标量读回、最后同步 | 完成，有限性true |
| 结果原子保存、worker完成 | 完成，退出0 |

17个调用边界均有BEGIN/CALL_RETURNED/END，共56阶段记录，无未完成调用/ERROR/半行。Mac外部镜像收到89条记录，其中56阶段记录与服务器逐条解析完全相同；镜像38,052字节。worker_result SHA独立比对通过，详见`gpu_operator_split_evidence_20261011_E26.json`。调用时间包含首次库启动和逐边界fsync，不能当纯GPU性能基准。

登记及运行15条健康记录CPU最高47/GPU33摄氏度，最低可用RAM31,289,495,552字节（29.14GiB），没有热停/内存保护。运行后复核CPU50°C不计入运行采样最高值。短操作跨越5秒采样间隔，不能排除漏采峰值。14次内核查询成功，期间均`-- No entries --`，表示没有收到测试期间内核条目，不保证突发掉电时日志能被保存。

只读版本核验驱动595.84（open kernel module），PyTorch2.8.0+cu128/CUDA12.8/RTX3090；未更改版本。本轮只说明一次1024同步操作及60秒停载观察没有复现E24重启。E24运行了连续30秒计划，本轮仅一次；新增fsync、CUDA_LAUNCH_BLOCKING=1、显式同步改变时序，不能把差异归因于某一个改动，不能排除软件或硬件，不能据此恢复A3。

## 保存与同步状态

登记发布已核验。最终README/索引/current state与结果、证据小清单按载荷审计后scoped推送main，最终提交号以Git历史及远端核验为准；原始日志LOCAL_ONLY，Mac只保留小记录。原hp3090/yuanqi-a3保持PAUSED、日研不变，本次unit已结束，无自动重试。

## 下一步与恢复入口

E26停止于既定结束点，不重复启动。下一独立实验应保留同样日志及负载次数，只切换CUDA_LAUNCH_BLOCKING以对照时序；随后才考虑延长普通矩阵循环至原30秒，避免同步方式与时长同时变化。本阶段未执行这些分支。恢复先读本README、state与phases，绝不重置已有实验，不跳到A3整链训练。
