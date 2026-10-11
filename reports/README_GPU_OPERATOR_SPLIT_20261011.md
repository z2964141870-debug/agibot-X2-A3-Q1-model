# GPU首轮调用拆分 E26

记录日期：2026-10-11。状态：准备登记，尚未启动GPU测试。

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
- 实际测试与启动结果待后文更新，不预记成功。

## 结果与边界

预检确认boot c7805e85-7467-4691-9037-b805ccae131f、GPU无计算进程、CPU47/GPU30摄氏度，可用RAM约29.6GiB。当前未开始E26运算。只能用最后完整调用和首个未完成边界缩小区间，不能由单次调用日志判定物理元件或驱动根因。

## 保存与同步状态

小型代码/README/登记清单按载荷审计后scoped推送main并核验；原始日志LOCAL_ONLY，Mac只保留小记录。原hp3090/yuanqi-a3保持PAUSED、日研不变，本次短测直接监测，无自动重试。

## 下一步与恢复入口

完成无CUDA安全契约检查、正式登记与发布后，仅启动一次E26。首次异常停止全部分支；成功则记录“单次同步诊断未复现”，不宣称长期稳定或A3可继续。恢复先读本README、state与phases，绝不重置已有实验。
