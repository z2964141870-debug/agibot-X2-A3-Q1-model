# A3持久化细日志诊断版本

记录日期：2026-10-11。状态：实现与5项无CUDA契约通过；原定E25短试跑未登记/未启动，因普通GPU对照E24确认新boot，按用户首故障停止约定暂停计算。

## 目标与最新授权

用户要求在A3训练中通过“开始/完成”日志找出重启触发区间。本轮构建独立诊断入口，保留原训练目标和官方初始化，不改vendor共享源码、不恢复旧E22/R01-R10、不启动长训或效果评测。不与E24计算并行。

## 完成实现

- `script/a3/phase_trace.py`：纯标准库日志，UTC/monotonic_ns/pid/thread/boot、trainer_global_step/optimizer序号、调用ID及父调用ID。每边界O_APPEND写入、fsync文件，首条再fsync目录。按进程分文件，避免混写；不保存输入张量或凭证。
- 每span依次BEGIN、PRE_SYNC_BEGIN/END、CALL_RETURNED、POST_SYNC_BEGIN/END、END。只有已初始化的CUDA才同步，不因记录主动初始化CUDA。异常留下ERROR且不写END；因此“Python返回”和“GPU同步完成”可分辨。缺失END、嵌套区间、损坏末行都保存于只读汇总，不把缺失当作异常已捕获。
- `traced_entry.py`在内存编译固定上游入口，只给Isaac App/环境入口加装饰器，保留源文件位置供Hydra找config。导入钩子覆盖SimulationContext创建/step、InteractiveScene创建/动作写入/状态更新、ManagerBasedRLEnv创建/step、ManagerEnvWrapper/reset/step及MotionLibBase创建/加载。记录实际HOOKS_READY，并启用faulthandler独立文件；整机掉电/复位通常不能期待Python异常处理完成，仍以最后持久边界为准。
- `traced_training.py`继承已验证trainer，记录初始化、官方权重加载、rollout/policy、batch读取、前向、loss、returns、backward、梯度裁剪、optimizer/zero_grad、统计/采样同步。保存callback记录checkpoint整体与torch.save/load、校验、SHA。所有包装只转发既有参数/返回值，诊断同步/fsync会显著改变时序，不能据此声称位级重现或效果改善。
- `run_phase_trace.py`预置独立E25：官方035、16env、4 minibatches、epochs1、lr尺度0.1、seed0、2更新，每更新保存。标准库supervisor先等E24全部15项通过、同boot、结果SHA及GPU空闲，再允许一次启动；900秒训练上限、5秒90/85°C连续2次、4GiB RAM、遥测缺失/内核异常停止，无开机恢复。登记保存源码/外部Isaac/官方PT/生成launcher SHA，生成副本只替换入口，不改vendor文件。

## 实际验证与限制

`/usr/bin/python3 -m unittest script.a3.test_phase_trace`：5项通过，覆盖同步顺序及元数据冲突、错误保留未完成、同步失败不得记完成、嵌套及末行截断、AST包装保留原返回。源码静态核验上述5模块类/方法存在，并核验两个IsaacLab路径指向同一项目；测试没有加载Torch/Isaac或A3权重，不是实际训练入口通过。

尚未实际运行Hydra/导入钩子/训练/保存新PT，原定E25目录及job没有创建。E24在1024矩阵首轮操作区间出现新boot，故停止全部后续，没有用整链A3再碰一次故障。CPU日志契约通过不能替代实际GPU验证；诊断入口运行前仍须核验HOOKS_READY和固定版本。

## 路径与恢复

代码和本README可供后续授权的诊断使用。原定E25输出`data/experiments/a3_phase_trace_20261011_E25/`、`data/training/a3_phase_trace_20261011_E25/`与`logs/a3_phase_trace_20261011_E25/`均尚未启动生成；无新模型可验收。现有register会拒绝E24 STOPPED_REBOOT，因此不能照旧命令启动，后续需根据新的排查方案另行编号与登记。不要重置E24来绕过此门槛。

只读边界汇总API：`script.a3.phase_trace.summarize_files(directory)`；它不导入torch或模型。记录解释先找到最内层未完成span及其最后事件，再结合独立健康/内核/boot，不把最后一行直接认定为软件错误或某块硬件损坏。

## 保存与结论

关键代码、README、索引/current state及E24停止小清单经diff审阅、Git载荷审计后scoped提交推送main并核验远端；原始日志服务器LOCAL_ONLY。停止的hp3090与yuanqi-a3 heartbeat保持PAUSED，日研不变。

本次持久标记已经在普通GPU分支留下比控制台更明确的区间：CUDA初始化/分配完成后、第一次1024矩阵操作完成之前。说明该取证办法有用，但A3日志版本未执行，真正根因仍UNKNOWN。证据见[普通GPU停止记录](README_A3_ISOLATION_20261011.md)。
