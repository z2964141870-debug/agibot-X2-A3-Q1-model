# GPU调用时序对照 E27–E29

记录日期：2026-10-11。状态：代码准备，尚未登记/运行新GPU任务。

## 目标与范围

用户“继续找凶手”授权继续普通GPU隔离排查。E24普通1024首操作附近重启，E26一次同步诊断通过，不能直接归因。此次固定三个独立编号，逐项完成保存后才登记下一项，首异常/新boot停止整链；无A3/Isaac/RL/模型加载，不改驱动、功率或BIOS，不复用历史job。

| 实验 | 与前项相比的主要改变 | 固定配置 |
| --- | --- | --- |
| E27 | E26的CUDA_LAUNCH_BLOCKING从1变0 | 一次1024 float32随机矩阵×单位矩阵、seed0、TF32关闭、全部额外同步与细日志 |
| E28 | 单次改为30秒循环 | blocking=0、额外同步与细日志保留，占空比目标25%、最少10ms sleep |
| E29 | 移除四个中间显式同步 | 同E28循环与校验，分配和每轮末尾同步保留，趋近E24原操作序列 |

E27单次模式计算路径与E26相同，新增循环计数/计时和固定比较入口有轻微CPU记录差异；不宣称两进程二进制完全相同。E28引入循环/间歇，不能把成功差异只解释为时长。E29仍保留逐调用fsync及细分finite.all/item，时序仍与E24不同，成功不排除原问题。每条BEGIN/CALL_RETURNED/END记录只表明相应调用边界，异步matmul返回不代表GPU完成；E29的allclose内部还会读回布尔值，不能细分其内部CUDA调用。

## 完成内容与关键决定

新比较入口绑定固定case、原boot、前项COMPLETED且无ERROR/未完成边界、5源码SHA。READY一次消费；所有unit非自启Restart=no，RuntimeMaxSec240、CPUQuota100%、MemoryMax4G，worker最长120秒。每项结束后60秒独立停载监测，后续登记重新核验GPU无计算进程/温度安全。CPU90/GPU85°C连续两次5秒采样、RAM保留4GiB；遥测/内核不可读、关键内核事件、计算异常、超时均停止。未知SSH失联不当作重启，不重复启动。

Mac只读SSH镜像每秒复制阶段/健康/内核记录，前1MiB逐条fsync，最后500条按收到的块fsync并替换小型滚动文件；完整原始日志留服务器、不入Git。短E27能完整复制，循环分支可能仅保留前缀与最后窗口，必须按实际覆盖报告，不宣称完整镜像。前项报告及清单先发布再进入下一项，避免仅在聊天中留中间结果。旧hp3090/yuanqi-a3仍PAUSED，日研不变，不需要开机恢复或新heartbeat。

## 版本、命令与证据

- 主机：ssh hp3090；根`/media/yu/FAFF-E9771/YUANQI_A3`。
- 代码：`script/diagnostics/gpu_operator_compare.py`、`collect_gpu_operator_compare.py`；不修改E26执行版本。
- 现有环境：PyTorch2.8.0+cu128/CUDA12.8/驱动595.84 open module，版本以每次实际日志为准。
- 登记：`/usr/bin/python3 -m script.diagnostics.gpu_operator_compare E27 register`；E28/E29替换case，必须前项通过，无重复/重置。
- 监督运行：同模块`<case> run`；只读镜像`<case> watch`；只读遥测总结`<case> summarize`。
- job/state/results：`data/experiments/gpu_operator_compare_20261011_<case>/`。
- 原始日志：`logs/gpu_operator_compare_20261011_<case>/`，LOCAL_ONLY。
- 小清单：`data/manifests/gpu_operator_compare_{registration,result,evidence}_20261011_<case>.json`。

## 实际验证与结论边界

开始前确认原boot c7805e85，CPU48/GPU30°C、无GPU计算、可用RAM约29GiB；Git main干净。新case尚未登记/执行，不能预记成功。后续实测记录追加于本README。

## 保存、同步与下一步

代码与预登记报告经diff/载荷审计后scoped提交推main并核验远端；大日志LOCAL_ONLY。先完成E27并发布，再根据实际结果进入E28、E29。首次故障即停，保存最后完成/未完成调用、确认boot区间和未做分支，禁止自动继续A3。
