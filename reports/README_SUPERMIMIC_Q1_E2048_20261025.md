# 2048环境与两小时训练

记录日期：2026-10-25，北京时间。阶段：RF04-E2048-UP01。状态：用户已授权，配置分析与资源预检准备中，正式训练尚未启动。

## 目标与范围

用户要求停止64环境路线，分析后采用2048环境实际训练至少2小时，并只开一个子agent分析位移；主代理负责2048配置、训练和监控。新实验独立创建，旧RF03已停止不恢复。

保留上一已确认的365帧双臂抬起加移动参考`train_3419f09ba93c91d75fc1`，先检查单动作可学习性。全局目标仍是SuperMimic for Q1，世界模型、H32 BPTT与critic，不增加新的数据硬门或直接改奖励/观测。环境数扩大提高独立状态、参考时刻和随机化上下文的并行覆盖，不等于每环境噪声幅度变大。旧配置已有domain randomization、观测噪声和动作N(0,0.1²)，不能写成没有噪声；低并行覆盖是否是主要根因仍需新结果。

## 配置依据与计划

原论文arXiv2610.28480 Table A.5记每GPU4096环境、世界模型每组10批×2048个8步窗口、策略4096个H32想象起点、UTD0.4、每环境1000行replay、动作噪声N(0,0.1²)。本次选择2048环境、WM batch1024×10、策略batch2048，保留UTD0.4，使每真实转移的训练采样量按论文同比例缩放；不只修改环境数而保留原64/128小batch。原64环境设置对应WM/真实转移与想象起点/真实转移均为该比例约2倍，但并行覆盖小。

先做有限2048实例、真实MuJoCo采集及一次WM/actor/critic更新的profile，检查数值与实际CPU/GPU内存，记录吞吐；这是运行可行性检查，不是数据或策略性能门槛。原后端为CPU MuJoCo/复用模型槽，2048指独立状态和随机化上下文，模型槽数不等于环境数。现场CPU cgroup额度24核，GPU3090/24GiB，停止后共享内存约22.1GB/OOM14；线程/槽数按实际资源选择，保留30GiB内存、100GiB余盘、12GiB单实验保护。

正式训练记录实际起止与活跃训练秒数，目标至少7200秒，正常达到目标后在完整采样/更新边界停止；另有明确最大采集轮安全预算，不靠等待凑时长。保护停止或异常导致不足2小时必须如实记失败/未完成，不能算满足要求或自动重启旧实验。

## 产物与验证

统一根`huanyu:/opt/data/private/ZHY`，沿四目录。预检和正式训练分别用独立产物目录，正式根拟为`data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025/`。每轮照常保存六宫格训练图；固定本轮初始/最终MuJoCo和上肢/位移分析，视频保存本地MP4，不建立网站。Mac不拉权重。

唯一子agent只读分析位移，不改训练代码、输入和共享入口，不跑新仿真/训练，不再分派；结果由主代理整合。训练期间保持源码和配置冻结。

已实现`collection_budget.py`与训练入口的可选限时预算：在完成的25轮报告边界且采集阶段实耗至少7200秒后正常结束；初始化、绘图和评测不计入，最大采集轮12000若先耗尽则报错，不能凑时长。原按步数训练保持原行为。模型槽/线程和batch均显式记录。4项预算边界控制和4项实际终态/配对控制在Mac通过；首次预算测试因Mac Python3.9不支持立即求值的联合类型失败，已用延迟注解修正，再跑4/4通过，服务器待验证。

新增`script/project_tools/profile_q1_2048.py`：2048独立状态、8线程/8槽、16采集轮，完整WM1024×10及actor2048/H32一次更新；仅profile加载既有RF02网络以避免把尚未预热的模型数值问题冒充资源不足，正式训练仍从零。输出真实吞吐、上下文/参考时刻覆盖、GPU峰值、共享内存和数值结果；无权重副本，不做跟踪效果评测。外部guard保留资源及进程回收保护。原训练/绘图入口复用，未改WM/BPTT/奖励/控制公式。

```sh
PY=data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
$PY script/project_tools/profile_q1_2048.py --root /opt/data/private/ZHY --output data/supermimic_for_q1/E2048_PROFILE_P01_20261025 --threads 8 --model-slots 8 --expected-oom-kill 14
$PY script/project_tools/run_q1_single_motion.py --root /opt/data/private/ZHY --output data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025 --num-envs 2048 --threads 8 --model-slots 8 --wm-batch 1024 --batch 2048 --steps 12000 --training-seconds 7200 --dashboard-label RF04-E2048 --code-commit PUBLISHED_40_CHARACTER_SHA --expected-oom-kill 14 --detach
```

当前2048 profile及正式训练尚未开始；具体发布SHA、服务器验证、日志、GPU/内存峰值和实际结果继续补记。下一步发布核SHA后执行profile，确认运行条件再启动已授权正式训练。
