# Q1双线逐子步观测

日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SUBSTEP_OBSERVATION。已授权的底座与PPO各一次观测均正常完成，各3180控制/12720子步，合计25440子步/7950策略批，0optimizer；CPU保存分析准备完成、实际分析待执行。普通自然步态目标仍未达，goal ACTIVE，采纳null。

## 实际观测结果

底座北京时间09:51:39至09:56:19、controller249353/start23874444与worker249360/start23874588；PPO10:34:33至10:39:23、controller253131与worker253138/start24131963。两guard均WORKER_COMPLETED/reap0；尝试/返回/核验完成数各12720，12片各完整/无提前跌倒，每路线3180控制未超界。底座3180策略批，PPO4770策略批，0优化更新。原四进程均已现场核不存在，10:43:46核原boot/GPU、15MiB/0%/无计算/OOM0。

两路线各6组配对已记录初态、首个策略输入、系数与RNG全部逐值一致，仍非完整求解状态等价或历史330环境重放。底座完成后临时本机18873及双master消失导致UNKNOWN，原启动会话返回0；按用户要求加载既有hp_han助手恢复Windows路由/原25386，取得终态后才启动PPO。没有恢复或重做底座，没有改变认证配置/服务器环境。缓存原工具一次正常低于8GiB阈值跳过，未删文件或修改维护任务。

14小工件654236字节已取回Mac并核三端大小SHA；36份原始NPZ/路线与检查点仍留服务器。8项新CPU分析测试检查原生时轴/状态连续性、子步计数、执行器ID映射/PD力重建、空条件与同刻低力mask；不创建策略或仿真。实际只读分析待执行，不能预称已隔离足部问题。执行与工件见[清单](../data/manifests/supermimic_q1_gait_goal_substep_observation_results_20261030.json)。下文“尚未启动”为授权登记当时历史。

## 授权与固定范围

[授权记录](../data/manifests/supermimic_q1_gait_goal_substep_authorization_20261030.json)对应原问题中底座与PPO各一次零优化观测。执行[已发布V02 plan](../data/manifests/supermimic_q1_gait_goal_substep_observation_plan_v02_20261030.json)，86121字节、282来源、SHA-256 c94cf9489c317ea6a2ca431af898bc4f8704e698067c5e927d17a389b1234942，文件不重新生成。原问题13测试为V01；V02只修正异步返回/核验完成和失败重复累计，17测试已通过，case、检查点、种子与全部上限相同。

底座比较原initial与301秒actor；PPO在同一原冻结教师上比较同600秒head的零/开状态，alpha.5、PD、限矩、参考均用plan冻结值。原固定停走320控制与起走210控制，各条件3次配对seed32000/32001/32002；每线12个单环境case，最多3180控制、12720子步、6360策略批调用；两线串行，合计最多6360控制、25440子步。每线1800秒停止阈值加10秒回收，不重试、不训练。图捕获、环境初始化/reset forward与rollout转移分列，不能当作训练。

唯一新根：data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_BASE_20261030及Q1_GAIT_GOAL01_SUBSTEP_PPO_20261030。只运行冻结入口probe_substeps.py supervise，不能修改plan或换名再做。若失败，保留异常、原始部分轨迹、尝试/返回/已核验数与实际资源回收证据。

## 发布与现场

先完成本阶段README、索引、授权及payload audit，推origin Q1并独立核40位远端，再经父版/当前内容守卫镜像Mac和25386并核大小SHA。发布旁收据为data/manifests/supermimic_q1_gait_goal_substep_authorization_publication_20261030.json；仅在实际完成后生成。

2026-10-30 09:45:08北京时间只读快照：原boot4d859008-4c83-4a6d-814d-cbb9dfe885c9、GPU-26028785-9fb9-45af-e614-4cc53af15d08，GPU15MiB/0%/无计算进程/OOM0；容器总5.18GB、cache4.31GB、RSS20.4MB，两个唯一观测根均不存在，V02 plan大小SHA一致。正式启动仍由原24GiB准入/30GiB守卫和独立GPU锁检查；快照不替代现场进程收据。

## 命令与时轴

```sh
substep_root=/opt/data/private/ZHY
"$substep_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$substep_root/script/supermimic_q1_hp/activate.py" \
  "$substep_root/script/supermimic_q1_gait_goal_analysis/probe_substeps.py" supervise \
  --plan "$substep_root/data/manifests/supermimic_q1_gait_goal_substep_observation_plan_v02_20261030.json" \
  --line base \
  --output "$substep_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_BASE_20261030" \
  --authorization-record "$substep_root/data/manifests/supermimic_q1_gait_goal_substep_authorization_20261030.json" \
  --code-commit PUBLISHED_40_HEX
```

PPO只将line改为ppo并用预登记PPO根，在底座退出并确认GPU释放后执行。实际代码提交、PID/start_ticks、guard返回和原始输出将在执行结果中补齐；当前不能称观测已完成。

每5ms保存pre/solve/post；当前后端先forward/solve再积分，实际力矩与接触、脚姿态属于积分前，post qpos/time属于后态。不能将不同时间点拼成因果关系，也不能将采样近限时长当连续饱和。每片保存首输入、模型系数、可用原生状态/RNG并比较；相等不证明完整求解状态等价，更不是历史330环境重放。

## 边界与下一步

大NPZ与权重留25386；后续仅取小JSON/CSV/图，核实际计数、关节力矩近限、接触/卸载和离地的时序，保留重复差异与失败。观测不能自动采纳checkpoint或新增训练；原底座301.1074845249823秒、PPO600.0709074319457秒与physical_regression结果保留，总901.178391956928秒不追加。原训练V02授权已用，当前是独立观测授权；goal ACTIVE、不结算，监督/缓存维护/其他自动任务及认证环境均保持，无agent。
