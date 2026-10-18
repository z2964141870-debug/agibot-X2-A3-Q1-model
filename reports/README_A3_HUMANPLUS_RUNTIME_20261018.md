# humanplus独立运行验收HP05

记录：2026-10-18北京时间10:39执行、10:40定位。阶段A3_HUMANPLUS_RUNTIME_20261018_HP05，状态STOPPED_ENVIRONMENT_BOOTSTRAP_EULA_EOF。4项CPU/fused CUDA事务控制和固定MJCF无策略10步通过；PhysX在许可提示读取EOF处退出、实际0步。此阶段不能记为完整环境通过、035基线或训练完成。

## 目标与范围

HP04已将63项输入逐文件大小/SHA验收、训练必需10/10。本阶段在humanplus@100.101.30.68的/home/humanplus/YUANQI_A3独立登记，候选/home/humanplus/anaconda3/envs/x2-sonic-isaaclab按只读使用。旧ZHY运行器、STOPPED账本、v1内存口径不复用；固定vendor为fe6868ba37034f89b912f0fb851bce19f120266d。

## 实现与决定

新增humanplus_supervisor.py，以已验humanplus_health的v2内存和全boot journal核验为基础，CPU固定读取x86_pkg_temp、GPU UUID/温度/利用率/计算PID及可见进程元数据。不可读即拒绝，退出中的已消失PID不冒充不可读活进程。启动三次、相隔5秒，可用>=8GiB、GPU利用率0、无其他Python/kit/isaac/stress/train或RSS>1GiB及GPU计算；运行保留4GiB、CPU90/GPU85连续两次5秒即停止，每阶段900秒。遥测在独立标准库子进程读取，单次4秒截止，避免日志或NVML卡住监控。只终止由本阶段start_new_session创建的进程组，异常不继续下一阶段，不恢复/覆盖终态目录。

新增humanplus_runtime.py，独立登记HP05代码、环境源文件、输入小清单、固定XML及96网格SHA和预算，再一次顺序执行tensor、mujoco、physx。tensor复用4项有限Adam事务控制，最多20原生optimizer调用、仅双元素合成参数；mujoco固定MJCF无策略10步；physx按本机IsaacLab的create_empty示例启动CUDA空场景10步。每阶段先重新做三采样启动门。空场景通过不证明机器人USD/训练资产闭包，10步MuJoCo不证明平衡。

无035策略模型加载、机器人策略rollout或训练optimizer；存在本次小张量/合成optimizer和MuJoCo机器人模型加载10步，不再声称0张量/0仿真。不改共享环境、驱动/功率/BIOS/内核权限，不清缓存，不停止其他任务。保留此前只读两采样收据，不复跑该计划。

## 版本、命令与证据

- 标准库测试：python3 -B -m unittest script.a3.test_humanplus_supervisor -v。Mac及新机各13项通过，覆盖其他计算/忙GPU、温度连续阈值、8/4GiB、OOM/内核、遥测丢失/截止、只停自有组、预检不启动/终态不可覆盖、运行中传感器失败和900秒期限。
- 登记：/usr/bin/python3 -B -m script.a3.humanplus_runtime register，在新根运行；登记存在即拒绝覆盖。运行：/usr/bin/python3 -B -m script.a3.humanplus_runtime run，只能执行一次，不能对已执行HP05复跑。
- 固定计划data/manifests/a3_humanplus_runtime_20261018_HP05_plan.json；结果同前缀_result.json。
- 终态独立审计同前缀_audit.json，由script/project_tools/humanplus_runtime_audit.py只读生成，含实际三个stage终态、代码未变、所有小日志/健康收据大小SHA和停止后健康；未改原plan/result/state。
- 登记及新机基准data/experiments/a3_humanplus_runtime_20261018_HP05/；原始分阶段health.jsonl、console.log、state.json、result.json位于logs/a3_humanplus_runtime_20261018_HP05/{tensor,mujoco,physx}/。
- 本次首次现场同boot afb6796b-3883-4df4-99ea-bc381c52ea13、可用约21GiB、GPU无计算、journal1410条无FATAL/新增OOM；此现场不冒充三采样验收。

## 结果与边界

HP05单次登记、单次run。tensor和mujoco各15.19/15.17秒完成；torch2.8.0+cu128、CUDA12.8、RTX5060能力12.0，4事务控制通过、CUDA allocator峰值12,288字节。MuJoCo3.3.7加载固定MJCF，nq72/nv65，10步合计0.010秒、状态有限。这些控制不证明完整策略加载/训练资产或闭环效果。

physx三采样门通过后，AppLauncher内层Kit导入提示EULA，stdin EOF使worker exit1；未进入SimulationContext、0PhysX步。新入口漏配既有Isaac Sim非交互bootstrap变量是适配遗漏，不是已证实机器故障。现有script/a3/fresh_batch_trial.py的worker使用OMNI_KIT_ACCEPT_EULA=YES；本次不临时写共享环境、改原登记代码或对停止HP05复跑。最小修正方案是经明确解除此次停止约束后，仅独立登记未完成的PhysX环境验收，并将启动变量写入新计划；已通过tensor/MuJoCo不重跑。不得借此恢复旧E44/E45/E49或伪造HP05通过。

12份监督采样同boot afb6796b-3883-4df4-99ea-bc381c52ea13，CPU最高81°C/GPU42°C，可用最小22,350,381,056字节，0FATAL/新增OOM，单采样最长0.122秒。停止后再次比较原基准无新健康问题，无GPU计算进程、记录的controller/worker均已退出。没有发现机器温度/OOM/内核异常；journal只证明实际可读记录。HP05已STOPPED，不自动续跑/重试或扩预算，zhy-a3按异常停止规则暂停；SSH失联仍只记UNKNOWN。

## 保存与恢复

原始日志/数据LOCAL_ONLY，无已核验云备份。代码/计划/结果/README按scoped审计提交；GitHub原SSH写权限未恢复，沿用LOCAL_COMMITTED_UNPUSHED，不重复同一失败推送。保留Mac并行研究记录及原未推送提交。

当前具体恢复入口是上节PhysX启动变量修正方案；先处理此次STOPPED约束，不能直接执行HP05 run/worker。未完成完整环境验收，因此新035基线/8更新训练均未登记或执行。完整验收后才独立登记六衣服/四官方035匹配基线，再原固定8更新及仅最终8的10项完整效果评测。输入迁移及旧停止分支不复跑，启动仍查实时资源。恢复先读本页、README_A3_HUMANPLUS_TELEMETRY_20261018.md及README_CURRENT_STATE.md。
