# A3 重启后恢复 E20 短试验

记录日期：2026-10-11。状态：E21两组200及44保存点核验通过；GPU回放再次重启后，用户明确暂停全部训练与评测，转[两小时空闲观察](README_A3_IDLE_WATCH_20261011.md)。前序入口：[E20/E21合同](README_A3_CONTROLLER_ADAPTER_20261010.md)。

## 目标与范围

用户通知电脑已重启，沿已授权工程主线恢复。只登记重启和检查产物，不开展重启根因排查、不追加长训。旧R01-R09/E10不修改；E21仅准备，暂不执行。

## 已完成内容

- SSH初次超时，随后恢复；实际boot `68093698-a432-4729-8e94-6bd02d40125d`，初次读到uptime66.04秒。与前次已确认boot不同。
- GPU温度31°C、显存77MiB、利用率0%；无E20服务在运行。RAM可用约30GB。这些是恢复时快照，非长期稳定证据。
- E20第二次启动实际保存角度MSE step200、latent step2；此前UNKNOWN更新为有保存进度。账本running为旧boot持久字段，不等于当前进程运行。
- 新增独立 `verify_adapter_recovery.py`：核验job登记的7文件SHA、训练数据大小/SHA/有限性、最新PT大小/SHA/CPU重载/模型和优化器有限性、连续更新计数、采样器状态，以及NPZ导出与PT参数一致。
- 2026-10-11 10:11:38北京时间核验通过。MSE step200：523913bytes，SHA `ff85ca4c1a4a36bce6141b6189037e86d2d80d8e99f7bbd645b02680042f2129`；latent step2：517513bytes，SHA `f84f76c74479a3f4ff02756474dded0ac049d137dc3922f148310d0d9ddb0694`。
- 训练窗口21,878条，数据SHA `831ced8193c19079c968d28226d42f1448958e67de8392049e56fd57ca65e820`保持。
- 原登记7源码已在服务器归档 `data/experiments/a3_controller_adapter_20261010_E20/registered_source_20261011.tar.gz`，SHA `e1864ed58b30f89c3a844932b51f6b4a1c97d7d68436bc0430edf21a5a3ace7d`。live源码与job逐项核验通过，未用本地E21新版覆盖。

## 决定与保护

第二启动有保存进度，因此先尝试继续E20原GPU条件。第三启动完成前置短阶段后，在latent200阶段又确认boot变化：`68093698-a432-4729-8e94-6bd02d40125d` → `3477d824-73f0-4216-9dfe-919d3d41f028`；最新sidecar仍MSE200/latent2，无新增保存。最后可读health为CPU42/GPU31°C，只登记事件，不据此归因或排除热问题。

按预案搁置E20 GPU分支，保留job/state及原7文件归档。state第三attempt仍running是旧boot持久字段，当前无该作业；不解除blocked，不修改历史起点或预算。新增独立E21 CPU两组从0至各200，CPU重新生成teacher标签，不沿用E20优化器/计数。原16训练动作、两档噪声、架构/超参/预算与效果门槛保持；GPU/CPU分支不能混合成目标损失的公平对照。

达到新boot后300秒、检查GPU空闲再启动E21。官方policy冻结，保留CPU90/GPU85连续2次5秒、CPUQuota100%、无保存进展2次及异常停止保护。MuJoCo仍按原入口串行GPU前向；若再次卡住，登记受影响项继续独立工作。

## 版本、命令与产物

服务器 `/media/yu/FAFF-E9771/YUANQI_A3`，main起点 `e9e5a877e1a328b8b696ba89c6c51480d8e0ae65`。登记源码版本以E20/job.json内SHA为准，vendor只读。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.verify_adapter_recovery
systemd-run --user --unit=yuanqi-a3-controller-adapter-e20-resume03 --property=CPUQuota=100% --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI_A3 --property=RemainAfterExit=yes --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 /media/yu/FAFF-E9771/YUANQI_A3/data/environments/a3-sonic/bin/python -m script.a3.run_controller_adapter run
```

2026-10-11 10:14:49北京时间启动前核验：uptime427.64秒、CPU40°C、GPU31°C、GPU空闲；systemd-run已返回 `Running as unit: yuanqi-a3-controller-adapter-e20-resume03.service`。这是启动确认，后续运行/保存进度以state和产物为准。

回执 `data/experiments/a3_controller_adapter_20261010_E20/restart_verification_20261011.json`；训练/完整动作进度 `state.json`；执行日志 `logs/a3_controller_adapter_20261010_E20/`。模型、窗口、归档均LOCAL_ONLY。

E21 CPU入口已实际传入服务器并注册，9项参考/旋转/残差契约检查全部通过；只允许固定E20/E21编号。训练数据和teacher标签由CPU重算，零初始化，预算各200；原E20代码以归档和前序Git提交保留。独立回放验证器改为从job读取实验编号，兼容归档旧入口；不改变实际策略输入或效果门槛。

```bash
A3_ADAPTER_EXPERIMENT=E21 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.run_controller_adapter register
systemd-run --user --unit=yuanqi-a3-controller-adapter-e21 --property=CPUQuota=100% --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI_A3 --property=RemainAfterExit=yes --setenv=A3_ADAPTER_EXPERIMENT=E21 --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 /media/yu/FAFF-E9771/YUANQI_A3/data/environments/a3-sonic/bin/python -m script.a3.run_controller_adapter run
A3_ADAPTER_EXPERIMENT=E21 OMP_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.verify_adapter_rollout
```

E21产物 `data/experiments/a3_controller_adapter_20261010_E21/`，日志 `logs/a3_controller_adapter_20261010_E21/`。实际已下发 `yuanqi-a3-controller-adapter-e21.service`；第一启动后boot `3477d824-73f0-4216-9dfe-919d3d41f028` → `dcb62ac7-6400-4c77-890d-ab8237ba8409`，未据服务inactive/ExecMainStatus0认定作业完成。

重启后发现最终CPU标签文件已保存：21,878窗口、60,763,817bytes，SHA `97786ab18a33dcbfb760f242520d3379f17d5fd6f2bbef1308d6edcc59f21ed7`。2026-10-11 10:26:23北京时间，`verify_adapter_recovery --experiment E21 --dataset-only`独立检查源码登记、数据大小/SHA/各数值字段有限性通过；尚无模型保存点，这不是teacher标签精度或模型效果通过。第一启动有已保存数据进度，冷却后原入口跳过标签准备继续。

E21源码/合同/阶段更新共9文件审计通过，提交 `1c678379d005e7a3bc24a6959099921bcb5f3c46`已推送main并比对远端一致。此后状态/数据核验更新待下一次同步。

### E21恢复与训练核验结果

`yuanqi-a3-controller-adapter-e21-resume02.service`实际启动，原账本登记第一次中断且已有数据进度，no_progress保持0。新boot dcb62ac7下两组分别完成从0→2冒烟、完整恢复至各200；CPUQuotaPerSecUSec=1s已读取确认，官方policy更新0。

`independent_verification.json`返回passed：44个PT保存点及各自NPZ导出核验、21,878实际训练窗口、96个CPU teacher标签独立重算、训练源与16/4划分核对通过。角度训练最后batch loss0.0013596158241853118；latent最后batch loss0.003380390116944909；两种损失单位不同，不能直接比较大小判断优劣。

此后GPU闭环阶段再次boot变化dcb62ac7→cb282100。最新持久manifest确认MSE/noise0下001/029/048三个完整动作passed，050原地踏步仍旧boot running；其余条件、效果门槛/实际输入复算未完成。用户10/11明确停止全部训练和评测，禁止继续此前恢复安排；已验证模型保留，未来恢复需用户新指示。

## 当前结果与下一步

E21训练与保存链路已通过。16完整回放、实际输入及效果门槛仍待完成；不能据训练进度宣称提升。四动作是重复诊断集，非独立最终验收。

当前只进行固定两小时空闲观察。用户以后重新授权计算时，跳过已验证完成的E21训练，从未完成的完整动作回放及实际输入/指标复算继续，读取固定门槛结果；不盲目增加预算。连续两次无保存进度停止保护保留。E20恢复必须使用归档内原登记源码，不能用新增E21配置回写旧job。

## 保存与同步

原登记源码/恢复验证脚本/恢复前README共13文件审计通过，提交 `a2a759b42693d157351880611e062c6625eb836c` 已推送main，HEAD与refs/heads/main一致。此启动后更新及后续结果仍待同步。本地学习/研究并行文件不覆盖服务器入口。
