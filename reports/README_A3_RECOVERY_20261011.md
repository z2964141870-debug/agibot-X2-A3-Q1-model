# A3 重启后恢复 E20 短试验

记录日期：2026-10-11。状态：恢复完整性通过，剩余执行待启动。前序入口：[E20/E21合同](README_A3_CONTROLLER_ADAPTER_20261010.md)。

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

第二启动有保存进度，继续E20原GPU条件；不改为CPU混合对照。达到boot后300秒、检查无其他GPU进程后，由原监督入口自动登记旧attempt中断并恢复最新保存状态。两组各累计200更新，官方policy冻结。保留CPU90/GPU85连续2次5秒、CPUQuota100%、无保存进展2次及异常停止保护；不自动解除blocked。

## 版本、命令与产物

服务器 `/media/yu/FAFF-E9771/YUANQI_A3`，main起点 `e9e5a877e1a328b8b696ba89c6c51480d8e0ae65`。登记源码版本以E20/job.json内SHA为准，vendor只读。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.verify_adapter_recovery
systemd-run --user --unit=yuanqi-a3-controller-adapter-e20-resume03 --property=CPUQuota=100% --property=WorkingDirectory=/media/yu/FAFF-E9771/YUANQI_A3 --property=RemainAfterExit=yes --setenv=OMP_NUM_THREADS=1 --setenv=MKL_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 /media/yu/FAFF-E9771/YUANQI_A3/data/environments/a3-sonic/bin/python -m script.a3.run_controller_adapter run
```

启动命令仍待实际确认，不能仅凭命令存在记为已启动。

回执 `data/experiments/a3_controller_adapter_20261010_E20/restart_verification_20261011.json`；训练/完整动作进度 `state.json`；执行日志 `logs/a3_controller_adapter_20261010_E20/`。模型、窗口、归档均LOCAL_ONLY。

## 当前结果与下一步

只证明保存状态可恢复。latent200、44保存点完整独立核验、16完整回放、实际输入及效果门槛仍待完成；不能据MSE200训练进度宣称提升。四动作是重复诊断集，非独立最终验收。

恢复顺序：原监督入口完成剩余latent更新和内部核验/闭环；再独立复算实际输入与指标，读取固定门槛结果。失败项保存，不盲目增加预算。若重启继续，核验保存进度后恢复；连续两次无保存进度停止保留。

## 保存与同步

本记录已保存；同步提交待执行，未经远端核验不记为完成。服务器旧的工程未提交修改属于本阶段前序源码，核验后scoped提交；本地学习/研究并行文件不覆盖服务器入口。
