# A3 训练包装器与梯度模式回放

记录日期：2026-10-12。独立实验 A3_ZHY_WRAPPER_20261012_E07。状态：6组回放完成并独立核验，重建包装器未复现真实训练偏差；0更新。

## 目标与依据

E06证明同配置/权重的直接4环境整段与逐帧前向均复现采样；真实训练mean仍相差RMSE0.010159。此轮只重放E05已验证捕获，比较直接Actor、原PPO包装器和Accelerate包装后的前向；分别关闭/开启梯度，共6变体，无环境、rollout、loss.backward或optimizer。

静态证据：固定vendor `train_a3_035_fromscratch.sh:242`显式 `--mixed_precision=no`；不能仅因存在Accelerate就认定BF16。使用实际runtime配置重建Actor/Critic，strict加载官方两网络；CPU16环境重建必须通过RMSE1e-5/ratio偏离0.001原门槛。

## 固定合同与实现

新机 `/opt/data/private/hechengzhu/ZHY`，官方035、E05的16×24输入和相同4组环境。通过原 `TRLPPOTrainer._get_mb_rollout_data`切片和 `_forward_model`调用包装器，逐键对比切片与Actor实际输入；记录dtype、device、shape、梯度/自动混合精度及TF32实际状态。所有TF32保持false，Accelerator指定mixed_precision=no，seed0，所有参数和Actor buffer摘要必须不变。实际runtime Critic启用running_mean_std，其buffer可随正常训练前向变化，单独记录全包装器buffer摘要；这不等于optimizer更新，不以此误判Actor权重变化。

入口 `script/a3/zhy_wrapper_probe.py`，输入证据测试 `test_zhy_wrapper_probe.py`。固定代码/模型/捕获SHA；READY同boot启动、worker只由monitor运行；CPU90/GPU85摄氏度连续2次5秒、内存保留4GiB、900秒上限、OOM/新boot/关键内核错误/遥测缺失停止；不重置、重复或开机恢复，计算串行，原机停止。

```bash
cd /opt/data/private/hechengzhu/ZHY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_wrapper_probe -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_wrapper_probe register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_wrapper_probe run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_wrapper_probe verify
```

## 产物与结果边界

输出 `data/experiments/a3_zhy_wrapper_20261012_E07/`，日志 `logs/a3_zhy_wrapper_20261012_E07/`，小型清单 `data/manifests/a3_zhy_wrapper_20261012_E07.json`。张量、模型与原始日志保持新机LOCAL_ONLY，Mac/Git只保留小型代码/报告/清单。

新机2项CPU输入证据测试通过（0.004秒），可以区分复制输入/值变更并拒绝维度错位。实际北京时间14:15:46–14:16:02完成，6组概率指标和实际首批差值独立CPU重载复算通过，state=VERIFIED，0optimizer。

| 路径 | 梯度模式 | mean RMSE对采样 | ratio最大偏离1 | 对真实首训练mean RMSE |
| --- | --- | --- | --- | --- |
| 直接Actor | 关/开 | 4.05e-7 | 1.72e-5 | 0.010159 |
| 原PPO包装器 | 关/开 | 4.05e-7 | 1.72e-5 | 0.010159 |
| Accelerate包装器 | 关/开 | 4.05e-7 | 1.72e-5 | 0.010159 |

24组切片与24次Actor真实前向输入逐键完全一致，dtype=float32；autocast=false、matmul/cudnn TF32=false、Accelerator mixed_precision=no。所有模型参数与Actor buffer摘要前后一致；Critic正常训练归一化统计buffer变化，已单独登记。六变体mean全部相同，原概率门槛内，未复现E05真实首训练的偏差。

**重建包装器和梯度模式均未复现真实偏差，不能据此宣布真实训练路径已修复。** 首要待查的是实际训练前向收到的输入/元数据及运行状态与离线重建是否一致。E04/E05只确认采样输入与存储一致，没有在真实首batch的Actor入口逐键核验，不能据此排除训练阶段输入变化。下一项需在真实入口保存首batch数据、参数/buffer/模型元数据/精度与调用选项，并在同一Actor同一小批次即时对照；仍0更新。

4健康样本同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`、CPU最高40/GPU30摄氏度、可用估计最低8.49GiB、OOM0、无新关键内核错误/遥测缺失。运行版本 `10630193ef6ed588acc3bfa359b43789911123e5`，捕获SHA `f1759c3e64df31ba54896584787d2861b6fc1d0740ef09e1eb5ef11dd589a133`；输入、调用记录和摘要详见小型清单。任务已结束，随后只读查询GPU无计算进程。

E05真实合同失败仍保留，不能解释全部历史微调退化或旧机硬件故障，也没有策略收益证据。Accelerate曾输出宿主kernel5.4低于建议5.5的告警，本轮未挂起；记录该限制，不把它当作已验证原因，也不修改宿主内核。

## 保存与下一步

结果已保存本README/索引/current state与小型SHA清单，按main scoped审计推送并核验远端；模型/张量/原始日志LOCAL_ONLY。下一项为独立真实入口零更新捕获，补足训练输入与即时前向证据，不复跑离线6变体、不开始长PPO或操作机器人。
