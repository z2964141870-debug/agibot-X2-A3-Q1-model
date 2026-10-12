# A3 训练包装器与梯度模式回放

记录日期：2026-10-12。独立实验 A3_ZHY_WRAPPER_20261012_E07。状态：预登记完成、2项CPU测试通过，回放尚未执行。

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

新机2项CPU输入证据测试通过（0.004秒），可以区分复制输入/值变更并拒绝维度错位；实际回放尚未运行。若重建包装器不复现真实偏离，不能据此判定真实训练路径已修复；下一项要在实际训练入口捕获输入/精度/前向差异。E05真实合同失败仍保留，不能解释全部历史微调退化或旧机硬件故障，也没有策略收益证据。

## 保存与下一步

CPU测试完成后立即保存预登记/索引/current state，scoped审计推送main并核验远端，之后登记运行。结果形成后立即记录，依据实测选择下一变量，不开始长PPO或操作机器人。
