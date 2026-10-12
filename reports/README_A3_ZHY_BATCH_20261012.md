# A3 小批次与训练调用路径对照

记录日期：2026-10-12。独立实验 A3_ZHY_BATCH_20261012_E06。状态：预登记完成、2项CPU测试通过，回放尚未执行。

## 目标与范围

E05关闭采样与学习的TF32后，16环境整批和逐帧回放复现原采样，但实际4环境训练前向仍偏离，概率合同失败。此轮复用E05捕获，区分直接4环境计算与训练调用路径。只在新机 `/opt/data/private/hechengzhu/ZHY` 重放，不创建Isaac环境、采样或优化器，不增加训练更新；原机保持停止。

## 固定设计与判据

使用E05保存的实际runtime配置重建Actor，strict加载官方035权重并沿用官方std限幅。先做CPU16环境整批重建门槛：mean RMSE不超过1e-5、ratio最大偏离不超过0.001；不能复现即停止，不做根因归因。

固定TF32=false、零optimizer，CPU/CUDA各比较aux=false/true下的16×24整批、四组4×24和四组4×1逐帧，共12变体。四组环境顺序来自E05保存的首实际批加三补充批，回填原环境位置；既不是新随机采样，也不是原学习loop的后续排列。直接4×24还与E05真实首训练mean逐元素对照，独立CPU重载复算全部概率指标和差值。

若直接4环境也偏离，进一步分离批形状/时间维与具体模块；若直接4环境复现采样、却不同于真实首批，检查训练路径的输入、梯度模式与包装器。此次回放使用no_grad，不能据此排除启用梯度时的前向差异。Actor参数摘要前后必须相同。

固定注册模型/输入/代码SHA，启动只允许READY且同boot，worker须由带保护的monitor启动。900秒上限，CPU90/GPU85摄氏度连续2次5秒、可用内存保留4GiB、OOM/新boot/关键内核错误/缺遥测停止；无重复、重置或开机恢复，计算串行。

## 实现、命令与产物

入口 `script/a3/zhy_batch_probe.py`；`test_zhy_batch_probe.py`核验打乱环境顺序后的逐帧/整段回填，以及拒绝缺失/重复环境。使用既有PyTorch环境和固定vendor `fe6868ba37034f89b912f0fb851bce19f120266d`。新机实际2项CPU测试通过，用时0.029秒；这只核验回放排列和覆盖，不代替模型结果。

```bash
cd /opt/data/private/hechengzhu/ZHY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_batch_probe -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_batch_probe register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_batch_probe run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_batch_probe verify
```

输入 `data/experiments/a3_zhy_likelihood_20261012_E05/`，输出 `data/experiments/a3_zhy_batch_20261012_E06/`，日志 `logs/a3_zhy_batch_20261012_E06/`，小型清单 `data/manifests/a3_zhy_batch_20261012_E06.json`。大张量、模型与原始日志留新机LOCAL_ONLY，Mac/Git仅代码、README和小型清单。

## 结果与边界

未运行；不得把实现完成写成概率合同通过或策略改善。E05失败和E03采集器失败原样保留。新机能运行已有工作负载不解释旧机具体故障器件，也不保证长期稳定。当前不追加长PPO。

## 保存与下一步

实际CPU测试后保存预登记README/索引/current state，scoped审计、提交推送main并核对远端，然后登记执行新机E06。阶段结果形成后立即更新相同入口；使用真实结果选择下一变量，不能由实验数量判断研究进展。更深目标的待确认理解见[实验依据与真实任务](README_A3_EXPERIMENT_DECISIONS_20261012.md)。
