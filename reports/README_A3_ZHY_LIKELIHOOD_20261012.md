# A3 新机PPO概率合同诊断

记录日期：2026-10-12。独立实验 A3_ZHY_LIKELIHOOD_20261012_E03。状态：3项CPU快照/覆盖测试通过，等待预登记后的单次执行。

## 目标与决定

此前E08在尚未更新参数时初始PPO ratio已经偏离1，E09发布配置重建不能复现采样；E10两次旧机初始化中断，无有效runtime诊断。原E10账本保持blocked。新机R11已成功初始化/训练，E02完成且独立核验后才允许这个新条件下的诊断；不是在原故障主机改名绕过停止保护。

本轮官方035初始化、16环境、24策略步、4minibatches、seed0，**禁止任何optimizer.step**。捕获采样时Actor实际收到的输入与输出，再与rollout存储对照；使用同一个runtime Actor重算首minibatch、GPU/CPU、TF32开关、train/aux/eval、逐帧/整批概率。CPU复制会同步并改变运行时序，明确是诊断配置，不能用它证明普通配置长期稳定。

新增采样快照显式复制CPU/GPU张量，避免只保存指向可变环境buffer的引用。输入是否错位、量化/精度是否改变结果均是待验证假设，不预先宣布原因。Actor允许官方get_std已确认的首次限幅，除此之外前后参数不得改变。Critic不进行参数更新；已有normalizer行为另外记录，不把参数SHA视为所有内部状态完全不变。

预定合同：采样输入与训练存储精确一致，采样mean与stored mean一致；首实际minibatch动作mean RMSE不超过1e-5，ratio最大偏离1不超过1e-3。CPU独立重载采样、实际首minibatch和全部完整变体，复算输入identity、动作误差与概率；每份捕获验证SHA。合同未通过则不扩展PPO长训，先按证据定位因素；不同精度模式分别报告，不能挑一个变体掩盖真实训练前向。

## 实现、保护与复现

新根 `/opt/data/private/hechengzhu/ZHY`；入口 `script/a3/zhy_likelihood.py`，诊断trainer `zhy_likelihood_trainer.py`。复用既有RuntimeLikelihoodTrainer与官方固定vendor，不修改历史源码或账本；代码/模型SHA在启动前固定，无重复、无开机恢复。CPU90/GPU85连续两次5秒、容器16GiB内可用估计保留4GiB、OOM/新boot/内核错误停止，最多900秒；不与E02或其他计算并行。

```bash
cd /opt/data/private/hechengzhu/ZHY
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_likelihood -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood register
data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood verify
```

产物 `data/experiments/a3_zhy_likelihood_20261012_E03/`，日志 `logs/a3_zhy_likelihood_20261012_E03/`，小型清单 `data/manifests/a3_zhy_likelihood_20261012_E03.json`。捕获张量与模型LOCAL_ONLY，不迁回Mac、不进Git。

## 当前结果与下一步

E02已完成64完整回放+3冒烟及独立核验，候选效果门槛全部失败，记录提交 `b2a9877dd987084e8070f6c70824f8c3ba581048` 已推送核对。原模型仍为基线。

新机CPU测试3项通过：显式快照不随环境原buffer原地变更、mean相同仍能检出输入错位、缺失sampling call拒绝验收。尚未执行runtime诊断，不能认定概率合同通过或根因已定位。预登记后单次执行本实验，检查采样输入与同runtime概率；以完整数据决定是否修复存储/前向一致性，再做单变量行为保持或冻结部分网络的短微调，不因为主机不重启就认为策略效果通过。

每阶段保存README/索引/current state、小型清单，检查diff、运行payload审计、scoped提交推送main并核验远端。原机服务保持停止，Q1/学习/每日研究不变，无机器人或驱动/功率操作。
