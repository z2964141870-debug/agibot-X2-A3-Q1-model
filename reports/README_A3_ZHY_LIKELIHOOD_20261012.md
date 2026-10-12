# A3 新机PPO概率合同诊断

记录日期：2026-10-12。实验 E03/E04。状态：E03采集器类型处理失败、已停止；修正后的E04准备预登记。

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

E03产物 `data/experiments/a3_zhy_likelihood_20261012_E03/`，日志 `logs/a3_zhy_likelihood_20261012_E03/`，小型清单 `data/manifests/a3_zhy_likelihood_20261012_E03.json`。E04用对应 `_E04/` 和独立清单。捕获张量与模型LOCAL_ONLY，不迁回Mac、不进Git；旧实验不重置。

## 当前结果与下一步

E02已完成64完整回放+3冒烟及独立核验，候选效果门槛全部失败，记录提交 `b2a9877dd987084e8070f6c70824f8c3ba581048` 已推送核对。原模型仍为基线。

E03于13:03:44–13:04:36执行到首轮采样完成、比较输入时停止，退出1。实际obs容器为TensorDict，原快照只接受dict，因而没有复制容器中的GPU张量，比较CPU存储时出现设备不一致；初始3项CPU测试只覆盖普通dict，未覆盖实际容器。该错误在新增诊断代码中，不是原训练代码/主机根因。完整日志包含 `_get_rollout_data -> sampling_identity` 的错误位置；未执行optimizer.step、未进入概率复算。已保存sampling输入含可变GPU引用，**无效**，不能据此判断输入错位或存储bug。

E03注册代码提交 `c0ccaff403c58a42e93f5eff0687a96ac8cb97b4`；原job、state、初始checkpoint、无效capture与日志保留。12个去重健康样本同boot，CPU最高40/GPU34摄氏度，可用最低5.86GiB，OOM0，无新增关键内核事件；没有主机保护触发或重启。本轮故障说明采集器需修复，不是停止保护的绕过依据。

修正使用标准Mapping接口递归显式复制TensorDict，追加真实TensorDict原地修改回归测试，累计4项CPU测试通过。E04是修正采集器后的独立零更新诊断，不重复或解锁E03。启动还要求E03已停止、错误清单确认采集器故障、0optimizer、同boot且无关键内核错误；模型/数据/预算/保护/概率门槛保持原登记。旧E10继续blocked。

E04执行入口：

```bash
data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood register --experiment E04
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood run --experiment E04
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_likelihood verify --experiment E04
```

E04尚未运行，不认定概率合同通过或训练退化根因定位。有效采集及同runtime概率证据形成后再决定修复或单变量微调；不因为主机不重启就认为策略效果通过。

每阶段保存README/索引/current state、小型清单，检查diff、运行payload审计、scoped提交推送main并核验远端。原机服务保持停止，Q1/学习/每日研究不变，无机器人或驱动/功率操作。
