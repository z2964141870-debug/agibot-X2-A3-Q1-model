# A3 修正入口完整集成与短试验

记录日期：2026-10-10。实验E03/R06与E04/R07。状态：E03集成失败并保留；E04完整训练2更新和独立核验通过，效果待评测。

## 目标与范围

从官方035 PT独立初始化，验证E02修正入口的完整Isaac训练契约，再用第1/2次更新在相同MuJoCo条件下对照官方及旧R05 step2。两项工程修正作为整体检验，不能把效果归因给单一修正。旧R01–R05、E01/E02账本、vendor和Q1保持各自原记录。

## 配置与关键决定

16环境、seed0、epochs5、4 minibatches，原16/4整动作划分seed42复用；辅助目标恢复官方AuxLoss PPO，KL控制实际学习率，记录每个optimizer.step。优化配置的总iteration字段保留200，保存回调在累计2次更新停止；此次不自动推进200。每次更新保存，额外保留step1供早期退化定位，未改变优化超参。

无累计启动/到期上限；连续两次无保存进度、异常停止、CPU90°C/GPU85°C连续两次5秒采样保护保留。重启后300秒冷却，从本R07独立校验checkpoint恢复，评测按完整动作恢复。服务CPUQuota100%，训练和评测依次运行。没有操作真机。

## 版本、命令与产物

服务器 `/media/yu/FAFF-E9771/YUANQI_A3`，main。vendor固定`fe6868ba37034f89b912f0fb851bce19f120266d`。入口`script/a3/corrected_trial.py`，修正trainer见[E02](README_A3_OPTIMIZER_20261010.md)。

```bash
data/environments/a3-sonic/bin/python -m script.a3.corrected_trial train --trial R07
data/environments/a3-sonic/bin/python -m script.a3.corrected_trial verify --trial R07
data/environments/a3-sonic/bin/python -m script.a3.corrected_trial evaluate --trial R07
data/environments/a3-sonic/bin/python -m script.a3.corrected_comparison
```

服务`a3-corrected-r07.service`仅训练2次更新，`a3-corrected-e04.service`仅MuJoCo评测。前者完成并独立verify通过后才启动后者。产物`data/experiments/a3_corrected_20261010_E04/`，checkpoint `data/training/a3_finetune_20261010/R07_*/`，日志`logs/a3_corrected_20261010_E04/`。E03/R06原目录和blocked账本保留，服务停用。模型、轨迹均LOCAL_ONLY，网盘未核验。

## 验证与结果边界

准备阶段重新核验服务器boot `6a7552b6-3c5a-4a51-929f-0a8d2d52dc79`，GPU空闲，旧R05服务inactive，main干净。E02九项CPU契约此前通过。

16:47:12–16:47:33 E03/R06完整入口在trainer初始化失败，0次更新、无checkpoint。确切异常`AcceleratedOptimizer object has no attribute _optimizer_step_pre_hooks`：Accelerate包装器继承torch Optimizer，旧isinstance判断未继续解包，钩子错误挂在包装层。不是服务器重启或模型数值异常。修正为沿`.optimizer`取底层后注册；新增真实`Accelerator(cpu=True).prepare_optimizer`契约验证。用unittest入口执行10项，避免改写E02原contracts文件；日志`logs/a3_corrected_20261010_E04/contracts.log`。R06异常停止保留，不解锁或重置；新R07独立初始化。

训练验收：step0网络等于官方且优化器重置；step1/2大小/SHA/CPU重载/网络与优化器有限；20/40实际优化器计数；实际LR日志及保存args/optimizer/scheduler一致；实时辅助loss日志有限。评测按20动作完整回放、120Hz CSV stride4→30Hz参考、50Hz policy、frame0初始化、无动作延迟、离线未来窗口。四个留出已用于诊断，不能称为未见最终测试。

16:53 E04/R07首次启动完成0→2，同boot、退出0，服务inactive后停用。独立CPU核验step0/1/2均通过；step0网络等于官方、优化器空，step1/2优化器计数20/40。40条实际optimizer.step审计，保存controller/optimizer/scheduler在step1/2均1e-5。辅助loss确实由完整入口计算与记录：total_aux_loss约0.00340557→0.00384782，g1_recon约0.202197→0.202330，a3_fast_g1_latent约0.00138360→0.00182452。未更改epochs或aux系数。

初次verification严查全部标量时发现首轮`Objective/length=NaN`；源码ppo_trainer.py:2085对空lenbuffer求mean，step1 checkpoint独立重载lenbuffer=[]且16环境长度全24，确认是没有完成回合的空统计。该项登记为未定义、清单用null表示，保留原TensorBoard，不称全部日志有限。第二轮length=31。损失/KL/LR和网络/优化器有限性仍严格验收，其他非有限标量仍拒绝。原失败日志verification.log保留，复核verification_retry.log和verification.json为最终证据。

step1 SHA `92b22ba2dd8a7fc487579438e4fccd3516b54b546a77180d79b24dd2f44b0166`；step2 SHA `78325364bedd6a5a473e6d56bd8cdd63b9d758b43a13736bf2830b9814b07f81`，各402076373字节，LOCAL_ONLY。训练时的job.json固定源码SHA与执行版本00b5ebc保存，之后只增强独立核验的空统计分类，未重跑训练；最终代码SHA见清单。

## 下一步与同步

先训练2次与独立verify，再完整selected20并生成跌倒、跟踪误差和共同跌倒前比较。通过链路不等于改善。阶段代码/报告审计后scoped提交推送main并核验远端；实际执行结果继续更新同一README。用户待办仍见[全链路记录](README_A3_OFFICIAL_TEST_20261010.md)。
