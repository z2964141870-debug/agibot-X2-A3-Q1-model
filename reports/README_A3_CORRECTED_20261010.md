# A3 修正入口完整集成与短试验

记录日期：2026-10-10。实验E03/R06与E04/R07。状态：本阶段完成；E03集成失败并保留，E04训练与评测链路通过，策略改善验收失败。

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

## 完整对照结果

17:06:49 E04评测完成：step1/2各20个完整动作、24304策略步，共40次回放/48608步；逐动作工件SHA、时长、reference采样和模型身份独立核验。官方与旧R05 step2复用已完成结果并重新核验SHA，4组共80次回放参与对照。当前4组均0跌倒，所以共同跌倒前窗口等于完整回放，不存在截断造成的误差下降。

| 模型 | 跌倒/20 | 全体关节RMSE rad | 留出关节RMSE rad | 根位置均值 m | 腕部局部误差 m | 腿部局部误差 m |
| --- | --- | --- | --- | --- | --- | --- |
| 官方 | 0 | 0.0590597 | 0.0663182 | 0.0841691 | 0.0119790 | 0.0189769 |
| 旧R05 step2 | 0 | 0.0915286 | 0.0987959 | 0.1528472 | 0.0333251 | 0.0262490 |
| 修正R07 step1 | 0 | 0.0799062 | 0.0858992 | 0.1250761 | 0.0234444 | 0.0226177 |
| 修正R07 step2 | 0 | 0.0831406 | 0.0892663 | 0.1364257 | 0.0219489 | 0.0250513 |

修正step2较旧step2整体关节误差降低9.16%，但比官方增加40.77%；20/20逐动作关节误差均高于官方，4个留出同样整体退化。step1→2腕部局部误差略降，关节、根和腿误差仍升高，不能据局部变化宣称整体更好。恢复官方辅助目标和一致LR控制减轻了这一次短试验的退化，但尚未找到优于官方的策略，也不能把变化归因给某一项修复或形成多种子结论。

## 测试矩阵与边界

| 任务 | 状态 | 证据与范围 |
| --- | --- | --- |
| E03/R06完整入口 | 失败 | 初始化钩子兼容问题，0更新，无模型；原blocked账本保留 |
| 真实Accelerate包装器与CPU契约 | 通过 | 10项测试，contracts.log；不覆盖所有分布式模式 |
| E04/R07官方初始化与2更新 | 通过 | 16环境、epochs5、seed0，单次启动正常完成 |
| step0/1/2独立保存校验 | 通过 | 大小/SHA/CPU/网络与优化器有限性；初始权重一致、计数重置 |
| 辅助目标与实际LR | 通过 | 40条optimizer.step，实际LR1e-5～2e-5；保存LR一致；aux loss非零有限 |
| 首轮回合长度统计 | 搁置 | 没有结束回合，空mean为NaN；明确标作未定义，未改vendor统计实现 |
| 两模型完整selected20 | 通过 | 40动作、48608步、0跌倒；仅仿真计算与当前动作稳定性 |
| 优于官方的效果验收 | 失败 | 全体/留出关节、根、腕、腿总体均更差 |
| Isaac独立指标/RKNN/动捕质量 | 搁置 | 既有卡点未重试，用户处理方式仍见全链路README |

本轮训练/评测103条保护采样CPU最高76°C、GPU最高38°C，最低可用RAM25248862208字节；同一boot，无热停或新重启。采样不证明硬件长期稳定。训练与评测服务inactive/disabled，R06失败服务disabled，GPU无计算进程；每日研究与旧暂停状态不变。

最终清单`data/manifests/a3_corrected_20261010_E04.json`记录模型SHA、数据划分、代码、训练job/state、E03失败、逐动作指标与轨迹SHA、封闭日志。生成中的comparison.log不参与自身哈希。详细标量与actual_optimizer_steps在verification.json；恢复完成项已重新调用train入口，独立核验目标后直接退出，不新增attempt。

## 下一步与同步

下一轮建议仅把epochs5→1，其余使用修正入口与同一数据/初始化/seed/评测协议，独立2更新（每次4个optimizer步），检验减少对同一小批数据的重复优化是否减轻早期退化。当前证据只支持这个待检验假设；本轮未启动epochs1或追加200更新，官方继续作为基线。进一步效果接受还需要未用于选择方案的独立动作和多种子，而不是沿本轮4留出反复择优。

阶段代码、报告、索引/current state和233KB左右清单审计后scoped提交推送main并核验远端，具体提交号以Git历史为准。大型模型/轨迹LOCAL_ONLY，不删除原件；Q1/vendor与旧账本未改。用户待办仍见[全链路记录](README_A3_OFFICIAL_TEST_20261010.md)。
