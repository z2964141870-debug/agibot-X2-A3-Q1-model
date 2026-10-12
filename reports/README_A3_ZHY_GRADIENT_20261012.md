# A3 修正精度后的更新与梯度排查

记录日期：2026-10-12。独立实验 A3_ZHY_GRADIENT_20261012_E11。状态：已实现，6项CPU检查通过，等待登记与实际运行；不能据此声称梯度或效果通过。

## 目标与范围

用户要求继续查，并记住分析此前问题。本轮针对E09修正计算一致性后，E10/R12两更新仍使四诊断动作平均关节误差增加4.63%的原因。先审计已有保存点，再在真实FP32入口完成一次16环境、24步、四批的零更新梯度诊断。使用原官方初始化、数据和奖励，分离policy、value、entropy和aux梯度。

工作在ZHY `/opt/data/private/hechengzhu/ZHY`；原hp3090只保存/同步代码和记录。旧停止账本不改写，不重复E09/E10、不追加长训；本轮不执行optimizer.step，模型与原始张量新机LOCAL_ONLY，无真机操作。

## 用户问题与证据跟踪

| 之前的问题 | 当前证据与判断 | 本轮要补什么 |
| --- | --- | --- |
| 是否硬件导致A3问题 | 原机普通CUDA/YOLO也出现新boot及现场供电保护提示；新机短训练可执行。旧机重启与当前策略退化分别记录，尚未定位旧机具体器件 | 继续原机停载，只观察新机健康；不把数值错误当成旧机重启的解释 |
| 旧机是否也遇到零更新偏差 | 10/10 E08梯度记录初始ratio约0.53–1.96；10/12新机E04/E05也失败。并非只在迁移后出现 | 当年未捕获真实精度，不能宣称已独立证明旧机同一BF16原因 |
| 参数未更新为何内部结果已变 | E08真实Actor权重/输入/buffer稳定；采样FP32、真实训练BF16。E09统一精度后通过 | 区分参数、输出和Critic归一化buffer，记录哪些状态能在无optimizer时变化 |
| 训练是否按预期优化 | E10真实8次optimizer和保存重载通过，但四动作效果门槛失败 | 拆损失和梯度、实际更新模块；确认目标与评测指标的关系 |
| 官方是否说明过 | 已有官方合同/issue核验：有不收敛讨论，所查公开内容未直接说明本地零更新异常；selected20不是035训练语料 | 复用已有来源记录，不把升级vendor当作未经证据支持的修复 |
| 能否用代码修到预期 | 初始概率一致性已修复，效果改善尚未实现 | 本轮只提供下一项单变量试验的依据，不把“不崩”“小漂移”当作更好 |
| 是否一直绕着更深的问题 | 工作理解：需要一条能判断每次实验是否缩小未知的可靠训练/验收链；尚待用户确认 | 每项诊断写清回答了什么、仍缺什么、下一步为何这样选 |

## 关键决定与实现

使用已有`GradientProbeTrainer`官方损失分解与禁止optimizer预钩子，叠加`FullPrecisionContractMixin`；新增`ZHYGradientTrainer`保存四个真实前向、Critic buffer及精度。没有重写训练损失或修改固定vendor。

新入口`script/a3/zhy_gradient.py`先核验R12初始/更新1/更新2保存点，随后一次真实零更新诊断；独立CPU复算概率和损失，审计R12每组真实参数/buffer变化。梯度范数是运行中的诊断证据，CPU复算不伪称独立重算了完整梯度。

CPU90/GPU85摄氏度连续两次5秒保护、保留4GiB内存、同boot、遥测/OOM/关键内核异常停止沿用；无自动恢复、旧目录重置或第二次启动。

## 版本、命令与产物

基于main `7d9cd3596de0a27a5f1e3796e5c87c4f934cfdde`；固定vendor `fe6868ba37034f89b912f0fb851bce19f120266d`。代码/输入SHA在登记时固定；启动前scoped检查、提交推送并核验。

```bash
cd /opt/data/private/hechengzhu/ZHY
PYTHONPATH=.:script/vendor/sonic_for_a3 OMP_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_zhy_gradient -v
data/environments/a3-sonic/bin/python -m script.a3.zhy_gradient register
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_gradient run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_gradient verify
```

实验`data/experiments/a3_zhy_gradient_20261012_E11/`；日志`logs/a3_zhy_gradient_20261012_E11/`；清单`data/manifests/a3_zhy_gradient_20261012_E11.json`。这些结果尚未生成。Mac只留小型代码/报告/清单。

## 结果、边界与下一步

6项CPU检查通过，覆盖独立损失的PPO/value裁剪和padding、辅助系数、参数/buffer区分、状态形状不匹配拒绝、禁止optimizer及旧目录重置拒绝。保存点std变化会分别报告从初始加载的总差异和已知限幅后的剩余差异，避免把限幅误当成优化器更新。

待实际诊断。此前源码已确认当前奖励包含身体/根跟踪、速度和动作平滑等项，不直接等于MuJoCo四动作关节RMSE；这是目标与验收需核对的依据，不是已经证明的唯一退化原因。

同一批零更新梯度只能说明当时目标推动哪些模块。不能证明R12所有更新的历史原因、每轮必改善、真实遥操/机器人效果或数据泛化。下一项由结果选择一个主要变量，保护官方基础能力并使用匹配基线。

## 保存与恢复

本README、索引/current state和代码在启动前保存；CPU检查、登记、运行与结果状态随后追加。尚未同步的内容不提前标成已推送。恢复先读本文件和E11 state/log；不重复已有实验。
