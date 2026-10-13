# A3有限更新能力保留：原文审查与E51短训练

记录：2026-10-14。研究与独立实现完成，E51/R20尚未登记、未执行；标准库3项控制和静态语法检查通过，ZHY张量控制待原资源门。官方035仍baseline，E50拒绝不改。

## 目标与已完成内容

用户要求改善官方035的衣服/SMPL完整动作表现，并在收益受限后优先读本地reports/research。E46–E50已充分说明固定新batch的Adam实际位移可向外、一阶投影不能保证有限teacher误差非增。此次结束该诊断链，设计可直接进入短训练、再完整闭环验收的一个新变量；没有继续加载E45 step4或E48 batch。

先读本地10/7持续学习、10/10 MimicX、10/13研究及既有专题记录；不重做MimicX/PriMo筛选。补读D2R方法和CPO作者固定源码。第三方PDF仅ZHY，源码仅服务器被忽略的研究目录，未安装、运行第三方代码或下载模型。

## 原文与源码结论

| 方法 | 原始版本与本次核验 | 对A3的适用边界 |
| --- | --- | --- |
| Continual Humanoid Motion Learning | arXiv 2610.04231v1，原提交2026-10-03 02:45:37 UTC；复用本地已读方法，匿名代码入口仍401/not_connected | 冻结旧任务列、任务标识和新增容量是保留机制的一部分；不能据此说共享dyn decoder加LoRA就不会遗忘。未获得可审查训练源码 |
| Dream to Remember (D2R) | arXiv 2610.13816v1，原提交2026-10-09 16:28:01 UTC；本次从摘要扩展到正文§3.1–3.4、式1/4/5 | §3.3把硬保留约束松弛为软KL项，并借冻结世界模型/参考策略反复产生旧任务状态。没有给本项目有限Adam步的硬上限保证；当前无兼容世界模型，不引入新框架 |
| Constrained Policy Optimization (CPO) | 原提交2017-05-30，PMLR70/2017；作者jachiam/cpo提交5c83925a84bc21dfc9f8bceb82f85daf2937e595（2017-06-07T01:42:57Z）静态审查 | 源码612–646行检查实际候选目标/约束并回溯，726–748行拒绝不合格步并恢复。它是自然梯度约束法；本项目只借有限候选接受机制，不是CPO复现、理论保证或新颖性声明 |

原始入口：
- https://arxiv.org/abs/2610.04231
- https://anonymous.4open.science/r/continual-humanoid-learning-35D3
- https://arxiv.org/abs/2610.13816
- https://proceedings.mlr.press/v70/achiam17a.html
- https://github.com/jachiam/cpo/tree/5c83925a84bc21dfc9f8bceb82f85daf2937e595

D2R正文PDF16906920字节、SHA256 `9efd28116d1f9c86f4d19c314b181c81a369aab2281148c022eb3b5851f38496`；位于ZHY `data/research/2026-10-14/a3_finite_retention/`。CPO所读源码35037字节、SHA256 `95d7f1b58ad48594ab98e36c66054a0fda319a56fee8ab6eef04f62efb727d22`；位于hp3090同名研究目录，固定树未发现LICENSE，不复制到Git。所有资产LOCAL_ONLY，未验证网盘。D2R项目站资源未核验，论文称公开不等于已取得代码。

## 三种门槛不能互换

1. 训练原保护：完整384官方teacher帧RMSE<=.05、maxabs<=.5，限制锚点偏差；不证明闭环能力不退化。
2. E50独立诊断：在原门之外要求相对step4的有限teacher MSE不增加（容差1e-9）。E50在此拒绝，原.05/.5仍过门。结果永久保留，不缩步重试该投影。
3. 效果验收：六衣服全部不倒、均关节RMSE比E26改善>=1%；原不倒动作及四官方关节退化<=2%，世界根/腕退化<=5%，官方不得新增跌倒。必须同条件完整回放。

新训练从官方teacher误差约0开始。若直接套每步误差非增，会把可用的锚点偏差预算几乎压到0；不等于数学上所有参数都不能动，但不是原训练要求。本实验保留原绝对上限，**不倒改E50，更不改变效果门**。

## E51/R20：唯一主变量与事务语义

从官方035重新暖启动Actor和Critic、空Adam；仍用E45四段开发MotionLib、seed0、16环境×24步，原reset/奖励/路由、cached实际tokens、soft teacher系数1、完整Actor+Critic全局clip .1、原KL名义LR规则均不变。旧E44/E45不恢复，不加载其训练checkpoint。

唯一算法变量：每次更新先在隔离参数和独立Adam状态上调用一次当前环境原生AdamW（保留原fused/flags），产生实际FP32候选；用固定序列 `1,1/2,1/4,1/8,1/16,1/32,1/64,1/128` 插值该实际位移，取第一个同时满足原完整teacher绝对门和同batch固定Critic值Actor目标不增（1e-7数值容差）的候选。Actor目标保留原PPO/entropy/aux/轮转96teacher，候选检查不更新名义LR、不重新采样。

这是一种预先登记的有限回溯更新算法，会改变每步有效位移；不能声称有效学习率没变。它没有teacher梯度投影，也不是给E50换编号后继续缩步。只检查固定锚点和当前batch，不声称全局约束或部署保证。

候选检查仅功能式替换Actor参数，不写在线模型；Critic只在原训练forward更新一次归一化统计，候选不再前向Critic。全模型Actor+Critic参数按同一比例提交，Adam历史矩/step仅提交一次；被拒绝候选不累计历史矩。全部8个不合格、非有限或异常即STOPPED，未提交的参数/optimizer保持，当前未提交batch的buffer恢复；不换batch/seed、扩序列、扫权重或再开同类诊断。

每步记录名义LR、实际scale、完整梯度范数、候选teacher/目标、接受前后状态SHA。保留原提交后teacher guard并对比接受时值。异常若已在提交后发生，明确记录已提交，不能伪称回滚。

## 固定预算与效果出口

- prepare：4个小型CPU/CUDA事务控制，原生Adam调用最多20次，只用2元素合成参数；0机器人模型/仿真。验证scale1与原生fused逐值相同、历史矩一次提交、拒绝无post hook/无状态改变、写盘异常恢复。三个已通过的标准库测试另行记录。
- train：最多8完整更新、32原生候选提案、32提交、3072交互；每提案最多8候选（总256）及1基准检查（总32）。候选检查每次2衣服Actor/decoder+4teacher forward，最多1728额外forward；不新增梯度计算。原训练及提交后原保护保留。
- 保存官方step0及完整通过的2/4/8；只对最终step8做一次六衣服+四官方完整评测（10回放），不按动作或checkpoint择优。短训练不是靠堆32轮绕过保留问题。
- 干净完成后独立验证原始暖启动、全部冻结参数/buffer、实际token、配置、32事务和Adam步数，再评测。任何STOPPED不评测早期checkpoint、不自动复跑。效果未通过即拒绝此预登记短预算；不把训练完成当改进。
- 七段衣服都是开发参考，zzx角色DEVELOPMENT_TRAIN、E23历史分区不改。通过后才另固定独立seed和真正未见会话。

## 版本、命令与保护

新文件 `script/a3/finite_{adam,retention_protocol,retention_trainer,retention_trial}.py` 及两个测试；不修改旧E46–E50登记源码。输入/运行库/源码SHA见 `data/manifests/a3_finite_retention_plan_20261014.json`；包括官方035、E11/E12锚点、已验证4MotionLib和原训练配置。运行版本由Git与登记记录固定。

所有张量/训练/仿真仅ZHY既有环境。登记/各阶段启动前3次5秒原available>=8GiB、无其他计算/大RSS、GPU空闲/遥测可靠；CPU90/GPU85连续2次5秒、4GiB保留、900秒/阶段、无开机恢复。首boot/OOM/内核/温度或计算异常停分支；SSH失败UNKNOWN。只在上述条件通过后按顺序各一次：

```bash
/usr/bin/python3 -m script.a3.finite_retention_trial register
/usr/bin/python3 -m script.a3.finite_retention_trial run --stage prepare
/usr/bin/python3 -m script.a3.finite_retention_trial run --stage train
/usr/bin/python3 -m script.a3.finite_retention_trial run --stage verify-training
/usr/bin/python3 -m script.a3.finite_retention_trial run --stage evaluate-cloth
/usr/bin/python3 -m script.a3.finite_retention_trial run --stage evaluate-official
/usr/bin/python3 -m script.a3.finite_retention_trial run --stage verify-effect
```

产物ZHY `data/experiments/a3_zhy_finite_retention_20261014_E51/`、`data/training/a3_zhy_finite_retention_20261014_E51/R20_001_s0/`，日志 `logs/a3_zhy_finite_retention_20261014_E51/`。Mac只小型代码/记录。尚未登记，命令不是已执行证明。

## 当前结果与下一步

研究/实现已完成；3项标准库测试、轻量plan入口与静态检查通过，0机器人训练/闭环回放。北京时间01:29恢复核验同boot/OOM0、GPU26°C/13MiB/0利用率，available约6.68GiB，低于启动线。先保存并发布方案，再按最新计划保护输入定向维护闲置cache、三次复验。资源不足WAIT、不登记、不改门；有余量执行prepare控制后继续。没有效果提升证据。

重要README/索引/current state/小清单随本阶段scoped审计提交推送，核验以实际命令输出为准；未推送前不称已同步。独立头环侧支修改保留，不接手其工作。

## 北京时间01:47实际资源与等待入口

01:46单次缓存维护从shell直接调用，在require_idle即因GPU忙拒绝，0释放/0删除；blocked收据`logs/zhy_file_cache/blocked_1791913605053288948.json`。此时新计划尚未传上去，使用E50保护计划；新计划会用的官方/锚点在终结扫描根以外，4MotionLib均小于1MiB不进入回收候选，且本次根本未进入扫描/释放阶段。没有重试维护。

01:47只读核验：同boot/OOM0、CPU41/GPU40°C；GPU19%/5286MiB，PID1183477可见RSS约5.84GiB，另有小型Python进程。原available估计约8.63GiB，但其他计算占用使启动门仍不满足；不终止或干预这些任务。**E51未登记、OUT/训练目录尚未创建、0本实验worker、4张量事务控制尚未执行**。状态WAIT_EXTERNAL_COMPUTE，不是机器故障，也不要求用户审批新方案。

恢复时低频重新核验现场；资源/GPU/其他进程三次原预检通过后只登记一次，依次prepare/train/verify/最终效果。若OUT已存在则读真实state，不重复登记。不因本次等待回到旧E50或另加零更新实验。小清单`a3_finite_retention_readiness_20261014.json`记录本次资源证据。
