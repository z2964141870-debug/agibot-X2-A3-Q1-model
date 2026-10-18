# A3 humanplus D02 独立实施

记录日期：2026-10-18。北京时间17:49实施阶段；D02已实现/标准库控制通过，尚未登记或运行科学阶段。拟HP18/R24，以实际单次登记为准。用户已授权持续优化。

## 目标与范围

只替换teacher损失样本为HP11四官方各96固定tick，共384个原preclip标签；fresh035、两key输出头/原Critic、空Adam、E45四段、seed0/16env24步、原loss系数/LR/clip/候选尺度和最终十项效果门保持。原E11完整384 teacher RMSE<=.05/maxabs<=.5独立硬门不变。官方四段明确DEVELOPMENT_REGULARIZATION，不能称未见测试。

## 完成内容与关键决定

新增humanplus_replay_protocol/trainer/loss/entry/verification及独立phase/stop审计。1570观察按原command580/orientation60/proprioception930桥接动态原生tokenizer schema，只选a3_fast；不改共享vendor/模型架构/部署路由。实际顺序从固定训练config核对为g1/teleop/smpl/a3_fast。

trainer继承原HeadOnlyTrainer精确两key所有权、冻结、Adam影子提案/提交/拒绝回滚；替换原teacher loss而非双加，新候选目标用同batch固定Critic和同quarter新loss，另算全部原teacher硬门。fresh035重建新384标签maxabs<=1e-5仅在未来登记的真实训练入口内检查，不做孤立模型诊断。

保存验证使用已修纯概率helper，完整0/2/4/8所有权/Adam/配方/token核验；最终8先生成真实PT硬链接+官方config包，再走实际checkpoint_metadata。回放捕获preclip FP32转float64再原±20 clip，与轨迹严格逐值核验；效果函数AST与原公式相同。旧HP02–HP17源码/账本不改。

## 版本、命令与证据

唯一计算根humanplus:/home/humanplus/YUANQI_A3。源HEAD d300a22cacb06d1028d0eb27fdf3e8db1ab2d7f1，提交后精确版本以Git历史为准。固定vendor fe6868ba37034f89b912f0fb851bce19f120266d，共享环境只读。

- Mac与服务器10项标准库边界控制通过：`python3 -B -m unittest script.a3.test_humanplus_replay_loss`。
- 首份readiness因后续配置顺序核对纠正而失效，保留data/manifests/a3_humanplus_replay_loss_readiness_20261018.json，仅为历史；未登记/未科学执行。
- 有效冻结：data/manifests/a3_humanplus_replay_loss_readiness_v2_20261018.json；307源码/451输入大小SHA/10环境源码均核对，源码compile通过。已有无关test_smpl_encoder.py invalid escape SyntaxWarning，不修改历史源码。
- 核定外层Actor/decoder上限2440=192rollout+96训练batch+4原初始teacher+128提交teacher+1728原候选+4新标签重建+288新候选quarter。8更新/32提案提交/3072交互、256候选不扩；调用计数是源码路径与收据推导，不冒称kernel插桩。
- 冻结命令：`/usr/bin/python3 -B -m script.project_tools.freeze_humanplus_replay_loss`已完成有效v2，禁止复跑覆盖。

## 结果与边界

本阶段0张量checkpoint加载/新forward/仿真/optimizer。标准库控制仅证明边界与编排，不证明真实桥接数值/训练/效果；这些留在独立登记阶段。035仍baseline，原输出头候选十项已拒绝，不续训/选早期点。

## 保存与同步

代码/本README/索引/current state/小清单scoped审查并载荷审计后本地提交。GitHub LOCAL_COMMITTED_UNPUSHED；不重复已知publickey失败push。模型/数据/原日志LOCAL_ONLY，网盘未核验。

## 下一步

先提交本实施与有效冻结，单次HP18登记；运行前实时三次5秒门，可用>=8GiB/无其他计算/GPU空闲/可靠遥测，运行4GiB/CPU90 GPU85连续两次/900秒、v2祖先及全boot日志保护。新科学STOPPED不续跑/扩预算/评早期；只有完整32提交和最终8验证通过才十项完整回放。首真实机器故障保存恢复入口并暂停heartbeat。HP11基线及旧终态不复跑。

## 17:50 HP18/R24单次登记

实现提交87dd1dd；HP18计划data/manifests/a3_humanplus_replay_loss_20261018_HP18_plan.json已单次生成，307源码/453输入/10环境SHA及大小核验。新384正则来自四官方各96，原384硬门不变。登记资源检查通过，仅是登记快照；实际run仍需三次5秒启动门。0新科学执行；下一单次train及独立审计，成功再保存模型验证和最终十项完整对照。不能重登记/复跑/覆盖或评STOPPED早期点。GitHub未同步。
