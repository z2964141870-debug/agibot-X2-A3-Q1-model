# A3 humanplus D02 独立实施

记录日期：2026-10-18。HP18/R24已完整实施、登记、训练、保存验证及十项效果，终态COMPLETE_MATCHED_EFFECT/独立verified=true但效果门false，候选拒绝，035仍baseline；18:11按用户授权清理该淘汰权重，见README_A3_HUMANPLUS_STORAGE_20261018.md。下文保留分阶段证据；已执行阶段禁止复跑。用户持续优化授权有效。

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

## 17:52 HP18真实训练完成

单次train COMPLETE且独立train审计verified=true：16环境cuda:0，fresh035/两key/空Adam链已留证，8更新/32提案32提交/3072交互，32候选均scale1、64含live检查、推导外层调用872（上限2440）。新384 preclip标签fresh035重建maxabs2.0265579223632812e-6，原teacher最高RMSE0.01708323135972023/maxabs0.08512035012245178。Kit专属路径/实际settings/native PID/关闭标记及原生退出0齐全。

13健康同boot，CPU最高81/GPU47°C、最低可用18947629056字节，0新增FATAL/OOM/自有进程组残留；仅本阶段连续证据。结果与审计data/manifests/a3_humanplus_replay_loss_20261018_HP18_{result,train_audit}.json，原始logs/a3_humanplus_replay_loss_20261018_HP18/train，模型data/training/a3_humanplus_replay_loss_20261018_HP18/R24_001_s0/milestone_{0,2,4,8}。仍无效果改善证据，下一成功保存状态验证后最终8十项，GitHub未同步。

## 17:54 保存状态验证通过

单次verify-training及独立审计verified=true，0/2/4/8真实PT大小SHA/原Actor+Critic架构/FP32/step0精确035+空Adam/后续仅两head非空变化/其余Actor参数与buffer不变、optimizer整数ID映射/矩形状/计数、actual sampled tokens和初始概率/原配方/routes及32个新loss quarter收据均通过。累计18健康无新增异常/残留，阶段间不作连续证明。训练验证data/experiments/a3_humanplus_replay_loss_20261018_HP18/training_verification.json；小审计a3_humanplus_replay_loss_20261018_HP18_verify-training_audit.json。下一最终8实际bundle/config核验后六衣服2994步及四官方5775步，0新增训练/不评早期点。035仍baseline，GitHub未同步。

## 17:56 六衣服完整效果

cloth单次6×499=2994步、独立审计verified=true，preclip FP32经固定vendor float64/±20 clip后逐值等于轨迹，实际最终8 bundle+官方config/checkpoint_metadata通过。5/6不倒，zzx仍倒；六项joint RMSE依次0.08989995096106186/0.12873518741433695/0.08986382888140042/0.09222030163952859/0.12724121954754097/0.3902342914701473rad，算术均0.1530324633190027rad，对HP11 .13932748084287783仍恶化。不能采用候选。累计29健康无新增异常/残留；下一只补原四官方5775步及效果总汇，不重训/重复cloth。保存HP18_result/cloth_audit与OUT/cloth，GitHub未同步。

## 17:59 四官方完整回放完成

单次official 637/882/1652/2604共5775完整步、4/4不倒，独立审计verified=true。加已完成六衣服总8769步/十项各单次；无额外训练或重复动作。累计42健康同boot/最高CPU81 GPU47°C/最低可用18947629056字节，0新增FATAL/OOM/残留；阶段间不作连续证明。下一只原效果公式effect汇总及审计，衣服已未达标，035仍baseline。用户新增授权可清理确定淘汰权重；先完成本轮全部收据，再按精确清单核对SHA/硬链接/FD/maps/用途后删除，保留035/必要输入/记录与指标；不把未备份淘汰权重删除称备份。

## 18:03 十项完整对照完成，候选拒绝

HP18终态COMPLETE_MATCHED_EFFECT、effect独立verified=true但matched_development_gate_passed=false。六衣服算术均RMSE0.1530324633190027rad，比HP11 .13932748084287783恶化9.836524993644447%，仍5/6不倒。四官方4/4不倒，关节相对基线singleleg+1.6022%/march+8.0104%/front_walk+4.9909%/rotate+3.1803%；后三项超+2%，march和front世界根/腕也超+5%。10动作8769步各执行一次，新增8更新32提交3072交互仅本轮一次。效果相较上轮部分回归减少不构成对035改善或因果证明；不采用候选，035仍baseline，D02该配方关闭不续训/扩预算/选早期点。

最终8：226206039字节/SHA dd7fd16aca2f1ab59b6a7af843e999e7a286264b61e1cb39f51c8c21818fc257，来自HP18/R24，仅历史拒绝候选。46健康同boot afb6796b-3883-4df4-99ea-bc381c52ea13，CPU最高81/GPU47°C、最低可用18947629056字节，0新增FATAL/OOM/残留；阶段间不作连续证明，无活动科学计算。HP18所有register/train/verify-training/cloth/official/effect/对应独立审计均单次完成，禁止复跑/覆盖。本轮不是机器故障，持续优化heartbeat保持ACTIVE。

恢复先本README及data/manifests/a3_humanplus_replay_loss_20261018_HP18_{plan,result,train_audit,verify-training_audit,cloth_audit,official_audit,effect_audit}.json。下阶段先按用户新存储授权审查清理闭环拒绝/永久关闭权重并留删除清单；必要输入/035/日志/配置/指标保持。其后仅从保存结果与既有研究去重选择实质不同方法，不能重做D01/D02或E46–E50同配方。GitHub LOCAL_COMMITTED_UNPUSHED，网盘未核验；终态大文件删除后以新存储清单作为实际存在状态，不把历史SHA收据当当前可用。
