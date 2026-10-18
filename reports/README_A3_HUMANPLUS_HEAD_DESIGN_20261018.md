# A3 冻结特征骨干、仅训练动作输出头的独立方案

记录日期：2026-10-18。设计编号：D01（不是科学实验编号）。状态：只读论证完成，设计可审查；未实现、未登记、未运行。

## 目标与范围

针对HP12有限缩步仍耗尽teacher余量的失败，提出实质不同的短训练对照：将可训练Actor从整个g1_dyn解码器缩小到既有最后线性层。计算根仍为humanplus@100.101.30.68:/home/humanplus/YUANQI_A3。本轮仅读取JSON、代码和已有研究，新增说明与小清单；没有加载张量、反序列化模型、运行仿真/优化器或重做旧审计。

HP12已经STOPPED：28提案/27提交、2688交互，第28次八比例均越teacher RMSE 0.05，最小0.050000153109196845，live为0.049996716421312044。原结果、step0/2/4及所有停止限制保留，不评测早期点，不补比例或恢复训练。此结果仅否定该次提案的固定候选集合，不证明其他参数方向可行或不可行。

## 完成内容与去重

- 核对服务器HEAD cfa0647dacd4c4735cbfac5838ddafa3348a4ce3及干净worktree；SSH可用。未采集新启动健康证据，旧健康快照不能用于未来启动。
- 复核HP12停止清单、HP11匹配基线合同，以及固定vendor的MLP构造、实际外部token解码路径、MuJoCo权重映射。
- 阅读已有E20/E21参考残差、E36解码器约束、E42/E44/E45/HP12失败链及持续学习/LoRA综述。E20/E21改变未来参考，不是动作输出头；不重开其账本。E46–E50投影/方向分支保持关闭。
- 在本地和服务器script/a3、reports检索head-only/output-layer-only/module.12及中文末层/输出头关键词，未检出已登记的同一参数范围；这只是本项目去重，不是文献新颖性证明。
- LoRA与持续学习已有先例；现有综述明确任务标识、旧分支与新增容量的边界。这里不复制任务分支，不引入rank选择或低秩因子化，也不声称冻结底座即可防遗忘。

## 关键决定与唯一变量

唯一科学变量：Actor可训练参数白名单改为下面两项，其余训练配方保持HP12不变：

```text
actor_module.decoders.g1_dyn.module.12.weight  # 29 x 512
actor_module.decoders.g1_dyn.module.12.bias    # 29
```

参数数目为29*512+29=14,877，属于静态源码推导，尚未通过实际模型收据确认。固定vendor的hidden_dims为[2048,2048,1024,1024,512,512]；BaseModule按Linear/SiLU交替构造，末层index12后无激活。若实际模型名称/形状/输出ABI不符，必须在首次更新前拒绝，不能搜索另一个可用层替代。

实际训练既有W、b，从035的W0、b0精确初始化，不新加模块、不做merge或另加残差分支。数学上可写为a=(W0+deltaW)h+(b0+deltab)，delta初值为零；运行仍是一遍原生Linear，避免新增加法或低秩合并的舍入与checkpoint映射差异。冻结所有前置解码层、编码器、其他Actor分支、std和Actor buffer；Critic仍按原配方训练。Actor/optimizer keyset、所有权、实际梯度及冻结摘要必须逐项留证。

对固定teacher输入，冻结特征h使精确算术下动作误差成为输出头参数的仿射函数，其平方误差门是凸约束；这是结构推导，不是本次实测或收敛保证。FP32舍入仍以原完整forward实测判定。该参数集合是原整个decoder集合的子集，可能限制有害漂移，也可能无法表达必要修复；没有证据表明必然存在能完成8更新并改善闭环的方向。实际rollout状态随策略变化，固定teacher特征并不固定闭环分布。

部署及评测均沿用相同观测和a3_fast路由，不读取文件名、衣服/官方标签或评测身份决定是否启用适配。所有十项回放使用同一修改后的输出头；不在官方回放切回035。

## 固定配方、预算与判据

从官方035 fresh Actor/Critic与空Adam开始；禁止用HP12保存点。seed0、16环境、24步、1epoch/4mini-batch；E45四段包含zzx，均为DEVELOPMENT_TRAIN。原reset、奖励、路由、实际采样token缓存、teacher系数1、全模型gradclip0.1和KL名义LR保持，不能为了新参数范围调整LR/损失权重/batch。裁剪范数因参数集合变化而改变是本变量的结果，不另外重标定。

继续使用隔离原生AdamW单次提案、固定比例1/2^k（k=0..7），按原顺序取首个通过者。每个候选用真实FP32原架构forward验证完整384帧teacher RMSE<=0.05/maxabs<=0.5和同batch固定Critic值的原Actor目标<=live+1e-7；不改成线性近似门或只评估head，不投影、不增加比例。通过后参数/矩/step一次提交，全部拒绝即终态STOPPED，并恢复未提交batch buffer；不跳batch、不续跑或追加诊断。

固定最多8整更新/32原生提案与提交/3072交互、256候选+32基线检查、1728候选机制额外Actor/decoder forward上限，保存0/2/4/8。现有初始化及提交后保护属于原配方，不能借此新增平行诊断；新增任何调用必须先展开并重新审查计数，不能静默超预算。没有独立空场景、合成optimizer或纯模型零更新阶段。

只有完成全部32提交且最终step8独立验收通过，才做六衣服四官方共10项完整匹配回放，与HP11同机合同逐项比较。六衣服全部不倒、算术均RMSE改善至少1%；原不倒衣服及四官方逐项关节误差不增超过2%、世界根/腕不增超过5%，官方无新增跌倒。保持完整窗口、共同前缀及原聚合口径，不混用算术均与pooled RMSE。任何科学STOPPED不做早期或最终效果，035仍baseline。即使通过也仅为开发适配，仍需独立seed和真正未见会话。

每阶段仍900秒；启动前独立三次5秒、可用>=8GiB/无其他计算/GPU空闲/可靠遥测；运行保留4GiB、CPU90/GPU85连续两次5秒保护，原cgroup v2与全boot journal连续核验保持。真实机器首异常留证并暂停heartbeat；科学候选拒绝不冒称机器故障。仅终止自有组，不改共享环境、vendor或主机配置。

## 版本、命令与证据

固定vendor：fe6868ba37034f89b912f0fb851bce19f120266d，本轮只读status干净。关键源码与合同SHA见data/manifests/a3_humanplus_head_design_20261018_D01.json。035为402050931字节，SHA 9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d；本轮引用HP04/HP12已验身份，没有重新读取大模型。实际新实验使用前必须重新核验其大小与SHA及完整输入闭包。

只读恢复命令：

```bash
ssh -S /tmp/yuanqi-humanplus-a3.sock -o BatchMode=yes humanplus@100.101.30.68 'cd /home/humanplus/YUANQI_A3 && git status --short && git rev-parse HEAD'
```

本轮没有运行新测试、旧审计器或任何科学register/run/verify/evaluation。没有新实验启动命令；D01不能作为可执行登记替代品。

## 实施缺口与下一步

设计已具备明确单一变量和固定判据，但执行实现未完成，没有可声称已冻结的新执行代码SHA。后续先只读审查以下实施要求，形成独立入口及审计方案后才可能进入实现；不得自动把HP12 run参数或全局DECODER常量改成另一层：

1. 独立trainer在create_optimizer前白名单冻结Actor，确保父类restrict_policy不会重新解冻整个decoder；所有optimizer组只拥有两项Actor加原Critic，原超参不变。不得修改旧HP12共享模块的行为。
2. 注册前锁定新代码闭包、四MotionLib、anchors/targets、035、HP11合同及环境源码SHA，重建新根路径；新实验编号须与实际账本核对，D01不占HP编号。
3. 首次真实训练内留证精确035暖启动、两key形状/14877计数、空Adam、原概率门和冻结摘要；每次提交/拒绝审计收紧到精确两key。不能照抄“所有DECODER前缀都可变”的旧验收。
4. Checkpoint保留原Actor架构/keyset，不合并/改写官方文件；0/2/4仅保存，最终8独立验证Actor变化只在这两key、冻结buffer相同、Critic和Adam步数及所有事务预算正确。
5. 沿用已验broker与正常关闭协议，但任务成功和科学拒绝退出分别记录；不给HP12补正常关闭收据。新行为的标准库控制应验证白名单、所有权和预算合同，不能重跑旧控制或以mock当作16环境实测。

方案不证明短训练会改善，也不声称LoRA/持续学习复现或新算法贡献。本轮没有可验收新模型，HP12停止状态不变。

## 保存与同步

本README、小SHA设计清单、索引及current state按scoped记录保存。Git提交号以实际历史为准；GitHub SSH publickey写权限此前已失败，本轮不重试相同失败push，状态LOCAL_COMMITTED_UNPUSHED。模型/原始轨迹及日志仍LOCAL_ONLY，无新增网盘核验。zhy-a3保持原30分钟ACTIVE，只通知实质结论；其他自动任务不变。
