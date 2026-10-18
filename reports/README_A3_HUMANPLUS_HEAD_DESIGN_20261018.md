# A3 冻结特征骨干、仅训练动作输出头的独立方案

记录日期：2026-10-18。设计编号：D01（不是科学实验编号）。状态：方案及14:46静态实施审查完成，设计可审查；未实现、未登记、未运行。

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

设计已具备明确单一变量和固定判据，但执行实现未完成，没有可声称已冻结的新执行代码SHA。下面五项已于14:46完成一次静态实施审查，具体决策见后节；后续须独立实现，不得只把HP12 run参数或全局DECODER常量改成另一层：

1. 独立trainer在create_optimizer前白名单冻结Actor，确保父类restrict_policy不会重新解冻整个decoder；所有optimizer组只拥有两项Actor加原Critic，原超参不变。不得修改旧HP12共享模块的行为。
2. 注册前锁定新代码闭包、四MotionLib、anchors/targets、035、HP11合同及环境源码SHA，重建新根路径；新实验编号须与实际账本核对，D01不占HP编号。
3. 首次真实训练内留证精确035暖启动、两key形状/14877计数、空Adam、原概率门和冻结摘要；每次提交/拒绝审计收紧到精确两key。不能照抄“所有DECODER前缀都可变”的旧验收。
4. Checkpoint保留原Actor架构/keyset，不合并/改写官方文件；0/2/4仅保存，最终8独立验证Actor变化只在这两key、冻结buffer相同、Critic和Adam步数及所有事务预算正确。
5. 沿用已验broker与正常关闭协议，但任务成功和科学拒绝退出分别记录；不给HP12补正常关闭收据。新行为的标准库控制应验证白名单、所有权和预算合同，不能重跑旧控制或以mock当作16环境实测。

方案不证明短训练会改善，也不声称LoRA/持续学习复现或新算法贡献。本轮没有可验收新模型，HP12停止状态不变。

## 14:46 静态实施审查收束

本次起点HEAD为671dafadeacd3fa32015ce35b047176e5c11c0b4，worktree干净、SSH可用。仅读取代码和小JSON，没有import项目训练模块/torch，也没有运行测试、旧register/run/审计或加载任何PT。下述是待实现要求，不是代码已通过的声明；不再为同一组缺口追加重复设计阶段。

### 优化器必须在父类初始化期间就固定

固定vendor的TRLPPOTrainer先在_init_trl构造PolicyAndValueWrapper，再调用create_optimizer_and_scheduler；随后构造callback handler、accelerator.prepare，最后才load_checkpoint。VerifiedFineTuneTrainer的step hooks在其super初始化返回后安装。因此不能等完整trainer构造完再删参数、替换optimizer或重新冻结，这会留下scheduler/callback/Accelerate/hook引用不一致。

ClothConstrainedTrainer.create_optimizer逐参数调用restrict_policy，会把所有DECODER前缀重新设为requires_grad=True。新trainer若先冻结head再调用FiniteRetentionTrainer.create_optimizer，仍会经过这一步，方案失效。选定的实施方式为独立HeadOnly trainer继承现有有限候选行为，但完整覆盖create_optimizer：先执行两key/形状/计数白名单，再显式调用固定Transformers Trainer.create_optimizer这一层通用创建器，绕过上述cloth范围重置；立刻核对原生空AdamW及两组超参，然后以原FiniteAdamW方式包装，必须在scheduler/callback/prepare之前返回最终optimizer。该直接祖先调用是有意的局部范围覆盖，须由固定源码SHA及标准库调用顺序控制防漂移；不改父类源码、不全局monkeypatch。

把wrapper参数映射到稳定名称：policy.<两head key>以及value_model中原可训练参数。优化器参数ID列表必须无重复、与此集合精确相等；不能只检查“没有冻结参数”，否则漏掉Critic或head仍可能通过。拒绝额外disc/reward模型、意外param override或不同optimizer类型。两组weight_decay、betas、eps、lr、fused/foreach等来自原通用创建器，除params外不能重新配置。再次调用create_optimizer时只允许返回同一已核验FiniteAdamW，不重置矩或重新创建；其他既有optimizer拒绝。

load_checkpoint继续走现有fresh035、空Adam、FP32、teacher/概率/token检查链，不能调用ResumableTrainer的resume=true分支。_cloth_scope必须记录两key，冻结摘要由原load_checkpoint覆盖所有其余Actor，包括前置g1_dyn层。首次batch及每次事务/提交前后检查requires_grad、完整所有权、冻结参数/Actor buffer摘要和冻结梯度None；这些检查不新增forward。Critic原有buffer可按原配方变化，不能误把整个wrapper buffer都要求静态；拒绝时仍恢复该未提交batch的完整wrapper buffer。

### 保存与停止审计必须独立绑定

humanplus_training_entry.environment/finish在函数体import humanplus_finite，humanplus_finite_verification.verify和两份HP12审计器也绑定该模块。只换trainer字符串会把USD、progress、checkpoint和关闭收据写回HP12目录；禁止直接复用这些绑定函数或修改HP12模块常量。新入口须独立绑定自身trial，并复用已验AST形状与broker/关闭协议，保持URDF/物理、环境与日志约束不变。共用纯函数可复用，不能通过调用旧register/run/审计获得新收据。

保存仍使用原架构原keyset，不新增adapter/merge。独立保存审计须记录checkpoint的optimizer group参数名顺序，建立序列化整数ID到完整名称的映射，核对只有两head加原Critic、无重复/漏项/额外矩，矩形状/有限性正确、所有step计数等于4*global_step。Actor实际变化集应为两key的非空子集，不强求每个head张量都变化；其余所有Actor参数与buffer逐值相同。step0必须精确035运行规范Actor、原Critic、空Adam。只有完整训练成功后才执行原verify-training及step8效果；STOPPED不加载保存PT。

HP12 stop审计写死了55事务记录、27提交、第28拒绝及7rollout，不能搬到新实验。新只读停止审计应从新登记预算和已有JSON收据推导k次COMMITTED、相配k次PENDING以及至多一次REJECTED，逐对验证顺序、原比例和首次通过、所有权/参数+Adam摘要、未提交buffer恢复、已发生交互及无禁止效果。科学拒绝不补正常关闭证明。正常完成仍须32提交/8更新、step8、CUDA同步、真实原生退出0和关闭日志，再独立验收效果。

### 调用预算展开

计数单位沿用原合同：一次外层policy.forward或一次实际缓存token的dynamics decoder调用；内部子层、kernel、参数摘要不冒充独立策略调用。保留原调用形状，不为head另加前向或缓存teacher特征。

| 训练路径 | 固定推导上限 | 口径 |
| --- | ---: | --- |
| 8次rollout，每次24次policy.rollout | 192 | 每次包含16环境，共3072交互 |
| 32训练batch：原policy、cached decode、teacher96 | 96 | 原配方3次/batch |
| 初始teacher384，4环境分组 | 4 | 原暖启动保护 |
| 32提交后teacher384，每次4组 | 128 | 原提交后第二保护 |
| 32提案的live基线+至多8候选，每次2衣服+4teacher | 1728 | 32*9*6，原候选机制额外上限 |
| 上述Actor/decoder外层调用合计 | 2148 | 设计成功路径上限，不含独立最终效果阶段 |

对应Critic只有原训练batch32次及每次rollout末尾24步/末状态的原价值估计；候选始终复用固定Critic值，ActorLossHarness不拥有optimizer且KL只记录，不新算Critic/更新LR。这里没有把训练基础调用挪入1728，也没有扩大候选、交互或效果预算。零位移候选不执行检查、提前接受或STOPPED只会减少调用。保存及成功后的原独立验证只读已有张量/JSON，不增加policy forward；新代码若产生另一条调用路径必须在登记前拒绝或说明，不能运行后补预算。

### SHA闭包与实施出口

本次新增静态阅读的16份来源SHA已加入同一D01清单，其中含实际Transformers trainer.py；它们是审查来源，不是已实现的新执行源码。原HP12登记中的input_sha256、environment_source_sha256和四MotionLib仅作来源合同，尚未在本轮重哈希大型输入或运行环境。新实验必须显式重建自己的代码闭包（包括新trainer/protocol/entry/verifier/auditor、继承模块、hook/candidate/loss及通用创建器），重查输入大小SHA、资产、环境源码和HP11合同；不能整体照搬旧code_sha256把新文件漏掉。

静态审查已一次收束，可进入独立实现及必要标准库控制；本轮只读范围内未写训练实现，仍没有执行代码SHA或新HP登记。实现时优先验证两key白名单、重复/缺失/其他模型拒绝、祖先调用和optimizer/scheduler绑定顺序、保存名映射、泛化停止事务及上表算术，不能重跑HP05/HP09/HP12控制。实际两key梯度、数值暖启动和16环境行为只能在之后完整登记的真实训练内留证，不能用mock充当实测。本设计不是科学启动凭据；新代码与输入SHA/预算未完整登记前不运行科学试验。

## 保存与同步

本README、小SHA设计清单、索引及current state按scoped记录保存。Git提交号以实际历史为准；GitHub SSH publickey写权限此前已失败，本轮不重试相同失败push，状态LOCAL_COMMITTED_UNPUSHED。模型/原始轨迹及日志仍LOCAL_ONLY，无新增网盘核验。zhy-a3保持原30分钟ACTIVE，只通知实质结论；其他自动任务不变。
