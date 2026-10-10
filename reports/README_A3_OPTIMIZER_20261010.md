# A3 优化器与训练目标诊断

记录日期：2026-10-10。实验E02，状态：诊断完成，独立修正入口CPU契约通过，策略效果未评测。仅CPU固定证据与小型控制流复现，不启动机器人策略训练/仿真，不改变R01–R05或E01账本。

## 目标与已完成证据

继[保存阶段退化定位](README_A3_REGRESSION_20261010.md)后，核验实际学习率控制、恢复契约及官方训练目标。服务器`/media/yu/FAFF-E9771/YUANQI_A3`，main，vendor只读，固定`fe6868ba37034f89b912f0fb851bce19f120266d`。初始Git干净，诊断服务inactive，GPU无计算进程。

1. **训练目标遗漏已复现。** 官方config trainer为`TRLAuxLossPPOTrainer`，R05为`FineTuneTrainer → ResumableTrainer → TRLPPOTrainer`，未继承辅助损失trainer。配置启用`compute_aux_loss=true`，g1_recon权重0.01、a3_fast_g1_latent权重1.0，模型返回aux但基础总损失忽略它。实际基础方法的哨兵梯度检查：主损失9、辅助损失25，辅助放大100倍总损失仍9；辅助叶子梯度None。不是官方模型缺辅助目标，而是项目适配器选择了不同trainer。
2. **学习率双重控制与恢复差异已复现。** 调用实际KL handler和已安装Transformers constant scheduler，20个合成KL/CPU标量优化步骤使args/优化器到1e-5，轮末scheduler把优化器写回2e-5、args保持1e-5。重放实际恢复适配器两条赋值，把args改为2e-5。相同下一KL：中性分支连续1e-5/恢复2e-5，高KL连续1e-5/恢复1.3333e-5，低KL连续1.5e-5/恢复3e-5。这是控制流复现，不是R05历史minibatch步长重建。
3. **critic_learning_rate不是有效独立参数。** 当前trainer实际继承HF `Trainer.create_optimizer`，CPU小模型的两个组分别为policy/value的weight和bias，两组都2e-5；R05保存组也同LR。配置里的critic1e-3未形成独立critic组；官方trainer也继承此构造，不据此单独引入critic学习率变更。
4. **没有观测到探索std突变。** R05保存std有效均值0更新0.387559、2更新0.387692、200更新0.393970。复用慢走100个固定输入，官方→当前高斯KL均值step2=0.744143、step200=18.655367；这是长期相对官方漂移，不是训练日志中旧采样→当前minibatch KL。少数raw std略超0.5按已有actor契约clamp，不改模型文件。

config结构对照发现24项路径差异，涉及16环境/单卡、200目标、目录/数据16训练动作、保存回调、缓存策略及trainer。辅助系数、奖励/终止、编码器和优化配置没有差异。R05显式motor_model.enabled=false，固定环境源码缺省同样false，不认定此额外映射改变了motor行为。小环境/数据规模变化仍可能影响效果，未证明具体因果。

## 产物与可复现命令

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.optimizer_diagnostic
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 PYTHONPATH="$PWD/script/vendor/sonic_for_a3:$PWD" data/environments/a3-sonic/bin/python -m script.a3.test_verified_finetune
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.optimizer_diagnostic --finalize
```

代码`script/a3/optimizer_diagnostic.py`；CPU复现只调用标量AdamW和哨兵autograd，不调用trainer.train或env.step，不优化机器人网络。证据`data/experiments/a3_optimizer_20261010_E02/diagnostic.json`与state.json，日志`logs/a3_optimizer_20261010_E02/diagnostic.log`；小型清单`data/manifests/a3_optimizer_20261010_E02.json`登记配置/源码/8个模型/事件SHA。模型和原大型工件LOCAL_ONLY。

首两次可用TensorBoard：KL1.329851/0.249387，clipfrac0.668229/0.517708，无aux loss标签，与实际loss路径一致；缺历史minibatch输入、每步LR和梯度，不能做完整历史重放或确定根因。

## 关键决定、边界与下一步

先修正可复现的训练契约，再讨论epochs或LR大小。下一实现采用独立、显式选择的新trainer，复用官方aux trainer并保留durable保存/恢复；KL自适应作为唯一LR控制源，恢复保存args自适应状态，增加每次真实optimizer.step的LR记录。旧入口及vendor保持原样，避免重写R05历史。修正不等于策略效果改善。

## 修正入口及最终核验

新增显式target `script.a3.verified_finetune.VerifiedFineTuneTrainer`，没有改现有job/service target。MRO为Verified→FineTune→Resumable→官方Aux→基础PPO；官方aux初始化、loss、stats及日志路径均继承，warm-start和durable保存/恢复仍走既有适配器。未手写或重新加一套辅助损失实现。

新入口用只计数而不改写组LR的scheduler，KL handler为唯一LR控制源；恢复后采用checkpoint.args.learning_rate并检查有限性/上下界，不再把旧轮末optimizer2e-5作为自适应状态。不会改变旧checkpoint、优化器moment或R05账本。每个底层PyTorch optimizer.step前检查LR与控制变量一致，返回后写入`actual_optimizer_steps.jsonl`并fsync；记录applied_group_lrs、全局step及本attempt优化步编号，日志也显示最近实际LR。若Accelerate跳过底层step则不生成成功step记录；日志不能证明策略改善或抵御物理断电丢失。

9项CPU回归通过：官方aux loss/stats选择、辅助系数和零权重梯度、禁用分支、scheduler不覆盖LR、旧checkpoint冲突修正、连续/恢复在零/低/中/高KL分支下相同LR与标量AdamW下一结果、非法LR拒绝、不支持scheduler拒绝、实际step日志与LR不一致时在更新前停止。结构化结果`contracts.json`，日志`contracts_receipt.log`。使用哨兵loss时主9+辅助25=34，辅助梯度10；对照旧基础方法辅助梯度None。未创建完整Isaac环境、未执行机器人网络backward/optimizer.step；完整trainer集成与效果仍待下一轮短试验核验。

证据限定：历史R05未保存每个minibatch LR/KL/rollout，不能事后确定真实全部步长。小型复现里相同高斯的vendor KL公式因epsilon给出约0.000290>0且低于0.005，handler会把2e-5增到3e-5；这揭示控制器行为，未证明历史首步就是3e-5。自适应下限1e-5也意味着只改初始LR到更低不能保证持续小步长，后续实验需共同审计边界。不能把这里的固定输入KL与训练minibatch KL混用。

官方/R05配置核对与所读源码SHA登记；本阶段小型清单登记修正代码、测试代码、官方aux方法及封闭日志SHA，finalize.log为生成清单时的活动输出不参与自哈希。原E01所有产物不改写。vendor固定PPO SHA `872b22d0ae6b1f124fe152e2831c496bfe6b4c80c92fde167129e0c3f1ce6fc1`，官方aux SHA `483539b22fca693091670a22f90bc21018867dce50f5e39b14b3303dfbea2388`。

阶段报告、索引/current state和小型清单保存，审计后scoped提交推送main并比对远端HEAD；同次提交号以Git历史为准。模型及原大型工件LOCAL_ONLY，未完成网盘核验；Q1、vendor和旧训练入口未修改。修正入口未启动，新训练与真机未执行。

下一步优先核验这个修正入口的完整Isaac训练契约，再从官方独立短试验并同条件评测；先保持epochs5和已有超参，检验“恢复官方训练目标+一致LR状态”的工程修正整体，再另立仅epochs5→1的对照。两处修正作为工程契约修复整体，不能将后续改善归因给其中单一因素；如需因果拆分应分别立分支消融。没有支持直接继续R05或上真机的证据。
