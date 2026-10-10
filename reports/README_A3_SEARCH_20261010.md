# A3 保守微调快速筛选与长期优化流程

记录日期：2026-10-10。E05/R08与条件分支E06/R09。状态：E05训练/核验/完整评测完成，效果仍退化；E06准备执行。

## 目标与选择依据

用户要求尽快摸出可长期优化的方法。前一轮修正入口减轻退化但仍比官方差40.77%，20个动作均更差；先检验对小批数据减少重复优化，再缩小实际学习步长。链路通过、减少退化、正向提升分别验收，不通过更新数宣称改善。

固定官方初始化、16环境、4 minibatches、seed0、官方辅助目标、同一16/4整动作划分。R08仅epochs5→1，独立2更新，计数为4/8 optimizer步。若仍退化，R09仅在R08基础上把actor初始LR、adaptive上下界按0.1缩放，保留KL控制规则与相对范围；不是只调初始LR后又被下限拉回。R09也从官方独立初始化2更新，不能接着R08算作单变量比较。

为了快速筛选，仅最终step2完整selected20；step0/1/2全部保存和独立核验，必要时再定位step1。结果同时报告训练/诊断留出、跌倒、完整与跌倒前误差以及共同窗口。已反复用于选择方向的4留出为验证集，不能作新的最终测试。

## 实施与恢复

服务器`/media/yu/FAFF-E9771/YUANQI_A3`，main，vendor固定fe6868b只读。复用corrected_trial的训练/保存/温度/异常/重启恢复契约，增加显式trial配置与动态计数校验；旧终止账本不改。无启动次数/到期上限，连续两次无保存进度停止，CPU90°C/GPU85°C连续两次5秒采样停止，重启300秒冷却；服务CPUQuota100%，训练和评测串行。

```bash
data/environments/a3-sonic/bin/python -m script.a3.search_trial train --trial R08
data/environments/a3-sonic/bin/python -m script.a3.search_trial verify --trial R08
data/environments/a3-sonic/bin/python -m script.a3.search_trial evaluate --trial R08
data/environments/a3-sonic/bin/python -m script.a3.search_trial report --trial R08
```

R09替换trial同理，只在R08证据保存后执行。训练服务`a3-search-train@R08.service`，评测`a3-search-eval@R08.service`。产物`data/experiments/a3_search_20261010_E05/`和E06，checkpoint`data/training/a3_finetune_20261010/R08_*/`与R09，日志`logs/a3_search_20261010_E05/`和E06。模型/轨迹LOCAL_ONLY，不操作真机。

## 预先固定的判据

快速诊断主要指标为验证4动作完整关节RMSE，跌倒不得新增；全20关节、根、腕、腿与跌倒统计均报告，不因单指标略好宣称整体改善。相比官方仍差的方案不能作为效果升级。训练配置差异必须与登记单变量一致，网络/优化器有限、实际LR与保存状态一致、PT完整重载为每轮前置门槛。

本轮最多先筛选两种保守更新方案；没有优于官方证据就不盲目接200/10000更新。长期流程候选为：官方基线固定→明确目标短板与可靠参考→一次只改一个变量→短训练/完整验证→训练回退或继续→新动作/新会话与多种子确认→导出及独立仿真验收。是否能提高效果仍以实测判断。

## 当前验证、同步与下一步

开始时main干净且HEAD7907b15，boot6a7552b6不变、GPU空闲、旧R07训练与E04评测inactive/disabled。先做入口/参数契约检查和R08试验。每个训练/评测阶段立即补同一README、索引/current state；审计后scoped提交推送main、核验远端。已搁置Isaac指标/RKNN/动捕质量依赖仍见官方全链路记录，不重复投入安装。

E05/R08执行版本c2038a6，3项新入口契约通过。首次训练0→2正常完成，同boot，无重启或热停；独立step0/1/2 CPU重载/大小/SHA/网络优化器有限性、初始化权重和计数重置通过，optimizer计数4/8，8条真实step已记录，step1/2保存args/optimizer/scheduler均1e-5。Aux目标日志非零且有限；首轮回合长度空统计仍单列。训练服务inactive/disabled。

实际Hydra配置逐项比较R07/R08仅`algo.config.num_learning_epochs:5→1`及3个输出路径变化，config_diff.json保留，不存在其他训练配置变量变化。step2 SHA `1d2b45c219fc06114a44ac69bab6e40517f3c3c295c65bc4bf6b152c4d63aff3`，402076373字节，LOCAL_ONLY。接下来仅该模型完整selected20；尚无策略效果结论。

## E05结果与分支决定

最终step2完整20动作/24304策略步、0跌倒，协议、逐动作SHA与时长核验通过。全体关节RMSE0.0772975rad、4验证0.0819503，低于epochs5的0.0831406/0.0892663，但高于官方0.0590597/0.0663182。腕局部0.0201953m、腿局部0.0237322m；根位置0.145100m高于epochs5的0.136426m和官方0.0841691m。减少epochs改善部分误差，未证明整体方向有效。

固定官方慢走前100个真实输入CPU探针重新核验输入SHA和官方输出一致；对官方的动作输出RMSE epochs5=0.0894935、epochs1=0.0716412，Gaussian KL均值0.727971→0.470217。此KL是单动作固定输入漂移，不是训练rollout KL，也不能作为闭环效果验收。结果fixed_input_drift.json登记。

E05训练/评测均inactive/disabled，未触发热停或重启；结果清单`data/manifests/a3_search_20261010_E05.json`。依预先规则进入E06/R09：epochs1保持，LR尺度0.1（初始2e-6、下限1e-6、上限2e-5），其余一致，从官方独立初始化。仍只2更新/8optimizer步，避免把少退化误判成长训有效。
