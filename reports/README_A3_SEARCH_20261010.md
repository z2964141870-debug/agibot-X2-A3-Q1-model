# A3 保守微调快速筛选与长期优化流程

记录日期：2026-10-10。E05/R08与条件分支E06/R09。状态：准备完成，试验尚未执行。

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
