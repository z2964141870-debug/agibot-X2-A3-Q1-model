# A3 微调退化定位

日期：2026-10-10。实验 E01；状态：诊断完成，尚未找到已验证的改善方向。服务器根目录 `/media/yu/FAFF-E9771/YUANQI_A3`，main。

## 目标与范围

比较同一 R05 训练轨迹中的 0、2、10、20、40、80、120 次更新模型，复用官方和 200 次更新的原评测，定位退化最早已观测阶段。只评测，不训练、不修改旧账本，不操作真机。留出动作从此也用于诊断，不能再作为全新最终验收集。

## 方案与依据

所有模型先独立核验大小、SHA、CPU 重载、网络/优化器有限性、16 环境与优化器更新计数；0 更新模型还核对两网络逐张量等于官方。相同资产、selected20、stride4/30Hz参考、50Hz策略、frame0初始化和0延迟完整回放。

记录各阶段逐动作首次跌倒、完整与跌倒前误差；额外按每个动作所有模型首次跌倒之前的共同窗口比较关节、根位置、手腕与腿部误差，避免不同截断长度造成误判。首次已观测退化不等于精确起点或因果归因；重启仅登记并按完整动作恢复。

独立服务 `a3-regression-e01.service`，CPUQuota100%，复用串行评测锁；CPU90/GPU85连续2次、5秒采样保护，重启后300秒冷却。普通失败/热停不自动解除；连续2次启动没有保存动作进度停止。没有累计启动次数/到期上限。

## 版本、命令与产物

vendor `fe6868ba37034f89b912f0fb851bce19f120266d`，新代码 `script/a3/regression_diagnostic.py`；版本以阶段 Git 提交为准。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
data/environments/a3-sonic/bin/python -m script.a3.regression_diagnostic verify
systemctl --user start a3-regression-e01.service
```

产物 `data/experiments/a3_regression_20261010_E01/`；日志 `logs/a3_regression_20261010_E01/`；小型最终清单 `data/manifests/a3_regression_20261010_E01.json`。全部大型工件 LOCAL_ONLY。

## 中间记录与边界（历史状态）

2026-10-10 12:41北京时间SSH超时，暂记连接UNKNOWN；最后已核验step0/2/10/20/40完整，step80有17个动作持久登记。尚未取得新boot，不认定断电或重启。后续连接恢复以boot/service/manifest重新核验，原已安装控制器保留300秒冷却。

12:42连接恢复，boot实际变为 `b13f532d-64fc-42eb-8a09-187639a79b3f`，uptime96秒、服务activating/exit75，GPU无计算进程。旧账本completed包含80且其完整20动作摘要可读：2/20跌倒、关节RMSE0.197315rad；120只有已初始化manifest、0完成。冷却后自动核验恢复，不人工解除blocked、不查重启根因。恢复及统计已有5项单元检查通过；完整动作真实SHA恢复仍以服务后续结果为准。

中间完整阶段：step10为1/20跌倒、关节RMSE0.151929rad；step20为0/20、0.135700rad；step40为0/20、0.142816rad。首次已观测跌倒位于2→10区间，但20/40恢复无跌倒，跌倒数不单调。所有这些阶段总体跟踪误差均高于官方；不是已经找到好的微调方向。80/120当时仍评测中，现已完成，最终结果以下方表格为准。

启动前服务器同 boot `9a9f35c1-c5b3-48d8-ab97-412730c8c937`、GPU空闲，Git main干净。0/2/10/20/40/80/120/200独立大小/SHA/CPU重载/网络和优化器有限性、计数与16环境均通过；step0网络逐张量等于官方。两个优化器参数组保存的LR所有阶段均为2e-5。评测尚未完成；已有官方0/20、step200为5/20跌倒是原实验结果。

step10所属R05_002_s2/config.yaml为0字节，原文件不改。已读固定vendor的A3Policy加载器，MuJoCo结构由权重形状和固定MLP构造、直接strict加载PT，不读取旁边训练YAML；故此次同条件仿真仍可执行，但空配置为历史记录缺损，不能用于独立复现该attempt训练配置。

## 只读诊断结果

CPU固定输入诊断 `script/a3/fixed_input_diagnostic.py` 已执行：读取官方慢走最初100真实策略输入，官方适配器与原模拟器动作通过atol1e-4/rtol1e-3，step0编码和动作均与官方完全相等。step2量化编码元素变化27.3594%，动作输出对官方RMSE0.0905764（原始policy单位，不是rad）。仅在推理时交换部件：当前decoder配官方token误差0.0742825；官方decoder配当前token误差0.0355094。两部分都有变化，此小样本中decoder变化影响更大；不能据此声称冻结encoder会修复策略。200次对应0.499309/0.482526/0.136428，均只限同一慢走前2秒固定输入，不是闭环或因果验收。证据 `fixed_input_diagnostic.json` 与 `fixed_input_outputs.npz`。

只读学习率审计 `script/a3/audit_finetune_updates.py` 已执行，证据 `update_audit.json`。step2/200的args.learning_rate均1e-5，optimizer两个参数组及scheduler._last_lr/base_lrs均2e-5；配置actor2e-5/critic1e-3/desiredKL0.01/adaptive下限1e-5。vendor KL handler逐minibatch写所有参数组，轮末lr_scheduler.step再次写入；故记录的自适应LR和checkpoint实际参数组LR不一致，不能把保存点2e-5认定为全程实际优化步长。该冲突值得下一轮前修正/验证，不是已经证明的退化根因。KL日志实际是旧采样策略对当前策略的高斯KL、按动作维求和；clipfrac为clipped surrogate更大的比例。

step0与step2均完整20动作/24304步、无跌倒。step0各聚合指标与已有官方一致；step2关节RMSE0.0590597→0.0915286rad，根位置0.0841691→0.1528472m，手腕anchor相对0.0119790→0.0333251m，腿部anchor相对0.0189769→0.0262490m。故最早已观测的跟踪退化在step2，位于0→2更新区间；没有step1保存点，不能断定发生在第1还是第2次。step2属于初次连续2更新冒烟，早于后续恢复attempt，不能将全部退化归因于后续重启。

审计同时读取全部可用R05 TensorBoard标量。首更新记录KL1.32985、clipfrac0.668229，第2次KL0.249387；Train/lr日志1e-5，与轮末scheduler/optimizer2e-5不一致。R05_002_s2和R05_006_s40事件缺所选标量，其他事件有尾部丢失；缺失不补造。训练标量不替代同条件评测，后续阶段评测仍进行中。

## 最终对照与结论

7个保存阶段各完整selected20，共新增140条动作回放；加官方和step200原结果，共核验9组、180条。每组24304策略步；固定A3资产、frame0初始化、0动作延迟、离线未来参考窗口。所有组协议、模型SHA、逐动作metrics/trace SHA与完整覆盖核验通过。另按每个动作所有9组最早跌倒前的相同帧区间比较，累计20598策略步。关节RMSE按步数加权平方后开根；位置类为加权均值。

| 更新数 | 跌倒/20 | 全回放关节RMSE(rad) | 共同跌倒前RMSE(rad) | 4留出关节RMSE(rad) | 留出跌倒/4 |
| --- | --- | --- | --- | --- | --- |
| 官方 / 0 | 0 | 0.059060 | 0.060458 | 0.066318 | 0 |
| 2 | 0 | 0.091529 | 0.091261 | 0.098796 | 0 |
| 10 | 1 | 0.151929 | 0.132468 | 0.135265 | 0 |
| 20 | 0 | 0.135700 | 0.134634 | 0.125488 | 0 |
| 40 | 0 | 0.142816 | 0.141728 | 0.132708 | 0 |
| 80 | 2 | 0.197315 | 0.161716 | 0.183742 | 1 |
| 120 | 1 | 0.193363 | 0.173912 | 0.172843 | 0 |
| 200 | 5 | 0.259152 | 0.199616 | 0.215507 | 1 |

最早已观测跟踪退化位于0→2区间，首次已观测跌倒位于2→10区间。共同跌倒前窗口同样退化，并非只由跌倒后回放造成。没有测试到整体优于官方的更新阶段；继续保留官方基线，微调模型效果未验收。没有step1，不能精确断定第1还是第2次更新；单条训练轨迹不证明因果、多种子、Isaac或真机表现。4留出仅对R05训练留出，不证明官方预训练未见过。

逐动作首次跌倒：step10左单腿4.46s；step80右单腿10.22s、侧弓步13.14s；step120侧弓步12.38s；step200右单腿4.70s、前弓步3.50s、侧弓步3.82s、双手上举10.06s、前伸11.08s。

对照图：[PNG](../data/experiments/a3_regression_20261010_E01/regression_curve.png)、[PDF](../data/experiments/a3_regression_20261010_E01/regression_curve.pdf)。横轴是保存阶段类别，不是等距更新数。原始聚合、共同窗口与逐动作数据见`comparison.json`。Mac小型副本为`data/sync/a3_regression_20261010_E01/`，未同步PT或大型trace。

学习率审计补充：KL handler在每个optimizer.step之前写学习率，scheduler在轮末再写；保存点2e-5不能认定为全程实际梯度步长。两个参数组未独立确认为actor/critic组。恢复适配器从保存optimizer LR恢复args LR，下一轮需验证此控制契约。这是值得检验的线索，不是退化根因定论。

## 最终运行核验与保存

最终state为complete，completed=[0,2,10,20,40,80,120]，finished_utc=2026-10-10T05:06:25.402803+00:00，即北京时间13:06:25。4次启动boot记录为`9a9f35c1…`、`b13f532d…`、`70e793b9…`、`6a7552b6…`；只按完整动作恢复，未展开重启根因排查。含NUL间隔的service日志和中断产物保留。最后独立服务已disable并停止，inactive/MainPID0/ExecMainStatus0，GPU无计算进程；没有新增训练，旧训练仍暂停。

全部8个R05模型及官方再次实际执行大小/SHA/CPU重载/网络与优化器有限性核验，计划不变。报告实际核验覆盖/协议/SHA通过；318项E01工件登记大小/SHA。349条健康采样CPU最高81°C/GPU38°C，最低可用RAM28268642304字节；health覆盖3个boot，不覆盖每次启动，不能证明硬件长期稳定或重启原因。报告解析支持登记并跳过损坏行，最终无被拒绝health记录。

恢复与聚合focused检查5项通过；PNG已目视检查无文字遮挡，代码编译通过。未重复与本轮无关的全链路测试。复核命令与终端日志：

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
data/environments/a3-sonic/bin/python -m script.a3.regression_diagnostic verify
data/environments/a3-sonic/bin/python -m script.a3.regression_diagnostic report
data/environments/a3-sonic/bin/python -m script.a3.plot_regression
data/environments/a3-sonic/bin/python -m unittest script.a3.test_evaluation_recovery script.a3.test_official_results
```

日志`logs/a3_regression_20261010_E01/{final_model_verification.log,final_report_verification.log}`。阶段代码另含`audit_finetune_updates.py`、`fixed_input_diagnostic.py`、`plot_regression.py`与service，SHA见最终清单，精确提交见Git历史。官方SHA `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`；step200 SHA `bba46271ef3c9962dcb01d525811a5b4f19b4d3cc2fe027afcf384e0f40e7a34`。原官方/200结果和seed42整动作16/4划分在`data/experiments/a3_fullchain_20261010/`。

报告/索引/current state、代码与SHA清单按阶段提交main，同次提交号由Git历史提供；同步结果以提交后远端HEAD核验为准。大型模型/trace留服务器LOCAL_ONLY，未完成网盘回下载核验。Q1只读，未改写并行内容。

## 下一步与未解决项

下一轮建议先明确并记录每个optimizer.step真正使用的LR，验证恢复前后的LR契约，再从官方权重做单变量短试验。例如在固定LR契约下仅把PPO epochs由5改1，保存2/10/20阶段并同条件评测，检验减少单批更新强度是否缓解早期退化。此方案是假设，不是已证明的改善方向；本轮未启动新训练。预先定义不增加跌倒且降低跟踪误差的通过条件，并重新固定诊断/最终验收集用途。

Isaac指标依赖、RKNN工具链、动捕质量/现场接口仍按[全链路用户待办](README_A3_OFFICIAL_TEST_20261010.md)搁置。本阶段已回答退化时点及可检验线索，尚未确定确切根因和最佳微调方向。恢复先读本README、当前状态与清单，不重复旧训练或已完成评测。
