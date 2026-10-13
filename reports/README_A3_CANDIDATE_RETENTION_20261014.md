# A3实际候选位移的能力保留约束：E49实现与独立检验

记录：2026-10-14。状态：**STOPPED**，已执行一次，不得复跑；投影尚未计算。官方035仍baseline。

## 目标与依据

E48固定新batch已见原始总Actor梯度下降向内、保存Adam位移向外，且有限teacher MSE增加7.63463e-5。候选尚未超过原.05/.5保护，所以不能称触发了历史停止。下一单变量是对**实际Adam位移**加入完整teacher的一阶非增约束，而非改梯度权重或增训练次数。

## 固定实现与预算

只读同E45 step4/moments和E48唯一保存fresh_rollout.pt内首batch/实际tokens，0新rollout/optimizer/checkpoint。完整Actor+Critic梯度、原0.1clip和原KL调LR复用，先核验原E48候选的标量和方向；不从另一个batch开始。

记g为完整384teacher MSE梯度，delta为原Adam实际位移。仅若g·delta>0，计算一次`delta_projected = delta - (g·delta)/(g·g) * g`。g范数为0或输入非有限即拒绝，不用极小数掩盖。FP32舍入后的真实位移重新核验一阶条件；登记前容差已固定，见下文。

随后完整384teacher有限步检查原RMSE<=.05/maxabs<=.5，另外要求MSE不增、固定同batch Actor目标不增。这是新候选诊断的更严格判据，不改变原训练效果门。投影到切平面不保证非线性有限步合格；若拒绝就记录，不自动缩步/扫LR/换权重。通过也不生成部署checkpoint或宣称闭环改善。

预算：1保存状态、最多16Actor/decoder forward（原2+4+4，投影teacher4和衣服2）、1Critic train forward、3autograd.grad、原候选1/投影候选1，900秒。没有仿真App，无关快速退出问题不会带入；最终progress/源文件核验需在正常退出前持久化。静态核实计数不变；运行中不扩展。

## 输入、保护与恢复

小清单`data/manifests/a3_candidate_retention_plan_20261014.json`包含源SHA与E48 fresh_rollout SHA `30a05c7fdad3c2cd7d022c567e33cd8a07b939b3344143127c9f9486dc8818c0`。缓存维护以本计划保护下一输入，勿将待用E48数据当闲置缓存处理。

所有计算仅ZHY既有环境，登记/启动各三次5秒原可用估计>=8GiB、无其他大进程/计算、GPU空闲/可靠遥测；4GiB运行保留、CPU90/GPU85两次5秒、900秒，首次异常停分支不复跑。E44/E45不恢复/不评测，E46–E48禁止复跑。当前0worker。登记和启动均完成原三次预检；执行已停止，下面命令仅保留版本复现说明，禁止再次运行E49。

## 保存与边界

实现/5项标准库控制与AST检查通过；实际执行在原映射JSON比较拒绝，0投影。原始工件ZHY LOCAL_ONLY，网盘未核验。本报告与清单按scoped审计提交推送；当前执行证据入口README_A3_FRESH_BATCH_20261013.md。若局部约束不合格先记录原因，不再连续堆同类零更新实验；若合格，另设短训练及完整匹配效果验证，官方基线及未见测试要求不变。

## 登记前固定的实现、容差与验证

代码为`script/a3/candidate_retention_{trial,probe,protocol}.py`，单独复用旧loss/映射/监控函数；未修改E46/E47/E48已登记源码。调用装饰器计数16Actor/decoder+1Critic，3次autograd.grad；禁止optimizer构造/step和torch.save。不创建仿真App，result/progress/退出前源文件核验正常落盘。

原E48对照：标量atol1e-7/rtol1e-5、方向atol1e-8/rtol1e-4；核验同Actor/批次/teacher/保存optimizer/原Critic buffers SHA、完整映射/28参数、loss/梯度norm/clip/KL/LR/原候选方向与有限误差。只有原对照通过才计算唯一投影。

投影用FP64计算，再一次舍入为FP32参数；实际舍入后一阶变化<=1e-8，有限teacher MSE变化<=1e-9，固定Actor目标变化<=1e-7；原RMSE<=.05/maxabs<=.5保持。固定Actor目标是原cached-token PPO/entropy/aux加首96teacher，Critic值在两次目标比较中取同保存常数；另报不含teacher项的衣服loss。所有门和容差在结果前固定，不扫参数。

五项标准库控制包括：去除外向分量、内向不改、切平面通过但非线性仍增的反例必须拒绝、舍入/目标各自拒绝、零范数/非有限/控制方向翻转拒绝、预算不允许第二次投影（其中部分合并为一测试）。Mac/原hp3090均0张量/模型执行。

```bash
# 仅标准库验证；没有加载模型
/usr/bin/python3 -m unittest script.a3.test_candidate_retention_protocol -v
# 以下仅ZHY；完成发布/同步与原预检后各执行一次，不作为重跑入口
/usr/bin/python3 -m script.a3.candidate_retention_trial register
/usr/bin/python3 -m script.a3.candidate_retention_trial run
```

源SHA/输入SHA/预算及容差已写`data/manifests/a3_candidate_retention_plan_20261014.json`，登记后不改此计划。预期产物`data/experiments/a3_zhy_candidate_retention_20261014_E49/`，日志`logs/a3_zhy_candidate_retention_20261014_E49/`，均ZHY LOCAL_ONLY。实现待scoped审计推送并核对，提交号由Git历史与registration记录，不预称执行成功。

## 实际停止与静态定位（北京时间2026-10-14）

E49在00:47:30–00:47:51单次运行后STOPPED，worker退出1，监控reason=null；真实错误为`ValueError: Original optimizer mapping or flags changed`（`candidate_retention_probe.py:204`）。实现提交`13749414180c277465cdd9d8e112af04d6dcabce`。1保存状态、10Actor/decoder+1Critic、3grad、1原始数学候选、**0投影/0optimizer/0新rollout/0checkpoint**。没有result.json，progress已写；不能记为投影效果拒绝/通过。

错误是本次新增校验实现问题：运行时Adam betas是tuple，历史JSON中的betas是list，Python直接结构比较为不等。标准库读取两份mapping.json，28参数和全部group字段逐项完全相同；只查看checkpoint ZIP内pickle操作码，betas(.9,.999)后为TUPLE2，未unpickle/加载模型。严格SHA复核99输入与源码文件均未变化。

停止前原始方向/梯度/clip/LR已精确复现E48：负Actor方向-.6137541928995537，Adam方向+6.256351608542673e-5，全梯度28.25528335571289、clip .0035391610581427813、LR6.75e-6。但有限原候选结果未持久保存，不能补称已完成其全部对照；投影有效性未知。运行中没有再次加载或重放。

6健康样本同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，OOM0，CPU41/GPU30°C，GPU最高111.56W，最低可用7.024GiB；0已记录内核/遥测异常。登记至stage启动存在时间缺口，健康最大间隔109.70秒，不能称无间断覆盖。未确认机器故障。

小清单`data/manifests/a3_candidate_retention_stop_20261014.json`；原始progress/mapping/pre_candidate/console与独立标准库static_failure_check保留ZHY LOCAL_ONLY。本地只保存小收据。原登记计划与E49代码不改，停止账本不重置。

## 校验修正与后续约束

新增独立`script/a3/optimizer_metadata_contract.py`：只允许betas的tuple/list表示统一；参数ID/shape/step/数量以及全部数值、字段和flags保持严格相同，布尔值不当作数字1。`test_optimizer_metadata_contract.py`含真实JSON往返、微小LR/eps/beta差异和flag/ID/shape/step/非法元数据拒绝控制；尚未接入已冻结的E49。

本次停止必须保留，不能仅换编号自动复跑。下一若重新执行修正版，需要单独明确继续这项已停止检验；先准备可审阅修正，保持0worker，不恢复E44/E45或其他旧实验。官方035仍baseline，当前无新增效果改善证据。

修正版E50已独立实现但**未登记/未启动**：`candidate_retention_followup_{trial,probe}.py`和`a3_candidate_retention_followup_plan_20261014.json`。与E49唯一实现差异为betas跨JSON表示比较；同输入、同16/1/3预算、同一次投影与固定容差。4项独立元数据回归控制通过；新控制器要求计划中的明确重试授权后才能登记，当时未填入授权或执行。不是给失败实验换编号自动重跑；E49旧代码完全保留。

实现与停止记录已推送核验main `ce79397b91f14c2589348f7fab77870e04773f06`并同步ZHY。新元数据比较在ZHY使用保存的JSON和静态确认的tuple表示再次标准库验证通过，0模型重载。E50仍未登记；已向用户询问是否允许这一次纠错接续，用户随后明确答复“执行修正版一次”。停止后缓存维护因Python父包装器被原空闲门拒绝，0释放；后续只读可用8.325GiB、无其他计算，详见当日内存README及接续readiness清单。

## E50单次接续授权

2026-10-14用户明确答复“执行修正版一次”，已在E50计划登记授权范围。E49仍STOPPED不改；E50按相同输入/预算/全部数值门槛独立登记与执行一次，不扩大候选、扫参数或自动复跑。此段写入时E50仍未登记。
