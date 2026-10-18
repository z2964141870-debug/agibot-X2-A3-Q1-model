# A3 humanplus 输出头训练显式 Kit 日志修复

记录日期：2026-10-18。独立HP14/R23。16:16北京时间完成实施与标准库冻结，尚未登记或科学运行。用户最新“继续推进呀。持续优化”明确恢复主线，覆盖HP13故障后的等待恢复安排。

## 目标与范围

修复HP13退出后混淆IsaacLab提示与Kit日志的验收错误，完成原输出头短训练及六衣服/四官方匹配效果。计算只在humanplus@100.101.30.68:/home/humanplus/YUANQI_A3；Mac仅小代码与记录。HP13源码、STOPPED账本及模型不改、不恢复或效果评测；新试验fresh035开始，固定8更新，不追加训练预算。

## 完成内容与依据

固定vendor的kit_args原为global error/file error；旧适配只将file改info，仍依赖stdout的Logging to file提示。当前提示属于IsaacLab，不是可靠Kit路径。已只读核对Kit kit-core.json的/log/file设置、AppLauncher参数传递及SimulationApp.close的三条关闭标记源码。

独立humanplus_head_logged入口与entry/验证/阶段审计绑定HP14。AST只改原三处存储/日志/关闭调用位点；日志显式global info/file info/output error和本次logs/.../train/kit.log，启动前拒绝已有日志或路径别名。初始化后读取实际carb设置并记录native PID；标准库broker记录真实退出码与PID，读取固定日志的有序三条关闭标记，保存大小/SHA，审计再次验证。不会从任意旧日志或IsaacLab提示推断关闭。

复用未改的HeadOnlyTrainer/保存callback和teacher/有限接受机制。独立标准库8控制Mac0.013秒、新机0.011秒各通过，涵盖真实AST函数执行、错误/重复设置、IsaacLab提示、错误PID、日志级别、缺失/倒序关闭和旧/符号链接日志。原HP05/HP09/HP13控制不复跑，无新空场景/模型诊断。新readiness核验286代码/435输入/10环境来源SHA，旧HP13全部冻结源码不变；新launch只变两存储路径，训练器/保存callback与全部科学参数相同。0张量/模型/仿真/optimizer。

## 版本、命令与证据

实施起点969911c0102c8649ffe268ade530b35bb91eae54；代码提交以Git历史为准。vendor固定fe6868ba37034f89b912f0fb851bce19f120266d、共享环境不改。冻结清单data/manifests/a3_humanplus_head_logged_readiness_20261018.json，SHA e402dbf57cbb76cdad1a21e61bdb44c54d04c4136f5a9558e2219cb04fe58052。

035固定402050931字节/SHA9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d。HP11合同SHA3be04e74cee457a8ad3b351347c4f1cc85b0233e6a1dbc4d4101c6105c7a7094，新登记重核输入实际大小/SHA。基础环境/home/humanplus/anaconda3/envs/x2-sonic-isaaclab只读。

预定入口（每项单次、须上一阶段成功与独立审计；未执行项不是已有证据）：

```bash
/usr/bin/python3 -B -m script.a3.humanplus_head_logged register
/usr/bin/python3 -B -m script.a3.humanplus_head_logged run --stage train
/usr/bin/python3 -B -m script.project_tools.humanplus_head_logged_audit --stage train
# 后续verify-training、cloth、official、effect各阶段与各自独立审计
```

计划/结果：data/manifests/a3_humanplus_head_logged_20261018_HP14_{plan,result}.json。
工件：data/experiments/a3_humanplus_head_logged_20261018_HP14、data/training/a3_humanplus_head_logged_20261018_HP14/R23_001_s0、logs/a3_humanplus_head_logged_20261018_HP14。

## 固定预算、结果与边界

唯一科学变量相对全decoder方法仍是原末层两key、14877参数；相对HP13仅日志基础设施修复。fresh035 Actor/Critic/空Adam、原四开发动作/seed0/16env24步/token/损失/LR/clip不改。固定8更新/32提案及提交/3072交互、最多256候选+32live检查、2148外层Actor/decoder调用上限；teacher RMSE<=.05/maxabs<=.5/同batch目标<=live+1e-7。保存0/2/4/8，只有最终8成功验收才做10完整效果，不评测早期或STOPPED模型。

HP11基线六衣服均RMSE0.13932748084287783rad、5/6不倒，四官方4/4不倒。效果门六衣服全不倒/均RMSE改善>=1%、原不倒与四官方关节<=+2%、世界根/腕<=+5%、官方无新增跌倒。开发通过仍不代表独立seed或真正未见会话。当前035仍baseline，无新闭环改善证据。

运行仍需实时三次5秒门：>=8GiB/无其他计算/GPU空闲/遥测可靠；运行>=4GiB、CPU90/GPU85连续两次5秒、900秒/阶段。cgroup v2及全boot journal严格保护不改，旧快照不是启动证明。首真实机器故障保存并暂停，仅终止自有组。科学拒绝不续跑/扫LR/比例/权重/batch；HP12/E46–E50仍关闭。

## 保存与下一步

本阶段README/索引/current state及新代码/小清单scoped审diff/载荷审计后本地提交。GitHub写权限已知未恢复，不重复同一失败push，LOCAL_COMMITTED_UNPUSHED；模型/原始日志LOCAL_ONLY，未验证网盘不称备份。

下一仅新登记与当前资源门，通过后单次train→audit→verify-training→audit→cloth→audit→official→audit→effect→audit，逐重要阶段更新本README与入口。前台期间原zhy-a3保持暂停以避免重叠，完成后按用户持续优化指示恢复原30分钟工作，旧hp3090/其他自动任务不变。

## 16:19 单次登记

执行代码152711e，单次register已完成，HP14计划REGISTERED；重新核对全部代码、输入大小/SHA、环境来源、HP11合同和原资源门。登记只读元数据，不是模型或未来三次启动门证据。下一仅本计划train，独立阶段审计通过后记录，再接最终训练验证与完整效果。


## 16:21 训练与原生关闭验收完成

登记提交98f8089后单次train完成，独立train_audit verified=true。实际16环境cuda:0、fresh035 Actor/Critic/空Adam、两head精确所有权/梯度收据；8更新/32提案及提交/3072交互、全部首比例1接受、32live基线，64候选机制检查。teacher最高及最终RMSE0.01891755871474743、maxabs最高0.08703947067260742，原门保持；源码与JSON推导804外层调用<=2148，不是kernel计数。

本次Kit运行设置/实际native PID2887701/自有日志路径绑定一致，三条有序关闭标记齐全、原生退出0，broker及独立审计均通过。Kit日志1002968字节/SHA c715ceb13c2fc13bb75bf8a0e14b65980cecae48f706ead07688453208a02c0c；只核本次自有日志。step8为226205975字节/SHA c508fe2abf6613d4c9842333b3843d7303523349b9ce65ae5f0f36ff5287626a。0/2/4/8保存，尚未成功后张量审计/效果。

12健康同boot afb6796b-3883-4df4-99ea-bc381c52ea13、CPU最高81/GPU47°C、可用最低18987188224字节、0FATAL/新增OOM/残留。原阶段日志与result/train_audit清单保存，LOCAL_ONLY/GitHub未同步。HP13原终态不变。下一仅本HP14成功训练后的verify-training，之后完整最终效果；不将训练通过当效果改善。

## 16:23 验证入口依赖故障保留

HP14 verify-training单次启动后在函数导入initial_probability时，经cloth_constrained_trainer→consistent_precision→verified_finetune导入gear_sonic，因该验证worker的PYTHONPATH只有工程根而ModuleNotFoundError退出1。异常发生在读取官方/保存PT之前；0保存模型反序列化、0效果回放。原train COMPLETE及关闭独立验收不变，整个HP14 result保留STOPPED，不重跑或修改其注册源码。verify-training_audit verified=false，累计16健康CPU最高82/GPU47°C、0FATAL/新增OOM/残留，无已证实机器故障。

用户已恢复持续推进。下一修复评估入口的纯数值函数依赖，在新独立只读模型评估合同中明确使用HP14已完成且独立验收的训练工件，不恢复HP14、不重新训练或追加交互。新合同登记前仅代码/标准库/SHA；原HP14失败验证没有已完成的张量检查可复跑。新评估仍须完整最终checkpoint/所有权/Adam/token/配方检查和全部10项匹配效果；任何科学失败保持拒绝，不因工程修复放宽门。HP13/HP12原停止模型仍不评测。

## 16:29 剩余评估独立合同实施

[HP15实施与预算](README_A3_HUMANPLUS_HEAD_ASSESSMENT_20261018.md)已保存：只读接受本HP14已独立验收的完整训练工件，在新合同完成此前未开始的保存PT验证与最终10项效果，0新训练/optimizer/交互。新6标准库控制Mac/新机各通过，来源493项SHA核验通过。纯概率公式/门及原效果/Adam验证AST保持。HP14原结果/代码/故障不改，不重跑原验证或训练。下一新合同登记并重查实时资源，未登记前0新增张量/模型/仿真；035无新改善证据。
