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

