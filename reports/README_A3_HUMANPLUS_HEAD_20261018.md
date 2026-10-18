# A3 humanplus 原动作输出头独立短训练

记录日期：2026-10-18。阶段：独立HP13/R22。15:44北京时间终态STOPPED_BROKER_LOG_PATH；实际8更新完成但关闭日志验收失败，不是已验收训练模型。下方15:35/15:37为历史阶段证据。

## 目标与范围

落实[D01方案与静态实施审查](README_A3_HUMANPLUS_HEAD_DESIGN_20261018.md)，唯一科学变量是Actor可训练集合缩到既有module.12.weight/bias，静态29×512与29共14,877参数。原架构、fresh035 Actor/Critic、原四开发动作、实际token、损失、LR、clip、teacher门、有限候选和HP11效果门保持。所有计算只在humanplus@100.101.30.68:/home/humanplus/YUANQI_A3；Mac仅小代码与记录。

## 完成内容

- 新humanplus_head_trainer完整覆盖create_optimizer：先冻结两key，再显式调用固定Transformers通用创建器，绕过会解冻整个decoder的cloth祖先。原生空AdamW组超参保留后立即包装FiniteAdamW，在scheduler/callback/Accelerate之前返回同一最终optimizer；拒绝晚替换、重复初始化、额外owner与任意optimizer override。
- 精确核对两head加全部原Critic的参数所有权、组顺序、形状、冻结梯度和Actor参数/buffer。首次训练及事务边界留存两head实际梯度证据，不增加forward。
- 新独立trial、entry、成功验证、阶段审计和泛化停止审计全部绑定新根；旧HP12及旧共享训练源码均未修改。纯AST/receipt/recipe帮助函数可复用，不调用旧register/run/verify/审计。
- 保存head_owners.json，记录实际组顺序、序列化整数ID→参数名、形状与checkpoint SHA；成功后核对两key的非空变化子集、其余Actor逐值不变、Adam矩形状/有限性/计数。停止审计只读JSON及文件SHA，不反序列化停止PT，按k次提交及至多一次拒绝推导实际计数，并核对未提交batch buffer摘要恢复。
- 最终14项新标准库控制Mac/新机各通过（0.024/0.028秒）；此前Mac12项中间版本通过后因增加保存回调/停止审计控制运行最终14项。控制直接执行新creator及callback的AST函数体，使用无张量fakes；这不是数值optimizer或16环境验收，不重跑HP05/HP09/HP12控制。
- 单次只读readiness核验278项目Python源码、433输入大小/SHA、8环境源码，固定vendor干净，vendor三处AST适配可编译。新launch与HP12只差训练器/保存callback及两个存储路径。未加载张量/模型、未运行仿真/optimizer、未采新资源启动证据。

## 关键决定与版本

实施起点Git HEAD为2470ae78faa90b1a9a519cf2e93a2bd976ba2dcd；本阶段提交以Git历史为准。vendor固定fe6868ba37034f89b912f0fb851bce19f120266d。没有修改共享环境、驱动或他人作业。

readiness为data/manifests/a3_humanplus_head_readiness_20261018.json，SHA b123704feef4950522846ab43a882fcb0d71044f82efdea9b3ef4636bfafbf96。控制收据为data/manifests/a3_humanplus_head_controls_20261018.json。前者是实施/输入冻结证据，尚非注册合同或未来健康证明；正式register仍须重核全部SHA和资源。

035：data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt，402,050,931字节，SHA 9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d。本轮重新哈希通过。HP11合同SHA 3be04e74cee457a8ad3b351347c4f1cc85b0233e6a1dbc4d4101c6105c7a7094。

## 固定预算与结果边界

最多8更新/32原生提案与提交/3072交互，候选256+live基线32，固定1..1/128，不扩比例/扫LR/换batch。外层Actor/decoder上限：rollout192、原训练96、初始teacher4、提交后teacher128、候选机制1728，共2148；保存审计使用JSON事件与固定源码路径推导，不冒称kernel计数。

保存0/2/4/8；只有32提交及最终8独立验证全部通过才做六衣服四官方10完整匹配效果。六衣服全不倒、算术均RMSE至少改善1%，原不倒与官方关节<=+2%、世界根/腕<=+5%、官方无新增跌倒。任何科学STOPPED不恢复/扩预算/早期效果。HP12及E46–E50保持关闭，035尚无新改善。

启动仍须当前三次5秒、可用>=8GiB、无其他计算/GPU空闲/遥测可靠；运行4GiB保留、CPU90/GPU85连续两次5秒、阶段900秒。新机cgroup v2和全boot journal保护不变。首真实机器异常留证并暂停heartbeat；仅终止自有进程组。

## 命令、产物与下一步

以下为服务器工程根本次已各单次执行的历史命令，终态STOPPED后禁止重跑；不能直接跳到worker/native或后续成功验证。

```bash
/usr/bin/python3 -B -m script.a3.humanplus_head register
/usr/bin/python3 -B -m script.a3.humanplus_head run --stage train
/usr/bin/python3 -B -m script.project_tools.humanplus_head_audit --stage train
```

科学阶段必须逐次保存记录并独立审计；本次STOPPED后只完成新的终态JSON收据审计，没有调用verify-training/效果。候选拒绝专用humanplus_head_stop_audit不适用于本次日志验收故障，未调用。

预定输出data/experiments/a3_humanplus_head_20261018_HP13、data/training/a3_humanplus_head_20261018_HP13/R22_001_s0、logs/a3_humanplus_head_20261018_HP13；登记前账本确认无HP13现存计划。实际两key形状/梯度/16环境尚未验收，不以fakes冒充。下一先scoped审diff/载荷审计/本地提交，再独立register及真实启动门。

## 保存与同步

README、索引/current state、代码与小清单scoped保存。GitHub SSH publickey写权限此前已失败，不重试同一失败push；LOCAL_COMMITTED_UNPUSHED。所有模型/轨迹/原始日志LOCAL_ONLY，无新增网盘核验。zhy-a3沿用30分钟ACTIVE、仅变化通知，旧其他任务不变。

## 15:37 单次登记

执行源码781639070298fc7e17d99b0f19e34e433565efc0，独立计划data/manifests/a3_humanplus_head_20261018_HP13_plan.json状态REGISTERED。register重新核对readiness全部代码与输入身份、D01继承来源和环境SHA；包含输入大小、8更新/32提案/3072交互和2148外层调用预算。原HP11合同/teacher/效果门及无恢复/早期效果规则不变。登记仅采元数据，没有加载模型/张量或运行仿真/optimizer；尚无未来阶段三次启动门证明。

下一仅本计划train，经原supervisor三次5秒资源门才允许创建16环境；每阶段及独立审计后保存README。任何科学STOPPED只读新停止收据，不调用成功验证或早期效果。

## 15:44 训练后日志验收故障与终态

登记提交86712ca后，单次train由原监督器三次5秒启动门通过再创建16环境cuda:0。fresh035 Actor/Critic精确匹配、空Adam、实际两head形状29×512/29与梯度、其余Actor冻结保护有运行JSON证据。完成8整更新、32原生提案/32提交、8rollout/3072交互；32候选均首比例1接受，加32live基线共64完整候选机制检查。提交后teacher RMSE最大及最终值0.01891755871474743，maxabs最大0.08703947067260742，均在原0.05/0.5门内。固定源码和事件推导外层Actor/decoder804次，低于2148上限；不是kernel仪表计数。

保存0/2/4/8，step8为226,205,975字节，SHA 63f6989b73b4b29639e854e36112a7cc5202f1b7aaa080dad1c1a925393fecd8。它是未最终验收的存证模型，不是已通过效果的新策略。停止后没有张量反序列化、模型验证或效果回放。

训练子进程留下native_training.json（8/32/3072、CUDA同步）、training_complete/context_clear_returned/native_close_begin，native_exit.json实际returncode=0；app.close未返回Python，因此没有native_close_returned。这与此前原生退出行为一致，但本次完整Kit关闭日志仍未验收，不能仅凭退出0补成功。

故障在标准库broker：复用了旧入口中从stdout匹配`Logging to file:`并要求唯一ENV/isaacsim/kit/logs路径的逻辑。当前匹配到的是IsaacLab环境日志`/tmp/isaaclab/logs/isaaclab_2026-10-18_15-39-35.log`，并非已证实的Kit日志，因此抛`ValueError: Exactly one authorized native Kit log required`，worker退出1、监督器终态STOPPED。没有发现足以证明当前Kit完整关闭的日志，不将/tmp日志重新命名/移动来满足门。实施时标准库控制覆盖了参数/调用/保存隔离，但没有覆盖日志提示所属组件与路径的这个组合；不能把此前控制通过当本次关闭已通过。

12健康样本同boot afb6796b-3883-4df4-99ea-bc381c52ea13，CPU最高79/GPU46°C、可用最小19,017,109,504字节，0FATAL/新增OOM/自有组残留；阶段55.164秒。单次阶段审计verified=false，after_issues和remaining_own_group_processes均空；没有已证实机器故障。这是运行/验收异常，按原异常停止约定保存并暂停zhy-a3，不当作teacher科学拒绝。

证据为data/manifests/a3_humanplus_head_20261018_HP13_{plan,result,train_audit,terminal_receipt_audit}.json。新增独立humanplus_head_terminal_audit.py只读现有JSON与checkpoint字节SHA，验证32事务/所有权/梯度收据、teacher门、8rollout、4份checkpoint sidecar及无后续阶段；不修改注册源码或终态、不加载PT、不复跑旧审计。其状态VERIFIED_STOPPED_BROKER_LOG_RECEIPTS_NOT_MODEL_OR_EFFECT_ACCEPTANCE，不能代替成功张量验证。原生progress/console/health及训练目录原样保留，模型/原始日志LOCAL_ONLY。

zhy-a3已通过应用工具更新为PAUSED，30分钟原节奏保留，其他任务不变。没有自有作业残留，没有开机恢复。HP13全部register/train/阶段审计/terminal审计已单次执行，禁止重跑、扩预算、verify-training和任何早期/最终效果；HP12和E46–E50同样保持关闭。

恢复入口仅先只读定位未来独立契约的Kit日志输出地址及可可靠绑定的关闭收据，必须显式约定自有日志路径或运行时身份，不扫描任意旧日志充当本次证据，不降低关闭验收。本轮尚未实施未来日志修复或登记HP14，不自动再次训练。035仍baseline，无闭环改善证据；后续推进须新的明确恢复指示，不能把这份故障审计当训练重试许可。
