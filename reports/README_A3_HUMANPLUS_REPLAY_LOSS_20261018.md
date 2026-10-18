# A3 humanplus 多动作保留损失方案

记录日期：2026-10-18。D02候选方案。16:58北京时间记录。未实现/未登记/未占HP编号，无新科学运行。用户已明确继续推进与持续优化；完成实现与登记后可按原资源和科学保护推进，不需重复索取迁移/认证或恢复许可。

## 目标与证据边界

HP14末层两key训练完整通过固定384原teacher门，HP15保存状态已验证，但HP16/HP17十项完整匹配效果拒绝：衣服均RMSE恶化13.13%、仍5/6不倒，四官方关节全超2%退化。这只支持固定锚点通过不足以证明闭环保留；不能证明缺覆盖是唯一原因，也不能保证扩大覆盖改善。该原head配方关闭，不重复同样训练/追加轮次或LR/权重扫描。

新单变量拟仅改teacher正则损失的样本覆盖：将原单rollout384帧替换为HP11四官方完整闭环中均匀取样各96帧，共384帧。Actor仍只训练原末层weight/bias，fresh035/Critic/空Adam、原四衣服训练段、seed/16env24步、损失系数1/LR/clip/有限接受/8更新32提交3072交互及原HP11效果门保持。原E11全部384硬teacher保护仍保留，不用新样本替代或放松旧门；不是把失败模型续训。

## 数据与分区

使用已保存HP11四官方NPZ的1570维实际观察与29维preclip Actor输出。每段固定tick floor(i*(N-1)/95)，i=0..95；覆盖首末，不根据当前模型误差择帧，不增加录制/模型采集。四段637/882/1652/2604步，具体384 tick与17来源文件大小/SHA在data/manifests/a3_humanplus_replay_loss_design_20261018_D02.json。当前只标准库读取合同/大小/SHA与整数选择，没有加载NPZ/张量或新forward。

四官方原本反复作为开发基准；用于新loss后明确成为DEVELOPMENT_REGULARIZATION，更不能当未见测试。十项仍固定开发门，真正未见会话与独立seed未提供。部署始终同一策略，不读动作身份决定是否启用适配。teacher标签是原baseline裁剪前动作，不把±20后的控制动作误当网络teacher。

## 实现要点与计数

原FiniteRetentionTrainer的候选objective同时使用其原teacher误差，因此不能只替换训练loss数据而保留旧objective：live候选必须重建与新实际loss相同的四分之一新loss，固定同batch Critic值；同时独立计算原384 teacher硬门。需要独立trainer/evaluator，不修改旧冻结源码或全局路线。保留baseline目标一致门、原参数/Adam影子提案和一次提交/拒绝回滚。

训练每batch新loss代替原同规模96帧loss，不双加loss或改系数；32batch/8更新/3072交互不扩。为保持原384硬门，候选每次增加1个新quarter前向，最多32×9=288次；fresh035新标签重建4次。因此旧外层Actor/decoder上限2148变为候选2440（+288+4），不是增加交互/提案/科学尝试数。该算术尚须实现源码路径核定，不能把草案当执行预算。

关键缺口是flat1570到固定原生a3_fast观察/路由的精确桥接，必须匹配原codec/量化/拼接与冻结行为，禁止在样本维度或eval身份切换策略。新384标签对fresh035的最大误差须<=1e-5，检查只纳入完整登记后的真实训练入口，不另做空模型/多轮诊断；旧teacher初始化、FP32/token/ownership/冻结检查链保留。

## 版本、保存与下一步

证据来源终态提交1e8027405bfc9165139478107a21cfc1b77c2b49；D02小清单冻结17源码/记录/NPZ身份和384整数tick，草案不是新执行代码SHA。计算唯一根humanplus:/home/humanplus/YUANQI_A3；Mac小代码/记录，共享环境/vendor不改，原模型/日志LOCAL_ONLY。GitHub未同步，不重复同一已知publickey失败push。

下一具体工作一次收束原生桥接与独立loss/candidate实现、标准库控制与实际源码调用预算，再固定新代码/输入SHA并登记。不要再增加一轮同类方案README或重复旧科学诊断。完整实施/登记前0新模型运行；登记后原三次5秒启动门/4GiB保留/CPU90/GPU85两次/900秒及cgroup v2/全boot journal保护保持。原HP12/HP13/HP14/HP15/HP16停止及HP17已完成阶段不复跑。新科学STOPPED不续跑/扩预算/早期效果，只有最终8全过门后十项完整效果。

本任务zhy-a3按用户持续优化指示恢复原30分钟ACTIVE；先实际Git/README和资源核对，状态未变静默，仅重要进展/故障/新阻塞通知。首真实机器故障或达原可验收模型目标保存并暂停；旧hp3090/yuanqi-a3保持PAUSED，daily yuanqi不变。

