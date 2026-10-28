# RF16保存接触相位短对照

## 首300秒检查

300.829688实训秒/71轮三重复已保存：验证0/0/0失败、两普通走与抬手全部3/3完整，但两走没有一致滑移/摆脚收益，原改善门0/3、严重退步1/3（repeat0停走倾斜比1.25062）。停走承重滑速相对未训练同alpha.5为1.3144/1.0127/.8825倍，起走1.2480/.9885/1.0526倍；不能用平均改善覆盖片间或重复间退步。上肢RMSE.2963-.2965rad基本保持，主要收益仍来自静态手臂混合，不归因head学习。

05:51:43原controller/worker/start_ticks/boot/完整命令一致，实训503.595399秒/120轮、11796480转移、critic2400/actor2200，GPU8179MiB/49度、共享18.72GB/OOM0。原learner继续600/1200门、硬1300秒，不扩时/重启或修改冻结来源；07:00物理/优化截止与09:00交付不变。

新增独立只读RF16审阅工具，help及真实6份初始/9份首门JSON集成通过，分别输出1980/2970逐片行，首门另有108训练/验证、27固定、63跟踪和72步态共同前缀行。初始/首门13小证据4026383字节Mac/25386大小SHA匹配；14审阅派生文件2132978字节回镜服务器并核SHA。工具不在1532冻结来源内，0新仿真/推理/optimizer。审阅入口`data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/review/stage_01/README.md`；视频待原pipeline终后生成，用户采纳null。[首门原值及现场收据](../data/manifests/supermimic_q1_RF16_stage01_20261029.json)。

## 正式唯一启动

05:35:23实际启动controller161302/start_ticks13696801；worker161313/start_ticks13696954，原boot/GPU一致。正式来源2c83b944059d4a007f0ca74fa54c0076fda3a357/最后4文件15385字节三端SHA及独立远端已核，CPU5反例通过，原模型/动作/奖励实现与预检一致。plan冻结1532项，含330metadata。05:36:01三状态RUNNING、selected_base repeat0/frame150、0实训/0optimizer，完整命令/start_ticks/mtime和GPU计算进程一致、GPU2729MiB/34度/共享17.49GB/OOM0；只是正常初始评测，不是效果改善。

只监督此原进程，约300/600/1200实训秒检查，硬1300/不扩时/不故障重试；全pipeline新源码冻结，报告提交不改变来源。07:00物理/优化截止，09:00非训练交付截止。原RF15已结束，不恢复；完整命令/身份/资源与来源见[启动收据](../data/manifests/supermimic_q1_RF16_launch_observed_20261029.json)。以下“尚未训练”仅前置准备历史。

## 05:33预检完成

正式开训前仅把独立guard的07:00截止明确限于物理/优化worker，非训练delivery仍受09:00保护；新增CPU边界反例核物理会停、已结束物理后的交付可继续。未改变已预检模型/adapter/reward/动作/训练实现，不重复物理预检；最终正式来源以本次后续冻结commit为准。

服务器既有activate.py环境4相位反例通过，11源AST通过；16/4096真实零optimizer预检05:29:40/05:31前正常COMPLETED，1532冻结来源（含330metadata）未变、双controller/worker已退出，05:32:08 GPU15MiB无计算进程/OOM0。4096小批24576概率差0，原生接触/重置前状态/人工截断bootstrap/自然终止不bootstrap、奖励分解/相位cache、原训练状态和CPU/CUDA RNG保护、冻结教师/归一化/actor参数均通过。

phase cache为300片77949帧，stance/swing/neutral约63.34%/33.86%/2.80%，与源/clip严格对齐；仅参考意图不是原生承重标签。预检各6小工件546929/546965字节Mac/25386 SHA匹配。代码8780110b07cdee3e0fb3e2faa20753fce9574995及15文件222765字节三端核验；正式source不再修改，仅报告提交不改变来源。尚未正式训练，以下准备“未预检”仅历史；开训后须现场身份/clock/guard核验。[预检原值收据](../data/manifests/supermimic_q1_RF16_preflight_20261029.json)。

2026-10-29 05:25，北京时间。夜间自由探索授权下独立新假设/新预算，不恢复RF15或补其剩余预算。尚未物理预检/训练；Mac系统Python因未安装torch未能导入单元测试，改用既有服务器activate.py环境，原始错误保留。11源AST已初查，正式CPU/16与4096预检及发布核SHA后才开训。

## 目标与变量

原相位审计证明reference声明支撑点基本锚定，但RF15踝点启发式与contact意图部分不同，尤其固定抬手右脚约11.8%帧相位相反。独立RF16仅把足端特征/奖励分段改成保存contact的保守30→50Hz映射：两邻源帧均on为stance、均off为swing、混合为neutral。标签仅参考意图，不当实际承重真值；不重转/筛选或修改原CSV/metadata。

固定RF14两小时stage02教师/归一化、手臂静态alpha.5、原300训练/30验证、4096真实MuJoCo Warp/8线程、原观测/目标/控制/PD/effort。新256/128 ELU14维腿腰head与critic，1420特征、std.1、物理残差.15tanh；复用RSL-RL PPO24步/5epoch×4小批、minibatch24576、lr5e-5且原生KL.01限到1e-5..5e-5，前10轮critic-only。原reward及.02stance/.02swing/.01stationary/-.002smooth/-.002residual系数全部保留；相位同时影响相应特征和奖励，不能分拆归因。

## 一次性预算

约300/600/1200实训秒三重复原330片及固定两普通走/抬手，软1200、硬1300秒，独立guard1240秒触发+45秒宽限；不得扩时、自动重试或复用旧优化器。07:00新优化/物理探索绝对截止，09:00交数据与本地MP4，评测/保存/媒体不计实训。严重退步沿用逐片滑移>=1.5或倾斜>=1.25或验证失败>=6、至少2/3即停；根XY仅诊断，改善也不扩预算。

## 核验与交付

先四CPU反例、实际16/4096零optimizer，检查原生接触、终止前/重置后、bootstrap、奖励分解、相位cache及clip时轴、冻结教师/归一化/状态RNG、同策略概率；正式源/metadata全冻结发布三端SHA。来源脚本独立script/supermimic_q1_contact_comp，原gait_comp依赖保留不改。新head初始零输出等于同静态alpha.5，只重建features不多抽一次观测噪声。

资源100GiB余盘/12GiB实验/30GiB共享/OOM0/86度/900秒实际无进展，单一GPU作业。独立根data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029。结束先真实JSON/NPZ/阶段决策，再六格/500iteration均值/PPO/分项、固定54秒8段6类原速本地MP4、失败hold保留；训练收益与静态上肢收益分列，WM/想象N/A，用户采纳null。权重/大NPZ只留服务器。

[预登记收据](../data/manifests/supermimic_q1_RF16_preparation_20261029.json)；[相位审计](README_Q1_REFERENCE_SUPPORT_20261029.md)。本次可能无改善，最终按具体数据/仿真效果交用户判断。
