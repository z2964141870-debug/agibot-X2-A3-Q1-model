# RF11：恢复RF09配置，仅修采样的一小时对照

最终状态：[RF11结果与交付](README_SUPERMIMIC_Q1_RF11_RESULTS_20261027.md)已完成训练、330片初末物理评测/共同前缀对照、六格图及本地初末54秒MP4。22:08:27全pipeline COMPLETED，五阶段正常reap；22:17原controller/worker均不存在，GPU已释放。验证0/30提前终止但根跟踪退步、上肢基本持平；36/19/12项训练、结果和媒体检查通过，29文件逐大小SHA匹配。下方RUNNING/待交付为历史，不自动续训、重试或开新训练；三端发布后暂停本轮监督。

21:55:05训练完成：实际3500.265989576秒、3271学习轮、33755136转移，WM32710/actor2771。训练guard于21:55:10正常reap退出0；训练worker17373已不存在，controller17370仍按原身份执行评测。36项保存/张量/更新计数/来源及参考身份审计全部通过，0新积分/推理/optimizer；[训练审计](../data/manifests/supermimic_q1_RF11_training_audit_20261027.json)、[阶段现场](../data/manifests/supermimic_q1_RF11_training_completed_20261027.json)。未追加训练，后续原自动评测继续。

21:57首轮330片物理JSON已保存：训练提前终止24/300、完成98.06%，验证提前终止0/30、完成100%。这是终止/完成证据，不等于已准确跟踪。全rollout验证根XY RMSE0.12498m、上肢0.93942rad，尚未做RF09共同前缀与结果算术审计，不能与不同执行时长均值直接宣称改善。六格图、初末各54秒MP4及本地交付仍待全pipeline结束。

21:04:01：已533学习轮/5730304转移/WM5330/actor33，实训436.950360秒；controller/worker身份、三状态与时钟匹配，GPU14619MiB/50°C/OOM0。已进入真实策略更新，尚无效果结论。原每10分钟监督`q1-rf08-4096`已通过工具更新为本RF11并实际ACTIVE，仅重要结果/异常通知，交付后暂停；仓库/每日研究任务未改。[监督与现场](../data/manifests/supermimic_q1_RF11_monitor_20261027.json)。

20:57:40现场：正式RF11已启动，controller17370/start_ticks1943459、worker17373/start_ticks1943487，boot/GPU与计划一致；三个状态RUNNING且真实命令/进程匹配。20:56:29 controller开始、20:56:44实际采集/训练开始，已58学习轮/864256转移/WM580/actor0（正常500轮预热）。4096原生obs80/goal590转移契约通过，GPU9897MiB/47°C、共享7.21GB、OOM0、盘余1.638TB。[启动现场](../data/manifests/supermimic_q1_RF11_launch_observed_20261027.json)。冻结提交`7a3567cd5b89e5cd6282a83aacb61f19590b0434`已独立核远端，15项部署SHA一致。预计21:55左右结束训练，评测/交付另需时间；实际以receipt为准。恢复只读状态，不重复启动或改冻结来源。

记录日期：2026-10-27，北京时间。状态：C02构建、CPU契约与原生16环境smoke通过，正式训练待发布启动。用户最新明确授权在改进方案下再训练一个小时；本轮独立4096环境，实际训练≤3600秒，不重启RF10。

## 问题与方案

[RF10](README_SUPERMIMIC_Q1_RF10_RESULTS_20261027.md)正常完成计算但物理表现崩溃：验证29/30提前终止，基线RF09近一小时为2/30；四片PD目标变化RMS1.91–2.36rad/20ms、动作变化罚压过正tracking。组合包含观测、奖励与采样，唯一因果未隔离。不能把训久或数据不足作为当前结论。

本次采用已稳定一些的RF09配方，只保留已证尾bin修复：按合法转移数建采样bin，保证起点f<clip_end，避免下一步跨动作参考。撤回C01的当前关节误差观测、直接关节tracking、朝向带宽和velocity权重变化。恢复obs80、原reward/噪声/goal590；控制/动作尺度/full effort/PD、reset、未来padding、H32和同300/30数据保持。这是采样bug修复的受控对照，不保证解决上肢持续bias或长程模型方向风险。

RF10已经包含同样采样修复，故不能说本次“修复了RF10未修的bug”；本轮检验的是撤回观测/奖励组合后，修复后的RF09是否稳定、是否较旧RF09有改善。与RF10比较仍是组合撤回，不能单独判定哪个字段致崩。

## 已完成验证

使用现有build_tracking_candidate.py的sampler_only分支，无新增训练公式。新独立runtime：
`/opt/data/private/ZHY/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027`，
源为RF09的AP01_runtime_20261026，固定HP`310244b73e45775744a75b789eb25a6c9c5a81c9`。

- CPU五组/四片12状态通过：参考张量逐位相同、合法bins、obs/goal/历史噪声和reward与原Base一致、旧80维schema兼容；0积分/0optimizer。
- 16环境真实Warp转移/奖励f+1、obs80/goal590/state57/context26通过，SMOKE_COMPLETED，0optimizer；不是4096完整profile或效果验证。
- sampler_only使用原BaseRewardCfg和原obs/noise，控制值只重定位runtime绝对路径。没有引入C01新tracking/误差字段。
- 启动前GPU15MiB/26°C、余盘约1.638TB、共享内存约3.74GB、OOM0；进程扫描匹配到检查命令自身，不能把该输出当训练存活。RF10已退出，RF11输出不存在。

小收据为data/manifests/supermimic_q1_RF11_C02_{candidate_source,cpu_contract,smoke_contract,smoke_receipt}_20261027.json。CPU测试只验证接口和公式，不证明新策略有效；正式worker启动时再做4096原生转移契约。

## 冻结配置与预算

用户自己的25386容器、唯一根`/opt/data/private/ZHY`，boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。独立输出：
`data/supermimic_for_q1/BONES_RF11_SOMA300_SAMPLERONLY_E4096_MAX01H_20261027`。

从零seed0，4096env/8线程、WM1024×10、actor1024/H32、critic5×4、UTD.4、BF16/compile。相同300/30清单SHA`ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337`。不从训崩RF10续训，也不复用smoke/profile权重。

软目标3500秒，外部3540秒保护加45秒宽限，实际训练硬上限3600；初始化/等待/评测/绘图不算训练。保留100GiB余盘、12GiB实验、30GiB共享cgroup/OOM0、86°C、900秒无进展保护。不自动重试、延长或并行第二训练。

## 运行命令与评测

在服务器根下，用现有Python执行：
`script/supermimic_q1_hp/run_trial.py --mode train --runtime /opt/data/private/ZHY/data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027 --library /opt/data/private/ZHY/data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF11_SOMA300_SAMPLERONLY_E4096_MAX01H_20261027 --expected-oom-kill 0 --code-commit <本次发布并独立核远端40位HEAD> --training-seconds 3500 --maximum-training-seconds 3600 --external-training-stop-seconds 3540 --video-suite /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027/evaluation/video_suite.json --baseline-evaluation /opt/data/private/ZHY/data/supermimic_for_q1/MG05_RF09_CKPT3500_BASELINE_20261027 --baseline-label RF09_ckpt3500_3424.267689s --comparison-name RF09_1H_RF11_comparison.json --trial-note RF09_recipe_with_only_legal_transition_bin_fix --detach`。

发布并核验源码和小记录SHA后才执行一次；随后用现场status/guard/run及boot/PID/start_ticks/命令/更新时间核身份，不用历史RUNNING断言训练中。运行冻结文件不热改。

自动初末330片评测、RF09共同前缀、六格PNG/PDF/CSV和初末各54秒同套多类MP4。评估提前终止/完成、根XY/腿/上肢/腕轨迹、实际更新数/时长；训练reward或WM loss下降不能代替物理改善。沿用RF09/10预选普通走/抬手片，不按学后结果换片。

若正常完成则补保存/算术/媒体审计、本地交付和RF09/RF10比较；若失败/保护停止保留真实终态，不自动再开。权重/大轨迹留服务器LOCAL_ONLY，小文件逐大小/SHA到Mac，不建网站。

## 保存与下一步

MG05/RF10已发布`e643c2913a1bb7ca8bd40b97a4f477cae2fcaa32`并独立核远端，22文件839774字节三端SHA一致；[收据](../data/manifests/supermimic_q1_MG05_RF10_publish_sync_20261027.json)。旧目标已完整结算，最新用户的一小时授权另行执行，无新增agent。

本阶段记录/预检收据显式暂存、payload audit、push Q1、独立比对40位远端并镜像后启动；发布身份由Git历史和实际plan给出。启动/重要结果及时更新本README及CURRENT_STATE/索引。效果未出前不宣称采样修复改善上肢或行走。
