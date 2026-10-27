# RF10：C01关节反馈与tracking一小时验证

记录日期：2026-10-27，北京时间。阶段：RUNNING，18:47:01 controller启动、18:47:19实际训练开始。属于[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)唯一正式试验，不续训RF09或profile，不自动重试/延时，不开agent。

冻结提交`83dcc76bf1cd02e4928493e17b4522cb9d5ee5a5`已push Q1并独立ls-remote匹配；21文件936839字节阶段payload通过，Mac/权威checkout/服务器SHA逐项相同。controller6852/start_ticks1166688、worker6855/start_ticks1166718，boot见下；18:55:37实际/proc身份匹配，三状态RUNNING，609学习轮/6508544转移、WM6090/actor109、实训498.182552秒。GPU14689MiB/53度、共享6351872000字节、OOM0、余盘1647293190144字节、本轮约283MB。[启动现场](../data/manifests/supermimic_q1_RF10_launch_observed_20261027.json)。这些是历史观测，恢复需重新核身份/进度，不按PID单独判存活。

原生4096/obs102/goal590/状态57/context26转移契约在正式worker再次通过。运行中不改冻结来源，后续报告提交不等于训练代码变化。预计19:45:40左右达到软预算，实际结束以receipt为准，不承诺此时已完成评测/视频。

## 方案与依据

采用[BD03/C01验证](README_SUPERMIMIC_Q1_MG05_NATIVE_CANDIDATE_20261027.md)的候选组合：合法转移bins、当前22关节误差追加历史观测至102维、直接上/下肢关节tracking、身体朝向std0.8及velocity权重重分配。最高tracking仍5；控制/动作平滑/重置/未来参考和H32保持。观测表达/奖励/已证采样bug同时变化，不能单因素归因。

依据是RF09上肢大部分误差为持续策略指令bias，短期模型关节方向较可靠，而原身体朝向reward近饱和；原生普通走存在支撑转移和明显接触点滑速，H32方向仍有风险。未把数据类型、PD、滑移或H32中的任何一个宣称唯一原因，也未称候选已改善。

候选CPU5组/12状态及4096完整GPU profile已通过，含原生转移、replay、H32 imagine、WM/actor/critic有限性。profile模型不进入正式初始化。16个构建/采样测试及新入口语法通过；实际单checkpoint评测路径已运行成功。

## 冻结计划与预算

- 用户自己的25386容器，根`/opt/data/private/ZHY`，boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。
- runtime `data/references/supermimichp/TRACKING_C01_runtime_20261027`，构建自固定HP`310244b73e45775744a75b789eb25a6c9c5a81c9`；正式源码使用发布并独立核远端后的40位HEAD，实际plan登记。
- 同300训练/30验证SOMA，`AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json`，清单SHA`ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337`。
- 从零，seed0、4096 MuJoCo Warp环境/8线程，WM1024×10、actor1024/H32、critic5×4、UTD0.4、BF16/compile。4096不是actor batch。
- 软目标3500秒，外部3540秒退出保护加45秒宽限，实际训练不得超过3600秒。初始化/等待/profile/评测/绘图不算训练。不自动重试或补足第二次训练。
- 100GiB余盘、12GiB实验、30GiB共享cgroup内存、OOM基线0、86度和900秒无进度保护保持；不动他人文件/任务。

唯一新根`data/supermimic_for_q1/BONES_RF10_SOMA300_C01_E4096_MAX01H_20261027`；启动前核不存在以及旧worker退出。冻结期间不改运行源码/config/参考。

## 启动与监督

服务器根下，用既有Python执行`script/supermimic_q1_hp/run_trial.py --mode train --runtime /opt/data/private/ZHY/data/references/supermimichp/TRACKING_C01_runtime_20261027 --library /opt/data/private/ZHY/data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF10_SOMA300_C01_E4096_MAX01H_20261027 --expected-oom-kill 0 --code-commit <已发布40位HEAD> --training-seconds 3500 --maximum-training-seconds 3600 --external-training-stop-seconds 3540 --video-suite /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027/evaluation/video_suite.json --baseline-evaluation /opt/data/private/ZHY/data/supermimic_for_q1/MG05_RF09_CKPT3500_BASELINE_20261027 --baseline-label RF09_ckpt3500_3424.267689s --comparison-name RF09_1H_RF10_comparison.json --trial-note C01_current_joint_error_joint_tracking_orientation_bandwidth_and_legal_transition_bins --detach`。

此为已实际执行命令，code-commit使用上述冻结40位SHA；恢复读真实status/guard/run状态与boot/PID/start_ticks，不重复执行。转移数/更新数与终态审计结束后补。初始化/profile不冒充正式进度。

## 评测与交付

固定330参考初末真实物理评测；主对照为RF09近一小时checkpoint（内部3424.267689秒/3500学习轮/36102144转移），另保留RF09五小时final作更强且预算不等的对照。披露实际预算/更新次数及重复轨迹差异，不宣称等初始化或完全确定性。

比较完成/跌倒、根移动/漂移、腿和上肢关节误差、腕部轨迹/抬手高度及时序；分别给共同前缀与完整执行。新旧reward公式分别在同轨迹上报告，不拿更换reward的数值判优。已有失败普通走保留，不根据学后表现换片。

自动pipeline生成初末≥30秒多类MP4、六格PNG/PDF/原值与500iteration均值CSV及来源收据，直接交付Mac对应实验目录，不建网站。权重与大轨迹留服务器LOCAL_ONLY，小记录核大小/SHA同步Mac与Git，重要阶段README/索引/CURRENT_STATE及时更新。最终好坏如实结算，全部证据交付后再完成MG05。

新增独立`compare_q1_hp_tracking_heads.py`在同物理状态分别计算旧/新analytic tracking，明确排除动作率/限位罚项；不调用策略、积分或optimizer。真实330片SELF预检通过，两份标签评分相同，上肢RMSE重算与保存结果误差小于1e-6。[收据](../data/manifests/supermimic_q1_RF10_tracking_heads_preflight_20261027.json)。结果审计入口已支持不重复同库benchmark的pipeline；尚未运行正式终后审计。这些独立project_tools文件不在运行冻结清单内，没有修改当前worker或MDP。

终后将复用原生诊断入口，在新权重上记录足地受力/滑速、H1/4/8/32方向，并保存已经执行的两次330片rollout及原始完成/关节/根误差，量化重复运行差异，不增加optimizer。诊断入口已去掉RF09硬编码标签，增加旧tracking头评分，区别新旧reward口径；它不在正式训练167项冻结源内（已实际核plan），未改冻结文件。重复统计函数已用RF09保存330片独立核算，与原结果的完成/上肢/根误差及失败数差0，[预检](../data/manifests/supermimic_q1_RF10_repeat_outcome_preflight_20261027.json)。正式终后结果仍待，最多一小时训练预算不变。
