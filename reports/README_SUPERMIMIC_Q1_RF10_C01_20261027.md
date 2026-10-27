# RF10：C01关节反馈与tracking一小时验证

记录日期：2026-10-27，北京时间。阶段：待发布冻结后启动；当前尚未训练。属于[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)唯一正式试验，不续训RF09或profile，不自动重试/延时，不开agent。

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

此命令仅为本轮计划；恢复读真实status/guard/run状态与boot/PID/start_ticks，不重复执行。训练controller/worker身份、实际开始/结束、转移数/更新数和冻结来源审计待启动后补。初始化/profile不冒充正式进度。

## 评测与交付

固定330参考初末真实物理评测；主对照为RF09近一小时checkpoint（内部3424.267689秒/3500学习轮/36102144转移），另保留RF09五小时final作更强且预算不等的对照。披露实际预算/更新次数及重复轨迹差异，不宣称等初始化或完全确定性。

比较完成/跌倒、根移动/漂移、腿和上肢关节误差、腕部轨迹/抬手高度及时序；分别给共同前缀与完整执行。新旧reward公式分别在同轨迹上报告，不拿更换reward的数值判优。已有失败普通走保留，不根据学后表现换片。

自动pipeline生成初末≥30秒多类MP4、六格PNG/PDF/原值与500iteration均值CSV及来源收据，直接交付Mac对应实验目录，不建网站。权重与大轨迹留服务器LOCAL_ONLY，小记录核大小/SHA同步Mac与Git，重要阶段README/索引/CURRENT_STATE及时更新。最终好坏如实结算，全部证据交付后再完成MG05。
