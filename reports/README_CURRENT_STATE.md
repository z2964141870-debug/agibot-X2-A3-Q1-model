# YUANQI 当前状态与下一步

2026-10-28 01:36，北京时间：RF12本轮监督连接中断，现场状态为 **UNKNOWN**。本机18873拒绝连接且无监听，容器与Windows桥控制socket均不存在；已做一次有界核查，未重启/重连训练、未改冻结来源或预算。最后成功快照为01:25:26：三状态RUNNING，A阶段1237.219683实训秒/287轮/28213248转移，critic5740/actor5540，OOM0、GPU9811MiB/55度。旧快照不能证明现在仍在运行或已停止。连接恢复后先核boot/PID/start_ticks、status/guard/clock与物理评测，当前无法同步25386小记录，监督保持原频率。[连接收据](../data/manifests/supermimic_q1_RF12_connection_unknown_20261028.json)。

2026-10-28 01:07，北京时间。[RF12独立PPO分阶段训练](README_SUPERMIMIC_Q1_RF12_PPO_20261027.md)已启动：4096环境/从RF11初始化；controller49990/start_ticks3416720、worker49995/start_ticks3416816，冻结684548c已发布/独立核远端、22文件三端匹配。三状态RUNNING，175.025实训秒/42轮/4128768转移，critic840/actor640，已通过10轮critic-only；1123冻结来源未变，GPU9811MiB/55度、共享10.29GB/OOM0。约3500/7100/10700实训秒固定三重复评测，整组物理门达标才自动追加最多5小时，总<=8小时；当前未证明效果。既有监督已ACTIVE每10分钟，不恢复旧实验或agent。[启动收据](../data/manifests/supermimic_q1_RF12_launch_observed_20261028.json)。

资源清理、AP03的883训练/原30验证、16/4096零optimizer与原生接触/活跃训练状态/RNG保护已完成；新增583动作只在C阶段渐进启用。六格/固定54秒视频已接入每阶段与最终交付，正式新效果媒体尚待检查点。

## 最新结果

独立4096 MuJoCo Warp环境、同300训练/30验证SOMA、seed0从零；恢复RF09的obs80/goal590和reward，仅保留已证尾bin合法转移修复。RF10也已有该修复，本轮主要撤回C01表示/奖励组合。冻结提交`7a3567cd5b89e5cd6282a83aacb61f19590b0434`。实际3500.265990秒/3271学习轮/33755136转移，WM32710/actor2771；36保存、19结果/CSV和12媒体检查全部通过。

RF09近一小时→RF11：训练提前终止54/300→24/300，验证2/30→0/30、完成95.70%→100%；共同前缀验证根XY0.07904→0.11446m、腿0.27621→0.26438rad、上肢0.93760→0.93577rad、腕世界轨迹0.18356→0.16549m。存活提高但根跟踪退步，上肢基本持平，不是完整全身跟踪成功；平均共同前缀5.1167秒，两轮更新数/时长不同。

固定普通走两片均完成，但近地移动/过冲仍明显；固定双手抬起片腕最高骨盆相对高度只有0.252/0.129m，参考0.534/0.560m，上肢RMSE1.29845rad。舞蹈例仍347/481步提前终止并保留失败标记。几何支撑不等于原生接触力，RF11未重测受力/滑移或完整actor目标H32梯度。详细边界和清单见[结果README](README_SUPERMIMIC_Q1_RF11_RESULTS_20261027.md)。

[RF10负面结果](README_SUPERMIMIC_Q1_RF10_RESULTS_20261027.md)/[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)保持已完成：RF10验证29/30失败、完成42.88%，组合的唯一致崩因素未隔离。旧目标不恢复；保留[RF09](README_SUPERMIMIC_Q1_RF09_RESULTS_20261027.md)/[RF08](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)。

## 本地交付

Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF11_SOMA300_SAMPLERONLY_E4096_MAX01H_20261027`。

`evaluation/initial.mp4`和`evaluation/final.mp4`各54秒、8个原速独立片段/6类，含固定普通前向走和抬手，不是连续54秒rollout。`dashboard/training_dashboard.png`及PDF/CSV为六格图与500iteration均值。29个服务器小工件26706642字节与Mac逐大小/SHA相同，全帧解码与目视核验完成。见[交付清单](../data/manifests/supermimic_q1_RF11_local_delivery_20261027.json)/[量化结果](../data/manifests/supermimic_q1_RF11_results_20261027.json)。

权重/大轨迹只在服务器LOCAL_ONLY，不拉Mac或宣称云备份。不建网站或重做媒体。

## 当前入口与来源

仅用户25386容器，`interactive40173`，唯一根`/opt/data/private/ZHY`及data/logs/script/reports。boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。22:17实际核controller17370/worker17373均不存在；GPU15MiB/26°C/0%，共享约4.87GB/OOM0，余盘1.631TB。阶段末实验约0.972GB。不得用旧PID或旧RUNNING断言仍在训练。

连接：`ssh -S /tmp/yuanqi-q1-25386.sock -T -p 18873 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25386' root@127.0.0.1`。保留Windows桥，不读写认证，不探测旧入口或师兄容器。[CS02恢复](README_Q1_CONTAINER_RESTORE_CS02_20261027.md)。

Python`/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python`经`script/supermimic_q1_hp/activate.py`加载既有CUDA/EGL；直接Torch缺libcudnn不等于环境损坏。RF11 runtime`data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027`，来源固定HP310244b。4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4/BF16/compile；师兄源默认1024env，不混同。

权威Git已切换持久副本`/Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028`的Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。旧`/tmp/yuanqi-supermimic-q1-20261023`部分文件与Git对象/config缺失，原因未证，不在旧目录修复或覆盖其他改动。结果记录按payload audit/显式push Q1/独立40位远端核验，再镜像三端。无关`script/supermimic_q1_sources/download_bones.py`不执行/修改/暂存/删除。

## 下一步与跟踪

RF11本轮授权已使用且不恢复；最新RF12授权覆盖本条此前“不再开下一轮”的范围，仅新增这一独立PPO实验及按客观判据的连续加训，不授权故障重启或其他试验。不开或激活agent。保持RF12冻结来源，下一步约3500实训秒核三重复真实评测/判据及阶段六格/54秒视频。原300/30仅A/B采样，新增583片仅C按已授权物理门渐进启用；不由reward或loss触发加训。

用户最新关切是连续多轮缺乏全面改善。现有证据确认上肢幅度/时序瓶颈持续，RF09五小时亦未解决；历史根轨迹曾改善，但不支持继续堆训练时长。先定位策略目标、实际控制和H32模型方向，再决定后续方案。

RF11结果`d98b14b9d97a1999e3289e9faea20b0d943285f5`及完成回执为历史已交付；其22:45暂停记录不再表示RF12监督现状。2026-10-28已通过工具将同一`q1-rf08-4096`更新为RF12并回读ACTIVE，每10分钟；正常无变化静默，异常/实质效果/完成通知。独立`supermimichp-x2`和`yuanqi`未修改，不新建替代任务。研究入口[RF12核验收据](../data/manifests/supermimic_q1_RF12_research_20261028.json)/[MorphMimic只读初评](README_MORPHMIMIC_REVIEW_20261027.md)。
