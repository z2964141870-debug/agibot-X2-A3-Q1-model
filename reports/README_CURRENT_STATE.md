# YUANQI 当前状态与下一步

2026-10-28研究：[五项简报](research/2026-10-28/README.md)保存四项新阅读与HoloBoxing v2对手协议更正；87请求83成功、9固定文本静态核验，main去重122→134、H001不变。完整历史由main维护，Q1工程状态按本分支最新记录；本次没有训练/仿真或新假设。[检查](README_RESEARCH_CHECK_20261028.md)/[同步回执](../data/manifests/research_watch_sync_20261028.json)。

2026-10-28 02:36，北京时间：**RF12改进目标失败并按预设规则停止，训练/评测/本地媒体交付已收尾。** 实训3503.474674秒/812轮/79822848转移，critic16240/actor16040；三重复共同前缀根XY误差为RF11的2.653/2.669/2.816倍，上肢增加13.3至14.0%、腕增加14.7至15.2%，验证失败4/6/3条、完成97.28至97.88%（RF11三次均0/30失败/100%），固定普通走均提前终止。改善门0/3、严重退步3/3，stop/physical_regression；B/C及五小时加训未执行，不自动重试或另开。[正式结果](README_SUPERMIMIC_Q1_RF12_RESULTS_20261028.md)/[完成收据](../data/manifests/supermimic_q1_RF12_completion_audit_20261028.json)。

02:28:14全pipeline COMPLETED，train/delivery worker均正常reap、返回0；02:33核原controller49990/start_ticks3416720和worker49995/start_ticks3416816均不存在，GPU15MiB/26度且无计算进程、OOM0。37项独立保存检查/12媒体检查通过，1123冻结来源286568437字节未变，stage到final的策略/critic/optimizer/RNG相同。结果6d77c58e9000b09739a4d8ef5a3e06dcd4f078ff已独立核远端、10记录246275字节三端匹配；02:45:16监督实际PAUSED并读回确认。冻结684548c不因报告提交而改变，Windows桥保留。

资源清理、AP03的883训练/原30验证及开训契约均已完成；本轮仅采样原300片，新增583片从未启用。六格/CSV/PPO诊断、初末54秒多类MP4已在Mac并核验，权重和大轨迹SERVER_LOCAL_ONLY。

## RF11历史基线

独立4096 MuJoCo Warp环境、同300训练/30验证SOMA、seed0从零；恢复RF09的obs80/goal590和reward，仅保留已证尾bin合法转移修复。RF10也已有该修复，本轮主要撤回C01表示/奖励组合。冻结提交`7a3567cd5b89e5cd6282a83aacb61f19590b0434`。实际3500.265990秒/3271学习轮/33755136转移，WM32710/actor2771；36保存、19结果/CSV和12媒体检查全部通过。

RF09近一小时→RF11：训练提前终止54/300→24/300，验证2/30→0/30、完成95.70%→100%；共同前缀验证根XY0.07904→0.11446m、腿0.27621→0.26438rad、上肢0.93760→0.93577rad、腕世界轨迹0.18356→0.16549m。存活提高但根跟踪退步，上肢基本持平，不是完整全身跟踪成功；平均共同前缀5.1167秒，两轮更新数/时长不同。

固定普通走两片均完成，但近地移动/过冲仍明显；固定双手抬起片腕最高骨盆相对高度只有0.252/0.129m，参考0.534/0.560m，上肢RMSE1.29845rad。舞蹈例仍347/481步提前终止并保留失败标记。几何支撑不等于原生接触力，RF11未重测受力/滑移或完整actor目标H32梯度。详细边界和清单见[结果README](README_SUPERMIMIC_Q1_RF11_RESULTS_20261027.md)。

[RF10负面结果](README_SUPERMIMIC_Q1_RF10_RESULTS_20261027.md)/[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)保持已完成：RF10验证29/30失败、完成42.88%，组合的唯一致崩因素未隔离。旧目标不恢复；保留[RF09](README_SUPERMIMIC_Q1_RF09_RESULTS_20261027.md)/[RF08](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)。

## 本地交付

RF12 Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF12_PPO_SOMA_AP03_E4096_MAX08H_20261028`。`evaluation/initial.mp4`与`evaluation/final.mp4`各54秒/8段6类原速合集，保留普通走失败hold；`dashboard/`含六格PNG/PDF/CSV原值+500iteration均值和PPO诊断，WM/想象N/A。43服务器工件48549915字节逐大小/SHA匹配，12项媒体与代表帧目视完成；[清单](../data/manifests/supermimic_q1_RF12_local_delivery_20261028.json)。以下RF11为历史保留。

Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF11_SOMA300_SAMPLERONLY_E4096_MAX01H_20261027`。

`evaluation/initial.mp4`和`evaluation/final.mp4`各54秒、8个原速独立片段/6类，含固定普通前向走和抬手，不是连续54秒rollout。`dashboard/training_dashboard.png`及PDF/CSV为六格图与500iteration均值。29个服务器小工件26706642字节与Mac逐大小/SHA相同，全帧解码与目视核验完成。见[交付清单](../data/manifests/supermimic_q1_RF11_local_delivery_20261027.json)/[量化结果](../data/manifests/supermimic_q1_RF11_results_20261027.json)。

权重/大轨迹只在服务器LOCAL_ONLY，不拉Mac或宣称云备份。不建网站或重做媒体。

## 当前入口与来源

仅用户25386容器，`interactive40173`，唯一根`/opt/data/private/ZHY`及data/logs/script/reports。boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。02:33原RF12 controller/worker均不存在；GPU15MiB/26°C/0%、无计算进程，共享约7.61GB/OOM0，盘余675969384448字节。不得用旧PID或旧RUNNING断言仍在训练。

连接：`ssh -S /tmp/yuanqi-q1-25386.sock -T -p 18873 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25386' root@127.0.0.1`。保留Windows桥，不读写认证，不探测旧入口或师兄容器。[CS02恢复](README_Q1_CONTAINER_RESTORE_CS02_20261027.md)。

Python`/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python`经`script/supermimic_q1_hp/activate.py`加载既有CUDA/EGL；直接Torch缺libcudnn不等于环境损坏。RF11 runtime`data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027`，来源固定HP310244b。4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4/BF16/compile；师兄源默认1024env，不混同。

权威Git已切换持久副本`/Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028`的Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。旧`/tmp/yuanqi-supermimic-q1-20261023`部分文件与Git对象/config缺失，原因未证，不在旧目录修复或覆盖其他改动。结果记录按payload audit/显式push Q1/独立40位远端核验，再镜像三端。无关`script/supermimic_q1_sources/download_bones.py`不执行/修改/暂存/删除。

## 下一步与跟踪

RF12终态/reap/GPU释放、三重复JSON、六格及本地初末54秒MP4均已核，最终结果已发布并三端核验，本轮heartbeat已暂停；仅本次暂停回执随同记录收尾。新增583片与B奖励从未执行，不能宣称已测试。RF12本次授权已结束，不自动重试或开启其他训练，不激活agent。

用户最新关切是连续多轮缺乏全面改善。现有证据确认上肢幅度/时序瓶颈持续，RF09五小时亦未解决；历史根轨迹曾改善，但不支持继续堆训练时长。先定位策略目标、实际控制和H32模型方向，再决定后续方案。

RF11结果`d98b14b9d97a1999e3289e9faea20b0d943285f5`为历史已交付。RF12在2026-10-28 02:45:16已通过工具将同一`q1-rf08-4096`实际PAUSED并读取配置确认，旧ACTIVE/每10分钟为训练期间历史；独立`supermimichp-x2`和`yuanqi`未修改，不新建替代任务。研究入口[RF12核验收据](../data/manifests/supermimic_q1_RF12_research_20261028.json)/[MorphMimic只读初评](README_MORPHMIMIC_REVIEW_20261027.md)。
