# YUANQI 当前状态与下一步

21:04:01 RF11实际533轮/5730304转移、WM5330/actor33、实训436.95秒，已过500轮预热，GPU14619MiB/50°C/OOM0。原每10分钟监督`q1-rf08-4096`已实际ACTIVE并只负责本轮RF11，正常静默、终后评测/六格/54秒MP4与三端记录交付后暂停；不改其他任务。[监督现场](../data/manifests/supermimic_q1_RF11_monitor_20261027.json)。

20:57:40：[RF11实际RUNNING](README_SUPERMIMIC_Q1_RF11_SAMPLERONLY_20261027.md)，20:56:44开始实际训练，最多3600秒。4096原生转移/obs80通过，58轮/WM580/actor0正常预热，15项部署SHA与冻结`7a3567cd5b89e5cd6282a83aacb61f19590b0434`一致；controller17370/start1943459、worker17373/start1943487，boot见下，须现场核身份。GPU9897MiB/47°C、共享7.21GB/OOM0、余盘1.638TB。RF10已退出，运行中不热改；本次仅一轮，不重试/延时。下方“待启动”是历史。

2026-10-27最新授权：[RF11独立对照](README_SUPERMIMIC_Q1_RF11_SAMPLERONLY_20261027.md)：用户要求针对RF10训崩再改进训练一小时。恢复RF09观测/reward，仅保留已证尾bin合法转移修复；独立4096环境从零，≤3600秒。C02 CPU五组/原生16环境smoke通过，待发布启动；不自动重启RF10。下方“无下一轮授权”是此前状态，当前授权仅此RF11一次。MG05已完成工具结算与22文件三端核验，RF10负面结果保持。

2026-10-27，北京时间。最新入口25386已恢复，原共享文件可见，无需迁移大数据。[MG05](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)诊断/候选/唯一短训练/评测与本地交付均已完成；[RF10结果](README_SUPERMIMIC_Q1_RF10_RESULTS_20261027.md)明显退步，停止采用C01，不自动续训或启动第二轮。工具最终结算以get_goal为准。下述信息覆盖历史paused、等待服务器和RUNNING记录，历史过程留阶段README/Git。

## 最新结果

RF10：4096 MuJoCo Warp环境、300训练/30验证SOMA、从零seed0；实际3500.992597437秒/3284学习轮/33890304转移、WM32840/actor2784。19:45:41训练结束、19:59:24全pipeline COMPLETED，五阶段正常reap；36保存与19结果/CSV检查通过，167冻结来源/330参考不变。冻结训练提交`83dcc76bf1cd02e4928493e17b4522cb9d5ee5a5`，来源和完整命令见[RF10运行](README_SUPERMIMIC_Q1_RF10_C01_20261027.md)。

同库RF09近一小时checkpoint（3424.267689秒）→RF10：验证提前终止2/30→29/30、完成95.70%→42.88%；共同前缀根XY0.04356→0.64030m、腿0.23554→0.67474rad、腕世界轨迹0.14524→0.54932m。上肢0.84637→0.81627rad的小降幅不能抵消失稳，RF09五小时对照亦无上肢改善。两次复评仍27/30、28/30失败。新旧tracking头均更差。

四片原生PD目标变化RMS1.91–2.36rad/20ms，动作变化罚远大于正tracking，明显指令振荡；C01多项组合的唯一因果未隔离。[BD03](README_SUPERMIMIC_Q1_MG05_NATIVE_CANDIDATE_20261027.md)已测原RF09真实单支撑/滑速及H32方向风险，不能简化为“不接触/不探索”。[上游核查](README_SUPERMIMICHP_Q1_UPDATE_20261027.md)已确认尾bin跨片bug及实际100/15清单，新增代码主要为归档/诊断；唯一子agent已结束，不再开agent。

## 本地交付

Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF10_SOMA300_C01_E4096_MAX01H_20261027`。

初末`evaluation/initial.mp4`、`evaluation/final.mp4`各54秒、8个原速独立片段/6类，含普通前向走与抬手，保留失败标记；不是一次连续54秒rollout。`dashboard/training_dashboard.png`及PDF/CSV包含六格与500iteration均值。30文件25639241字节逐大小/SHA核对，12媒体检查/全解码和19结果/CSV检查通过。完整比较JSON也在本地根，不重渲染、不建网站。见[交付清单](../data/manifests/supermimic_q1_RF10_local_delivery_20261027.json)、[完成审计](../data/manifests/supermimic_q1_MG05_completion_audit_20261027.json)。

原RF09及RF08工件保留，入口[RF09结果](README_SUPERMIMIC_Q1_RF09_RESULTS_20261027.md)/[RF08结果](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)。权重/大轨迹服务器LOCAL_ONLY，不拉Mac或宣称云备份。

## 当前服务器与版本

仅用户25386容器，`interactive40173`，boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。唯一根`/opt/data/private/ZHY`及data/logs/script/reports。20:20实际核RF10 controller6852/worker6855均不存在；20:27 GPU15MiB/26°C/0%、盘余约1.640TB。恢复时重新核现场，不凭旧PID判运行，不探测25245/25252或师兄容器。

连接：`ssh -S /tmp/yuanqi-q1-25386.sock -T -p 18873 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25386' root@127.0.0.1`。Windows桥socket`/tmp/yuanqi-windows-25386.sock`；保留桥，不读写认证。恢复证据见[CS02](README_Q1_CONTAINER_RESTORE_CS02_20261027.md)。

Python`/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python`，通过`script/supermimic_q1_hp/activate.py`加载既有CUDA库；直接Torch导入缺libcudnn不能误判环境损坏。RF09 runtime`data/references/supermimichp/AP01_runtime_20261026`保留，C01独立runtime`data/references/supermimichp/TRACKING_C01_runtime_20261027`。实际4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4/BF16/compile；师兄源默认1024env，勿混同。

权威Git`/tmp/yuanqi-supermimic-q1-20261023`，Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。本次结果小记录显式暂存、payload audit、push并独立比对远端40位SHA，再镜像Mac/25386核SHA；提交身份由Git历史及同步收据提供。无关`script/supermimic_q1_sources/download_bones.py`不执行/修改/暂存/删除。

## 下一步与独立跟踪

本目标所授权的唯一短训练和证据已交付。下一方案先拆开尾bin修复与表示/奖励组合，并检查动作幅度/reset/长程方向，不自动开下一轮、重试或延长。保留100GiB余盘、12GiB实验、30GiB共享cgroup、86°C/900秒无进展保护，不动他人文件/任务。

本轮未创建RF10自动任务；既有`supermimichp-x2`每小时仓库跟踪及`yuanqi`研究任务保持，旧RF08监督已PAUSED。研究入口[2026-10-27简报](research/2026-10-27/README.md)；[MorphMimic只读初评](README_MORPHMIMIC_REVIEW_20261027.md)作为后续资料，未替换主线或安装执行。
