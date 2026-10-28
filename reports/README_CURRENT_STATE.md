# YUANQI 当前状态与下一步

RF15逐片门准备735464329a227027c5db0d3928d53cb4ab6eb116已独立核远端/7文件155221字节三端SHA；监督更新读回ACTIVE每10分钟。新媒体验收入口支持RF14初末/selected三片，实际检查待本轮媒体生成，RF15 GPU预检/正式训练均未开始。[准备发布](../data/manifests/supermimic_q1_RF15_per_walk_published_20261028.json)。

RF14两小时记录5072f6f3f3a3a567d9a9961c83dc0ad724da2d22已独立核远端，7文件234612字节三端大小SHA一致。RF15开训前延长门已对齐文档为两固定走逐片通过，不再用均值掩盖退步；8项CPU门测试及已保存轨迹集成验证通过，RF14全部1145来源再次核SHA未变。GPU预检/训练仍待RF14退出择优。21:26:13现场RF148921.123888累计秒/7968轮、同身份/实际进度正常/OOM0，继续原冻结预算。[小收据](../data/manifests/supermimic_q1_RF15_per_walk_gate_20261028.json)。

20:58已核RF14两小时门7201.145359秒/6469轮三重复完成：验证1/0/0失败，固定两走及抬手均3/3完整。普通走滑移/足高部分改善，但上肢RMSE增加8.5%至9.0%、相对腕增加25.9%至26.6%，倾斜/抬手漂移增加；当前排序仍选RF11_initial，不称整体改善。原controller110563/worker110573及boot/start_ticks/命令一致，现场7288.130549秒/6545轮/67551232转移、WM65450/actor6045、OOM0，继续冻结的累计14300秒检查/硬14400秒。5小JSON共1128459字节Mac/25386大小SHA匹配，权重/大NPZ留服务器。[两小时原值](../data/manifests/supermimic_q1_RF14_stage02_20261028.json)。

RF15准备已发布b5344ced0362f2782b0515ca6961370cd3b505b9并独立核远端，19文件244351字节三端大小SHA一致；9项CPU测试、804帧真实参考FK/导入通过，尚未GPU预检/训练。RF14全部1145冻结来源再次核SHA未变，原learner继续两小时/四小时物理门；一小时候选退步、仍选原RF11。监督原任务ACTIVE每10分钟并更新回读，其他任务未改。[最新现场与发布](../data/manifests/supermimic_q1_RF15_preparation_published_20261028.json)。

20:28：[RF15补偿准备](README_SUPERMIMIC_Q1_RF15_GAIT_PREP_20261028.md)已有独立实现/9项CPU测试与导入验证，GPU预检和训练均未执行，等待RF14退出/择优。首小时须真实防滑/摆脚改善且固定片3/3完整才延长，根XY不设硬门。RF14原进程在20:24:49已累计5406.248秒/新增1905.982秒/4909轮，原boot/PID/start_ticks/命令一致，GPU14847MiB/50度、共享15.10GB/OOM0；一小时仍无全面改善，当前保留原RF11。

RF14累计一小时stage01（3600.601031秒/3340轮）已保存三重复，验证0/1/0失败；固定走后停一次271/320步跌倒，另两次完整但过冲/滑移加重。当前排序仍选RF11_initial，不能称改善。原learner继续累计两小时门，不改冻结源码或预算；权重/大轨迹服务器留存。[阶段](../data/manifests/supermimic_q1_RF14_stage01_20261028.json)。

2026-10-28 19:48:09：[RF14](README_SUPERMIMIC_Q1_RF14_RF15_GAIT_20261028.md)唯一4096底座续训流程已启动，controller110563/start_ticks10173450、worker110573/start_ticks10173556/原boot与GPU一致。冻结4d87269d7ac815cd75a0424dcf7be56ae2c2661d已push/独立远端核验，13文件218822字节三端大小SHA一致。初始RF11三重复真实评测中，初查累计3500.265990秒/新增0，不是效果改善。原q1-rf08-4096已实际ACTIVE每10分钟并回读，其他任务未修改；底座累计硬<=14400秒，RF15独立首小时后依真实改善决定最多4小时。以下RF14“尚未启动/监督暂停”为准备历史。[启动收据](../data/manifests/supermimic_q1_RF14_launch_observed_20261028.json)。

2026-10-28 19时后：用户新目标已active，执行[RF14底座续训与RF15步态补偿](README_SUPERMIMIC_Q1_RF14_RF15_GAIT_20261028.md)。RF11底座累计最多4小时（暂按包含原3500.265990秒），1/2/4小时固定三重复普通走物理评测择优；后续新PPO补偿最多4小时，先一小时实测改善才延长。仅新独立实验，不恢复RF12/RF13、不故障重试/agent；旧不再训练条款由本次授权覆盖。RF11模型与优化器可恢复，replay/仿真/RNG需重新初始化。代码与16/4096零optimizer预检准备中，尚未开训；唯一25386及原boot/GPU已重新核为空闲。原监督仍暂停，正式启动后复用。

2026-10-28 18:42：**RF13交付与记录收尾完成，基本展示目标未稳健达标。** 3500.385426实训秒，正常预算结束/双guardreap0/原进程退出/GPU释放；12保存检查与1137冻结来源、42文件28862203字节Mac大小SHA、12媒体检查和9图目视均通过。两固定普通走完整3/3且真实前进；抬手176/271、271/271、271/271，完成者仍显著漂移，上肢收益主要在未训练static075已有。结果854e406506afcf7a1d6e96bf8cb58b11109a975f已独立核远端，14项记录512622字节三端匹配；原q1-rf08-4096已18:42:44实际PAUSED并读回。完成回执随本次收尾提交镜像，详细最终发布以Git及收据为准。其他任务未修改，不追加训练/重启/agent。[结果](README_SUPERMIMIC_Q1_RF13_RESULTS_20261028.md)/[完成收据](../data/manifests/supermimic_q1_RF13_completion_audit_20261028.json)。

2026-10-28 16:18，用户最新优先级：**先保证固定普通前走/自然步态与上肢运动的基本展示，精确根轨迹暂列次位。** 检查真实前进/连续换脚、姿态/滑步及抬手幅度/时序/全程稳定，根XY保留诊断，不单因旧根阈值否定展示。唯一[RF13](README_SUPERMIMIC_Q1_RF13_BALANCE_20261028.md)源码/内部停止规则冻结，硬<=3600实训秒，不热改、重启或加训。第二门2401.480799秒/748轮：验证0/0/0失败、两普通走三次完整且有真实前进；抬手repeat0为167/271步终止，另两次完整但腿/滑移/振荡恶化，展示尚未稳健达标。旧整组目标0/3、严重退步0/3，原learner继续3500秒最后门。16:12:30同一controller93989/worker93994与start_ticks/boot/命令一致，2547.881秒/795轮/78151680转移，GPU8267MiB/52度、共享12.34GB/OOM0。4小文件1154178字节已Mac大小SHA匹配；1137冻结来源16:02全核未变。[第二门/优先级](../data/manifests/supermimic_q1_RF13_stage02_demo_priority_20261028.json)。监督同任务ACTIVE每10分钟，其他任务保持；最终视频/六格尚待。

2026-10-28 15:54：[RF13首个物理门](README_SUPERMIMIC_Q1_RF13_BALANCE_20261028.md)1201.398777实训秒/373轮三重复完成。验证0/0/1条提前终止、完成100%/100%/99.8%，固定两普通走/抬手均完整；相对RF11上肢约降83%/相对腕降56%，根XY仍高45%至61%、振荡高23%至28%。相对未训练静态75%根变+4.8%/+2.1%/-2.5%，尚无一致学习收益；完整目标0/3、严重退步0/3，按原预算继续2400/3500秒评测，硬<=3600秒，不扩时/重试/agent。15:54:58同一controller93989/start_ticks8571724、worker93994/start_ticks8571852、boot/命令/mtime一致，RUNNING/clock ACTIVE；已实训1622.969秒/504轮/49545216转移、critic10080/actor9880，GPU8267MiB/53度、共享12.06GB/OOM0、盘余1.672TB。冻结51d4f00/1137来源不变，原监督ACTIVE每10分钟，其他研究任务保持。阶段4个小证据1142555字节已Mac大小SHA核，六格/媒体待终后。[阶段收据](../data/manifests/supermimic_q1_RF13_stage01_20261028.json)/[启动](../data/manifests/supermimic_q1_RF13_launch_observed_20261028.json)。

2026-10-28 15:03：[UR04准备完成](README_Q1_BALANCE_ADAPT_UR04_20261028.md)，按用户“可以，先做”及“continue”准备[唯一RF13](README_SUPERMIMIC_Q1_RF13_BALANCE_20261028.md)。五测试/16与4096真实0optimizer预检/独立评测的训练环境、门控、接触及RNG保护通过；混合截断子集索引错误已修复，原失败保留。三重复回退根只改善约3%、上肢0.159→0.410rad且失败未减，正式关闭回退，仅14维腿腰补偿+手臂课程/根腿奖励。旧1123来源未变；尚未正式训练，发布后三端核验仅一次4096/实训<=3600秒，无旧实验恢复/五小时扩时/故障重启/agent。监督当前暂停，其他研究任务保持。[收据](../data/manifests/supermimic_q1_UR04_balance_prep_20261028.json)。

2026-10-28 13:35：[UR03全身折中](README_Q1_UPPER_TRADEOFF_UR03_20261028.md)五档33环境三重复0optimizer完成。75%参考混合上肢0.9441→0.1607rad、相对腕0.1234→0.05345m，验证0/0/0失败且固定两走/抬手全完成；根XY0.1237→0.1906m、腿约增5%、振荡约增25%，不是全面达标。推荐保留该仿真候选并研究上肢课程下的下肢独立适应与平衡回退，不仅调全局reward。COMPLETED/reap0/进程退出/GPU释放、1123旧来源未变、26小工件2953599字节Mac/25386匹配。RF12授权结束/监督暂停，不新增训练、agent或自动任务。[量值](../data/manifests/supermimic_q1_UR03_upper_tradeoff_20261028.json)。

2026-10-28 12:49：[UR02平衡耦合](README_Q1_BALANCE_COUPLING_UR02_20261028.md)完成0optimizer诊断与候选实现。保存动作重建误差0，上肢观测交换使下肢/腰目标变0.153–0.165rad；固定同一下肢命令的物理手臂干预也改变根/接触，但开环对照本身21–23/30失败，不能分解原闭环根因。冻结教师+参考手臂+14维下肢残差代码、5测试及16/4096前向通过，尚未接入学习器或训练；不称已解决全身平衡。V03 COMPLETED/reap0/进程退出/GPU释放，1123冻结来源SHA未变。RF12/监督仍结束/暂停，其他研究任务保留。[小记录](../data/manifests/supermimic_q1_UR02_balance_coupling_20261028.json)。

2026-10-28 11:44：[UR01上肢参考控制实测](README_Q1_UPPER_REFERENCE_UR01_20261028.md)完成0optimizer、33环境三条件三重复。参考前馈把验证上肢RMSE约0.937→0.070rad、相对腕约0.122→0.055m；但共同前缀根XY恶化1.74–1.90倍、验证2/2/2失败，固定抬手152/271/271步。限幅补偿也未保住平衡。已落实独立上肢目标修正，下一候选收窄为“参考前馈+单独下肢平衡适配”，尚未训练/证明全身成功；RF12及监督保持结束/暂停，不增加训练授权。[量值](../data/manifests/supermimic_q1_UR01_reference_control_20261028.json)。

2026-10-28研究：[五项简报](research/2026-10-28/README.md)保存四项新阅读与HoloBoxing v2对手协议更正；87请求83成功、9固定文本静态核验，main去重122→134、H001不变。完整历史由main维护，Q1工程状态按本分支最新记录；本次没有训练/仿真或新假设。[检查](README_RESEARCH_CHECK_20261028.md)/[同步回执](../data/manifests/research_watch_sync_20261028.json)。

2026-10-28 02:36，北京时间：**RF12改进目标失败并按预设规则停止，训练/评测/本地媒体交付已收尾。** 实训3503.474674秒/812轮/79822848转移，critic16240/actor16040；三重复共同前缀根XY误差为RF11的2.653/2.669/2.816倍，上肢增加13.3至14.0%、腕增加14.7至15.2%，验证失败4/6/3条、完成97.28至97.88%（RF11三次均0/30失败/100%），固定普通走均提前终止。改善门0/3、严重退步3/3，stop/physical_regression；B/C及五小时加训未执行，不自动重试或另开。[正式结果](README_SUPERMIMIC_Q1_RF12_RESULTS_20261028.md)/[完成收据](../data/manifests/supermimic_q1_RF12_completion_audit_20261028.json)。

02:28:14全pipeline COMPLETED，train/delivery worker均正常reap、返回0；02:33核原controller49990/start_ticks3416720和worker49995/start_ticks3416816均不存在，GPU15MiB/26度且无计算进程、OOM0。37项独立保存检查/12媒体检查通过，1123冻结来源286568437字节未变，stage到final的策略/critic/optimizer/RNG相同。结果6d77c58e9000b09739a4d8ef5a3e06dcd4f078ff已独立核远端、10记录246275字节三端匹配；02:45:16监督实际PAUSED并读回确认。冻结684548c不因报告提交而改变，Windows桥保留。

资源清理、AP03的883训练/原30验证及开训契约均已完成；本轮仅采样原300片，新增583片从未启用。六格/CSV/PPO诊断、初末54秒多类MP4已在Mac并核验，权重和大轨迹SERVER_LOCAL_ONLY。

## RF11历史基线

RF14在19:53:00现场已经累计3600.601031秒/新增100.335041秒、iteration3340/WM33400/actor2840；stage01约一小时三重复评测repeat1/frame100，训练时钟暂停，原进程身份一致/GPU14847MiB/42度。尚未形成完整阶段比较，不宣称改善；顶端19:48的“初始评测”只是启动快照。

RF14准备19:38已通过3项单测与16/4096真实Warp零optimizer预检：模型/优化器恢复一致，评测保护训练现场及RNG，双guardreap0/OOM0。详见[准备收据](../data/manifests/supermimic_q1_RF14_preparation_20261028.json)；正式优化须发布/三端核验后启动，原监督暂未激活。

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

仅用户25386容器，`interactive40173`，唯一根`/opt/data/private/ZHY`及data/logs/script/reports。boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。RF13历史controller93989/start_ticks8571724、worker93994/start_ticks8571852均已退出，终后身份与资源见结果收据。原RF12亦已退出，不用旧PID/旧RUNNING推断仍在训练。

连接：`ssh -S /tmp/yuanqi-q1-25386.sock -T -p 18873 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25386' root@127.0.0.1`。保留Windows桥，不读写认证，不探测旧入口或师兄容器。[CS02恢复](README_Q1_CONTAINER_RESTORE_CS02_20261027.md)。

Python`/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python`经`script/supermimic_q1_hp/activate.py`加载既有CUDA/EGL；直接Torch缺libcudnn不等于环境损坏。RF11 runtime`data/references/supermimichp/TRACKING_C02_sampler_runtime_20261027`，来源固定HP310244b。4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4/BF16/compile；师兄源默认1024env，不混同。

权威Git已切换持久副本`/Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028`的Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。旧`/tmp/yuanqi-supermimic-q1-20261023`部分文件与Git对象/config缺失，原因未证，不在旧目录修复或覆盖其他改动。结果记录按payload audit/显式push Q1/独立40位远端核验，再镜像三端。无关`script/supermimic_q1_sources/download_bones.py`不执行/修改/暂存/删除。

## 下一步与跟踪

当前按本页顶端新授权推进RF14/RF15，旧RF13/RF12已完成及旧“不再训练”描述仅限制旧实验；不得据此取消新目标，也不得扩展为额外未授权训练。RF14/RF15进度与下一条命令集中在对应新阶段README。

RF13量化/媒体验收、结果发布与三端记录核验完成，原监督已实际暂停；失败与旧来源全部保留。当前不启动新训练、追加或重启，不激活agent；训练目标改变须另获用户授权。保留static075/initial和stage01早期证据，不能把更稳的未训练参考混合称新head训练成功。下一方案应针对抬手原地平衡/滑移/残差偏移，根精确轨迹次位，尚未执行新方案。

RF12终态/reap/GPU释放、三重复JSON、六格及本地初末54秒MP4均已核，结果已发布并三端核验；其阶段B奖励/新增583片从未执行，不能宣称已测试。RF12授权已结束，旧暂停记录仅历史，不恢复或重试。

用户最新关切是连续多轮缺乏全面改善。现有证据确认上肢幅度/时序瓶颈持续，RF09五小时亦未解决；历史根轨迹曾改善，但不支持继续堆训练时长。先定位策略目标、实际控制和H32模型方向，再决定后续方案。

RF11结果`d98b14b9d97a1999e3289e9faea20b0d943285f5`为历史已交付。RF13期间同一`q1-rf08-4096`的ACTIVE仅历史；18:42:44本轮交付后已工具设为PAUSED并读回，独立`supermimichp-x2`和`yuanqi`未修改，不新建替代任务。研究入口[RF12核验收据](../data/manifests/supermimic_q1_RF12_research_20261028.json)/[MorphMimic只读初评](README_MORPHMIMIC_REVIEW_20261027.md)。
