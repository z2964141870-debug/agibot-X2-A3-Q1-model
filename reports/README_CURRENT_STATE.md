# YUANQI 当前状态与下一步

2026-10-27最新：用户要求先别继续训练，后续训练保持停止，不自动开启RF10、续训或恢复旧目标。MG04已完成，实际get_goal为null，本轮监督已PAUSED。[BD01瓶颈诊断](README_SUPERMIMIC_Q1_RF09_BOTTLENECK_20261027.md)/[小收据](../data/manifests/supermimic_q1_RF09_BD01_diagnosis_20261027.json)完成：冻结策略/保存轨迹496次前向、0积分/0optimizer。验证8条gesture上肢策略目标到参考1.18438rad、实际到目标0.06185rad；首先是策略目标偏差。手势仍获约69.10%最高步奖励，身体朝向项仅满值2.60%，奖励敏感度/模型动作梯度优先检查，尚未锁定唯一根因。

两条普通走参考本身双脚近地85.57%/80.99%，几何比例不能证明真实双足受力；Q1所有关节仍为模型统一推导PD15.79/1.01，非已完成逐关节整定。保持有界诊断，先区分接触/制动、控制与模型方向问题，不再默认加数据/时长。既有[RF09结果](README_SUPERMIMIC_Q1_RF09_RESULTS_20261027.md)、[量化汇总](../data/manifests/supermimic_q1_RF09_results_20261027.json)、[本地交付](../data/manifests/supermimic_q1_RF09_local_delivery_20261027.json)保留；旧RUNNING、待评测和目标active均为历史。

## 最新实验与结论

RF09仅在用户25252，4096 MuJoCo Warp环境、从零，实际18000.93146751076秒/17895学习轮/183504896转移、WM178950/actor17395。02:20:12开始训练、07:20:14正常结束、07:33:52全pipeline COMPLETED；七阶段guard退出0并reap，全部已登记controller/worker退出。36项训练保存、31项评测算术/六格图核验通过，153冻结来源与330参考保持。V01仅tuple/list序列化比较误报，失败原件保留，V02通过。

固定RF08原115参考共同执行片段：验证根XY RMSE0.40151→0.09232m，walking三片沿目标平均1.12341→2.55001m（参考2.39430m），原地gesture漂移0.63250→0.11589m。上肢整体仅小幅改善，gesture更差；验证提前终止0→2/15，旧训练walking3/28→5/28。新库验证29/30完成，但不能替代旧基准。预选两条普通前向走分别5.82/2.84秒倒地，视频保留失败，不按学后效果换片。尚不能称全身跟踪或稳定步态合格。

[AP02](README_SUPERMIMIC_Q1_AP02_SOMA_RETARGET_20261027.md)完成300训练/30验证、82/11演员无交集，原115源全部保留。Q1静止pelvis0.419m重复原点修正，同源D02腕误差3.52→3.12cm/峰高4.11→3.04cm；参考改善与策略跟踪分开，未宣称引入新的学习式重定向模型。数据、转换、预算共同改变，不作单因素归因。

## 本地工件与资源

Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027`。最终视频`evaluation/final.mp4`、初始`evaluation/initial.mp4`各54秒/1350帧/25fps/1280×480，8段6类，含两条普通前向走、双手抬起与原地手势。各片独立初始化、原速串接，终止后显式保持标记，不冒称连续54秒rollout。12项媒体核验/全帧解码、16段中点和两条失败结尾已目视核对。

六格图`dashboard/training_dashboard.png`及PDF/CSV均交付，原值/500学习iteration尾随均值，106870条数值/分母/窗口/实际转移核对通过。41文件共39411623字节Mac/服务器大小SHA一致。RF08旧基线和新增55.24秒长视频保留，入口[RF08结果](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)。

07:46:45 GPU14MiB/26度、共享内存10254839808字节/OOM0、盘余1591808323584字节，实验约3.006GB。100GiB余盘/12GiB实验/30GiB共享内存等保护未放宽，不动其他任务。权重/大轨迹留服务器LOCAL_ONLY，无网盘已备份声明；不拉Mac、不建网站。

## 版本与位置

权威Git `/tmp/yuanqi-supermimic-q1-20261023`，Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。冻结训练提交`a553b6a5a3c44167042355ff2f2fbd42ff558342`；训练终后审计发布`bbf05063f60d0a6729c4b550d9f81b27c8d989a7`已独立核远端、7文件142940字节三端大小SHA一致。后续报告提交不改变冻结来源。运行完整命令在[RF09运行记录](README_SUPERMIMIC_Q1_RF09_SOMA300_5H_20261027.md)，不能为恢复重复执行。

仅用户25252，唯一根`/opt/data/private/ZHY`及data/logs/script/reports。SSH：`ssh -S /tmp/yuanqi-q1-25252.sock -T -p 18869 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25252' root@127.0.0.1`。保留Windows桥，不读写认证，不操作师兄容器；SSH失败记UNKNOWN，不据此重启。

Python `data/environments/supermimic_q1_hp_20261026/bin/python`，经`script/supermimic_q1_hp/activate.py`。runtime `data/references/supermimichp/AP01_runtime_20261026`、上游`310244b73e45775744a75b789eb25a6c9c5a81c9`。RF09库`data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json`。4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4、BF16/compile、seed0、heading XY/朝向历史及完整未来根/关节参考，Q1 full effort/模型推导PD15.791367/1.005310。师兄实际1024环境，不称全部条件等同。

## 收尾与下一步

结果已提交push Q1、独立核远端40位SHA并镜像Mac/服务器。原`q1-rf08-4096`于08:00:27工具更新并回读为PAUSED；独立`supermimichp-x2`仓库跟踪与`yuanqi`研究任务保留。MG04已结算完成，当前无活动目标，不创建新目标。本次五小时不自动续训/重试，不创建或激活子agent。用户最新要求先诊断，覆盖旧训练推进授权。

后续若另行授权试验，优先处理普通走的制动/支撑和上肢幅度/相位，固定参考与预算再作单变量对照。不重复解包/hash45GB原包、转换330片、profile或已有媒体/轨迹分析。无关`script/supermimic_q1_sources/download_bones.py`不执行/修改/暂存/删除。
