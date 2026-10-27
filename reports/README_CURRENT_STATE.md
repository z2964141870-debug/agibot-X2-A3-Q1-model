# YUANQI 当前状态与下一步

2026-10-27 17:37最新：[CS02](README_Q1_CONTAINER_RESTORE_CS02_20261027.md)已恢复用户25386，hostname`interactive40173`、boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`，原ZHY共享挂载未变。RF09权重与清单SHA一致、330参考路径无缺失，CUDA/模型/权重加载通过，GPU换为3090 `GPU-26028785-9fb9-45af-e614-4cc53af15d08`，检查后15MiB/26度，OOM0、盘余约1.65TB。停服小结果/日志已取回并核SHA，实际日志有backend/attrs，139具体退出阶段仍未明；原生接触/H32与候选契约待。本次0新积分/0optimizer/无训练/无agent，MG05工具仍paused且未完成；下方等待新入口为历史。[收据](../data/manifests/q1_container_restore_CS02_20261027.json)。

2026-10-27：[MorphMimic只读初评](README_MORPHMIMIC_REVIEW_20261027.md)已保存。它扩展跨机器人世界模型/策略共享和五点输入；默认缺位置纠偏、29关节完整映射及上游报告显存占用均需Q1适配，不能直接替换现主线。未安装/仿真/训练，也未恢复MG05；等待用户新服务器入口。

2026-10-27最新：用户确认25245服务器已经停止，等待新SSH/密码，期间继续可行离线分析。旧入口不再连接、不自动重启；本轮无正式训练。MG05本次get_goal实际为paused，四项目标未完成，不创建/恢复旧目标。[离线分析与恢复顺序](README_SUPERMIMIC_Q1_MG05_SIGNAL_ANALYSIS_20261027.md)保存：验证手势81.72%上肢平方误差为片内持续bias；[上游核查](README_SUPERMIMICHP_Q1_UPDATE_20261027.md)确认train15/300片尾bin跨片。候选独立构建器、关节误差/奖励与采样overlay已写，16项离线测试及实际MDP公式AST检查通过；服务器真实runtime尚未生成或部署，Torch/物理契约未运行。

独立BD02物理探针V02已退出0，但两条普通走没有复现原生轨迹，接触力不当作原摔倒归因；局部H1/4/8动作方向已测、H32闭环待。旧scope纯FP32说明已勘误为继承checkpoint AMP；原生检查139段错误日志和完整result待新入口取回。先核实际模型/摩擦/时序和原生力，再验证当前关节误差观测/跟踪奖励候选。唯一上游子agent已完成，不再开agent。A部分完成、B部分准备、C至E待；新一轮仍4096环境最多3600秒，不续训RF09。Mac/Git保存新小记录，服务器同步待恢复。

2026-10-27最新授权：[MG05诊断与tracking修正目标](README_SUPERMIMIC_Q1_MG05_DIAGNOSE_TRACKING_20261027.md)active，get_goal已核为用户四项完整目标。先诊断、选方案，再独立4096环境最多一小时验证训练；唯一子agent只读核上游更新，主代理负责25245。覆盖下方暂停/无目标/不开agent旧状态，不启动旧实验。当前尚无新训练，RF09冻结来源保持。

2026-10-27 10:20：用户已将工作入口改为25245，仍暂停训练。[CS01容器切换](README_Q1_CONTAINER_SWITCH_20261027.md)已验证：`interactive63856`共享原`PRIVATE/zhy`挂载，ZHY/环境/330参考/权重现存，不复制大数据；CUDA张量、MuJoCo/Warp导入、Q1模型/最终权重加载通过，0积分/0optimizer。GPU3090空闲14MiB/26度、盘余约1.589TB、新容器OOM0；这些是当时采样，后续仍需复核。恢复统一使用下方25245命令，不再使用历史25252入口。

2026-10-27最新：用户要求先别继续训练，后续训练保持停止，不自动开启RF10、续训或恢复旧目标。MG04已完成，实际get_goal为null，本轮监督已PAUSED。[BD01瓶颈诊断](README_SUPERMIMIC_Q1_RF09_BOTTLENECK_20261027.md)/[小收据](../data/manifests/supermimic_q1_RF09_BD01_diagnosis_20261027.json)完成：冻结策略/保存轨迹496次前向、0积分/0optimizer。验证8条gesture上肢策略目标到参考1.18438rad、实际到目标0.06185rad；首先是策略目标偏差。手势仍获约69.10%最高步奖励，身体朝向项仅满值2.60%，奖励敏感度/模型动作梯度优先检查，尚未锁定唯一根因。

两条普通走参考本身双脚近地85.57%/80.99%，几何比例不能证明真实双足受力；Q1所有关节仍为模型统一推导PD15.79/1.01，非已完成逐关节整定。保持有界诊断，先区分接触/制动、控制与模型方向问题，不再默认加数据/时长。既有[RF09结果](README_SUPERMIMIC_Q1_RF09_RESULTS_20261027.md)、[量化汇总](../data/manifests/supermimic_q1_RF09_results_20261027.json)、[本地交付](../data/manifests/supermimic_q1_RF09_local_delivery_20261027.json)保留；旧RUNNING、待评测和目标active均为历史。

2026-10-27研究：[五项简报](research/2026-10-27/README.md)已保存；Q1当前MotionLib静态存在末尾补帧复用末帧速度的模式，实际影响未测、未归因跌倒。另记SONIC朝向修复、第三方资产和三篇方法边界；无新增训练/仿真或原创假设。完整历史/去重由main维护，工程以Q1最新状态为准；[检查](README_RESEARCH_CHECK_20261027.md)/[同步回执](../data/manifests/research_watch_sync_20261027.json)。

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

当前仅用户25386，唯一根`/opt/data/private/ZHY`及data/logs/script/reports。SSH：`ssh -S /tmp/yuanqi-q1-25386.sock -T -p 18873 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25386' root@127.0.0.1`。Windows桥控制socket`/tmp/yuanqi-windows-25386.sock`；不读写认证，不操作师兄容器。25245/25252均为历史入口，不再探测，不把历史断线当丢数据依据。

Python `data/environments/supermimic_q1_hp_20261026/bin/python`，经`script/supermimic_q1_hp/activate.py`。runtime `data/references/supermimichp/AP01_runtime_20261026`、上游`310244b73e45775744a75b789eb25a6c9c5a81c9`。RF09库`data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json`。4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4、BF16/compile、seed0、heading XY/朝向历史及完整未来根/关节参考，Q1 full effort/模型推导PD15.791367/1.005310。师兄实际1024环境，不称全部条件等同。

## 收尾与下一步

结果已提交push Q1、独立核远端40位SHA并镜像Mac/服务器。原`q1-rf08-4096`于08:00:27工具更新并回读为PAUSED；独立`supermimichp-x2`仓库跟踪与`yuanqi`研究任务保留。MG04已结算完成，当前无活动目标，不创建新目标。本次五小时不自动续训/重试，不创建或激活子agent。用户最新要求先诊断，覆盖旧训练推进授权。

下一步按CS01诊断计划：先在冻结策略中核上肢目标/奖励及动作梯度，再用少量固定普通走状态区分真实足地受力、滑移与制动，最后核模型方向与物理微扰是否一致。尚未执行这些新增物理试验；先给出证据与修改方案，再决定训练。后续若另行授权训练，固定参考与预算作受控对照，不自动续训。不重复解包/hash45GB原包、转换330片、profile或已有媒体/轨迹分析。无关`script/supermimic_q1_sources/download_bones.py`不执行/修改/暂存/删除。
