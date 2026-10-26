# YUANQI 当前状态与下一步

当前MG02-D：[唯一RF06腿部限矩对照已启动](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)，2026-10-26北京时间17:22:03、controller3301/start_ticks3629852873、worker3304/start_ticks3629852891，boot保持d994e71f-bae0-46ba-a5f7-6c2163771c6c。冻结代码`0b1da033148fff0b705fd85cdac5a773b5a7f332`、6来源SHA通过；4096/WM2048×10/actor4096-H32，从零同seed/参考，12腿限矩50%/其余25%、动作尺度25%/PD50-2，软3000秒/最大3600秒、训练阶段3500秒+55秒清理。17:23:36真实进程/命令匹配、RUNNING step0初始化、OOM0/GPU2704MiB/37度/内存7.62GB/盘余1.789TB；尚无更新或效果结论。RF05全流程及本地交付已完成，A/B/C完成；不重启RF05/RF06、不热改冻结来源，不开子agent。

17:10:45，[RF05全流程与本地交付完成](README_SUPERMIMIC_Q1_E4096_20261025.md)：7237.6279秒/4608000转移、WM399/actor349，9阶段WORKER_COMPLETED、controller989/worker990均退出、GPU14MiB/OOM0/盘余1.789TB。11项1723401字节PNG/PDF/CSV及初末MP4已在Mac对应RF05目录逐大小/SHA通过；最终365步不倒但仅净移1.87cm，沿参考方向-1.68cm、双支撑99.178%、单支撑0，未行走，抬手同步仍0。腿部边界限矩饱和0.274%/髋横滚0，不能把RF04的限矩归因直接外推为本轮不走的唯一解释。独立RF04/RF05共同122步比较已保存，腿部/上肢误差均较RF04差。MG02 A/B/C完成，RF06尚未启动，下一复核资源后执行一次一小时腿部限矩对照；不重启RF05。

17:05：[RF05训练及初末动力学结果](README_SUPERMIMIC_Q1_E4096_20261025.md)已出：实际7237.6279秒、1125轮/4608000转移、WM399组/actor349次，保存13项及初末算术审计通过。最终365步不倒但净位移0.018737m/参考1.601504m；回报0.475380→0.500930，完整段腿部/上肢误差均略恶化，两侧肩上同步仍0。初始已365步不倒，不能把存活全部算训练收益。管线仍render_initial，worker990退出/GPU14MiB/OOM0，controller989身份仍匹配；先完成小产物交付，未启动RF06。

[MG02-C独立RF06入口](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)实现/验证/发布部署完成：8测试通过，C02真实CPU评测/算术审计/全身/腿部/两帧编码退出0，365步状态/动作与D06B误差0、无优化器更新/无CUDA；C01媒体入口故障已保留，复用已有字体/ffmpeg后通过，RF05的78项冻结来源不变。代码`0b1da033148fff0b705fd85cdac5a773b5a7f332`已独立核远端，13项Mac/25252大小SHA一致，目标一小时预算已核。RF06未启动，固定腿部限矩50%/其余25%，最多3600秒、软目标3000秒；RF05全流程完成并释放资源后才按RF06 README准确命令顺序执行。

16:25现场RF05仍RUNNING：750轮/3072000转移、WM249/actor199，controller989/worker990逐boot/start_ticks匹配，GPU40度/17274MiB、共享内存7.56GB、OOM0、盘余1.461TB。该轮只验证环境2048→4096及更新批量同步扩大，控制/参考/奖励未改，不是腿部特调；更多环境不自动提高噪声幅度或参考种类，效果须等真实终后评测。原两小时预算不变，下方较早进度与阶段计划保留为历史。

最新预算：用户要求今后每次训练最多1小时，先试效果。当前已运行RF05仍原两小时基线；下一轮RF06改为**训练最多3600秒、软目标3000秒**，独立训练阶段3500秒时限/55秒退出清理保护，不自动延长。MG02目标卡旧RF06“两小时”文字由此次新指示覆盖；后续具体入口与验收以[MG02 README](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)为准。RF05/RF06预算不同须如实报告，不能称等时消融。

[MG02-B闭环诊断D06B](README_SUPERMIMIC_Q1_LEG_CONTROL_20261026.md)已完成：原25%状态/动作精确重现122步倒地，**仅12腿关节50%**同一固定策略完整365步不倒，但净位移仅0.040410m/目标1.601504m、沿目标方向0.013741m、双支撑99.18%，仍未行走。全身50%无明显额外收益，抬手肩上同步仍0；852控制步/0optimizer/无CUDA。首尝试D06的变量覆盖故障及基线NPZ保留，D06B原样收据已接收。RF05源码/输入不变，接下来准备独立RF06训练/评测控制契约一致性。

当前正式目标已更新为[MG02腿部闭环与独立训练](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)，get_goal实际active，不重复创建。RF05保留原25%限矩/PD/奖励/参考，尚未针对腿部特调；15:27实读200轮/819200转移、WM29/actor0，真实进程匹配、OOM0。先完成三组闭环CPU诊断并准备独立RF06，仅12腿关节限矩改50%、动作尺度与上肢/腰保持；RF05全pipeline结束和释放资源后才启动RF06，4096环境至少7200秒，从零同seed/参考。阶段状态与具体命令以MG02 README为准，旧MG01 paused标签不再是当前目标。

15:10:43增量观测：[RF05启动/进度收据](../data/manifests/supermimic_q1_RF05_E4096_launch_observation_20261026.json)记录75轮/307200真实转移，WM0/actor0仍在128轮采样预热。controller与worker均逐boot/PID/start_ticks匹配，GPU38摄氏度/2704MiB，共享内存6860914688字节、under_oom0/OOM0，盘余1532036804608字节；本实验约69.29MB。实际训练尚未完成，不将日志seconds当两小时达标。原`q1-2048`已通过工具更新并核配置为ACTIVE，每10分钟监督25252新作业，正常静默；MG01目标工具的paused标签与监督自动任务状态分开记录。

15:02:22最新现场：[CM01迁移与运行验证完成](README_Q1_CONTAINER_MIGRATION_20261026.md)，自己的25252容器已实际启动[RF05-E4096](README_SUPERMIMIC_Q1_E4096_20261025.md)，4096环境/WM2048×10/actor4096-H32/UTD0.4、原365帧参考/seed/从零、实际训练至少7200秒、最大12000轮。发布部署版本`abda76e7bc873405b4321b4d9a6f5991e7290828`已独立核远端及27项目标文件；新train源码SHA`0a8128f6d4bfb5908436373d850c5edd7e51b808a0a731bbb5d94150ef92767f`。controller989/start_ticks3629014760、worker990/start_ticks3629014771，boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`；guard/管线RUNNING，初始step0，仅为启动和身份核验，未宣布效果。冻结运行源码/输入，不再覆盖。

25252连接：`ssh -S /tmp/yuanqi-q1-25252.sock -T -p 18869 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25252' root@127.0.0.1`，保留Windows桥；唯一实体根仍`/opt/data/private/ZHY`。本机socket为临时会话，不含认证材料。新容器OOM基线0，保留100GiB余盘/12GiB实验/30GiB共享内存/86摄氏度/900秒无进度保护；15:01启动前内存2.43GB、盘余1.53TB、GPU14MiB，真实CUDA/三模型/EGL通过。28.4万文件327.64GB全部内容SHA和原树一致，必要外部626文件及七处链接完成，历史360断链集合保持。

以下为各时点历史恢复记录；“仅迁移、不启动RF05”与旧容器OOM14均已被最新用户授权和25252现场覆盖。MG01工具最近仍返回paused，未伪报active或重复创建；用户已明确恢复执行D/E，监督任务后续按新入口更新。

最新授权：用户要求边迁移边分析腿部，运行条件具备可直接训练，覆盖下方“仅迁移、不启动RF05”的旧范围。正式RF05只在自己的25252容器执行，4096环境/WM2048×10/actor4096-H32/UTD0.4、同参考seed/从零、至少7200秒；现尚未启动。MG01工具14:44仍返回paused，不能虚记active，不重复创建目标。

[D05腿部动力学对照](README_SUPERMIMIC_Q1_LEG_CONTROL_20261026.md)已完成：原保存动作逐步精确重现；25%限矩腿部边界饱和10.25%，左右髋横滚68.03%/47.54%。50%同122步开环动作未触发倒地但净位移0.08721m、双支撑97.54%，未证明闭环或完整行走；降低参考26.53mm的直接PD仍早倒。探针0optimizer/无CUDA/源文件不变，收据和六轨迹已在25252独立目录。RF05先保留原控制/参考，完成环境规模对照，不由探针同时改多个变量。

14:46迁移核验：源283995普通文件SHA生成退出0，目标已逐字节验证217029文件/284278342761字节，主tar仍运行；SONIC外部626文件/1084496445字节独立验证完成。目标Python链接和运行脚本尚在迁移尾段，当前不能启动；后续补兼容链接、CUDA/MuJoCo及资源现场检查，然后冻结部署启动，不把非运行数据的尾段迁移设为训练硬门。

10月26日用户澄清25651属于师兄，自己的正式入口为`root@59.77.15.15:25252`，经Windows桥接、本机18869。[CM01全量迁移](README_Q1_CONTAINER_MIGRATION_20261026.md)进行中：源项目约327.64GB逻辑内容、283995个普通文件，采用容器间SSH直传到自己的`/opt/data/private/ZHY`，源原件保留；另补必要SONIC外部依赖。完整性和运行检查尚未完成，不能视为可训练环境已通过；不使用下方25651“无需搬运”的历史判断。MG01仍paused，不启动RF05。

当前正式目标：[MG01两阶段训练监督](README_SUPERMIMIC_Q1_MONITOR_GOAL_20261025.md)为paused，10月26日13:31已由get_goal重新核验；换服务器入口和效果询问未恢复训练/监督。阶段A/B/C已完成，RF05及最终比较未完成。以下02时等盘满/active表述仅为历史，不是启动授权。

最新入口为经原Windows桥接的`root@59.77.15.15:25651`，本地18868，13:32现场已确认原`/opt/data/private/ZHY`、Python环境、BONES原包和RF04权重可见；RTX3090/24GiB空闲，盘余1866774560768字节，约1.87TB。先前25636是不同private子树，不再使用其缺文件结论；25061未尝试。无需搬运已有资产，但内存/OOM及完整运行条件仍待重新核验。RF05目录不存在，未启动。

[13:36同参考诊断](README_SUPERMIMIC_Q1_FAILURE_REVIEW_20261026.md)发现RF04实际输入整段抬高36.44mm，90.14%帧双脚足底均高于15mm，前122帧52.46%均高于30mm；来源为walk_forward_stop，根目标确含前进。优先验证重定向后的足地几何、控制执行及接触切换学习链；只读FK/0积分/0更新，不证明唯一根因、不增加数据门。新收据与可复现执行记录见阶段README。

10月26日01:23:37，[RF04全流程COMPLETED](README_SUPERMIMIC_Q1_E2048_20261025.md)：实际训练7244.1069秒、1775轮/3635200转移、WM659/actor609，13项保存检查和初末算术身份审计通过；六宫格和初末MP4已到Mac对应实验目录，11项1512643字节逐大小/SHA通过。最终产生0.456592m位移但122/365步、2.44秒提前倒地，整片回报0.47538→0.19674；共同122步上肢RMSE0.68489→0.60841rad，完整抬手主峰未到达，不称跟踪成功。RF04完整结束，不重启或追加预算。

用户要求[后续4096环境对齐论文采样规模](README_SUPERMIMIC_Q1_E4096_20261025.md)，01:35:18完整profile已通过：4096环境/8线程/8槽、16轮65536转移、WM2048×10及actor4096/H32/critic实际更新有限，CUDA分配/保留峰值15.60/16.54GiB，guard WORKER_COMPLETED且worker回收，OOM14。短测不是正式训练或效果结果，不重复profile。RF05仍用同一参考/seed、从零与至少7200秒；仅环境和批量规模对齐论文。RF04冻结结束后仅修正manifest采样规模声明，语法、64/2048/4096文案和AST对比通过，随本阶段发布部署。

当前阻碍是共享磁盘：[10月26日快速增长排查](README_Q1_STORAGE_GROWTH_20261026.md)两次private一级目录/总量完全相同，共703216345088字节，但底层整盘余量持续下降，增长超出容器当前可见范围，不能点名具体其他项目。01:52:03实际余32179216384字节，约30GiB，低于100GiB保护；RF05目录不存在，尚未启动，不记训练失败。此时GPU27摄氏度/14MiB、共享内存16655953920字节、under_oom0/OOM14，RF04 worker与profile进程已退出；旧RF04 controller为匹配start_ticks的僵尸，不是活跃训练。未删除文件或停止他人任务。沿用原q1-2048监督，待平台释放同盘空间或提供统一根下挂载后重新核资源与来源身份，顺序启动RF05；不降低保护、不重复实测、不新增子agent。

RF04历史启动于10月25日23:21:31，2048环境/8槽/8线程、WM batch1024×10、actor2048/H32、UTD0.4，保留原365帧抬手移动参考与方法、从零初始化。冻结代码f03db7566fcd590912b4cc9b24621d32b19319a2，终后审计确认输入/来源保持；controller885444/start_ticks3623369662、worker885445/start_ticks3623369674仅作为已结束作业身份，不再视为当前RUNNING。后续可改元数据须另发布身份；`q1-2048`每10分钟监督后续4096，正常静默，禁止自动重启/扩预算。

02:02:16最新复查：共享盘可用空间已为0，RF05仍不存在，未启动。元数据修正与profile/磁盘记录已在Q1提交`81db2e1fee2828e9ce59d0271c7d2f2cde56cee1`发布并独立核远端；服务器因无可用空间尚未部署/同步，新代码SHA`0a8128f6d4bfb5908436373d850c5edd7e51b808a0a731bbb5d94150ef92767f`，服务器仍旧SHA`4a1431570c80cae8e3377c80bc25a14992f66e1a1af58aaffddd63c6643c519b`。最新报告和小收据先保留GitHub与Mac，待盘余恢复后补服务器同步/身份核验再启动。既有本地曲线与MP4不受本次服务器盘余归零影响。

唯一子agent已完成[位移诊断](README_SUPERMIMIC_Q1_DISPLACEMENT_REVIEW_20261025.md)，没有GPU或新仿真。旧RF02参考/实际速度约0.327/0.019m/s，腿部幅度约参考26.5%、双支撑97.86%，目标速度一致性已核；目前支持站立附近局部最优，不确定单一原因。先看本轮固定动作动力学比较，再考虑完整闭环梯度及同源脚部相位检查，不增加数据硬门。

原[RF03-UP01](README_SUPERMIMIC_Q1_RF03_UP01_20261025.md)因用户调整已于22:51:46 STOPPED_SIGNAL，667轮/42688转移、WM216/actor166、971.10秒；guard确认worker回收、GPU恢复14MiB/OOM14，不恢复、不执行原计划初末评测。final.pt为停止快照，不是正式完成。旧RUNNING收据均为历史快照。

最新完成：[RF02全身诊断D02](README_SUPERMIMIC_Q1_RF02_FULLBODY_20261025.md)。固定4条上肢片肩肘0.63631→0.28769rad，但中途抬手事件实际幅度仅目标8.45–13.16%、目标高于肩时均未达到；1024来源全部locomotion桶，源G1高于肩片数左5/右4、Q1左4/右3。原8片根位置奖励贡献几乎未增，双脚承重片均97.86%、腿力矩饱和0.51%。6状态H1/8/32局部开环奖励方向各12/12一致，剥离精确动作率项仍成立，但H32微调根XY仅约0.07mm，不能证明会迈步或完整actor梯度正确。全部10483控制步/209660子步，4聚焦单测及原数组审计通过，0optimizer更新；工件约6.73MB，D02全部作业退出/OOM14。其单动作建议现已作为上方独立RF03启动，原RF02不恢复。

## 本轮终态与证据

- [整轮结果与交付](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)：冻结1024个BONES train来源，326883帧/108.62分钟，120→50Hz原速Q1转换；复用128，新增896，仅使用可读、维度、时轴、数值等运行条件。旧几何flags保留诊断，不恢复代理额外硬门。
- RF02为64环境、12000/12000采集轮、768000真实转移，主训练6小时31分35秒。WM4749组/actor4699次；最终保存11项核对通过，三Adam47490/4699/93980、输入和源码保持。pipeline在20:36:58完成，16阶段退出0，未追加预算。[训练收据](../data/manifests/supermimic_q1_BONES_RF02_training_receipt_20261025.json)、[保存核对](../data/manifests/supermimic_q1_BONES_RF02_training_audit_20261025.json)、[流程终态](../data/manifests/supermimic_q1_BONES_G02_final_status_20261025.json)。
- 固定相同8条train参考，初始/最终整片归一化回报0.379982→0.654285、完成82.4645%→100%、提前终止2/8→0/8，共同前缀关节RMSE0.43952→0.27418rad（7/8改善）。最终平均根净位移仅0.02035m/参考1.81529m，**尚未学会行走和移动**；结果不代表未见动作或真机。[对比](../data/manifests/supermimic_q1_BONES_RF02_comparison_20261025.json)、[名单/视频身份核对](../data/manifests/supermimic_q1_BONES_RF02_final_evidence_audit_20261025.json)。
- 用户最新要求：视频直接保存本地MP4，自行打开，不再额外搭网站。Mac目录`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF02_replay_checks_20261025/`已保存`Q1_BONES_RF02_initial.mp4`及`Q1_BONES_RF02_final.mp4`，合计412323字节，大小/SHA与服务器一致，末帧目视通过。首片初始87/244步倒地，最终244/244步但只净移动0.01514m。已关闭本轮服务器18867和Mac18863视频转发，保留18864桥；[交付收据](../data/manifests/supermimic_q1_BONES_RF02_local_mp4_delivery_20261025.json)。偏好已写入三端各自AGENTS。
- [每轮六宫格训练图](README_SUPERMIMIC_Q1_TRAINING_DASHBOARD_20261025.md)：最终PNG/PDF/CSV及收据已保存服务器和Mac的`data/supermimic_for_q1/BONES_RF02_training_dashboard_20261025/`。1425点独立重算通过、PNG目视通过；本轮未记录前三项回合统计，明确留缺。下一新实验使用`script/project_tools/train_q1_with_dashboard.py`记录完整六项并自动出图，细原始/粗500训练iteration均线。

## 平台与资源

唯一实体根`huanyu:/opt/data/private/ZHY`，四目录`data / logs / script / reports`。Python为`data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`。现有18864 SSH桥仍监听；boot基线`d994e71f-bae0-46ba-a5f7-6c2163771c6c`、OOM基线14，恢复必须重新核现场。

最新资源与磁盘阻碍见上方01:52观测；以下均为历史记录。RF04启动后23:35现场：GPU40摄氏度/9150MiB，共享容器内存23.366GB、under_oom0/OOM14，盘余835.629GB，本实验分配约69.3MB。共享盘余变化不等于本实验新增量。2048预检完整WM/actor/critic更新有限、CUDA峰值约8.49GB；D02及下方20:43/20:50不能作为当前空闲证明。

20:43终后观测已无活动训练，GPU28摄氏度/14MiB；本轮训练目录分配2.748GB、ZHY总分配325.008GB。20:50实际cgroup v1补充观测内存23.525GB、under_oom0/OOM14、盘余1.682TB；初次cgroup v2探测的null原样保留。[分配/进程观测](../data/manifests/supermimic_q1_BONES_RF02_final_resource_observation_20261025.json)、[真实v1健康观测](../data/manifests/supermimic_q1_BONES_RF02_final_health_observation_20261025.json)。100GiB余盘、单实验12GiB及温度/内存/OOM保护继续执行。

最终权重在服务器`data/supermimic_for_q1/BONES_RF02_12000_20261025/run/final.pt`，205912083字节/SHA`fa5bdc4542986df522fd3aeb96a59f2c121eba3688f6527805f3fb98b254c8fb`；初始、1000至12000保存保留。Mac只接收代码、小收据、曲线和小视频，不拉模型。大型资产仍为LOCAL_ONLY，不声称网盘备份。

BONES原包`data/references/bones_seed/archives/g1.tar.gz`，23.50GB、SHA`52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`；完整SHA/gzip CRC/142220 CSV目录覆盖通过，仅提取冻结1280候选，无全量展开或原包副本。[接收记录](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)。AMASS/16维SMPL-X、PHUMA及必要环境继续保留，见[存储归拢](README_Q1_STORAGE_CONSOLIDATION_20261025.md)。

## 后续与边界

背景讨论见[奖励、重定向与接触梯度归因](README_SUPERMIMIC_Q1_RF02_GRADIENT_REASONING_20261025.md)；原建议的本轮分项/接触/局部收益方向已在D02完成。现有局部收益方向结果不能外推脚支撑切换、critic或全闭环；当前不凭讨论直接更改奖励/接触参数。保持SuperMimic优先和无新增数据门。

原RF02训练、评测和绘图都已结束，不重启或自动追加预算；RF03已停止，RF04完成，后续4096的RF05尚未启动。[只读方法审阅](README_SUPERMIMIC_Q1_RF02_METHOD_REVIEW_20261025.md)登记的500步reset回报缺bootstrap/截断样本排除问题仍待，86/1024段超过500帧、影响未量化；本次单动作365帧先避开这一触发条件。方法核心来自第三方固定SuperMimic-Reproduced，不能称作者官方训练源码。

旧PHUMA R02D磁盘保护停止、旧A3及所有FAILED/STOPPED实验不恢复、不改终态。最多一个子agent；本轮结束没有执行中的子任务。RF01的500轮失败及各时点RUNNING快照见整轮README和报告索引，不作为当前状态。

## Git与恢复

权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。当前RF04训练冻结版本`f03db7566fcd590912b4cc9b24621d32b19319a2`；`d66ac1cd21de1682ca6f069227c070ebf98da02f`仅为旧RF02历史版本。文档及独立4096资源入口的后续提交不代表运行中训练源码改变。各阶段记录显式推送Q1并独立比对远端，服务器/用户镜像同步报告、小收据和各自AGENTS新增偏好。唯一既有未跟踪`script/supermimic_q1_sources/download_bones.py`不执行、不提交、不删除。

恢复先读此文件、[RF04-E2048](README_SUPERMIMIC_Q1_E2048_20261025.md)、[后续RF05-E4096](README_SUPERMIMIC_Q1_E4096_20261025.md)和[磁盘排查](README_Q1_STORAGE_GROWTH_20261026.md)，重新核资源/进程/Git状态。RF04为COMPLETED，不重启、覆盖或追加预算；`E4096_PROFILE_P01_20261025`已正常完成，不重复profile。先核`BONES_RF05_UP01_E4096_T02H_20261025`是否出现及现场状态，未启动时须实际余盘恢复100GiB后才执行，使用已发布部署身份。RF03仅为停止历史；RF02历史证据由整轮README恢复。详细历史见[报告索引](README.md)、[论文优先前快照](README_CURRENT_STATE_BEFORE_PAPER_FIRST_20261025.md)和[更早归档](README_CURRENT_STATE_ARCHIVE_20261025.md)。

- 每日研究最近检查10/26：46研究请求全部成功，关注源无实质更新；两份10/23旧v2仅HTML元数据等变化，去重111→113、H001保持，静默、不新增简报。最新有内容仍为[10/24](research/2026-10-24/README.md)；见[检查](README_RESEARCH_CHECK_20261026.md)，main/Q1文档已推送，09:23:32独立内容核验通过。当前Q1工程以自身最新状态为准，日报不调度计算、不重复既有资源问题。
