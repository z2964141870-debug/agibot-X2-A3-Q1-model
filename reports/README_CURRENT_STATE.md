# YUANQI 当前状态与下一步

当前正式目标：[MG01两阶段训练监督](README_SUPERMIMIC_Q1_MONITOR_GOAL_20261025.md)保持active，阶段A/B已完成，4096实测/独立两小时训练及最终比较待。10月26日01:23:37，[RF04全流程COMPLETED](README_SUPERMIMIC_Q1_E2048_20261025.md)：实际训练7244.1069秒、1775轮/3635200转移、WM659/actor609，13项保存检查和初末算术身份审计通过；六宫格和初末MP4已到Mac对应实验目录，11项1512643字节逐大小/SHA通过。最终产生0.456592m位移但122/365步、2.44秒提前倒地，整片回报0.47538→0.19674；共同122步上肢RMSE0.68489→0.60841rad，完整抬手主峰未到达，不称跟踪成功。01:24:30 worker已退出/GPU14MiB/OOM14，盘余183.36GiB持续下降，本实验仅约0.483GB；已询问平台侧空间，100GiB保护保留。下一步一次4096完整批量实测，不能以RF04性能失败冒充方法成功；本轮训练预算正常完成，可以按授权继续对照。

最新追加指示：当前2048环境两小时RF04继续，用户要求[后续4096环境对齐论文采样规模](README_SUPERMIMIC_Q1_E4096_20261025.md)。RF04完成评测/交付及释放资源后，顺序一次4096完整WM2048×10、actor4096/H32容量实测，允许后直接独立RF05；暂沿用相同参考、从零与7200秒预算。4096尚未实测/训练，18GB仅估计。独立profile入口已扩展；训练manifest文案待RF04冻结结束后修正，再发布部署。沿用同一监督任务，不并行训练，不增加子agent。

RF04历史启动于10月25日23:21:31，2048环境/8槽/8线程、WM batch1024×10、actor2048/H32、UTD0.4，保留原365帧抬手移动参考与方法、从零初始化。冻结代码f03db7566fcd590912b4cc9b24621d32b19319a2，终后审计确认输入/来源保持；controller885444/start_ticks3623369662、worker885445/start_ticks3623369674仅作为已结束作业身份，不再视为当前RUNNING。后续可改元数据须另发布身份；`q1-2048`每10分钟监督后续4096，正常静默，禁止自动重启/扩预算。

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

RF04启动后23:35现场：GPU40摄氏度/9150MiB，共享容器内存23.366GB、under_oom0/OOM14，盘余835.629GB；最近guard本实验分配约69.3MB，保护正常。共享盘余变化不等于本实验新增量，继续按真实余量监督。2048预检完整WM/actor/critic更新有限、CUDA峰值约8.49GB，profile已退出。D02及下方20:43/20:50是此前终后历史观测，不作为当前空闲证明。

20:43终后观测已无活动训练，GPU28摄氏度/14MiB；本轮训练目录分配2.748GB、ZHY总分配325.008GB。20:50实际cgroup v1补充观测内存23.525GB、under_oom0/OOM14、盘余1.682TB；初次cgroup v2探测的null原样保留。[分配/进程观测](../data/manifests/supermimic_q1_BONES_RF02_final_resource_observation_20261025.json)、[真实v1健康观测](../data/manifests/supermimic_q1_BONES_RF02_final_health_observation_20261025.json)。100GiB余盘、单实验12GiB及温度/内存/OOM保护继续执行。

最终权重在服务器`data/supermimic_for_q1/BONES_RF02_12000_20261025/run/final.pt`，205912083字节/SHA`fa5bdc4542986df522fd3aeb96a59f2c121eba3688f6527805f3fb98b254c8fb`；初始、1000至12000保存保留。Mac只接收代码、小收据、曲线和小视频，不拉模型。大型资产仍为LOCAL_ONLY，不声称网盘备份。

BONES原包`data/references/bones_seed/archives/g1.tar.gz`，23.50GB、SHA`52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`；完整SHA/gzip CRC/142220 CSV目录覆盖通过，仅提取冻结1280候选，无全量展开或原包副本。[接收记录](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)。AMASS/16维SMPL-X、PHUMA及必要环境继续保留，见[存储归拢](README_Q1_STORAGE_CONSOLIDATION_20261025.md)。

## 后续与边界

背景讨论见[奖励、重定向与接触梯度归因](README_SUPERMIMIC_Q1_RF02_GRADIENT_REASONING_20261025.md)；原建议的本轮分项/接触/局部收益方向已在D02完成。现有局部收益方向结果不能外推脚支撑切换、critic或全闭环；当前不凭讨论直接更改奖励/接触参数。保持SuperMimic优先和无新增数据门。

原RF02训练、评测和绘图都已结束，不重启或自动追加预算；RF03已停止，RF04完成，后续4096的RF05尚未启动。[只读方法审阅](README_SUPERMIMIC_Q1_RF02_METHOD_REVIEW_20261025.md)登记的500步reset回报缺bootstrap/截断样本排除问题仍待，86/1024段超过500帧、影响未量化；本次单动作365帧先避开这一触发条件。方法核心来自第三方固定SuperMimic-Reproduced，不能称作者官方训练源码。

旧PHUMA R02D磁盘保护停止、旧A3及所有FAILED/STOPPED实验不恢复、不改终态。最多一个子agent；本轮结束没有执行中的子任务。RF01的500轮失败及各时点RUNNING快照见整轮README和报告索引，不作为当前状态。

## Git与恢复

权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。当前RF04训练冻结版本`f03db7566fcd590912b4cc9b24621d32b19319a2`；`d66ac1cd21de1682ca6f069227c070ebf98da02f`仅为旧RF02历史版本。文档及独立4096资源入口的后续提交不代表运行中训练源码改变。各阶段记录显式推送Q1并独立比对远端，服务器/用户镜像同步报告、小收据和各自AGENTS新增偏好。唯一既有未跟踪`script/supermimic_q1_sources/download_bones.py`不执行、不提交、不删除。

恢复先读此文件、[RF04-E2048](README_SUPERMIMIC_Q1_E2048_20261025.md)、[后续RF05-E4096](README_SUPERMIMIC_Q1_E4096_20261025.md)，重新核资源/进程/Git状态；RF04为COMPLETED，不重启、覆盖或追加预算。先查看`E4096_PROFILE_P01_20261025`与`BONES_RF05_UP01_E4096_T02H_20261025`目录是否已存在及其状态，避免重复profile/训练。RF03仅为停止历史；RF02历史证据由整轮README恢复。详细历史见[报告索引](README.md)、[论文优先前快照](README_CURRENT_STATE_BEFORE_PAPER_FIRST_20261025.md)和[更早归档](README_CURRENT_STATE_ARCHIVE_20261025.md)。每日研究最近10/25检查无实质更新，静默；自动研究不调度训练。
