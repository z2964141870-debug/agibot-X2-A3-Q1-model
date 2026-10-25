# YUANQI 当前状态与下一步

更新日期：2026-10-25，北京时间。本轮BONES-G02/PF02/RF02有限预算基线已完成；仍沿Q1上的SuperMimic可微世界模型、想象BPTT和critic方法复现。完成基线不代表行走或论文完整复现成功。

## 本轮终态与证据

- [整轮结果与交付](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)：冻结1024个BONES train来源，326883帧/108.62分钟，120→50Hz原速Q1转换；复用128，新增896，仅使用可读、维度、时轴、数值等运行条件。旧几何flags保留诊断，不恢复代理额外硬门。
- RF02为64环境、12000/12000采集轮、768000真实转移，主训练6小时31分35秒。WM4749组/actor4699次；最终保存11项核对通过，三Adam47490/4699/93980、输入和源码保持。pipeline在20:36:58完成，16阶段退出0，未追加预算。[训练收据](../data/manifests/supermimic_q1_BONES_RF02_training_receipt_20261025.json)、[保存核对](../data/manifests/supermimic_q1_BONES_RF02_training_audit_20261025.json)、[流程终态](../data/manifests/supermimic_q1_BONES_G02_final_status_20261025.json)。
- 固定相同8条train参考，初始/最终整片归一化回报0.379982→0.654285、完成82.4645%→100%、提前终止2/8→0/8，共同前缀关节RMSE0.43952→0.27418rad（7/8改善）。最终平均根净位移仅0.02035m/参考1.81529m，**尚未学会行走和移动**；结果不代表未见动作或真机。[对比](../data/manifests/supermimic_q1_BONES_RF02_comparison_20261025.json)、[名单/视频身份核对](../data/manifests/supermimic_q1_BONES_RF02_final_evidence_audit_20261025.json)。
- 用户最新要求：视频直接保存本地MP4，自行打开，不再额外搭网站。Mac目录`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF02_replay_checks_20261025/`已保存`Q1_BONES_RF02_initial.mp4`及`Q1_BONES_RF02_final.mp4`，合计412323字节，大小/SHA与服务器一致，末帧目视通过。首片初始87/244步倒地，最终244/244步但只净移动0.01514m。已关闭本轮服务器18867和Mac18863视频转发，保留18864桥；[交付收据](../data/manifests/supermimic_q1_BONES_RF02_local_mp4_delivery_20261025.json)。偏好已写入三端各自AGENTS。
- [每轮六宫格训练图](README_SUPERMIMIC_Q1_TRAINING_DASHBOARD_20261025.md)：最终PNG/PDF/CSV及收据已保存服务器和Mac的`data/supermimic_for_q1/BONES_RF02_training_dashboard_20261025/`。1425点独立重算通过、PNG目视通过；本轮未记录前三项回合统计，明确留缺。下一新实验使用`script/project_tools/train_q1_with_dashboard.py`记录完整六项并自动出图，细原始/粗500训练iteration均线。

## 平台与资源

唯一实体根`huanyu:/opt/data/private/ZHY`，四目录`data / logs / script / reports`。Python为`data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`。现有18864 SSH桥仍监听；boot基线`d994e71f-bae0-46ba-a5f7-6c2163771c6c`、OOM基线14，恢复必须重新核现场。

20:43终后观测已无活动训练，GPU28摄氏度/14MiB；本轮训练目录分配2.748GB、ZHY总分配325.008GB。20:50实际cgroup v1补充观测内存23.525GB、under_oom0/OOM14、盘余1.682TB；初次cgroup v2探测的null原样保留。[分配/进程观测](../data/manifests/supermimic_q1_BONES_RF02_final_resource_observation_20261025.json)、[真实v1健康观测](../data/manifests/supermimic_q1_BONES_RF02_final_health_observation_20261025.json)。100GiB余盘、单实验12GiB及温度/内存/OOM保护继续执行。

最终权重在服务器`data/supermimic_for_q1/BONES_RF02_12000_20261025/run/final.pt`，205912083字节/SHA`fa5bdc4542986df522fd3aeb96a59f2c121eba3688f6527805f3fb98b254c8fb`；初始、1000至12000保存保留。Mac只接收代码、小收据、曲线和小视频，不拉模型。大型资产仍为LOCAL_ONLY，不声称网盘备份。

BONES原包`data/references/bones_seed/archives/g1.tar.gz`，23.50GB、SHA`52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`；完整SHA/gzip CRC/142220 CSV目录覆盖通过，仅提取冻结1280候选，无全量展开或原包副本。[接收记录](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)。AMASS/16维SMPL-X、PHUMA及必要环境继续保留，见[存储归拢](README_Q1_STORAGE_CONSOLIDATION_20261025.md)。

## 后续与边界

最新讨论见[奖励、重定向与接触梯度归因](README_SUPERMIMIC_Q1_RF02_GRADIENT_REASONING_20261025.md)：当前全身/根奖励存在，根位置项占正奖励理论上限10%，身体位置项使用xy/yaw重对齐；低状态误差不证明动作收益方向正确。旧R03开环方向检查不作为RF02结论；下一先分解本轮真实轨迹奖励、核参考/策略接触对应，再比较模型建议微调与真实回报/位移变化。此次只读核对，0新物理/模型前向/更新，不以讨论直接更改奖励或接触参数。

本轮训练、评测和绘图都已结束，不重启或自动追加预算。后续优先核参考移动与动作/奖励对应，处理[只读方法审阅](README_SUPERMIMIC_Q1_RF02_METHOD_REVIEW_20261025.md)登记的500步reset回报缺bootstrap/截断样本排除问题；86/1024段超过500帧，实际影响仍未量化。方法核心来自第三方固定SuperMimic-Reproduced，不能称作者官方训练源码。新实验保持论文优先，不新增代理自定数据门。

旧PHUMA R02D磁盘保护停止、旧A3及所有FAILED/STOPPED实验不恢复、不改终态。最多一个子agent；本轮结束没有执行中的子任务。RF01的500轮失败及各时点RUNNING快照见整轮README和报告索引，不作为当前状态。

## Git与恢复

权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。训练冻结版本`d66ac1cd21de1682ca6f069227c070ebf98da02f`；本轮结果随Q1阶段提交保存并独立比对远端。服务器/用户镜像同步报告、小收据和各自AGENTS新增偏好。唯一既有未跟踪`script/supermimic_q1_sources/download_bones.py`不执行、不提交、不删除。

恢复先读此文件、整轮README与服务器流程`BONES_BASELINE_G02_20261025/status.json`、训练`BONES_RF02_12000_20261025/run/receipt.json`，重新核资源/进程/Git状态。详细历史见[报告索引](README.md)、[论文优先前快照](README_CURRENT_STATE_BEFORE_PAPER_FIRST_20261025.md)和[更早归档](README_CURRENT_STATE_ARCHIVE_20261025.md)。每日研究最近10/25检查无实质更新，静默；自动研究不调度训练。
