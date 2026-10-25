# YUANQI 当前状态与下一步

更新日期：2026-10-25，北京时间。当前目标：在Q1上复现SuperMimic的可微世界模型、想象BPTT和critic训练方法，优先推进原方法复现。入口：[论文优先主线](README_SUPERMIMIC_Q1_PAPER_FIRST_20261025.md)。

## 用户纠正已落实

代理额外制定的90%帧15mm支撑、30mm最大间隙、3mm自穿透、统一2rad/s、IK残差及30分钟/1小时、独立测试配额等不再阻止数据或训练。旧19/22门与失败收据仅作历史诊断。新入口只保留可读、数值有限、Q1维度/身体顺序及原速时间轴等运行必要条件，原几何flags保持原样。原论文过滤的来源和适用范围须明确，不能冒称已完整复现官方G1过滤代码。

## 最新执行

- [BONES PF02完整合库](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)已完成：复用128并新增896，1024/1024个冻结train来源、326883帧/6517.18秒=108.62分钟，0转换/IO错误；独立来源集合缺项/额外均0、全部120→50Hz原速、1024数组链接。新7批总分配1.031GB；原false几何标记保留，不阻止训练。

- RF02于14:02:53进入独立训练，已保存350轮/22400真实转移快照、WM89组/actor39次，H32想象BPTT及critic真实进入更新，梯度/损失有限；初始模型69122731字节实存。仍按64环境12000采集轮/768000转移及固定前8train初始/最终真实动力学比较、回放；完整目标未完成。控制器861470/start_ticks3619874227，包装862406/3620017857，worker862408/3620018050，boot保持，三实际进程已读；目前无运行子agent，最多一个。

- 发布代码`d66ac1cd21de1682ca6f069227c070ebf98da02f`已独立核远端Q1，12个服务器reproduction源码逐SHA与发布本一致。恢复先读`data/supermimic_for_q1/BONES_BASELINE_G02_20261025/status.json`、当前阶段guard/progress和日志，核实boot/PID/start_ticks，不因连接超时重启既有流程。启动[plan](../data/manifests/supermimic_q1_BONES_G02_plan_20261025.json)/[launch](../data/manifests/supermimic_q1_BONES_G02_launch_20261025.json)/[状态快照](../data/manifests/supermimic_q1_BONES_G02_launch_status_20261025.json)已保存。

- 当前训练根`data/supermimic_for_q1/BONES_RF02_12000_20261025/`，实际worker日志`logs/BONES_RF02_12000_20261025.log`。恢复读其`guard.json`及`run/status.json`；[训练manifest](../data/manifests/supermimic_q1_BONES_RF02_admission_20261025.json)/[合库收据](../data/manifests/supermimic_q1_BONES_PF02_1024_admission_20261025.json)/[早期更新快照](../data/manifests/supermimic_q1_BONES_RF02_early_update_status_20261025.json)已取回。16份静态原样文件与服务器逐SHA一致。早期实测OOM14、GPU40摄氏度、内存27.87GB、盘余2.091TB；资源保护保持，尚无RF02效果结论。

- RF01 E01实际动力学评测及算术复算完成：固定前2片初始/最终整片回报0.27296→0.14400、完成比例62.03%→32.49%、提前终止1/2→2/2，执行段关节RMSE0.44609→0.44551rad。第一片最终视频真实倒地、71/266步提前终止，不能称整体改善或有效步行。五份[原样收据](README_SUPERMIMIC_Q1_BONES_BASELINE_20261025.md)已取回且服务器SHA逐一一致；结果仅覆盖500轮小试两片训练动作。

- PF01已完成：冻结train前128个BONES动作，128/128转换并通过基本IO、0转换/IO错误；39694帧/791.32秒=13.19分钟。120Hz至50Hz保持原速，取消统一速度cap，Q1实际XML关节范围保持。实验分配142741504字节。
- RF01已完成：12:42:02至12:53:46，64环境/4模型槽、500/500采集轮/32000真实转移，训练计时689.58秒。三网络随机初始化，世界模型8步、H32想象BPTT；WM149组/actor99次，guard完成且worker已回收。
- [保存核对](../data/manifests/supermimic_q1_BONES_RF01_training_audit_20261025.json)通过：三网络参数真实改变且有限，三Adam实际1490/99/1980步，输入/源码保持。此核对0动力积分/optimizer更新，不是新增数据质量门。最终模型205621779字节/SHA`6e467822f7e6c800bcb12f00b90e9029998c2d26a629d83df6e44c2325a17bee`，保留服务器。
- RF01目录：`data/supermimic_for_q1/BONES_RF01_500_20261025/`；终态`guard.json`为`WORKER_COMPLETED`、`run/status.json`和`run/receipt.json`为`TRAINING_FINISHED_PENDING_EVALUATION`，日志`logs/BONES_RF01_500_20261025.log`。`run/admission.json`是启动快照，不是当前运行状态。
- 训练源码发布：`22962d3f09fa8c29345761e54f93888874358b11`；主入口`script/supermimic_q1_reproduction/run_reproduction.py`。PF01报告发布`e7ee33791d4b6f63cd3bcbfe99b5834cf4f7a0b0`已独立核Q1远端。

本轮证明取消额外数据门后的有限训练链路已完成，但RF01真实回放仍失败，尚无有效跟随证据，也不是原论文完整数据/算力配置复现。下一沿原方法扩库、增加交互训练，再做同参考初始/最终比较；不再为自定几何门或配额等待。RF01原训练五份收据与E01另五份收据分别保存，完整命令和边界见阶段README。

## 平台、资源与恢复

唯一实体根：`huanyu:/opt/data/private/ZHY`，沿用`data / logs / script / reports`。Python：`data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`。连接可经现有18864 SSH转发；boot基线`d994e71f-bae0-46ba-a5f7-6c2163771c6c`、OOM基线14，恢复时重新读取现场，凭据不落盘。

RF01结束盘余2094026186752字节，约2.094TB；本轮唯一分配274984960字节，采样最高GPU39摄氏度、OOM14保持。100GiB余盘、单实验12GiB及温度/内存/OOM保护继续作为资源管理。Mac只放代码和小收据，不拉模型。原包及大型产物`LOCAL_ONLY`，不声称网盘已备份。

BONES原包位于`data/references/bones_seed/archives/g1.tar.gz`，23499973647字节，SHA`52580ea8bced72ea9e2ff1e7b68f01c51c7f1099581e9a46b7c87e1dec106d8a`；完整SHA、gzip EOF/CRC及142220 CSV目录覆盖通过，仅提取冻结1280候选，没有全量展开或原包副本。见[接收](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)。AMASS/SMPL-X、PHUMA及必要依赖仍保留于统一根，见[存储归拢](README_Q1_STORAGE_CONSOLIDATION_20261025.md)。

旧PHUMA R02D已因磁盘保护停止，不重启或修改终态；保存的累计10282策略与父5000开发配对均0/6严格跟随，不能称已有移动改善。见[停止记录](README_SUPERMIMIC_Q1_R02D_BOUNDED_20261025.md)及[DEV-B历史结果](README_SUPERMIMIC_Q1_RELOCATED_DIAGNOSTICS_20261025.md)。旧A3停止实验不恢复。

## Git与历史

权威副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。阶段结束更新README/索引，核对diff和暂存区、运行payload audit、显式push Q1并独立核远端，再同步用户镜像及服务器。唯一既有未跟踪`script/supermimic_q1_sources/download_bones.py`不执行、提交或删除。

详细旧记录见[本次完成前快照](README_CURRENT_STATE_BEFORE_PAPER_FIRST_20261025.md)、[更早归档](README_CURRENT_STATE_ARCHIVE_20261025.md)和[报告索引](README.md)。每日研究最近10/25检查无实质更新，静默；自动研究不调度训练。
