# 4096环境论文规模配置

2026-10-26最新安排：用户自己的25252容器正接收[CM01全量迁移](README_Q1_CONTAINER_MIGRATION_20261026.md)，已明确允许同期腿部分析及运行就绪后直接训练。RF05配置和原参考/控制保持；新容器OOM基线为0，启动命令改为`--expected-oom-kill 0`，不沿用下方旧容器14。新的真实启动时间、代码部署和进程身份须重新登记。[D05限矩/足高短时对照](README_SUPERMIMIC_Q1_LEG_CONTROL_20261026.md)不改原RF04终态，也不改本轮规模对照变量。原4096完整批量实测已通过，不重复整套profile。

记录日期：2026-10-25，更新至2026-10-26，北京时间。阶段：E4096-P01完成，RF05-E4096-UP01待资源。状态：RF04训练和终后交付完成，4096完整批量实测通过且worker回收；02:02共享盘可用空间为0，低于100GiB保护，RF05目录尚未创建或启动，服务器部署暂缓。

## 目标与决定

用户在RF04训练期间要求后续切换4096环境，直接对标原论文。按“后续”执行为顺序新实验：先完成RF04的至少7200秒真实训练和既有动力学评测/交付，再释放其计算资源，实测4096完整批量，资源允许后直接启动RF05。没有授权同时占用GPU开第二训练，也不修改正在运行的RF04源码、配置或参考。

原论文arXiv2610.28480v1，Table A.5，PDF第16页，2026-10-25已通过官方项目链接重新打开全文并目视核表：每GPU4096环境、WM每组10批×2048个8步窗口、actor4096条H32轨迹、UTD0.4、每环境1000行replay、动作噪声标准差0.1。沿用Adam/critic及当前Q1方法，不增加数据准入要求。原始来源：`https://arxiv.org/pdf/2610.28480`；项目：`https://supermimic.github.io/`。

| 配置 | 已完成RF04 | 待启动RF05 | 原论文Table A.5 |
| --- | ---: | ---: | ---: |
| 环境数 | 2048 | 4096 | 每GPU4096 |
| WM窗口/组 | 1024×10 | 2048×10 | 2048×10 |
| WM窗口长度 | 8 | 8 | 8 |
| actor batch/H | 2048/32 | 4096/32 | 4096/32 |
| UTD | 0.4 | 0.4 | 0.4 |
| 每环境replay | 1000 | 1000 | 1000 |
| 实际训练预算 | 至少7200秒 | 暂沿用至少7200秒 | 320000 iterations |

用户最新指示明确的是环境与论文规模；下一轮暂沿用同一两小时预算作为有限对照，不把它记成320000轮全量复现。保留同一365帧/50Hz双臂抬起与移动参考、seed2026102513、从零初始化，便于先判断更大采样规模是否有效；尚未扩大到全数据集。Q1与原G1、CPU MuJoCo/模型槽与原MuJoCo Warp/H100、Q1控制参数及reset噪声等差异继续列明。环境数对齐不等于全部实验条件对齐。

## 资源实测与执行

独立资源实测入口`script/project_tools/profile_q1_2048.py`支持`--num-envs`和可选`--checkpoint`，按环境数配置WM半数batch、actor同数batch及1000行replay。默认2048行为保持。4096实际采16轮/65536真实转移，再完成一组WM2048×10及actor4096/H32/critic更新。RF04冻结来源不包含此独立profile入口；实测未改变RF04源码或工件。

profile加载已完成RF04的权重，只为代表性时序/资源测量；新正式训练仍从零，profile时间不计入新两小时。用户估算约18GB显存现已有下方实测，不能把分配峰值、保留峰值或单次采样混用。100GiB余盘、30GiB共享内存、OOM基线14、86摄氏度、12GiB单实验及无进展保护继续；容量失败不静默缩小批量冒充4096对标。

RF04终态必须先为COMPLETED，原始`run/receipt.json`必须满足`training_seconds>=7200`。如它保护停止、FAILED或STOPPED，先报告实际问题，不自动以4096覆盖失败。等待RF04 worker退出及GPU资源释放后，按现场OOM基线核对并执行一次profile。恢复时若profile或RF05目录已存在，先读状态及进程，避免重复启动；不恢复旧科学停止/失败实验。

计划目录：

- profile：`/opt/data/private/ZHY/data/supermimic_for_q1/E4096_PROFILE_P01_20261025/`。
- 正式RF05：`/opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF05_UP01_E4096_T02H_20261025/`。
- 日志继续使用统一`logs/`；目录日期为计划登记日，实际跨日启动时间另记，不能把计划日当实际启动日。

profile已按以下绝对路径命令执行一次，恢复时不重复运行。RF05命令仍待盘余恢复及来源发布核验，`CODE_COMMIT`取权威Q1 checkout已推送且独立核远端的40位HEAD。首次相对输出路径调用在参数解析阶段被拒绝，未创建实验；随后绝对路径调用为唯一实际profile，不是失败实验重试。Python为既有环境。

```sh
PY=/opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
$PY /opt/data/private/ZHY/script/project_tools/profile_q1_2048.py --root /opt/data/private/ZHY --output /opt/data/private/ZHY/data/supermimic_for_q1/E4096_PROFILE_P01_20261025 --num-envs 4096 --checkpoint /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025/run/final.pt --threads 8 --model-slots 8 --expected-oom-kill 14
$PY /opt/data/private/ZHY/script/project_tools/run_q1_single_motion.py --root /opt/data/private/ZHY --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF05_UP01_E4096_T02H_20261025 --num-envs 4096 --threads 8 --model-slots 8 --wm-batch 2048 --batch 4096 --steps 12000 --training-seconds 7200 --dashboard-label RF05-E4096 --code-commit "$CODE_COMMIT" --expected-oom-kill 14 --detach
```

RF04全部冻结阶段结束后，已修正`train_reproduction.py`中固定“smaller environment and batch”的元数据文案，按实际环境数、WM与actor批量声明，避免4096的manifest仍声称缩小采样。语法、64/2048/4096文案取值和单项AST对比通过，除该声明外AST相同；未改训练、奖励或观测公式。修正随本阶段显式暂存、payload审计和Q1发布，部署后核源码SHA，RF05启动必须使用已发布身份。此项不是数据门槛。

元数据修正已随`81db2e1fee2828e9ce59d0271c7d2f2cde56cee1`推送Q1并独立核远端，9项133708字节payload及diff检查通过。新训练源码SHA为`0a8128f6d4bfb5908436373d850c5edd7e51b808a0a731bbb5d94150ef92767f`。02:02:16盘余为0，因此未尝试服务器部署；服务器仍为原SHA`4a1431570c80cae8e3377c80bc25a14992f66e1a1af58aaffddd63c6643c519b`。恢复空间后先同步最新报告/小收据和新代码并逐SHA核验，再以已发布40位HEAD启动RF05。当前不能称三端已经同步新修正。

## 验证、工件与监督

配置/独立profile发布提交`9f7debdb1d095820bd71deff8432c2f23aee0642`已推送Q1并独立核远端；profile源码SHA为`5c55e9b7bc2471aad50810d5c44245622705a73c8f484aac395be5b38cc31674`，与真实收据一致。RF04于01:23:37全流程完成并释放worker/GPU后，profile于01:33:33至01:35:18执行，`receipt.json=PROFILE_COMPLETED`、`guard.json=WORKER_COMPLETED`、worker return code 0并正常reap。controller891502和worker891503在终后及01:52复查均已不存在，GPU回落14MiB，OOM仍14。没有第二正式训练或新增子agent。

| 实际测量 | 数值 |
| --- | ---: |
| 环境/线程/模型槽 | 4096/8/8 |
| 初始参考帧/随机化上下文 | 365/4096种 |
| 采集轮/真实转移 | 16/65536 |
| 采集耗时/吞吐 | 79.0634秒/828.904转移每秒 |
| WM2048×10更新耗时 | 3.7756秒 |
| actor4096/H32及critic更新耗时 | 9.6714秒 |
| replay张量 | 2224226304字节 |
| CUDA已分配峰值 | 16746968576字节，15.60GiB |
| CUDA保留峰值 | 17760780288字节，16.54GiB |
| 结束时共享内存采样 | 17232601088字节，非峰值 |
| OOM计数 | 14，未增加 |

完整WM、actor及critic指标均为有限数，表中为短时容量测量，不证明新策略性能或两小时训练已完成。[profile原始收据](../data/manifests/supermimic_q1_E4096_profile_20261026.json)、[guard原始收据](../data/manifests/supermimic_q1_E4096_profile_guard_20261026.json)已保存；服务器对应实验根保留`receipt.json`和`guard.json`，小型工件约16KiB。使用的RF04权重SHA为`9988bbfaaac922bf27e0807992a7eaf197c8bce07b0eb232b71309415dcf3a7e`，仅加载测量，不用于RF05初始化。

当前阻碍为共享磁盘：01:52:03可用约30GiB，02:02:16已为0，低于100GiB保护。[两次目录盘点](README_Q1_STORAGE_GROWTH_20261026.md)显示可见private总量703216345088字节完全相同，而底层整盘余量持续下降；增长者超出当前可见范围，需平台侧定位。RF05目录不存在，不把它记为训练失败，也不降低保护或另选平级目录。

继续使用本聊天的`q1-2048`监督任务，更新其流程覆盖“RF04完成与交付→4096实测→RF05训练与交付”，不重复创建自动任务，不新增或激活子agent。普通进度保持安静，只有异常、需要用户处理及重要最终结果才通知。每轮均保存既有六宫格与初末MP4；MP4交付本机对应`data/supermimic_for_q1/`目录，不建网站、不拉权重，核大小/SHA。

RF05完成后分别给出位移/路径/速度、上肢关节误差、抬手幅度和时序、完成比例、真实训练时长与转移数。RF04与RF05是同一动作的有限训练比较，不是全库或真机结果。交付完成或用户停止监督后暂停该heartbeat。当前容量实测已通过，尚无RF05正式模型、图表、MP4或效果结论。

恢复时先读[当前状态](README_CURRENT_STATE.md)、[RF04](README_SUPERMIMIC_Q1_E2048_20261025.md)，重新核实际作业/盘余/OOM与Git状态；旧[位移诊断](README_SUPERMIMIC_Q1_DISPLACEMENT_REVIEW_20261025.md)继续保留为原因假设，不能据扩大环境数提前宣布解决。
