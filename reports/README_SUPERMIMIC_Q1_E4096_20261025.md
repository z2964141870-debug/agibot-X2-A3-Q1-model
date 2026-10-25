# 4096环境论文规模配置

记录日期：2026-10-25，北京时间。阶段：RF05-E4096-UP01准备。状态：配置及资源实测入口已准备，4096尚未实测或启动训练；当前2048环境RF04继续原两小时预算。

## 目标与决定

用户在RF04训练期间要求后续切换4096环境，直接对标原论文。按“后续”执行为顺序新实验：先完成RF04的至少7200秒真实训练和既有动力学评测/交付，再释放其计算资源，实测4096完整批量，资源允许后直接启动RF05。没有授权同时占用GPU开第二训练，也不修改正在运行的RF04源码、配置或参考。

原论文arXiv2610.28480v1，Table A.5，PDF第16页，2026-10-25已通过官方项目链接重新打开全文并目视核表：每GPU4096环境、WM每组10批×2048个8步窗口、actor4096条H32轨迹、UTD0.4、每环境1000行replay、动作噪声标准差0.1。沿用Adam/critic及当前Q1方法，不增加数据准入要求。原始来源：`https://arxiv.org/pdf/2610.28480`；项目：`https://supermimic.github.io/`。

| 配置 | 当前RF04 | 后续RF05 | 原论文Table A.5 |
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

已扩展独立资源实测入口`script/project_tools/profile_q1_2048.py`，增加`--num-envs`和可选`--checkpoint`，按环境数自动配置WM半数batch、actor同数batch及1000行replay。默认2048行为保持。4096会采16轮/65536真实转移，再完成一组WM2048×10及actor4096/H32/critic更新，保存真实上下文覆盖、吞吐、CUDA峰值和共享内存。旧RF04冻结来源不包含此独立profile入口；当前训练与评测代码保持未改。

用户估算约18GB显存仅登记为待验证估计，不作已测容量。profile会加载已经完成RF04的权重，只为代表性时序/资源测量；新正式训练仍从零，profile时间不计入新两小时。现有100GiB余盘、30GiB共享内存、OOM基线、86摄氏度、12GiB单实验及无进展保护继续；容量失败则保存实际错误与终态，不静默缩小批量冒充4096对标。

RF04终态必须先为COMPLETED，原始`run/receipt.json`必须满足`training_seconds>=7200`。如它保护停止、FAILED或STOPPED，先报告实际问题，不自动以4096覆盖失败。等待RF04 worker退出及GPU资源释放后，按现场OOM基线核对并执行一次profile。恢复时若profile或RF05目录已存在，先读状态及进程，避免重复启动；不恢复旧科学停止/失败实验。

计划目录：

- profile：`/opt/data/private/ZHY/data/supermimic_for_q1/E4096_PROFILE_P01_20261025/`。
- 正式RF05：`/opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF05_UP01_E4096_T02H_20261025/`。
- 日志继续使用统一`logs/`；目录日期为计划登记日，实际跨日启动时间另记，不能把计划日当实际启动日。

以下命令只能在上述顺序条件满足后运行，当前不执行；`CODE_COMMIT`取权威Q1 checkout已推送且独立核远端的40位HEAD。Python为既有环境。

```sh
PY=data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
$PY script/project_tools/profile_q1_2048.py --root /opt/data/private/ZHY --output data/supermimic_for_q1/E4096_PROFILE_P01_20261025 --num-envs 4096 --checkpoint /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF04_UP01_E2048_T02H_20261025/run/final.pt --threads 8 --model-slots 8 --expected-oom-kill 14
$PY script/project_tools/run_q1_single_motion.py --root /opt/data/private/ZHY --output data/supermimic_for_q1/BONES_RF05_UP01_E4096_T02H_20261025 --num-envs 4096 --threads 8 --model-slots 8 --wm-batch 2048 --batch 4096 --steps 12000 --training-seconds 7200 --dashboard-label RF05-E4096 --code-commit "$CODE_COMMIT" --expected-oom-kill 14 --detach
```

实际训练前仍需修正`train_reproduction.py`中固定“smaller environment and batch”的元数据文案，使4096的manifest不再错误声称缩小采样；只改声明逻辑、不改训练公式。该文件在RF04冻结来源内，因此当前只登记待办，必须在RF04及评测全部退出后编辑、检查、提交推送、部署并重新核SHA，再启动RF05。此项不是数据门槛。

## 验证、工件与监督

本地CLI帮助及Python语法检查通过；未执行4096仿真或GPU更新，不能宣称实际约18GB或能启动成功。服务器部署与CLI验证、三端小文件SHA、代码审查及payload audit随本阶段同步记录；运行期证据待实测后补充。

继续使用本聊天的`q1-2048`监督任务，更新其流程覆盖“RF04完成与交付→4096实测→RF05训练与交付”，不重复创建自动任务，不新增或激活子agent。普通进度保持安静，只有异常、需要用户处理及重要最终结果才通知。每轮均保存既有六宫格与初末MP4；MP4交付本机对应`data/supermimic_for_q1/`目录，不建网站、不拉权重，核大小/SHA。

RF05完成后分别给出位移/路径/速度、上肢关节误差、抬手幅度和时序、完成比例、真实训练时长与转移数。RF04与RF05是同一动作的有限训练比较，不是全库或真机结果。交付完成或用户停止监督后暂停该heartbeat。当前仍未有RF05模型、图表、MP4或容量结论。

恢复时先读[当前状态](README_CURRENT_STATE.md)、[RF04](README_SUPERMIMIC_Q1_E2048_20261025.md)，重新核实际作业/盘余/OOM与Git状态；旧[位移诊断](README_SUPERMIMIC_Q1_DISPLACEMENT_REVIEW_20261025.md)继续保留为原因假设，不能据扩大环境数提前宣布解决。
