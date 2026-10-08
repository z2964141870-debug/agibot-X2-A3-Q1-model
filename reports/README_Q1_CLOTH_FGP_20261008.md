# Q1 衣服原始 IMU 经 FGP 转换测试

记录日期：2026-10-08。阶段：Q1-P4。状态：走路 E003 转换完成，输入 6 项检查通过；Q1 E003 被 20:26 重启中断，单腿站立尚未开始。

## 目标与范围

用户指定 Mac `/Users/yu/Human+/data` 的衣服输出数据，要求核对服务器 FGP、转换并用于 Q1。沿用测试范围，不启动训练或机器人控制。用户确认上一阶段两段录制本来就没有移动，`trans=0` 不当作该录制的故障；本次另查包含腿部/移动动作的 IMU 样本。

开发在 hp3090 `/media/yu/FAFF-E977/YuanQi_Q1`，分支 `Q1`，开始 HEAD 为 `b0c1fabe5cc915531bb6862dd92ec62d12e1cc49`。服务器既有 PredActor 两份暂存记录独立保留。

## 已完成与关键决定

- Mac 数据只有 `train_released.zip`，1,896,104,195 字节；服务器已有 `/media/yu/FAFF-E9771/data/fgp/CLOTHO_train_data/train_released.zip`，SHA-256 与 Mac 相同：`8015e969dc2ef9629e3a4e02d16d2846449d7db2ebd2eb94ca063c814358f952`。不重复迁移或在 Mac 解压 3.15 GB 数据。
- 包内 206 段，含 `acc.pt/rot.pt`、规范化后的 IMU、`device2bone.pt`、`pose.pt/joint.pt` 和原始 `sensor_data.txt`，不是只有衣服字节流。选择走路 `G_0113_zj_train/183926_zj_walk` 与单腿站立 `G_0109_yy_train/193310_yy_oneLegStand` 进行代表性测试；不宣称已验证全部 206 段。
- `acc.pt [T,11,3,1]`、`rot.pt [T,11,3,3]`，按既有 `process_data.py/cano_eval.py` 去掉槽 5，保留 `[0,1,2,3,4,6,7,8,9,10]`；采用发布包预处理张量合同。原始文本与 PT 常不同帧数，不静默截断或把文本时间戳套给 PT。30 Hz 来自已有数据代码合同，不称作设备实测时序。
- 现有 FGP 在 `/home/yu/projects/x2-teleop/FGP-main`；速度扩展在 `/home/yu/projects/fgp/scripts/clotho_velocity`，其数据在 `/media/yu/FAFF-E9771/data/fgp`。模型与代码仅就地调用，不复制第三方权重/SMPL 资产到 Git，不执行历史 README 中的训练/硬件命令。
- 新增 `script/q1/infer_clotho_fgp.py`：实际重跑 FGP canonicalizer → LIP → SMPLight FK → 已有速度头；包内 `pose/joint` 只用于另外的标签对比，不进入推理。规范化采用 30 帧因果窗口最后一帧、单步 residual 和 0.7 EMA，首 29 帧及速度暖机 30 帧不进入 Q1 输出。
- LIP 仅从第一个完整规范化窗口开始，避免未规范化暖机帧污染其递归状态；每段重新初始化。记录原始 IMU 旋转误差及投影修正幅度，超过 0.05 Frobenius/行列式偏差或反射则拒绝；首/中/末三个窗口检查批量与单独推理差异。
- 根位移是已有学习式速度头及脚接触修正的估计，非直接测得的世界轨迹；既有模型可能见过该数据，对包内标签的对比不是独立泛化评测。脚接触概率不是力板真值。
- 仍沿用上一阶段首帧相对诊断校准和人工支撑 PD，只测试数据/关节接口，不把根部轨迹直接强加给动力学机器人。

## 版本与复现入口

Python `/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python`，当前已确认 PyTorch 2.8.0+cu128 与 CUDA 可用；GPU RTX 3090 24 GB，开始显存 38 MiB。规范化、LIP 与速度 checkpoint 的身份在实际实验摘要登记。本轮不修改公共环境。

```bash
cd /media/yu/FAFF-E977/YuanQi_Q1
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/infer_clotho_fgp.py \
  --archive /media/yu/FAFF-E9771/data/fgp/CLOTHO_train_data/train_released.zip \
  --sequence G_0113_zj_train/183926_zj_walk \
  --fgp-root /home/yu/projects/x2-teleop/FGP-main \
  --velocity-root /home/yu/projects/fgp/scripts/clotho_velocity \
  --velocity-checkpoint /media/yu/FAFF-E9771/data/fgp/velocity_checkpoints/clotho_v2_displacement/last.pt \
  --output-dir data/experiments/q1_cloth_fgp_20261008_E001
```

已有输出目录拒绝复用。失败目录与 stdout/stderr 保存在 `data/experiments/`、`logs/`；成功后 `body_only.npz` 可交给 `validate_egolocate.py`。PT 使用 `weights_only=True`，不执行数据包附带指令。

## 结果与保存状态

输入身份及走路 E003 实际 FGP 转换通过，Q1 回放未完成。输入和全部衍生测试标记 `TEST_ONLY/training_allowed=false`。网盘未新上传，状态 `LOCAL_ONLY`，原件保留。用户最新要求：大文件传输较慢时停止自动重试，提供具体文件、大小和目标路径，由用户用飞书等第三方软件传输；本阶段所需大包和模型已在服务器，无需用户新传文件。

19:37 恢复核验：原 SSH 转换命令因连接中断未启动，E001 日志/输出目录和推理进程均不存在；现已重新连接，保留资源核验结果后启动转换。不把前一次命令发出当成运行成功。

### 离线恢复检查点

- 已实际确认 `FGP_IMPORTS_OK`：当前环境可导入 canonicalizer、SMPLight 与既有 LIP 推理入口。没有新增依赖安装。
- 最新推理脚本与输入适配器 SCP 已返回退出码 0，报告及共享索引/状态检查点也已传入服务器。上传源码 SHA-256：`infer_clotho_fgp.py` 为 `e1d6cee65aa9165cd286c5eb54a1d9c57aff5c91c2b86960af28be5b6c28ae2b`，`egolocate_input.py` 为 `233dd400415231a87a921470f81c97c562bb41988ad7bb40a52f27a40fc548da`；恢复后仍要重核服务器文件哈希。
- 走路 E001 通过 `nohup` 在服务器启动，SSH 返回 `walk_pid=2944`；日志为 `logs/q1_cloth_fgp_20261008_E001.log`，目标目录为 `data/experiments/q1_cloth_fgp_20261008_E001`。PID 是启动记录，不证明当前仍在运行。独立启动使 SSH 中断不必然终止推理；主机重启仍会终止。
- 已发出服务器 `test_egolocate_input.py -v` 命令，目标日志 `logs/q1_cloth_input_tests_20261008.log`，但读取结果的连接中断；本轮不能记为 6/6 通过。Mac `py_compile` 两份代码通过，Mac 缺少本测试的 SciPy/PyTorch/MuJoCo/Mink，没有安装它们或改在 Mac 运行大模型。
- Tailscale 由在线转为离线，`LastSeen=2026-10-08T11:40:00.1Z`（北京时间 19:40）；随后 SSH/Tailscale ping 连续超时。走路成功/失败、输出数值与批量一致性结果均未读取；单腿站立及两段 Q1 回放尚未启动。不能从命令已启动推断转换可用。
- HTTPS 已核验远端 Q1 仍为开始提交 `b0c1fabe5cc915531bb6862dd92ec62d12e1cc49`；Mac GitHub SSH 返回 `Permission denied (publickey)`。本阶段尚未推送，服务器两份 PredActor 暂存修改原样保留。Mac 只准备小型 Git 检查点，不复制大数据、模型或第三方项目。

19:56 用户通知恢复后重新连接：主机 uptime 约 3 分钟，显存 38 MiB，无本阶段推理进程；E001 日志/目录及输入测试日志均不存在。两份服务器源码哈希与上述上传版一致，分支/HEAD 仍为 `Q1/b0c1fab`，已有 PredActor 暂存内容保留。前次 PID 返回只能证明启动请求得到回复，不能证明推理实际执行；本轮改为 Python `Popen(start_new_session=True)`、由启动进程先创建日志，并立即核对日志/进程。现无输出可复用，因此使用尚不存在的 E001/E002。

### 20:00 再次断联检查点

- 最新阶段 README 已上传成功。采用 Python `Popen` 启动 E001，由启动进程先以 `open("x")` 创建日志，再创建独立会话的推理子进程；SSH 返回退出码 0、`pid=2838` 和 `logs/q1_cloth_fgp_20261008_E001.log`。这次确认日志创建和进程创建成功；随后检查实际进度的 SSH 中断，不能据此确认推理完成。
- 再次发出输入检查命令，但返回结果仍未读到。读取共享报告快照的 SCP 失败，没有将部分下载当作服务器最新基线。
- Tailscale 再次显示 `Online=false`、`LastSeen=2026-10-08T12:00:00.1Z`（北京时间 20:00），IPv4/已核验同机 IPv6 SSH 均超时。尝试读取休眠/前次启动内核记录也未连上，故断联原因未确认；不能直接归因于 GPU、休眠或网络。
- 新增本地 `script/q1/run_clotho_tests.py` 批处理入口，串行执行输入检查、两段 FGP 推理和两段 Q1 重定向/支撑回放，每步单独记录命令、退出码和日志，并原子更新 `driver_summary.json`。遇到失败停止，已有目录/日志拒绝覆盖，全部 TEST_ONLY。Mac 语法检查通过，尚未传入服务器或实际执行，不算新增通过项。
- 恢复后先查 PID/日志/摘要。若已有完整结果则继续使用；若失败/部分结果保留，并用批处理新编号 E003/E004（仅在确认本阶段无遗留推理进程后）。CPU 模式可用于后续隔离诊断，不修改公共环境或全局 GPU 设置。

批处理复现入口（尚未执行，恢复后按上述规则决定是否需要）：

```bash
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/run_clotho_tests.py \
  --run-id q1_cloth_pipeline_20261008_R001 --first-experiment 3
```

将此入口以独立会话启动，stdout 为 `logs/q1_cloth_pipeline_20261008_R001.log`；内部各步日志为 `logs/q1_cloth_pipeline_20261008_R001_<step>.log`，状态为 `data/experiments/q1_cloth_pipeline_20261008_R001/driver_summary.json`。不依赖持续 SSH 会话，但主机断电/重启仍会终止。该批处理没有训练或真机控制入口。

### 20:24 CPU 实际执行检查点

20:21 再次恢复：uptime 4 分钟，前次 E001 日志/输出/进程仍不存在。日志目录位于 ext4 NVMe，并非已发现的 tmpfs。`journalctl --list-boots` 记录 19:34、19:53、20:17 的多次启动；前次日志筛查没有得到可归因的 OOM、NVIDIA Xid、休眠或 panic 证据，重启原因仍未确认，不把它等同于网络问题或 GPU 故障。

本轮不再启动 CUDA 大模型，使用相同 checkpoint 的 CPU 推理，4 个 CPU 线程、batch size 8。新增各模型阶段进度日志及执行设备记录，不改变神经网络算法。批处理入口和新版推理脚本已上传成功，实际以 `--device cpu --first-experiment 3` 启动 R001，独立进程 PID 66294（历史启动标识）；启动日志先写入并 fsync，完整状态由 `driver_summary.json` 原子更新。输入检查步骤退出码 0；已实读到走路 E003 加载 1825 帧、checkpoint 载入以及规范化窗口 0/160/320 的进度。单腿站立 E004 和 Q1 E003/E004 由批处理串行执行，结果待追加。

```bash
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/run_clotho_tests.py \
  --run-id q1_cloth_pipeline_20261008_R001 --first-experiment 3 --device cpu
```

上述目录已存在，不能直接重复该命令。恢复先读状态及各步日志；失败/中断保留目录并选择新的 run ID 和实验编号。MuJoCo 运动学/PD 为 CPU 运算，抽帧渲染仍使用已有 EGL，不把 CPU 神经推理说成全部图形渲染也不使用 GPU。

恢复连接后先核验，不重跑已有完整输出：

```bash
cd /media/yu/FAFF-E977/YuanQi_Q1
git status --short --branch
pgrep -af '[i]nfer_clotho_fgp.py'
tail -n 60 logs/q1_cloth_fgp_20261008_E001.log
ls -l data/experiments/q1_cloth_fgp_20261008_E001/summary.json
tail -n 12 logs/q1_cloth_input_tests_20261008.log
```

若 E001 有完整摘要，核对有限值、`velocity_valid_all_exported`、旋转修正和批量差异后交给 `validate_egolocate.py --render`。若有失败/部分输出，保存原目录和日志，使用新实验编号重跑；不要直接重复 E001。单腿站立使用 `G_0109_yy_train/193310_yy_oneLegStand`，独立目录 `q1_cloth_fgp_20261008_E002`。Q1 输出使用 `q1_cloth_q1_20261008_E001/E002` 新目录。

重要代码、报告、审核后的小日志与清单按用户授权提交并推送 `Q1`，以实际远端 SHA 核验为准。共享 README 按服务器最新内容合并 Q1 条目，保留 Mac 未同步研究记录。

服务器离线期间，Mac 小型版本检查点位于 `data/sync/q1_cloth_fgp_20261008/checkpoint_repo`，从已核验 `origin/Q1` 浅克隆，只含代码/报告/清单，约 1.6 MB。只保存本阶段七份文件；共享报告使用服务器基线合并 Q1 条目，Mac 的其他研究记录不纳入。检查点提交号由该副本 Git 历史给出；本阶段仍未推送成功。恢复后先读服务器最新状态，再合并本阶段差异，不用该副本覆盖服务器共享报告或已有暂存内容。

### 20:38 实读完成产物与重启中断

- 恢复连接后确认无 `infer_clotho_fgp.py/run_clotho_tests.py/validate_egolocate.py` 遗留进程。服务器 20:35 uptime 8 分钟，与已核验 20:26 启动相符；重启原因仍未确认。
- R001 的输入检查实读为 6/6 PASS；走路 E003 步骤退出码 0。1825 输入帧去除 59 暖机帧后导出 1766 帧，CPU/4 线程/batch 8，推理耗时 78.7235 秒，输出有限，所有导出速度帧有效。原始旋转正交误差最大 `7.4824e-7`；批量/单窗口推理差异最大 `2.0862e-7`。
- 包内标签比较：局部旋转误差 P95 `0.306584 rad`，根相对关节位置误差 P95 `0.377489 m`，相对根轨迹 RMSE `0.657945 m`。估计 SMPL XYZ 根范围 `[2.68649,0.21216,2.03748] m`，标签 `[2.20800,0.058074,1.35200] m`。这些值不支持高保真跟随结论，也不是独立泛化评测；待进一步分解姿态/骨长/轨迹误差。
- 成功产物：`data/experiments/q1_cloth_fgp_20261008_E003/summary.json`；`body_only.npz` 2,876,717 字节，SHA-256 `8142ea18889cba34cf2d582d88adbbf91d91a0cae79dfb83b786666db90e3d8c`；`fgp_trace.npz` 3,266,448 字节，SHA-256 `ed2d37483bf1f95e9b71d23d895b9191bee34778723979345df1e93429f28d9c`。源码 SHA-256 `ff93a6783f29b2012915c9959e34d5d2e47164c3617d24d1ec235a36bda9e4c7`。
- Q1 E003 只有身体/关节参考，`kinematic_trace.npz` 为 0 字节，无 supported replay、summary 或渲染产物。保留该中断目录；R001 的 `RUNNING` 是重启前残留状态，不是实时进程状态。单腿 E004 未启动。
- 后续复用完整 E003 身体包，Q1 用新目录 E005；单腿 FGP 使用未占用 E004，Q1 使用新目录 E006。准备增加 CPU 真实 mesh 抽帧渲染及进度记录，以减少本次测试的图形环境依赖；该渲染修改尚未实际验证。不从发生重启推断 EGL/GPU 是故障原因。

## 下一步

1. 复用走路 E003，完成新目录 Q1 E005；单腿 E004 再推理，并完成 Q1 E006，不复跑成功步骤或覆盖中断目录。
2. 检查 CPU 抽帧图、关节限位、支撑回放及标签误差来源；测试结果不等于无支撑动态行走，不启动训练/真机。
3. 更新本 README、索引/当前状态，审查提交并核验 `origin/Q1`。
