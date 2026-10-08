# Q1 衣服原始 IMU 经 FGP 转换测试

记录日期：2026-10-08。阶段：Q1-P4。状态：两段 FGP 转换、Q1 数值重定向/支撑回放与抽帧生成完成；姿态精度、无支撑平衡及主机持续稳定性未通过验收。

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

输入身份、走路 E003 实际 FGP 转换和 Q1 E005 运动学/支撑回放通过，尚未证明姿态保真或无支撑平衡。输入和全部衍生测试标记 `TEST_ONLY/training_allowed=false`。网盘未新上传，状态 `LOCAL_ONLY`，原件保留。用户最新要求：大文件传输较慢时停止自动重试，提供具体文件、大小和目标路径，由用户用飞书等第三方软件传输；本阶段所需大包和模型已在服务器，无需用户新传文件。

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

### 走路 Q1 E005 已完成

- 复用 E003 身体包，30→50 Hz 无外推重采样为 2942 帧，完整运动学求解耗时 3.09 秒；人工支撑 PD 回放 58.82 秒，有限值通过、限位违例 0，最大关节速度 1.999998 rad/s（到配置上限）、3.01% 关节样本接近限位。
- 对诊断校准后 IK 目标的位置误差 P95 `0.014338 m`；最大关节回放 RMSE `0.012856 rad`，最大绝对误差 `0.031155 rad`，力矩饱和比例 0。这些是对变换后目标/关节参考的结果，不是对真实人体标签的误差，也不等于同姿跟随验收。
- 输出在 `data/experiments/q1_cloth_q1_20261008_E005/`；小型摘要副本为 `data/manifests/q1_cloth_q1_20261008_E005_summary.json`。CPU 抽帧使用编译模型的真实 mesh 顶点/三角面及世界变换，未重新绘制机器人或安装环境；生成 `body_q1_overview.png` 340,985 字节，图像视觉检查待追加。模型共 24 meshes、509,188 faces，CPU 渲染较慢但已完成；没有以图形接口替换运动学/动力学算法。
- 成功步骤日志为 `logs/q1_cloth_q1_20261008_E005.log`；此前 E003 0 字节中断文件保留。
- 转换/中断检查点已提交并推送：`3bc7815f1f1e10cdeec2e0fefeb749a6138da016`，本地 HEAD 与远端 `refs/heads/Q1` 完全一致。既有两份 PredActor 暂存内容保留，未混入本阶段提交。新增渲染代码/当前结果待后续阶段提交。
- 单腿 E004 CPU 推理已发出独立会话启动，日志 `logs/q1_cloth_fgp_20261008_E004.log`，实际进程/输出还需下一次检查；未称作已完成。

走路 Q1 复现（已有 E005 目录不可覆盖，复现时使用新目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python -u script/q1/validate_egolocate.py \
  --input data/experiments/q1_cloth_fgp_20261008_E003/body_only.npz \
  --output-dir data/experiments/q1_cloth_q1_20261008_E005 --render --render-backend cpu
```

### 单腿 E004 推理已完成与误差初查

单腿 `G_0109_yy_train/193310_yy_oneLegStand` 实际 1826→1767 帧，CPU 83.17 秒，暖机/有限值/速度有效性检查通过，batch/individual 最大差 `2.3842e-7`。包内标签比较：旋转 P95 `0.646459 rad`，根相对关节 P95 `0.393520 m`，根轨迹 RMSE `0.536074 m`。估计根范围 `[0.63139,0.36240,1.13386] m`，标签 `[0.80600,0.33700,0.28000] m`；“单腿站立”样本标签自身也有移动，不能把整个录制当零位移静态真值。摘要已保存 `data/manifests/q1_cloth_fgp_20261008_E004_summary.json`，产物服务器 E004 目录，准备交给 Q1 E006。

步行标签误差初查（只读取产物与标签）：同一 SMPLight 骨架用标签 pose FK 与包内根相对 joint 的 P95 差仅 `2.74e-7 m`；用预测 pose 对同一模板标签 FK 的 P95 仍为 `0.377489 m`，因此本次误差不是两套骨长导致。去掉两者根朝向后姿态关节 P95 仍有 `0.245619 m`，根朝向 P95 为 `0.932490 rad`。腕关节位置 P95 左/右 `0.462931/0.450856 m`，踝 `0.346368/0.325341 m`。这个分解支持优先检查 FGP 重建/根朝向与校准，不能因 Q1 IK 残差小就认为输入姿态已准确。只读计算尚需落为可复现诊断脚本/JSON 后同步。

已视觉检查走路 CPU 抽帧：真实 Q1 mesh 非空、肢体与头脚在视野内，抽帧存在姿态变化；这是运动学姿态图，未显示无支撑动力学。颜色较浅，细小表面细节不足，不据其验证完整高保真。

### 20:50 单腿回放后的再次重启

E006 实读日志显示 2944 帧、IK 3.43 秒、kinematic trace 已保存、58.86 秒 supported replay 已保存，随后开始 CPU 抽帧渲染。20:50:38 重新 SSH 的 uptime 为 0 分钟，已有四份 NPZ 文件保留，无抽帧图/summary，进程已消失；不能宣称整项验证完成。此次也没有 GPU 神经推理；重启原因未确认，不能从时序推断渲染、CPU 或硬件故障。

诊断脚本第一次 SCP 因连接 reset 未传入服务器，随后的诊断命令退出码 2（脚本不存在），没有诊断 D001 产物。准备重传小型代码，并从 E006 的已保存文件独立核验/汇总数值，保留中断记录；不反复重跑模型与同一回放。若没有完整数值，则明确记录缺失。暂不重试 CPU mesh 渲染。

### 更换供电后的恢复核验

用户说明已把供电从排插移至墙壁插座，要求继续尝试。20:58/21:01 实读 uptime 为 1/3 分钟，无遗留推理或回放进程；这只是一次恢复状态，不能证明此前原因或长期稳定性。未改动系统电源设置。

新增 `script/q1/recover_q1_validation.py`，只加载已保存数值文件，检查 TEST_ONLY、有限值、完整帧数/时序、身体输入一致性、关节参考与 qpos 一致性、限位及回放 target 对齐；不重跑神经网络/IK/动力学。先在完整走路 E005 对照原 summary，9 项数值完全相同（最大绝对差 0），再恢复单腿 E006。

E006 四个 NPZ 的完整解码和上述检查均通过：2944 参考帧、58.86 秒支撑回放、关节限位违例 0，IK 位置 P95 `0.095351 m`，最大关节回放 RMSE `0.030482 rad`、最大绝对误差 `0.227515 rad`。恢复摘要为 `data/experiments/q1_cloth_q1_20261008_E006/recovered_summary.json`。原始全物理子步力矩饱和计数没有持久化，恢复报告对此记为 null；已保存的末子步力矩仅可核验在限幅内，不能据此声称整个回放饱和比例为零。

误差诊断 D001 已实际完成（成功日志 D002，第一次 D001 日志保留脚本未传入的失败），脚本 `script/q1/diagnose_clotho_output.py`，摘要清单 `data/manifests/q1_cloth_diagnostics_20261008_D001_summary.json`。单腿同模板 pose-FK/joint 标签差 `2.95e-7 m`，预测误差 P95 `0.393520 m`，去根朝向后 `0.254600 m`，根朝向 P95 `0.987265 rad`。两段仍存在显著人体重建误差，模型/规范化/校准各自贡献尚未确定；不擅自用标签修正推理。

新增独立 `render_q1_recording.py`，在恢复摘要及产物哈希核验后，只补抽帧图和渲染回执。这样渲染再中断也不会丢失数值摘要；准备以 CPU 后端补单腿图，不重复完整推理/回放。

### 本轮最终数值结果与补图回执

| 指标 | 步行 FGP E003 / Q1 E005 | 单腿 FGP E004 / Q1 E006 |
| --- | --- | --- |
| 导出 SMPL 帧数 | 1766 | 1767 |
| 神经推理 CPU 耗时 | 78.72 秒 | 83.17 秒 |
| 对包内人体标签：根相对关节位置 P95 | 37.75 cm | 39.35 cm |
| 对包内人体标签：相对根轨迹 RMSE | 65.79 cm | 53.61 cm |
| 对诊断校准后 IK 目标：位置 P95 | 1.43 cm | 9.54 cm |
| Q1 参考限位违例 | 0 | 0 |
| Q1 支撑回放时长 | 58.82 秒 | 58.86 秒 |
| Q1 支撑回放最大关节 RMSE | 0.01286 rad | 0.03048 rad |
| Q1 支撑回放最大绝对关节误差 | 0.03115 rad | 0.22752 rad |

两类位置误差的比较对象不同：Q1 数值对诊断校准/缩放后的目标，人体数值对包内标签；不能用前者小来证明人体重建准确。标签 pose/joint 不进入推理，既有模型可能见过这些样本，因此这不是独立测试集成绩。此包的对齐 IMU 张量可用于离线通路测试；`sensor_data.txt` 到同一标定/对齐张量的转换、现场服装校准和实时接口尚未核验。

单腿 R002 在 21:05 完成独立 CPU 抽帧（未重新运行推理/IK/PD）：`body_q1_overview.png` 343,853 字节，SHA-256 `1d2bddc32c0cc0bce40305174331f60e800a26952ceb664239d2980c35e673ca`；`render_summary.json` 保存后端、源码/数值产物及图像身份，清单副本 `data/manifests/q1_cloth_q1_20261008_E006_render_summary.json`。恢复数值摘要中的 `render_completed=false` 是 21:01 的历史状态，最新补图结果以独立回执为准，不回写历史证据。小图下载后哈希与回执一致，已实际视觉检查：mesh 非空、视野完整，人体抬腿与机器人对应姿态可见，但机器人抬腿幅度不足，不能认为已经复现单腿站立；与数值残差偏大的结论一致。两张小图在 Mac `data/sync/q1_cloth_fgp_resume_20261008/`，未下载 NPZ/模型/大包。

换插座后的本次渲染完成，但 journal 启动记录仍有 `20:57:59→21:05:47`、随后 `21:06:17` 新启动；21:10:11 实读 uptime 3 分钟。最后一轮日志/内核筛查未得到可归因的 OOM、NVIDIA Xid、panic 或正常关机证据；传感器枚举时的 thermal 读取失败不能直接证明过热。是否人工重启、供电与其他硬件原因仍未确认，不能宣称换插座解决问题。当前没有本阶段遗留模型/仿真进程。

新增脚本服务器语法检查通过，五份成功摘要 TEST_ONLY 边界核验通过；输入 6/6 检查、恢复对照 9 项完全一致及实际 NPZ 解码结果均已有证据。没有新安装依赖、训练、真机控制、网盘上传或大型数据迁移。

复现入口（输出/日志已存在时拒绝覆盖；复制复现须用新实验目录）：

```bash
cd /media/yu/FAFF-E977/YuanQi_Q1
# 重建推理时沿用上面的 infer_clotho_fgp.py 参数，并指定 --device cpu 与新目录。
# 仅恢复已有数值，不重新执行神经推理或物理回放。
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/recover_q1_validation.py \
  --input data/experiments/q1_cloth_fgp_20261008_E004/body_only.npz \
  --output-dir data/experiments/q1_cloth_q1_20261008_E006 \
  --original-validator-sha256 ad2b79e750e9644e60ed452ba7e7a2f66cb7ee03b909d448fb784c823a186b3c
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/render_q1_recording.py \
  --output-dir data/experiments/q1_cloth_q1_20261008_E006 --backend cpu
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/diagnose_clotho_output.py \
  --archive /media/yu/FAFF-E9771/data/fgp/CLOTHO_train_data/train_released.zip \
  --fgp-root /home/yu/projects/x2-teleop/FGP-main \
  --experiments data/experiments/q1_cloth_fgp_20261008_E003 data/experiments/q1_cloth_fgp_20261008_E004 \
  --output-dir data/experiments/q1_cloth_diagnostics_20261008_D001
```

工件清单集中在 `data/manifests/q1_cloth_*_summary.json` 与 `q1_cloth_input_20261008.json`，源码身份由各摘要登记；原始/衍生 NPZ、模型、大包均留服务器，`LOCAL_ONLY`。尚没有需要用户通过第三方传输的新文件。本轮 15 份代码/报告/清单/小型执行记录已提交并推送 `40c8f95572568c6788c5787eea0ca289cd08262f`；`git rev-parse HEAD` 与 `git ls-remote origin refs/heads/Q1` 实读完全一致。暂存 payload 审核通过，其他两份 PredActor 暂存内容原样保留。发布核验回执见 `logs/session_records/q1_cloth_fgp_publication_20261008.json`；回执自身提交号以 Git 历史为准，不循环写入自己的提交号。

## 下一步

1. 本轮离线测试及结果发布已完成；恢复先核对报告/发布回执与实时主机状态，无需再跑两段转换或 PD 回放。服务器稳定性未确定，不自动启动训练。
2. 优先检查 FGP 输入预处理、根朝向、服装标定和配对站姿校准，降低人体重建误差；单腿重定向残差也需改善。
3. 再建立地面/脚接触、无支撑动力学跟踪与训练任务。当前测试数据保持 TEST_ONLY，不用于训练，不操作真机。
