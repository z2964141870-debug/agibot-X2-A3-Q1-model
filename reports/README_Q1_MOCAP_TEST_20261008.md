# Q1 真实动捕服 FGP→SMPL 录制测试

记录日期：2026-10-08。阶段：Q1-P3 / E001、E002。状态：两段输入审查、服务器核验、完整离线重定向/支撑回放与抽帧渲染复核完成。输入通路可用，准确同姿跟随和动力学移动未通过验收。

## 目标与范围

用户提供 `hand_follow_20260923_191240/egolocate.jsonl` 和 `hand_follow_20260924_150132/egolocate.jsonl`，要求测试动捕服→FGP→SMPL 输出是否可用，明确只用于测试。本轮所有输入与衍生产物标记 `TEST_ONLY` / `training_allowed=false`，不作为训练集、不更新神经网络、不操作机器人。

开发主机与目录为 hp3090 的 `/media/yu/FAFF-E977/YuanQi_Q1`，分支 `Q1`。Mac 原件只读，不新增大型原始副本；全帧身体字段提取为约 1.9 MB/份的小包，相机 IMU/日志及手部网格不在本次 Q1 测试范围内。仅提取测试所需字段，不能称为原 JSONL 已完整迁往服务器。

## 已完成内容与输入证据

- 原始文件共 386,590,910 字节，流式逐行检查，共 3,892 帧，无解析、字段形状或非有限身体数值错误。
- 格式：外层 `host_monotonic_s`、`elapsed_s`，内层 `payload`。身体字段包含 `pose_matrix[24,9]`、`axis_angle[72]`、`parents[24]`、`joints[24,3]` 和 `trans[3]`。两段 SMPL parent tree 一致。
- 已读生产代码：`EgoLocate/egolocate_fgp_web_demo.py` 的 `pose_joints` 调用 SMPL `forward_kinematics(..., calc_joint=True)` 并加 `trans`；`articulate/model.py` 将输入旋转定义为父节点坐标系中的局部旋转。此处作为资料核对，未执行历史硬件/机器人命令。
- `joints` 已有米制全局关节位置，当前输入测试无需下载或复制 SMPL 人体模型。全局身体朝向通过现有局部旋转与 parent tree 组合获得。
- `pose_matrix` 与 `fused_pose_matrix` 在两段所有帧完全相同，Q1 测试使用原身体字段；不用 HaMeR 手指网格替换身体或腕部。
- 原件身份、完整帧数及身体包身份见[输入清单](../data/manifests/q1_mocap_test_inputs_20261008.json)。身体值均可无损表示为原有 float32；时间戳保留 float64，不量化。

| 录制 | 有效帧数 | 源时间跨度 | 中位帧间隔 | 最大间隔 | 缺失源 frame index |
| --- | --- | --- | --- | --- | --- |
| 20260923_191240 | 1969 | 75.0162 s | 0.033814 s | 0.157886 s | 1 |
| 20260924_150132 | 1923 | 75.0806 s | 0.033977 s | 0.177951 s | 1 |

没有重复/倒序时间、重复/倒序 frame index，均无超过 0.2 秒的间隔。中位约 29.5 Hz 不等于平均帧率；不按 JSON 内即时 `fps` 字段强行生成等间隔时间轴。

## 关键决定及风险边界

- **根部位移全为零**：两段每帧 `trans=[0,0,0]`，`joints[0]` 与 `trans` 一致。因此可测试姿态输入，但不能用这两段验证行走位移跟随；不能人为补出位移后称作原始输出。
- 原始局部旋转并非精确 SO(3)：最大 `||RᵀR-I||F` 为 0.001676 / 0.001651，最大 determinant 偏差约 0.00126。骨长变化约 0.86 / 0.94 mm，逆变换恢复的静止骨向量偏差约 0.99 mm。转换时必须记录受限 SO(3) 修正，明显坏矩阵应拒绝，不静默当作完美旋转。
- 首帧不是已确认的人体/机器人配对校准姿势。若采用首帧相对基准，只能当作输入联调诊断；真实人体形体缩放、骨段对应、姿势保真和接触一致性仍须验证。
- 输入基准采用明确 SMPL 轴转换，并核对静止骨向量与可视化；设备的世界重力/朝向没有独立标定证据。转换不能让未知坐标系变成已验证的物理坐标。
- 只在时间戳支持的区间内进行 50 Hz 位置插值与旋转 Slerp，保留间隔统计；不跨越超过 0.2 秒的缺口。原件没有此类长缺口。

## 版本、命令与证据

- 开始时 Q1 HEAD：`4e954bf411e23f6eb596b025ac1ca002ceefd24f`，服务器工作区干净。
- E001/E002 执行时基线为 `38a7805086203f54f7b9a1b12653d3ecc6d89285`；期间存在其他专题记录更新，未覆盖或纳入本阶段。运行源码/配置的精确 SHA-256 保存于实验 `summary.json`，未提交代码不只用基线号代表版本。
- 原创流式审查脚本：`script/q1/inspect_egolocate.py`；本轮提取时源码 SHA-256 为 `c0e6523db4bf47daee5bd3e9a66c50a4738d3a1ebd04d307290c10614ae255cb`。
- Mac 原始执行记录：`logs/q1_mocap_inspect_20260923_20261008.log`、`logs/q1_mocap_inspect_20260924_20261008.log`。
- 本地小包及完整检查摘要：`data/sync/q1_mocap_test_20261008/<recording_id>/`；目标服务器位置：`data/test_inputs/q1_mocap_20261008/<recording_id>/`。

原件流式检查命令（Mac，第二段只替换记录 ID；已有输出目录不可复用）：

```bash
python3 script/q1/inspect_egolocate.py \
  --input /Users/yu/Documents/ChatGPT/YAMAHA/packaging/original-cloth/data/hand_follow_20260923_191240/egolocate.jsonl \
  --output-dir data/sync/q1_mocap_test_20261008/hand_follow_20260923_191240
```

实际迁移使用 `script/q1/transfer_mocap_test.sftp`，OpenSSH SFTP 小窗口 `-R 4 -B 8192`。初轮连接/传输停滞，终止了本轮失效传输；通过 SSH multiplex 恢复，逐份核验部分文件与本地前缀 SHA-256 后才用 `reput` 续传。两段完整大小与 SHA-256 均已一致，输入清单记为 `BODY_BUNDLES_SERVER_SIZE_SHA256_VERIFIED`。原件并未整体迁移。

服务器执行命令（E002 只替换输入记录 ID 与实验编号，必须使用新的输出目录）：

```bash
cd /media/yu/FAFF-E977/YuanQi_Q1
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/test_egolocate_input.py -v
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/validate_egolocate.py \
  --input data/test_inputs/q1_mocap_20261008/hand_follow_20260923_191240/body_only.npz \
  --output-dir data/experiments/q1_mocap_20261008_E001 --render
/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python script/q1/summarize_egolocate_geometry.py \
  data/test_inputs/q1_mocap_20261008/*/body_only.npz
```

环境沿用 MuJoCo 3.3.7、Mink 1.2.0、qpsolvers 4.12.0、quadprog 0.1.13、NumPy 1.26.0、SciPy 1.15.3；本轮没有安装或修改公共依赖。RTX 3090 当前盘点为 24,576 MiB，总显存使用 38 MiB；测试主要为 CPU 数值/仿真与 EGL 渲染，不是 GPU 训练。

## 结果与保存状态

输入检查最终 6/6 通过，覆盖非有限/错误字段拒绝、SO(3) 修正预算、父节点旋转组合、轴变换、非均匀时间插值和 TEST_ONLY 导出。初次两项失败是零值附近未设置绝对浮点容差，差异不超过 `2.22e-16`；只修正测试的 `atol=1e-12`，没有改动转换算法。首次及最终日志分别为 `logs/q1_mocap_input_tests_20261008.log`、`logs/q1_mocap_input_tests_final_20261008.log`。

| 检查 | E001：09/23 | E002：09/24 |
| --- | --- | --- |
| 输入解码/有限输出 | 通过 | 通过 |
| 源平均采样率 | 26.234 Hz | 25.599 Hz |
| 50 Hz 输出 | 3751 帧 / 75.00 s | 3755 帧 / 75.08 s |
| 人体→Q1 腿链比例 | 0.45710034 | 0.45710104 |
| 身体位置残差 P95 / 最大 | 3.719 / 6.895 cm | 6.506 / 11.997 cm |
| 身体朝向残差 P95 / 最大 | 0.366013 / 0.530637 rad | 0.607770 / 0.865132 rad |
| 关节参考越界数 / 最大速度 | 0 / 1.999998 rad/s | 0 / 1.999998 rad/s |
| 关节样本距预留限位 <0.001 rad 比例 | 9.42% | 2.93% |
| 支撑回放时长 | 75.00 s | 75.08 s |
| 最大单关节 RMSE / 最大瞬时误差 | 0.009742 / 0.033719 rad | 0.027567 / 0.083568 rad |
| 力矩饱和比例（每物理步、每关节） | 0 | 3.027e-6（0.000303%） |

残差相对于首帧相对映射构造的机器人目标，**不是人体姿态真实精度**；首帧被映射到机器人中立位，不能据首帧零误差宣称校准正确。姿态 P95 分别约 20.97° / 34.82°，第二段右腕位置 P95 为 11.826 cm，不能称作高保真跟随。参考速度受到现有 2 rad/s 限制；不可达/过快动作产生的误差保留在轨迹中。接近预留限位的样本也可能来自中立位的伸直关节，不能只凭该比例判断碰撞或精度。50 Hz 由插值产生，不增加原始采样的信息，也不是已验证的设备实时频率。

E001/E002 目录 `data/experiments/q1_mocap_20261008_E00{1,2}/` 各保存 `body_reference.npz`、`joint_reference.npz`、`kinematic_trace.npz`、`supported_replay.npz`、`body_q1_overview.png` 和 `summary.json`。终端输出为 `logs/q1_mocap_20261008_E00{1,2}.log`。摘要及完整工件清单的 Git 副本为 [E001](../data/manifests/q1_mocap_result_20261008_E001.json)、[E002](../data/manifests/q1_mocap_result_20261008_E002.json)，其 SHA-256 分别为 `c67f1cd680effb143bdc25cd1ba73e8198434d7720e611211138bdefa9d299b0`、`e413a143c511d0f245c34c37291dfa67f93e336314244c860f99b7ea939f9562`。IK 分别耗时 4.0604 / 4.0332 秒，只表示该次离线求解速度，不是端到端实时延迟。

### 源动作与可视化检查

几何诊断以骨段间转角定义膝/肘弯折，0° 表示伸直，日志为 `logs/q1_mocap_source_geometry_20261008.log`；脚踝/腕部位置变化以人体根部为参考，不当作世界位移。

| 录制 | 首帧左/右膝弯折 | 全段左/右膝范围 | 左/右脚踝最大相对位移 |
| --- | --- | --- | --- |
| 09/23 | 103.72° / 115.12° | 64.35–111.21° / 75.29–118.96° | 0.332 / 0.256 m |
| 09/24 | 9.59° / 16.46° | 6.52–32.92° / 12.30–30.16° | 0.253 / 0.211 m |

两段各 4 帧比对图已复核，选择首帧、左/右腕位移最大帧与末帧；不是逐帧视频人工验收。图像非空、模型网格可见，第一段源首帧显著屈腿，而 Q1 首帧为中立姿势，证实相对校准消掉了绝对姿态差异。这是当前映射的实质限制。不能仅凭记录判定此屈腿是真实人体姿态还是动捕偏差；需录制视频/标准站姿对照。第二段能看到上肢变化传入 Q1，但较大腕部/朝向残差仍未解决。两段存在腿部输出变化，但没有世界根部移动信息，不能从脚踝相对运动推断行走跟随通过。

渲染使用 IK 运动学 qpos，不是支撑动力学回放画面。两张图 SHA-256 分别为 `53b09b23a0276a1bc3fe3e14212b00f2cbd76ff787ad2add7a836a4236f0a4d1`、`65e7a4978b892cd2c3146818b21074bfc59f29eec7b5b33abd670318a16cff7a`；Mac 的 `data/sync/q1_mocap_test_20261008/overview_E00{1,2}.png` 是核验过的约 0.5 MB 预览副本，大轨迹未复制到 Mac。

当前证据支持“真实录制身体姿态可进入离线 Q1 输入/重定向/支撑关节回放流程”。骨盆在回放中人工固定，根部朝向仅在 IK qpos 中保留，未在动力学回放中执行；真实配对校准、脚接触/自碰撞、自平衡、行走移动、在线传输延迟、训练及真机均未验证。没有启动训练。

两段实验已结束，无本阶段遗留仿真/训练或机器人作业。记录已落盘，重要执行摘要见[阶段日志](../logs/session_records/q1_mocap_20261008.log)；首次/最终输入测试与源几何 stdout 也保存于 `logs/session_records/q1_mocap_{input_tests,input_tests_final,source_geometry}_20261008.log`。

代码、README、审核后的小日志与清单按 `Q1` 分支规则提交、显式推送并比对 HEAD / `refs/heads/Q1`；实际同步以 Git 历史及 Mac `logs/q1_mocap_publish_20261008.log` 回执为准。共享报告只合并 Q1 条目，Mac 其他研究待同步项与服务器其他专题暂存修改分别保留。数据与网盘状态为 `LOCAL_ONLY`，原件保留，没有把大数据或第三方 SMPL 模型放入 Git。

## 下一步与恢复入口

1. 下一工程目标是基于已知标准站姿的绝对姿态校准与逐骨段尺度映射，检验腕/脚残差和接触条件，不把诊断校准当作完整跟随验收。
2. 后续校准与动力学任务需要另外准备：已知标准站姿、抬腿/迈步/转身样本、有效根部移动信息及坐标/地面定义。两段保留测试用途，不转为训练集。实时接入另需确认传输方式/端口、时钟与硬件坐标；离线读文件通过不等于网络实时通过。
3. 测试确认了现有 FGP→SMPL 字段可以继续做输入适配；后续可以训练 Q1 动作跟踪策略，但还需训练数据/任务、接触与动力学验证。本轮没有训练，也不把这两段自动改为训练数据。
4. 恢复先读本报告、[报告索引](README.md)、[当前状态](README_CURRENT_STATE.md)与[前阶段](README_Q1_RETARGET_20261008.md)，重查实际作业、Git 和清单哈希，不重复基础迁移或环境安装。
