# Q1 横移 V2、Recovery V2 与 FGP 仿真离线验收

2026-11-22，3090 主机。按[训练交接](DEPLOY_HANDOFF_Q1_LATERAL_AND_RECOVERY_V2_20261122.md)准备隔离候选并验证三个入口。本轮只运行 CPU 推理、MuJoCo、录制数据回放及本机 HTTP。ROS 桥接回归取代了 DDS 发布器，实际运行中的机器人控制和实时 BLE 穿戴留待现场验证。

## 横移 V2

| 检查 | 原版 | 横移 V2 |
| --- | ---: | ---: |
| 扰动与 10 ms 延迟，2 ms 物理步，通过数 | 4/24 | 24/24 |
| 同组最坏横滚 | 49.52° | 6.32° |
| 同组最小步宽 | 0.01 cm | 8.38 cm |
| Card 04 站立、前走、转向、蹲起、停步 | 完成 | 完成 |
| Card 04 停步末段漂移 | 0.003 m | 0.026 m |
| Card 04 前走步宽 | 16.38 cm | 14.31 cm |

V2 名义横移 12/12 无跌倒，四项前后和转向检查完成，两版均完成 17.5 秒交替横移。PD 越限为 0。测试使用训练仓库的配对评估器及原有 50 Hz 目标限幅。原 Planner SHA-256 前缀 6a989b2b，隔离候选 ce0aa0a4。沿用原部署目录可继续选择旧版。

## Recovery V2

桥接以显式开关选择 V2，候选默认关闭。策略 50 Hz，发送端按每 2 ms 逐拍逼近位置目标，最大增量 0.2 rad。发布调用成功后，下一次推理才读取实际发送目标对应的 prev_action。交还需要 100 ms 内的参考，模拟的 101、150 ms 和未来时间戳均被拒绝，95 ms 的参考触发交还，仅重锚和重置 tracker 各一次。交还保留发送目标与动作历史。无 FGP 时保持 Recovery 持控或在直立基态使用零速度 Planner。Recovery 独立起身模式不传遥操参考。

合并候选五个基础 ONNX 的哈希匹配，四模态 1,000 次随机输入的 PD 估算越限为 0，增量冒烟 11 项控制权检查通过，Recovery ONNX SHA-256 前缀 5d38cfcb。实际 ROS 调度及板端执行误差留待真机测量。

另用训练评估器的 2 ms 物理、11 ms 反馈延迟、500 Hz 位置目标斜坡和部署版 owner 复验 seed 92017。评估器以本地仿真回调替代 ROS 桥接。

| 输入 | 功能结果 | 柔和度与控制权 |
| --- | --- | --- |
| 俯卧、仰卧、侧卧，各 1 个，无遥操 | 3/3 起身后自主保持到结束 | 站稳确认 7.40–9.58 s，遥操调用与交还均为 0 |
| 24 个静态倒姿 | 24/24 站稳并交还 | 23/24 同时满足速度与接触阈值，最大地面穿透 10.42 mm |
| 10 个受推倒地 | 10/10 起身并满足交还后 1 s 的高度和直立阈值 | 2/10 同时满足速度与接触阈值，峰关节速度 17.27 rad/s，峰地面/臂腿穿透 10.08/9.92 mm |

三组估算 PD 越限与误交还均为 0。受推试验末帧有 4/10 满足更严格的静止、角速度与接触联合判据。训练正式验收的 120 个未见静态倒姿为 116 个及时、114 个满足柔和度，30 个未见受推样本均恢复。触地冲击使严格检查仅通过 15/18，现场默认关闭有实际依据。

## FGP 录制回放与 HTTP 仿真

历史输入由真实 BLE 穿戴生成。来源由 session 的 live 模式、stats 的 live_ble 和全部 30,379 帧的 ble_receiver_monotonic_proxy 核对。仿真修正 pelvis free joint 局部角速度后，三种 4 秒静站参考均完成。录制开头 4 秒完成，峰倾角 6.68°。本机 HTTP 注入 0.65 秒 503、再恢复并切换标定代次，GUI 入口完成两段，52 个空闲拍里暂停物理，收到有效参考后重锚并继续。修复了 MuJoCo 被动窗口退出时清理 GLFW 的竞态，最后一次 GUI 回归进程正常退出。

| 同一录制的回放窗口 | 根部 XY 反馈 | 结果 |
| --- | --- | --- |
| 0–20 s | CGFA | 19.42 s 高度触发终止，峰倾角 36.72° |
| 0–20 s | 零 | 完成，峰倾角 14.37°，末端位移误差 0.82 m |
| 0–20 s | MuJoCo 根位置真值 | 完成，峰倾角 14.45°，末端位移误差 0.12 m |
| 0–40 s | MuJoCo 根位置真值 | 完成，峰倾角 17.26°，末端位移误差 0.017 m |
| 0–70 s | MuJoCo 根位置真值 | 47.04 s 倾角触发终止 |
| 60–80 s，独立初始化 | CGFA、零、MuJoCo 真值 | 三组分别在 4.90、3.58、5.88 s 触发终止 |

第一段原 CGFA 运行的末端位移误差为 2.54 m。第 16 秒重置后单独回放 4 秒，原 CGFA 也能完成，峰倾角 12.89°。这组对照把前段失稳缩小到根部 XY 反馈及闭环历史。第 60 秒附近输入骨盆约 0.47 m、倾角约 34–36°，固定录制初始站立骨盆高度 0.978 m 后仍失败。CGFA 保留为默认，sim-truth 和 zeroed 是显式仿真对照。当前证据支持短段遥操仿真和 HTTP 接入，长段跟踪仍有失败。

## 复现与工件

~~~bash
cd /home/yu/projects/Q1_deploy
candidate="$PWD/verification/q1_deploy_v2_candidate_20261122"
project=/home/yu/projects/agibot-X2-A3-Q1-model
py=/home/yu/projects/SuperMimicHP/.venv/bin/python
recording=/home/yu/projects/FGP_mocap/results/live_device_status_r1/smpl.jsonl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$py" -B "$candidate/code/smoke_test_deploy_bundle.py"
"$py" -B -I "$candidate/code/smoke_test_deploy_v2_addon.py"
"$py" "$project/script/q1_h003_planner_strafe/eval_strafe_suite.py" \
  --new-onnx "$candidate/onnx/q1_tok_planner_cmd.onnx" --out-dir /tmp/q1-strafe-recheck

MUJOCO_GL=egl "$py" "$candidate/code/sim_q1_fgp_human_smpl.py" \
  --smpl-jsonl "$recording" --duration 20 --xy-feedback sim-truth --video \
  --output /tmp/q1-fgp-recheck

"$py" "$project/script/q1_deploy_offline/check_fgp_http_sim.py" \
  "$candidate" "$recording" /tmp/q1-fgp-loopback-recheck

source /opt/ros/humble/setup.bash
source /home/yu/ros2_ws_aimdk/install/setup.bash
/usr/bin/python3 "$project/script/q1_deploy_offline/check_recovery_bridge.py" "$candidate" "$PWD"
~~~

有桌面时，本机 HTTP 回归可附加 --viewer。现场穿戴时，可将仿真入口换成 --fgp-url 指向提供 /api/smpl 的 FGP 主机，并使用 --viewer。只启动 MuJoCo，此入口没有 ROS 节点。仿真依赖 SuperMimicHP 的 Q1 网格资产和 Python 环境。候选目录包含代码实文件、配置、六个 ONNX 和场景 XML，实验运行的原始输出见 [工件与指标清单](../data/manifests/q1_deploy_v2_offline_20261122.json)。大工件与原始录制保留在本机，状态 LOCAL_ONLY。

下一次现场可以先验证穿戴到 MuJoCo 的实时路径。横移 V2 的真机表现仍有待测量。Recovery 的受推冲击和 FGP 困难片段需要继续验证。

## 2026-11-22 实时仿真入口修复

之前的实时入口存在两个实际缺陷。`--duration 0` 被拒绝，且重复使用同一个输出目录会因 `FileExistsError` 退出。已修复候选入口和 Git 交接快照。

- 使用 `--fgp-url` 时，省略 `--duration` 或设置 `--duration 0` 会持续运行，直到关闭 MuJoCo 窗口或按 `Ctrl-C`。
- `Ctrl-C` 会正常保存 `report.json` 和 `trace.npz`，报告中的 `stop_reason` 为 `interrupted`。
- 已存在的输出目录会自动使用 `_run02`、`_run03` 等新目录，保留历史记录。
- 本机 HTTP 连续运行回归通过两次，分别处理 124 和 72 个策略步，均正常中断并保存报告。

## 实时衣服到 Q1 仿真复核

2026-11-22 启动真实 FGP BLE 服务后，两件衣服均连接，服务输出约 30 Hz，`tracking` 有效。首次实时仿真暴露出适配器问题。`/api/smpl` 只返回最新帧，客户端正常跳过中间序号却被旧代码判为 `source_gap`，造成参考段重置。修复后，实时仿真 `reference_segments` 从 9 段降为 1 段，序号跳跃合同检查通过。该修复已同步到 `q1_fgp_smpl.py` 和 FGP HTTP 入口。

修复后仍需区分输入姿态与策略能力。当前衣服会话的标定状态为 `static_window_passed=true`、`tpose_geometry_verified=false`。前 20 秒输入骨盆高度约 `0.534–1.028 m`，根部倾角最高约 `34.2°`。在这段非站立起始输入下，仿真约 `3.78 s` 触发倾角阈值，策略目标误差 P95 `1.22183 rad`。

同一会话中选取 `15.1–35.1 s` 的低倾角直立片段，原始输入倾角 P95 `6.79°`、最大 `8.76°`。相同策略和 CGFA 链路完成 `20.0 s` 仿真，最大 Q1 倾角 `8.686°`，目标误差 P95 `0.47546 rad`，没有空闲拍或参考段重置。

当前结论是实时数据链已经接通，策略在接近站立的输入上可运行。非站立人体起始姿态与 Q1 默认站立初始化之间仍存在明显 handoff 分布差异。不能把前一段失败直接归因成所有 human_smpl 跟踪能力失效，也不能把直立片段通过外推成任意姿态跟踪通过。下一步应先重复合格 T-pose 标定并从低倾角站立片段开始，再单独设计人体姿态到 Q1 初始状态的过渡。
