# Q1 FGP `human_smpl` 遥操适配对齐记录

**日期**：2026-11-21
**范围**：FGP SMPL HTTP → Q1 Pure V2b `human_smpl` → 低层桥接器
**控制状态**：本轮只读验证，没有发布任何关节命令

## 已完成实现

- 新增 [code/q1_fgp_client.py](../code/q1_fgp_client.py)。它独立线程轮询 FGP `/api/smpl`，使用现有 `FGPSMPLStream` 做状态、序号、会话、标定代次、断流和 50 Hz 重采样处理。
- `FGPHttpSource` 使用有界 tick 队列，记录 HTTP 错误、丢弃 tick 和 FGP stream 事件。新鲜度门限为 150 ms。
- [code/q1_ros2_lowlevel_bridge.py](../code/q1_ros2_lowlevel_bridge.py) 增加两个入口。
  - `--fgp-readonly --fgp-url ...`，读取 Q1 反馈并运行 `human_smpl` 推理，不创建 HAL 命令发布者。
  - `--human-smpl --fgp-url ...`，为后续低层策略接管预留，当前没有进行真实主动测试。
- FGP segment 变化时会重置 Q1 policy history、CGFA 状态、根参考和上一动作。
- Q1 的 `obs_80[72:80]` 与 FGP 根位置、根朝向共用 `p_ref_xy` 和 `R_ref`，人体特征与机器人本体参考保持同一 heading。
- 500 Hz HAL 发布线程与 FGP HTTP 线程分离。FGP 输入不会阻塞控制线程。

## 数据路径

```text
FGP /api/smpl
  → FGPHttpSource
  → FGPSMPLStream
  → SMPL24 → Q1 坐标变换
  → 10 帧 human_smpl 430D
  → Q1 obs_80 + human_smpl tokenizer
  → Pure V2b action
  → 目标位置缓存
  → 500 Hz HAL 发布
```

FGP 坐标约定沿用现有适配。

```text
FGP：+Z 前、+X 左、+Y 上
Q1 ：+X 前、+Y 左、+Z 上
```

FGP 的 SMPL24 输出使用 `joints_m`、`pose_mat`、`floor_y_m`、`session_id`、`calibration_id` 和 `sequence`。跨主机时不使用 `source_ns` 与 Q1 单调时钟直接相减，HTTP 到达时间由客户端单独记录。

## 已完成验证

| 检查 | 结果 |
|---|---|
| Python 语法 | `q1_fgp_client.py` 和 `q1_ros2_lowlevel_bridge.py` PASS |
| ONNX 部署 smoke | 5/5 SHA-256，4 个模态均 PASS，OOB=0 |
| FGP 录制 → `FGPHttpSource` | 161 个 50 Hz tick，输出 feature shape `(430,)` |
| 现有 FGP 只读 replay | 89 个策略步，反馈最大年龄 `2.862 ms`，推理 P95 `8.917 ms` |
| 新 `--fgp-readonly` 入口 | 本地 HTTP fixture 运行 4 秒，返回码 0 |
| HAL 命令发布 | `Publisher count: 0`，设备管理器仍有 1 个订阅者 |

只读入口的复现形式如下，命令中的 URL 应指向当前 FGP 服务。

```bash
source /opt/ros/humble/setup.bash
source /home/yu/ros2_ws_aimdk/install/setup.bash
python3 code/q1_ros2_lowlevel_bridge.py \
  --fgp-readonly \
  --fgp-url http://127.0.0.1:8090 \
  --duration 30
```

## 当前边界

- 本轮没有启动真实 FGP BLE 服务，也没有使用 `--human-smpl` 主动发布模式。
- 当前验证使用 FGP 已保存的 SMPL JSONL 作为 HTTP fixture，覆盖了格式、历史、坐标和只读桥接路径。
- FGP 实时服装连接、真实 Q1 承重状态下的人体动作跟随、目标角安全裕量和主动低层交接仍未验收。
- 现有历史只读结果中，未经关节限位投影的 `human_smpl` 输出曾出现左踝、右踝和右肩目标超出配置位置范围。主动真机前需要保留现有位置步进、PD 力矩和关节限位保护，并单独记录目标投影统计。

## 下一阶段

1. 在 FGP 服务真实 `tracking` 状态下重复 `--fgp-readonly`，记录 `session_id`、`calibration_id`、源年龄、tick 丢弃和 430D 特征统计。
2. 只读稳定后，在吊架保护下启用 `--human-smpl`，先保持零动作人体输入，观察目标角、IMU 和关节反馈。
3. 通过站立承重门后，先测试上肢动作，再测试小幅躯干动作，最后测试移动。
4. 真实主动测试必须单独保存 FGP 帧、`obs_80`、430D 特征、策略动作、Q1 反馈和目标投影结果。
