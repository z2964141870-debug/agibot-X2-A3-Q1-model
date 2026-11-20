# Q1 自主起身模块

在 50 Hz 控制循环中，由一个调度器决定每一拍的唯一关节目标。解压后依赖 Python、NumPy 和 ONNX Runtime。此包生成关节目标与估算 PD 力矩。实测范围为 MuJoCo。

```python
from pathlib import Path
from q1_autonomous_recovery import Q1AutonomousRecovery
from q1_v2b_runtime import Q1V2bOnnxController

root = Path("/path/to/unpacked_bundle")
teleop = Q1V2bOnnxController(root, ("robot_pose",))
branch = Q1AutonomousRecovery(
    teleop, root / "onnx/q1_recovery_expert.onnx",
    root / "robot_model/supermimic.xml",
)
# 新 episode 中用实测关节角调用 teleop.reset(q)，同时调用 branch.reset()。
# base_step 用这一个 teleop 实例处理本拍参考，返回 (target, tau, info)。
# reset_anchor 把外部参考、足锚及历史一起重置到当前实测站姿。
target, tau, recovery_owns = branch.step(
    R_imu, omega_body, q, qd, previous_sent_action,
    base_step, reset_anchor,
    sample_time_s=sample_time_s, teleop_stamp_s=latest_usable_reference_time_s,
)
previous_sent_action = (target - teleop.default_q) / teleop.action_scale
```

- 旋转矩阵为机体到世界，角速度为机体坐标（rad/s），关节顺序沿用冻结 V2b 配置。动作历史应使用上一拍实际发送的目标。两个时间戳均用同一单调时钟的秒数。外部接收器只提供通过质量检查的最新参考时间，断流时传 `None`。交还时的 `base_step` 应消费与时间戳对应的同一快照。
- 遥操拥有控制权时，连续三拍严重倾斜或低位触发接管。没有新鲜遥操参考且已检测到倒姿时可在第一拍接管。正常遥操时的断流处置仍由遥操接收器负责。
- Recovery 独占时，先运行既有起身 ONNX，再转入私有站立策略。后者使用冻结 V2b 的独立历史、足锚、固定站姿和朝向。上层只消费返回的一个目标。起身及自主站立期间，外部遥操回调调用次数为零。
- 站稳要求内部站立策略已经完全接管，腿链有效、高度超过 0.36 m、直立余弦超过 0.95、角速度模长小于 0.70 rad/s、关节速度 RMS 小于 0.85 rad/s、双足相对根部的平面速度各小于 0.15 m/s。连续 0.50 秒合格后，最近有效参考还须晚于或等于本次接管、年龄不超过 100 ms。任一站稳条件失效或相邻采样间隔超过 35 ms，重新计时。模拟器真实骨盆速度、接触力和真值位姿仅用于验收。
- `owner` 为 `recovery` 或 `teleop`。恢复时 `phase` 为 `rise`、`settle` 或 `ready`。站稳后遥操仍缺失时保持自主站立，持续检查稳定性。交还当拍先重置遥操控制器和足锚，再运行一次遥操，无两路目标叠加。再次摔倒会重新启动。非有限测量或时间倒序时该次调用报错，调用方负责处理未产生命令的情况。
- 此版本针对地面上发生的跌倒。低滑移时可以从足端运动估计水平移动；悬吊、双脚一起滑动等情况下，仅靠当前本体量无法证明地面支撑。
