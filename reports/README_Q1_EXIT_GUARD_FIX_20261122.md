# Q1 低层退出保护修复记录

日期：2026-11-22

## 事故现象

本轮使用 `--keyboard-switchable --exit-passive` 运行在 Q1 低层运控模式。按下 `q` 后，桥接日志显示：

```text
PASSIVE_DEFAULT was not requested: SetMcAction service unavailable
```

随后桥接发送退出阻尼帧，所有关节的 `stiffness=0`、`damping=5`。机器人当时处于自主承重站立状态，因此退出阶段失去位置保持并摔倒。

真机记录目录：

`/home/yu/projects/Q1_deploy/real_logs/q1_fgp_control_v2_real_20261122_run02`

末三帧均为 `mode=damping`，记录中的 `stiffness` 全为零。FGP 和 policy 记录没有显示退出前发生目标跳变。

## 根因

`SetMcAction` 属于官方 MC 服务。低层运控模式关闭 MC，因此该服务不可用。低层模式下不能把 `PASSIVE_DEFAULT` 当作桥接退出的可用后端。

## 修复

候选桥接文件：

`/home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/code/q1_ros2_lowlevel_bridge.py`

- `q` 在 `PASSIVE_DEFAULT` 服务可用时才真正退出。
- 低层模式检测不到该服务时，`q` 切回零速度 Planner，并继续保持 PD 命令。
- `x` 作为强制退出键。强制退出前必须由人工或吊架承托机器人。
- `_switch_to_passive()` 现在返回成功状态，只有被官方服务确认后才认为退出到被动模式完成。

当前候选文件 SHA-256：

`a4617373d4402812734fbd5e14de01578a813d5c7c7129c4ae1eb17aa0a27f90`

## 离线验证

- `SWITCHABLE_BRIDGE_PASS`
- `FGP_V2_ADDON_SMOKE_PASS`
- `SMOKE_TEST_ALL_PASS`
- `torch_loaded=false`
- `mujoco_loaded=false`
- `oob_total=0`

本修复尚未再次上真机。下一轮必须在吊架和人工承托下先验证 `q` 进入安全保持，再验证 `x` 的强制退出路径。
