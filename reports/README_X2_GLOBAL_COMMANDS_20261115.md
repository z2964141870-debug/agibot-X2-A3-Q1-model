# X2 衣服遥操全局命令

> **暂停实控。** 20:23 容错版真机短测出现异常运动。以下控制命令仅保留为历史记录。当前仅做只读和离线排查。[事故记录](README_X2_TOLERANT_MOTION_INCIDENT_20261115.md)

在 SoC1 的任意目录均可运行，无需 `cd`。每个终端先 `ssh run@10.0.1.41`，使用交互密码。

终端2，策略（先确认未运行）：

```bash
x2-teleop policy
```

终端1，衣服和网页（先确认未运行）：

```bash
x2-teleop fgp
```

打开 http://10.0.1.41:8768，等待上下衣接收，采集T-pose（3秒准备、3秒采集），等待SMPL更新。

终端3，先检查MC并只读检查：

```bash
x2-teleop mc inspect
x2-teleop observe 6
```

确认MC为Running/STAND_DEFAULT且只读数据连续、现场支撑及急停就绪后，现场操作者才可选择短时控制：

```bash
x2-teleop control 10 --confirm-live
```

等待定时会话自然结束，确认MC_STAND_CONFIRMED后再关终端1、2。FAULT_DAMPING时保留终端，使用对应现场恢复流程；`x2-teleop recover`需要当前事故自己的参数，不能复用旧PID。查看帮助 `x2-teleop help`。

本次在机器人 `/usr/local/bin/x2-teleop` 安装，原R6二进制和20ms配置保持，30ms候选尚未通过有效人体数据验证。/tmp验证命令可发现、help成功、无确认的control拒绝，未启动电机；MC inspect命令可运行但服务返回 `MC service unavailable`，当前MC模式未确认。FGP可能已经运行，勿重复启动；确认当前服务状态后再使用。
