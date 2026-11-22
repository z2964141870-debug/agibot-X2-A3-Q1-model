# Q1 遥操版本里程碑

版本标识：`Q1-TELEOP-20261122`。本记录冻结 Q1 分支当前已经打通的 Planner / FGP `human_smpl` 单进程切换入口和同步数据采集器。

## 冻结内容

- Git 分支：`Q1`
- 本里程碑前分支 HEAD：`493dc62493186f815ca9ae5a82722ca331c89722`
- 版本标签：`q1-teleop-20261122`
- Planner 候选 SHA-256：`ce0aa0a49666515e0b47dd200995fecc8980cd3bcdd41149fbc4824fb92679e5`
- human_smpl tokenizer SHA-256：`7dab4d9d056e797c1ae8631d6c15ffd5fc3324234c86b821b253fcbb46adfb0e`
- Recovery：关闭
- 低层发送：500 Hz
- policy 推理：50 Hz

候选代码快照位于 `Q1_deploy/verification/q1_deploy_v2_candidate_20261122/code/`。对应 Git 记录位于 `script/q1_deploy_offline/`。

| 文件 | SHA-256 |
| --- | --- |
| `q1_ros2_lowlevel_bridge.py` | `b828a69e846fa8f2d917df771a5330958639bd2443cb066b7fe1816ed3de21e7` |
| `q1_runtime_recorder.py` | `882543dab147166ca792f0c986cc6300fa9d102fe7677d49d80bb721734d8886` |
| `q1_fgp_client.py` | `e3039c38e48655ee06ffe579713ff5f353c509089c45700358fed91ff0f3b47d` |
| `q1_fgp_smpl.py` | `834f46bcb34ade4a167455187674c0733ca46f542914af1ab60a590f16b534a0` |

## 操作协议

启动后默认 Planner。键盘含义如下。

- `w` 前进
- `s` 后退
- `a` 左移
- `d` 右移
- `e` 右转
- `t` 切换到 human_smpl
- `p` 切回 Planner
- 空格停止 Planner 命令
- `q` 退出并发送阻尼帧

`human_smpl` 切换要求 FGP 最新帧处于新鲜时间窗。切换在 50 Hz 推理边界执行，并重锚参考和 tracker。单个进程拥有 500 Hz HAL 发布循环，避免两个控制程序竞争命令话题。

## 已保存的真机运行数据

数据目录为 `/home/yu/projects/Q1_deploy/real_logs/q1_switchable_record_20261122/`。运行约 197.74 秒，包含以下同步流。

| 流 | 记录数 |
| --- | ---: |
| FGP 接受帧 | 5,855 |
| HAL 状态，约 500 Hz | 98,861 |
| policy，约 50 Hz | 9,887 |
| 实际命令，约 500 Hz | 98,866 |

异步写盘 `dropped={}`，队列未完成任务为零。human_smpl 阶段约 97.76% 的 policy 拍有有效 FGP 元数据，另有 12 个短暂失效区间。数据可以用于版本回放、故障分析和按有效性筛选的训练准备，完整效果评估另行处理。

详细数据报告见 [Q1 切换真机运行数据报告](README_Q1_SWITCHABLE_RUNTIME_DATA_20261122.md)。原始 JSONL 文件保持本机 `LOCAL_ONLY`，没有进入 Git 历史。

## 复现入口

```bash
cd /home/yu/projects/Q1_deploy
source /opt/ros/humble/setup.bash
source /home/yu/ros2_ws_aimdk/install/setup.bash
unset ROS_LOCALHOST_ONLY
export ROS_DOMAIN_ID=0

/usr/bin/python3 -B \
  verification/q1_deploy_v2_candidate_20261122/code/q1_ros2_lowlevel_bridge.py \
  --bundle verification/q1_deploy_v2_candidate_20261122 \
  --keyboard-switchable \
  --fgp-url http://127.0.0.1:8090 \
  --keyboard-vx 0.25 --keyboard-vy 0.05 --keyboard-wz 0.60 \
  --keyboard-z-ratio 1.0 --keyboard-reach-x 0.04 \
  --record-dir real_logs/q1_switchable_record_20261122
```

FGP 服务使用上一轮已核验的两件衣服地址和本机 `8090` 接口。启动真机前仍需确认低层模式、安全吊索、人工扶持和急停准备完毕。
