# Q1 FGP Control V2 合入记录

日期：2026-11-22。按照 `DEPLOY_HANDOFF_Q1_FGP_CONTROL_V2_20261122.md` 将 A2 适配方案合入 `/home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122/`。本次只修改隔离候选，没有启动 Q1 真机。

## 合入内容

- `code/q1_fgp_smpl_v2.py`
- `code/q1_fgp_client_v2.py`
- `code/smoke_test_fgp_v2_addon.py`
- `config/q1_deploy_config.json` 的 `fgp_v2` 参数块
- `code/q1_ros2_lowlevel_bridge.py` 的 V2 FGP 客户端、fallback guard、paced tick 和热切换重锚补丁
- 原桥接、配置的备份
  - `code/q1_ros2_lowlevel_bridge.py.bak_pre_fgp_v2`
  - `config/q1_deploy_config.json.bak_pre_fgp_v2`

冻结 ONNX 没有替换。`q1_tok_human_smpl.onnx` 保持 `7dab4d9d...`，Planner、RobotPose、TokenHead 和 Recovery 文件保持现候选版本。

## 验证结果

- 独立包 `SHA256.json`：13/13 通过
- 独立 `python -I` FGP V2 smoke：`FGP_V2_ADDON_SMOKE_PASS`
- 模拟合入 FGP V2 smoke：`FGP_V2_ADDON_SMOKE_PASS`
- 五模态部署 smoke：`SMOKE_TEST_ALL_PASS`
- Switchable bridge offline regression：`SWITCHABLE_BRIDGE_PASS`
- PyTorch 与 MuJoCo 导入：均为 false
- PD OOB：0
- Recovery 默认：false

已验证的合入候选入口与本机之前的同步采集器、`--exit-passive` 退出流程共存。可选 FGP 服务端 2 行防抖补丁没有施工。

## 当前真机命令

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
  --record-dir real_logs/q1_fgp_control_v2_real_20261122 \
  --exit-passive
```

切换入口的按键是 `w/s/a/d/e` Planner、`t` 进入 human_smpl、`p` 回 Planner、空格停止 Planner 命令、`q` 退出。Recovery 测试使用 `--recovery-offline --exit-passive`，不要与该 switchable 命令同时运行。

## 回滚

```bash
cd /home/yu/projects/Q1_deploy/verification/q1_deploy_v2_candidate_20261122
cp code/q1_ros2_lowlevel_bridge.py.bak_pre_fgp_v2 code/q1_ros2_lowlevel_bridge.py
cp config/q1_deploy_config.json.bak_pre_fgp_v2 config/q1_deploy_config.json
```

这份合入记录只证明隔离候选合入和本地验证通过，不替代新的真机安全小幅度验收。
