# X2 容错版 · 10 秒短测

1. 保持衣服和策略终端运行，确认网页 SMPL 实时更新，机器人有支撑、急停就绪。
2. 在机器人终端三从任意目录粘贴第一段。确认 `Running` / `STAND_DEFAULT`。第二段先删去 `--control --confirm-live X2_ULTRA_MORPH_CONTROL` 做只读检查。见到 `mc_capture_complete=true`、`ready_target=true` 且人体参考没有反复过期后，再完整执行第二段开始短测。

```bash
base=/home/agi/x2_morph_human_20261113
sh "$base/script/x2_morph_backend/run_orin.sh" mc inspect
```

```bash
sh "$base/script/x2_morph_backend/run_orin.sh" control \
  --binary "$base/data/analysis/X2_TOLERANT_R1_20261115/build/x2_morph_native" \
  --profile "$base/data/analysis/X2_TOLERANT_R1_20261115/profile_30ms.json" \
  --endpoint tcp://127.0.0.1:5765 \
  --output "$base/data/analysis/X2_TOLERANT_CONTROL_$(date +%Y%m%d_%H%M%S)" \
  --seconds 10 --control --confirm-live X2_ULTRA_MORPH_CONTROL
```

3. `CONTROL_ACTIVE` 后做小幅动作。等待 `MC_STAND_CONFIRMED` 自然结束。
4. 出现 `CONTROL RECOVERY REQUIRED` 时保留终端三和现场支撑，记录当次输出。全局 `x2-teleop control` 目前仍会启动旧版 R6。

2026-11-15 恢复记录。原阻尼 writer 已退出，MC 从 `Idel` 经官方管理接口恢复，实读 `Running(64010)` / `STAND_DEFAULT`。

恢复后两次只读均捕获到 MC 命令，均无电机发布。6 秒接受 0/296，12 秒接受 519/583，后者仍有 48 次人体参考过期、11 次人体插值间隔超限、5 次 RPC 超过 30ms。先复查人体输入连续性。证据位于板端 `data/analysis/X2_TOLERANT_POST_MC_OBSERVE_20261115_2018` 和 `data/analysis/X2_TOLERANT_POST_MC_OBSERVE_20261115_2020`。
