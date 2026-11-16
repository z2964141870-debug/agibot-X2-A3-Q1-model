# X2 SONIC 真人只读联测工具

用于“衣服 → SMPL → SONIC tracking”的隔离只读联测。所有脚本都不发布电机命令，也不改动 MC、kplanner、生产站立后端，以及生产 token 服务使用的端口（5557、5558、5563、5573、5574）。隔离端口为 15557、15558、15566、15573、15574、15575。结果说明见 [README_X2_SONIC_LIVE_HANDOFF_20261116](../../reports/README_X2_SONIC_LIVE_HANDOFF_20261116.md)。

## 板端使用（agi@10.0.1.41）

把本目录和 `script/x2_fgp_preview/` 同步到板端同级目录（例如 `/home/agi/x2_sonic_live/script/`）。

```bash
# 1. 一次性构建诊断后端（重启后仍保留）
bash script/x2_sonic_live/build_diag.sh

# 2. 现场按正常流程拉起 kplanner 和站立后端，机器人稳站

# 3. 在 agi 终端前台启动衣服（Enter 后 3 秒倒计时摆 T-pose；Ctrl-C 退出时自动释放蓝牙）
bash script/x2_sonic_live/run_garment_fg.sh

# 4. 穿戴者自然站直后，另开一个终端开始录制：默认 2.5 s 后接入、保持 12.5 s、总时长 19.5 s
bash script/x2_sonic_live/run_shadow_live.sh
#   板端 adapter 和 token 服务还没打补丁时：LEGACY=1 bash script/x2_sonic_live/run_shadow_live.sh
```

录制目录为 `/home/agi/x2_sonic_live/captures/<时间>/`，结束时自动输出 `handoff_audit.json`。如需完整摘要，在 Mac 上对拷回的目录运行：

```bash
python3 -m script.x2_sonic_live.summarize_capture <capture_dir>
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `build_diag.sh` | 核对生产 `x2_deploy_onnx_ref.cpp` 的 SHA，在独立工作区应用 cycle trace 和来源标记补丁并编译，同时生成带来源标记的 watchdog |
| `run_shadow_live.sh` | 依次启动来源标记 watchdog（订阅 kplanner 的 :5556）、隔离 token 服务（`--planner-cmd-port 0`）、诊断后端（`--no-publish --cycle-trace`），按时间表写入并删除 engage 哨兵，最后运行审计 |
| `run_garment_fg.sh` | 衣服前台入口，intent 发到 15573，engage 哨兵为 `/tmp/x2_shadow_pico_intent.engage`，启动和退出时都断开两件衣服的蓝牙 |
| `garment_sonic_relay.py` | 只给未打补丁的 adapter 用：15573 → 15575，SMPL Y-up 转 SONIC Z-up |
| `summarize_capture.py` | 汇总审计结果、各阶段关节目标、planner 段来源、穿戴者手腕运动和原件 SHA |
| `onboard_patched/`、`patches/`、`install_onboard.sh`、`test_onboard_fixes.py` | 三个生产问题的正式修复、unified diff、带 `.bak_20261116_pre_sonic_fix` 备份的板端安装脚本，及离线回归测试 |

衣服进程本身大约占 2 个 CPU 核，会和现场 50 Hz 控制循环抢资源。不录制时请按 Ctrl-C 退出衣服进程。
