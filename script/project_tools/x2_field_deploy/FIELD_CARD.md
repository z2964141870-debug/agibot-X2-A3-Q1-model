# X2 键盘行走 · 现场任务书 v3（2026-10-23 首走成功版）

机器人: AgiBot X2 Ultra · 策略: sonic_x2_v0.2 (HUMANPLUS_X2_PLUS)
**2026-10-23 17:13 / 17:43 两轮完整点火, 18:0x 键盘 w 前进 0.35 m/s 走通。**

## ⛔ 核心纪律（v3 血泪教训, 违反任何一条 = 整机断电重来）
1. **SoC0 只读**。绝不杀/冻/重启 SoC0 上的进程: SIGSTOP 冻结 MC → 总线握手崩;
   手动重启 hal_ethercat → 核心转储; `systemctl restart agibot_software` → em 崩。
   MC 的停/启只由 deploy 自己编排(EM HTTP)。
2. **电池断电重启 = 唯一完全恢复**。`sudo reboot` 会跳过厂商上电编排(开机会拉起
   hal/MC/em 的 start_apps RPC)。搞坏了别 reboot, 直接拔电池。
3. **一个开机周期只做一次"停 MC 接管"**。反复 stop/start MC 会让 hal 的 joint
   state 数据流对外消失(SHM 传输系统性卡死, 2026-10-23 全天排查结论)。要重来
   就整机断电, 不要连续重跑仪式。
4. **站立模式是点火前置**。preflight 需要关节状态流, 而 PASSIVE(刚开机)模式下
   hal 不对外发布数据。上电等 ~90s → 遥控器 **L2+X**(位控站立) → 才能跑仪式。

## 终端速查（桥接机直连, 全部在 SoC1; SoC0 不开终端）
| 终端 | 用途 |
|---|---|
| A | 仪式启动 + deploy 监视 |
| B | 键盘控制 |

断电重启后主机密钥会变: `ssh-keygen -R 10.0.1.41` 再连。
账号 agi@10.0.1.41, 密码见团队备忘。

## 阶段 0 —— 开机(现场)
1. 电池装好, 长按电源 5s
2. 等 ~90s(厂商栈自启, 别碰机器人)
3. 遥控器短按 **L2+X** → 腿绷紧到站姿 = 成功
   (应急替代: SDK 设模式, 见文末)

## 阶段 1 —— 仪式 + 接管监视(终端 A)
```bash
ssh agi@10.0.1.41
bash ~/gear-sonic/ritual_start_kb.sh
# 等: pose stream OK (4220b)  ← 不过就等30s重跑一次
tmux attach -t x2_deploy      # Ctrl-B D 退出监视
```
绿灯序列: `Preflight PASS` → `stopping MC`(轻微下沉, 正常) → `CONTROL tick` 每秒刷。

⚠️ 陷阱: 部署中止后重跑仪式可能 "already running -- SKIPPED"(尸检标记被 RTPS
噪音冲出 200 行窗口)。手动 `tmux kill-session -t x2_deploy` 再跑仪式。

## 阶段 2 —— 键盘行走(终端 B)
```bash
ssh agi@10.0.1.41
~/walk/t2_keyboard.sh
```
| 键 | 动作 |
|---|---|
| `w` / `s` | 前进 0.35 m/s / 后退 0.1（按一下锁定）|
| `a` / `d` | 原地左 / 右转 0.65 rad/s |
| `空格` | 停(意图清零) |
| `q` | 退工具(先空格) |

- 工具以 30Hz 持续喂意图, **终端全程开着**; 工具一断 kplanner 0.3s 后回 IDLE
  (机器人站定, 不会摔), 重开即可。
- 每次按键必须看到回显 `cmd vx=...`。
- 字段协议(2026-10-23 修复): target_velocity 元组 = **(yaw, 侧移, 前进, 髋高)**,
  旧工具发 (vx,0,0,yaw) 会把 w 变成原地左转——已修, 勿回退。

## 阶段 3 —— 收工
1. 终端 B: `空格` → `q`
2. 终端 A: attach deploy 面板 → **Ctrl-C 一次** → 看 RAMP_OUT → start_app +
   SetMcAction(STAND_DEFAULT) → deploy 干净退出(机器人回 MC 力控站立)
3. 现场按官方流程关机(趴下调零力矩再断电)

## 中止阶梯
1. `空格`(每一拍可停) → 2. `q` → 3. Ctrl-C 交接 MC → 4. **物理急停(零力矩, 机器人会倒, 真紧急才用)**

## 附: SDK 模式应急通道(SoC0, 只读查询/应急设置)
```bash
# 查询/设置 MC 模式(vendor rc python 类型, 已验证)
ssh run@10.0.1.41 不可用; 用桥接机: ssh run@10.0.1.40
sudo bash /tmp/mc_vendor_run.sh            # 查询当前模式
sudo bash /tmp/mc_vendor_run.sh set JOINT_DEFAULT   # 站位准备(等价 L2+X)
```
模式名: PASSIVE_DEFAULT / SOFT_EMERGENCY_STOP / DAMPING_DEFAULT /
JOINT_DEFAULT(=L2+X 位控站立) / STAND_DEFAULT(=R2+X 力控站立)。
SetMcAction 走 action_desc 名字(不是 value 数值), source=deploy_x2。

## 参考
- `x2_end_to_end_runbook.md` 全流程终端命令单(桥接机 2 终端版)
- `x2_keyboard_walk_runbook.md` 键盘速查卡
- `reports/X2_MISSING_JOINT_TOPICS_20261023.md` joint 话题可见性排查全记录(已结案)
- v2 的 T3 SIGSTOP 冻结流程(pause_mc.sh)**已废弃**, 文件仅存档
