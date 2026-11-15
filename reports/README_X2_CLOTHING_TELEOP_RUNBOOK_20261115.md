# X2 衣服遥操——从开机到收工操作书

版本：2026-11-15。适用：当前 X2 Ultra / SoC1，Mac 网线直连。衣服→FGP S1 GPU→SMPL24→Morph human→X2 后端。

> 当前已验证到“真人 SMPL 送入策略＋真实机器人反馈生成关节建议目标”。重启前6秒只读测试271次请求、205次候选通过、66次拒绝，MC命令捕获完整，电机发布0。**这不等于真机动作效果通过，也不是稳定50Hz认证。**
>
> 首次接管在现场保护架/悬吊、急停可立即操作的条件下做短测。机器人同时有人使用时，先协调唯一控制者，结束其他遥操部署；不要同时运行 Sonic 与 Morph writer。

---

## 0. 开机和网络

1. 按厂家流程上电，等系统启动稳定；本机程序不会自动启动电机控制。
2. 用官方流程让 MC 进入 `STAND_DEFAULT`。不要根据“看起来站着”替代模式检查。
3. Mac 有线网卡 `USB 10/100 LAN 3` 已固定 `10.0.1.10/24`；机器人 SoC1 为 `10.0.1.41`。Wi-Fi 保持上网。
4. 三个终端均登录 SoC1，分别保留运行。每个终端先执行：

```bash
ssh run@10.0.1.41
cd /home/agi/x2_morph_human_20261113
mkdir -p data/analysis
```

SSH 使用交互密码，不写进操作书。下面命令在机器人 shell 执行；复制代码框，不要把反斜杠复制成 `\\`，续行符后不能有空格。

**每次机器人重启：需要重开三个终端、重连衣服、重新标定。模型和修复文件会保留，进程不会保留。**

---

## 1. 先启动终端2：Morph 策略服务

先检查有无已启动的同一服务，避免重复占用端点：

```bash
pgrep -af 'script.x2_morph_human serve'
```

没有服务时启动：

```bash
sh script/x2_morph_human/run_orin.sh serve \
  --endpoint tcp://127.0.0.1:5765 \
  --max-freshness-ms 120 \
  --max-interp-gap-ms 100
```

看到服务监听/ADVISORY_ONLY 信息后保持终端运行。`publish_allowed=false`是正常设计：策略服务只生成建议，终端3才管理实际发送权限。

**不要为了通过就绪检查改成3000ms新鲜度或2000ms插值间隔。** 之前现场服务曾带宽松参数，重启前205个候选的只读结果不能当作这套严格参数已认证；若严格参数拒绝，查具体原因。

---

## 2. 终端1：衣服、S1 GPU、网页和人体转发

上衣/裤子上电并放在机器人附近，关闭其他接收衣服的程序。地址已内置：上衣 `F5:C6:03:E6:4B:30`，裤子 `DA:1B:78:C1:56:1F`。

```bash
bash script/x2_fgp_preview/run_live_orin.sh
```

此命令同时启动两个进程：

- S1 CANO TensorRT/CUDA Graph → LIP/GP/原生物理 → SMPL 数据。网页仅保留状态/标定/重连。
- 网页 `/api/smpl` → `HumanFrame` → 终端2。

看到 `FGP_MORPH_BRIDGE_READY` 表示转发进程已启动，**不表示标定已通过或机器人可接管**。保持终端运行。日志路径会打印在终端里。

Mac 浏览器打开：<http://10.0.1.41:8768>

1. 等上下衣都显示“接收中”。首次扫描允许最多180秒，耐心等。
2. 点击“采集 T-pose”。
3. **前3秒准备**：摆好T-pose；**后3秒采集**：保持不动。
4. 当前稳定性门：姿态抖动≤5°、加速度标准差≤0.8m/s²、至少60对有效样本。静态通过仍需自己观察T-pose形态是否正确。
5. 标定成功后等待30帧历史积累，确认状态receiving、frame_seq持续递增。遥操优先模式关闭3D网格，空白人体区是预期现象。需要观察人体质量时先退出此模式，改用run_orin.sh smpl；它只做可视化，不转发接管。

同一终端暂时没有空闲输入时，可在终端3执行只读检查：

```bash
curl -fsS http://127.0.0.1:8768/api/state | python3 -c 'import json,sys;s=json.load(sys.stdin);print({k:s.get(k) for k in ("reconstruction","smpl_inference_ms","smpl_age_ms","smpl_realtime_valid","error")});print(s.get("model_profile"));print(s.get("calibration",{}).get("state"))'
pgrep -af 'script.x2_fgp_preview.live_morph_bridge'
```

应看到 `reconstruction=receiving`、`static_passed`、帧持续更新、后端 `TensorRT CUDA Graph`，且有转发进程。网页动但没有转发进程时，终端3仍收不到人体。

**如果已经运行 `run_orin.sh smpl`：它只启动网页，不包含转发。** 先在原网页终端Ctrl-C退出，再使用上面的 `run_live_orin.sh`；不要同时开两套网页/衣服接收。

---

## 3. 终端3先做只读检查

### 3.1 检查 MC

```bash
sh script/x2_morph_backend/run_orin.sh mc inspect
```

应显示 MC `Running(...)` 和 `STAND_DEFAULT`。此命令只查询。未达到时按官方流程处理后再检查。

### 3.2 只读验证完整数据链，绝不接管

先确认网页标定成功和人体在更新，再执行：

```bash
sh script/x2_morph_backend/run_orin.sh control \
  --binary data/analysis/X2_MORPH_NATIVE_BUILD_R6_20261113/x2_morph_native \
  --profile data/analysis/X2_MORPH_NATIVE_SENSOR_V2_20261113.json \
  --endpoint tcp://127.0.0.1:5765 \
  --output "data/analysis/X2_LIVE_OBSERVE_$(date +%Y%m%d_%H%M%S)" \
  --seconds 6
```

这里的子命令虽然叫 `control`，但**没有 `--control` 标志，因此是observe**。约6秒后退出并打印状态。

| 字段 | 需要确认 |
| --- | --- |
| `mode` | `observe` |
| `ready_target` | `true`；`candidates`应大于0 |
| `mc_capture_complete` | `true` |
| `transport_failed` | `false` |
| `hardware_commands` / `sdk_publish_calls` | 均为0 |

只读样本偶尔就绪不保证接管后持续就绪。若存在大量 `rejected`，先检查 `native/native.jsonl` 的 `rejection_reason` 和 `reply.reason`，不要直接反复尝试接管。

---

## 4. 显式接管——现场决定后执行

前面检查通过，穿戴者先保持稳定姿态。第一次用10秒短测；后续现场确认后可改成60秒。

```bash
sh script/x2_morph_backend/run_orin.sh control \
  --binary data/analysis/X2_MORPH_NATIVE_BUILD_R6_20261113/x2_morph_native \
  --profile data/analysis/X2_MORPH_NATIVE_SENSOR_V2_20261113.json \
  --endpoint tcp://127.0.0.1:5765 \
  --output "data/analysis/X2_LIVE_CONTROL_$(date +%Y%m%d_%H%M%S)" \
  --seconds 10 \
  --control \
  --confirm-live X2_ULTRA_MORPH_CONTROL
```

程序先完成ROS初始化，再在最多20秒内等待连续1秒目标就绪、MC命令完整且无新增拒绝；随后仍重新检查所有条件和MC站立，执行阻尼→停MC→确认总线静默→Morph渐入。

| 看到什么 | 含义/动作 |
| --- | --- |
| `CONTROL_ACTIVE` | 交接已完成，定时控制开始；先做小幅动作 |
| `RETURNING_TO_MC` | 到时正在正常归还；保持终端和终端1、2运行 |
| `MC_STAND_CONFIRMED` | MC模式和总线交还已确认 |
| `SESSION_COMPLETED`（summary.json）/正常退出 | 正常会话结束 |
| `CONTROL RECOVERY REQUIRED` | 异常恢复，按下节处理 |

当前启动阶段可能暂时没有终端输出；不要把安静当断联，也不要再开一份writer。需要查看时，在额外只读终端读取本次输出目录的 `native/status.json`，不要使用旧目录。

---

## 5. 正常收工与异常处理

### 正常收工

1. 保持第一、第二终端及衣服连接，**让第三终端的10/60秒定时会话自然结束**。
2. 等 `MC_STAND_CONFIRMED` 且第三终端正常退出，确认机器人已由MC控制。
3. 再在终端1Ctrl-C停止衣服/转发，在终端2Ctrl-C停止策略。
4. 之后按厂家流程关机。

### 异常

当前Morph版本与参考Sonic操作书不同：**第三终端Ctrl-C不是正常渐回MC的快捷键**，它可能锁定故障并保持阻尼writer。不要将“按Ctrl-C退出”写作正常收工。

- 动作明显异常：现场立即使用物理急停；不要依赖网页按钮停止电机。
- 出现 `FAULT_DAMPING` / `CONTROL RECOVERY REQUIRED`：保留终端，保持现场保护，按官方及现场恢复流程明确恢复MC并核总线归还。不要 `kill -9`、重跑控制脚本或自动重启MC来掩盖未完成交接。
- 衣服断流/策略退出/目标过期/反馈过期/RPC超时都可能进入故障；重新连上数据不代表控制会自动恢复。

---

## 6. 报错对照表

| 现象 | 定位与处理 |
| --- | --- |
| `Failed to fetch` | 浏览器访问不到8768服务。先查Mac有线地址、网页进程和路由；服务重启后刷新页面 |
| 上下衣“连接错误” | 点击网页“重新连接衣服”，等扫描/连接完成；再标定 |
| 标定不通过 | 查看最大姿态抖动/加速度波动，5°门；不要边采集边按键或走动 |
| 网页人体不动 | 查看是否static_passed、历史预热/断流、frame_seq是否递增 |
| 推理接近1秒 | 查model_profile，应为S1 TensorRT CUDA Graph；旧B/U100h CPU是错误入口 |
| `no frames or receiver clock mismatch` | 首先查转发进程。此次真实原因是网页独立运行、转发未启动/启动异常；不是单看网页会动就已送到Morph |
| `JSONEncoder ... unexpected keyword argument 'flush'` | 本次转发脚本启动日志括号错误，已修；出现说明仍用旧文件 |
| `Startup not ready ...` /旧`SMPL/feedback/MC command capture did not become ready` | 看ready_target、mc_capture_complete、transport_failed；进一步查本次native/native.jsonl。不要只延长等待时间 |
| `rpc_deadline` | 本机策略完整RPC超过20ms；保持门，先诊断资源竞争，勿用旧平均值称通过 |
| `Address already in use` /网页锁占用 | 相同服务已运行；检查具体PID/归属，在原终端正常退出。不要泛杀机器人进程 |
| `MC must be running in STAND_DEFAULT` | 官方MC尚不满足交接条件，按官方流程处理，不能跳过检查 |

---

## 7. 日志在哪里

- 终端1：打印的 `data/analysis/X2_FGP_LIVE_MORPH_<时间>/`；`fgp_server.log`、`bridge.log`（JSONL），`preview/`含标定和SMPL。
- 终端3：本次 `X2_LIVE_OBSERVE_<时间>` / `X2_LIVE_CONTROL_<时间>`；`stdout.log`、`stderr.log`、`native/status.json`、`native/native.jsonl`、成功时`summary.json`，启动超时时`startup_not_ready.json`。
- 文件已经存在时换新时间目录，不覆盖旧失败记录。

## Done / To do

**Done**：S1 GPU网页、137B上下衣协议、重连按钮、3+3秒单人标定/5°门；修复人体转发的启动错误。重启前SMPL→Morph→真实反馈→建议目标只读验证已成，0电机输出。

**To do**：此次重启后由操作者依次启动终端2→终端1→标定→终端3只读门，再现场决定是否进行10秒接管。真机动作效果、持续稳定和严格120/100ms数据门仍需现场验证。

## 16:30就绪失败实录与当前限制

本次控制启动38请求/23候选/15拒绝，MC capture完整、transport正常、0电机；末3次均为人体帧间隔超过100ms，另6次RPC超过20ms。先出现一次ready不代表之后能保持；原逻辑还在一次ready后初始化ROS，再复查时已丢目标。

现已将ROS初始化提前，并在20秒内要求连续1秒就绪、请求推进且拒绝计数不增加，所有原安全检查保留。板端及Mac共15检查通过。新的10秒只读检查483请求/360候选/123拒绝，连续1秒稳定窗为false、MC capture true、0硬件发布。**当前应先改善输入/推理连续性，停止反复尝试真实接管；此修复不等于真机可用。**

Agent to do：利用保存的native.jsonl区分人体流超过100ms、RPC超过20ms、反馈及其他拒绝；结合BLE收包/网页/bridge逐帧时间定位。暂不扩大新鲜度或插值间隔，不自动运行--control。

## 遥操优先模式

用户允许移除影响遥操的辅助功能，run_live_orin.sh现默认--no-live-mesh --no-smpl-log：完全跳过逐帧SMPL网格/FK显示，保留原姿态和根平移送入策略；逐帧SMPL文件关闭，转发日志每30次请求记录一次。网页保留衣服状态、标定和重连。未改模型/物理/人体预览/反馈/MC及deadline。修改后需要重新启动第一终端并标定；实际频率收益须标定后实测，不能由删代码推算。

## 16:42交接未完成：不能把Morph零发布当作MC未被操作

X2_LIVE_CONTROL_20261115_164224最终capture_frozen=true、STANDBY、hardware_commands=0，没有CONTROL。最终1238请求/1175候选/63拒绝，原生日志没有control_rejected事件。MC后续查询为Running/28597、PASSIVE_DEFAULT，未确认STAND恢复。只能确定捕获后、Morph接管前未完成；Python原终端异常未落盘，无法从最终PASSIVE推断具体失败点或MC实际停启是否成功。现场不要重复接管，也不要将此状态记为正常交接成功。

新增交接分步骤日志handoff_steps.jsonl及handoff_failure.json，记录CHECK_CONTROLLER_READY、CHECK_MC_RUNTIME_AND_STAND、FREEZE_RETURN_COMMAND、REQUEST/CONFIRM_DAMPING、REQUEST_MC_STOP、CONFIRM_MC_STOPPED、CONFIRM_BUS_SILENT、REQUEST_MORPH_ARM、CONFIRM_MORPH_CONTROL，及异常/当时原生状态。原运行的缺失异常不能事后恢复，需用户原终端Traceback才能进一步定位；不为获取日志重跑真实控制。原门限、状态机、MC调用和恢复逻辑未改。15项本机/板端测试情况见后续实际回执。
