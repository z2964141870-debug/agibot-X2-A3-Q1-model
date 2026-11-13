# X2 双 Orin 桥接连接实测

最新记录时间：2026-11-13 12:42:53 北京时间。状态：Windows 与 SoC0/SoC1 均已原生交互密码登录成功。下方 12:27 的超时保留为历史，不代表当前不可达。

## 最新恢复结果

用户提供新的恢复条件后，原样执行 `ssh sshuser@100.92.227.62`，立即收到密码提示并登录 Windows `DESKTOP-LQL43SI`。随后从 Windows 分别执行 `ssh agi@10.0.1.41` 和 `ssh run@10.0.1.40`，交互认证后均进入 Linux。没有使用转发入口或额外 SSH 参数，没有重建转发。

| 设备 | 真实账户与地址 | 只读硬件结果 | 系统 |
|---|---|---|---|
| Windows 桥接机 | `sshuser@100.92.227.62` | `DESKTOP-LQL43SI` | Windows `10.0.26200.9711` |
| X2 Ultra SoC1 | `agi@10.0.1.41` | `NVIDIA Jetson Orin NX Engineering Reference Developer Kit Super V2Board` | Ubuntu 22.04.5；`Linux 5.15.148-tegra aarch64`；L4T R36 revision 4.3 |
| X2 Ultra SoC0 | `run@10.0.1.40` | `RK3588 AGI Board` | Ubuntu 22.04.5；`Linux 6.1.99-rt36 aarch64`；没有 `/etc/nv_tegra_release` |

两台 Linux 登录横幅均标明 X2 Ultra、序列号 `X220025BBZ0029`，分别为 SoC1/SoC0。用户拓扑图中的“双 Orin”称呼与设备树不一致：SoC1 是 Orin NX，SoC0 是 RK3588，不能给 SoC0 套用 Jetson CUDA/TensorRT 环境。

本轮只有登录与只读 `id -un`、`uname -smr`、`head -n 1 /etc/nv_tegra_release`、`cat /proc/device-tree/model`。没有 sudo、服务变更、文件上传、模型部署或硬件控制。SSH 会话 `92286` 当前停在 SoC0 `run@agi:~$`，SoC1 已正常 `exit` 回 Windows 后才登录 SoC0；会话未来是否仍存活须重新核验。

Windows 初始 shell 是 `cmd.exe`：终端输入须用回车 CR 提交。第一次 LF 输入仅回显，已取消自己的未提交行，再以 CR 正常进入第二跳；不能把无提交命令的静默当成连接中断。密码仅通过交互提示提交，不写入记录或命令行。

## 早先未恢复时的历史

用户随后要求忽略转发排查、直接裸连。本轮已原样执行以下命令，没有添加额外参数；客户端最终返回 `Operation timed out` / 255，未出现密码提示，未进入 Windows。下面三个带显式等待参数的尝试只保留为此前检查的历史证据。

```bash
ssh sshuser@100.92.227.62
```

## 目标与范围

按用户新指定的桥接机与现有转发尝试原生交互密码 SSH，只读验证登录和设备身份。本阶段没有变更 SSH 配置、Windows 转发或机器人服务，没有发出硬件命令。

| 入口 | 用户指定目的端 | 本次实际结果 |
|---|---|---|
| `run@100.92.227.62:22040` | SoC0 `10.0.1.40:22` | 等待 90 秒后 `Operation timed out`，退出 255；未出现密码提示 |
| `agi@100.92.227.62:22041` | SoC1 `10.0.1.41:22` | 等待 90 秒后 `Operation timed out`，退出 255；未出现密码提示 |
| `sshuser@100.92.227.62:22` | Windows 桥接机本身 | 等待 90 秒后 `Operation timed out`，退出 255；未出现密码提示 |

没有提交任何密码；未登录 Windows 或 Orin。SoC0 的 SSH 密码是否与其 sudo 密码相同尚未验证。

## 网络与客户端证据

- 本机至 `100.92.227.62` 的路由为 `utun4`。
- Tailscale 本机状态 `Running`，本机和目标 `DESKTOP-LQL43SI` 均显示在线。
- 三次 Tailscale peer ping 全部收到回复，延迟 10、17、13 ms；另一次 ICMP 模式收到 12 ms 回复。
- `ssh -G` 核实实际目标仍为 `100.92.227.62:22041`、用户 `agi`，`ConnectTimeout=90`、`ConnectionAttempts=1`、`ControlMaster=false`，没有 `ProxyCommand` 或 `ProxyJump`。
- 这些证据支持桥接机可响应网络探测，以及本次三个 TCP SSH 入口均连接超时；不能据此断言机器人离线、整个桥接网络中断、密码错误，或判定防火墙/访问策略/监听/转发中的某一项为唯一原因。

复现命令以用户提供的三个端点为准：

```bash
ssh -tt -o ConnectTimeout=90 -o ConnectionAttempts=1 -o ServerAliveInterval=60 -o ServerAliveCountMax=5 -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no -o StrictHostKeyChecking=ask -p 22040 run@100.92.227.62
ssh -tt -o ConnectTimeout=90 -o ConnectionAttempts=1 -o ServerAliveInterval=60 -o ServerAliveCountMax=5 -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no -o StrictHostKeyChecking=ask -p 22041 agi@100.92.227.62
ssh -tt -o ConnectTimeout=90 -o ConnectionAttempts=1 -o ServerAliveInterval=60 -o ServerAliveCountMax=5 -o PreferredAuthentications=password,keyboard-interactive -o PubkeyAuthentication=no -o StrictHostKeyChecking=ask sshuser@100.92.227.62
```

## 恢复步骤

后续连接遵循用户最新要求，使用上面的裸连入口。12:42 已成功恢复，下面 Windows 本地 PowerShell 核对项仅为早先诊断备选，本轮没有运行，也无需为了本次连接另行执行：

```powershell
Get-Service sshd
Get-NetTCPConnection -State Listen -LocalPort 22,22040,22041
netsh interface portproxy show all
```

早先三个 SSH 尝试及裸连失败会话均已结束；当前使用新的成功裸连会话，不探旧句柄。若后续再次连接，沿 Windows 裸连及机器人内网原生 SSH 交互认证，核实时态后继续。

## Done / Todo：用户简版

- Done：Windows 裸连成功，并从 Windows 直接登录两台设备；确认 SoC1 为 Orin NX，SoC0 为 RK3588。
- Todo：后续部署按实际硬件分别准备；本次没有上传或运行控制程序。

## Done / Todo：Agent 恢复版

- Done：早先三个显式等待入口和一次裸连失败均保留为历史。新恢复条件后 Windows 裸连与两次内网 SSH 均密码认证成功；已核相同机器人序列号、账户、内核、设备树，SoC1 实核 L4T R36.4.3；SoC0 非 Jetson。记录 CR 提交 Windows cmd 的差异。凭据不落文件，未变更 SSH 配置、服务或转发。
- Todo：后续先复用当前会话并核时态；模型部署的设备环境核查以 SoC1 Orin NX 为候选，SoC0 依据 RK3588 实际能力区分。保持 `publish_allowed=False`、`hardware_commands=0`；本次仅连接验证，不因登录成功启动训练或电机控制。

机器可读记录：`logs/session_records/x2_bridge_access_20261113.json`。本地 README、索引及当前状态随本阶段提交；GitHub 推送以实际命令及独立远端 SHA 核验为准。Windows/Orin 镜像未尝试。
