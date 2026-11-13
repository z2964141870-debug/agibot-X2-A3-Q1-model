# X2 双 Orin 桥接连接实测

记录时间：2026-11-13 12:27 北京时间。状态：网络探测成功，SSH 登录未完成。

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

后续连接遵循用户最新要求，使用上面的裸连入口。Windows 本地 PowerShell 的以下只读核对项仅为早先诊断备选，本轮尚未实际运行：

```powershell
Get-Service sshd
Get-NetTCPConnection -State Listen -LocalPort 22,22040,22041
netsh interface portproxy show all
```

然后核对适用于 Tailscale 接口/配置文件的入站防火墙与访问策略。收到真实的新恢复条件后再尝试用户提供的入口；成功认证后只读核设备身份、系统架构和资源。当前三个 SSH 进程均已结束，无远端作业。

## Done / Todo：用户简版

- Done：按最新要求执行 `ssh sshuser@100.92.227.62` 裸连，收到 `Operation timed out`，仍未进入认证。
- Todo：入口有新的恢复条件后继续同一裸连方式，密码提示出现后交互认证，先核 Windows 身份。

## Done / Todo：Agent 恢复版

- Done：早先三个入口各一次原生 SSH，等待 90 秒，均退出 255；用户随后要求裸连，已另执行一次不带额外参数的 Windows SSH，也退出 255。全部零认证提交、零远端命令。已核实 `utun4` 路由、Tailscale 在线及 peer/ICMP 回复、SSH 无代理及无连接复用。凭据不落文件，不把旧入口规则套用到用户这次新提供的 X2 桥接机。
- Todo：从 Windows 本机监听/服务/转发及规则证据定位具体失败层；未有新条件时不重复认证或换入口。能够登录后先核 SoC0/SoC1 主机名、账户、架构、JetPack/系统版本，继续保持 `publish_allowed=False`、`hardware_commands=0`；训练与模型部署未获本轮执行指令。

机器可读记录：`logs/session_records/x2_bridge_access_20261113.json`。本地 README、索引及当前状态随本阶段提交；GitHub 推送以实际命令及独立远端 SHA 核验为准。Windows/Orin 镜像未尝试。
