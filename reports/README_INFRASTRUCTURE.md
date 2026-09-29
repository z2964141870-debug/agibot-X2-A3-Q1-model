# 训练服务器与固定目录

用户于 2026-09-29 指定后续训练在 `ssh hp3090` 上运行，项目放在 `/media/yu/FAFF-E9771/YUANQI`。

## 本次实测

| 项目 | 结果 |
| --- | --- |
| 主机 | `yu-MS-7D31`，Linux x86_64 |
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB |
| NVIDIA 驱动 | 595.84 |
| Python | 系统 Python 3.10.12 |
| Git | 2.34.1 |
| 磁盘 | `/dev/nvme0n1p1`，ext4，读写挂载到 `/media/yu/FAFF-E9771` |
| 初始化前剩余空间 | 634,076,459,008 字节，约 634 GB / 590.5 GiB；后续会变化 |
| GitHub | SSH 读取用户指定仓库成功，初始化前无远端 refs |

GPU 已可识别，但未据此声称 CUDA/PyTorch/Isaac 版本组合通过训练验证。

## 既有资源候选

只查看了目录名称，未修改或启动：

- `/home/yu/projects/IsaacLab`、`/home/yu/IsaacLab`
- `/home/yu/projects/gr00t-wbc-x2`、`a2a-x2`、`x2-sonic-sim`、`x2-rl-deploy`
- `/home/yu/miniconda3/envs/x2-sonic-isaaclab`

它们是 P0 的资源核验入口，准确版本、依赖、资产和训练能力仍待确认。不要复制整套旧项目或改动其运行环境来完成目录初始化。

## 百度网盘入口排查

- 已安装官方 Linux 包 `baidunetdisk 8.7.0`。
- 可执行文件：`/opt/baidunetdisk/baidunetdisk`，具备执行权限。
- 桌面入口：`/usr/share/applications/baidunetdisk.desktop`。
- 用户配置目录存在，但未读取认证文件，也未据配置目录推定登录有效。
- 排查时没有发现正在运行的百度网盘进程。
- 存在本地图形会话与 X11 socket；尚未启动客户端检查其界面。
- 未发现 PATH 中的 BaiduPCS-Go、bypy 或 rclone 等命令行入口。

结论：**有官方图形客户端入口，尚未验证可传输；没有已确认的自动化上传方案。** 当前阶段无需新增安装。第一次大型工件备份前，再通过客户端正常登录状态和一个小文件往返确认通路；若 GUI 不满足自动化需求，按用户要求讨论替代方案。

## 日常操作

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
git status --short --branch
python3 script/project_tools/inventory_host.py
```

依赖环境按选定训练框架在后续工作中配置；项目初始化不包含 GPU 作业或机器人连接。
