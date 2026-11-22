# humanplus 主机只读探测

日期：2026-11-22。目标地址 `100.101.30.68`，本轮只通过 SSH 读取系统信息，没有上传文件、安装软件、修改配置或启动 Q1 控制。

## 目标主机

| 项目 | 只读结果 |
| --- | --- |
| 主机名 | `Legion-Y9000P` |
| 系统 | Ubuntu 24.04.5 LTS |
| 内核 | Linux 7.0.0-42-generic x86_64 |
| CPU | 24 个逻辑 CPU |
| 内存 | 30 GiB，总可用约 16 GiB |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB，总显存；探测时使用约 1,426 MiB，温度约 47°C |
| 驱动 | 595.99.02 |
| 磁盘 | 根分区 937G，已用 881G，可用约 8.9G，使用率 100% |
| 网络 | Tailscale `100.101.30.68`，SSH 端口可达 |

磁盘空间是当前最直接的迁移阻塞条件。继续复制模型、环境或录制数据前需要先清理或扩容。

## 软件与现有进程

- `/opt/ros/jazzy` 已存在，目标机使用 ROS 2 Jazzy。
- 默认 shell 环境的 Python 为 Anaconda Python 3.13.9。
- `/usr/bin/python3` 为 Python 3.12.3。
- 当前运行了 HumanPlus 相关 teleop 窗口、相机、椅子检查和 FGP worker 进程。
- 已发现现有环境目录 `assist_teleop_migration_20261027` 和 `assist_teleop_baseline_20261111`，其中包含 Sonic、FGP 等既有运行环境。

## 与当前 Q1 部署环境的差异

当前 Q1 3090 运行路径使用 Ubuntu 22.04、ROS 2 Humble、Python 3.10 和 Q1 候选 bundle。humanplus 主机的 Ubuntu 24.04、ROS 2 Jazzy 和 Python 3.12/3.13 不能直接视为兼容环境。目标机本轮没有验证 `aimdk_msgs`、Q1 HAL 话题、Q1 bundle、ONNX Runtime 版本或 MuJoCo Q1 资产。

RTX 5060 Laptop GPU 的显存和当前空闲资源支持做受控的 CPU/GPU 仿真可行性检查。完整迁移还需要先完成依赖隔离、磁盘清理、ROS 消息包核验和 Q1 资产清单核验。

## 本轮边界

- 没有复制 Q1 项目、模型或数据。
- 没有安装依赖或修改 shell、ROS、GPU 和网络配置。
- 没有启动 Q1 仿真、真机桥接或电机控制。
- 没有读取密码以外的凭据、私钥或用户数据。

本报告只记录目标主机探测结果，不代表迁移已经开始或目标机已经具备 Q1 运行条件。
