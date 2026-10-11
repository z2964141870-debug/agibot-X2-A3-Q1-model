# NVIDIA 驱动切换准备与阻塞记录

记录日期：2026-10-11。阶段 / 实验编号：DRIVER_SWITCH_20261011。状态：阻塞（目标“1404”未对应可安装的 NVIDIA 包，且安装需要主机管理员密码）。

## 目标与范围

本阶段只改变 NVIDIA 驱动版本，用于区分当前 595.84 驱动分支与此前普通 CUDA/PyTorch/YOLO 重启现象的关系。A3、强化学习、Isaac、YOLO 和任何 GPU 训练在驱动切换期间均保持停止；不改 PCIe、供电、功率限制、BIOS 或硬件设置。

## 完成内容

- 只读核验主机、内核、GPU、驱动包和自动恢复状态。
- 确认当前是 Ubuntu 官方包 `nvidia-driver-595-open`，不是无法追踪的 `.run` 安装。
- 评估 apt 候选：`nvidia-driver-550` 在当前 Jammy 源中解析为 580.178.04 运行包；因此计划明确切换到 `nvidia-driver-580`，避免命令名与实际版本不一致。
- 生成一次性切换登记，包含回退命令和切换前证据；尚未执行安装或重启。

## 关键决定及依据

- 选择 580.178.04：当前系统可直接从官方源获取整套包，能替换 595.84 的用户态库、内核模块和固件，且可回退到已安装的 595.84。直接使用 580 包名比 550 元包更可复现。
- 不选择直接修改驱动参数或功率：这些无法区分软件分支，且不属于用户本轮授权范围。
- 安装需要 root 权限。远端 `sudo -n true` 返回需要密码；不在聊天或日志中收集密码，因此本阶段在执行前暂停。
- 用户补充“1404”，但当前 Jammy 软件源没有名称或版本包含 `1404` 的包；它可能是 Ubuntu 14.04，也可能是驱动版本的简称。Ubuntu 14.04 不能在现有 Ubuntu 22.04/6.8 内核上作为一次驱动切换直接安装，因此未猜测执行。

## 版本、命令与证据

- 主机与工作目录：`hp3090`，`/media/yu/FAFF-E9771/YUANQI_A3`。
- 切换前：Ubuntu 22.04.5，内核 `6.8.0-138-generic`，RTX 3090，NVIDIA 595.84，模块 `/lib/modules/6.8.0-138-generic/kernel/nvidia-595-open/nvidia.ko`。
- 当前 boot ID：`7f9e3cd2-cfcf-4ca0-9493-527fc78a418d`。
- apt 模拟结果：安装 `nvidia-driver-580` 会移除 595 系列并安装 580.178.04 及 DKMS/firmware 依赖；没有发现会删除根文件系统或非 NVIDIA 工作负载的计划。
- 待管理员执行（主机终端）：

  ```bash
  cd /media/yu/FAFF-E9771/YUANQI_A3
  sudo apt-get install nvidia-driver-580
  sudo reboot
  ```

  安装结束后保持不启动任何训练，等待本任务核验。
- 回退命令（若 580 无法正常加载，需在可用 TTY/SSH 中执行）：

  ```bash
  sudo apt-get install nvidia-driver-595-open=595.84-0ubuntu0.22.04.1
  sudo reboot
  ```

  回退前应确认 `apt-cache policy nvidia-driver-595-open` 仍提供该版本；若源已变化，先保留当前包状态，不强制删除驱动。
- 原始大型包和安装日志保留在主机 apt/dpkg 记录中，不复制到 Mac 或 Git。

## 结果与边界

- 当前只完成盘点和可回退方案登记，**没有声称驱动已经切换**。
- 在管理员执行命令前，不能比较 595 与 580 的重启行为。
- 驱动切换本身会触发一次计划内重启；该重启不应计为负载触发故障。切换后首先只做版本、模块、boot、服务和 GPU 空闲核验。

## 下一步与恢复入口

1. 若目标是 580.178.04，在 `hp3090` 本地终端执行上面的安装和一次重启命令；若“1404”另有所指，先给出完整包名或版本。
2. 用户通知“已重启”或主机恢复 SSH 后，核验新 boot、`nvidia-smi`、`modinfo nvidia`、DKMS、GPU 温度/利用率及所有 A3 自动服务仍 disabled/inactive。
3. 先运行只读/低风险对照，再决定是否允许任何普通 GPU 测试；首个新 boot 或计算错误仍停止，不自动恢复 A3。
4. 继续前重读本文件、`reports/README_CURRENT_STATE.md`、`data/manifests/driver_switch_registration_20261011.json`。

## 保存与同步状态

本文件与登记清单在当前提交中保存；大型驱动包、apt 缓存与原始安装日志保持主机 `LOCAL_ONLY`。
