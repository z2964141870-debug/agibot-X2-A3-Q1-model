# NVIDIA 驱动切换记录

记录日期：2026-10-11。阶段 / 实验编号：DRIVER_SWITCH_20261011。状态：完成（已切换到 580.178.04，尚未进行负载复测）。

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
- 安装需要 root 权限，已由用户在当前会话提供密码后完成；密码未写入文件、日志或 Git。
- 用户所说“1404”是 sudo 密码，不是目标版本；目标仍按本记录选择 580.178.04。

## 版本、命令与证据

- 主机与工作目录：`hp3090`，`/media/yu/FAFF-E9771/YUANQI_A3`。
- 切换前：Ubuntu 22.04.5，内核 `6.8.0-138-generic`，RTX 3090，NVIDIA 595.84，模块 `/lib/modules/6.8.0-138-generic/kernel/nvidia-595-open/nvidia.ko`。
- 当前 boot ID：`7f9e3cd2-cfcf-4ca0-9493-527fc78a418d`。
- apt 模拟结果：安装 `nvidia-driver-580` 会移除 595 系列并安装 580.178.04 及 DKMS/firmware 依赖；没有发现会删除根文件系统或非 NVIDIA 工作负载的计划。
- 已执行安装（主机终端）：

  ```bash
  cd /media/yu/FAFF-E9771/YUANQI_A3
  sudo apt-get install -y nvidia-driver-580
  sudo systemctl reboot
  ```

  安装期间未启动任何训练；随后执行了一次计划内重启。
- 回退命令（若 580 无法正常加载，需在可用 TTY/SSH 中执行）：

  ```bash
  sudo apt-get install nvidia-driver-595-open=595.84-0ubuntu0.22.04.1
  sudo reboot
  ```

  回退前应确认 `apt-cache policy nvidia-driver-595-open` 仍提供该版本；若源已变化，先保留当前包状态，不强制删除驱动。
- 原始大型包和安装日志保留在主机 apt/dpkg 记录中，不复制到 Mac 或 Git。

## 结果与边界

- 计划内重启后 boot ID 从 `7f9e3cd2-cfcf-4ca0-9493-527fc78a418d` 变为 `560a7f8b-e70c-45a7-b3d7-5388308339cb`；这是驱动切换动作，不计为负载触发故障。
- `nvidia-smi` 已报告 `580.178.04`；`modinfo` 显示 DKMS 模块 `/lib/modules/6.8.0-138-generic/updates/dkms/nvidia.ko`，`dkms status` 为 installed。
- 新系统已启动，GPU 29°C、23.28W、0%利用率、77MiB显存；未发现 A3/YOLO/Isaac/PyTorch 计算进程，A3 自动服务仍 inactive/disabled。
- 切换后只完成空闲核验，尚未在 580 上重跑会导致重启的 GPU/YOLO 工作负载；因此目前只能确认驱动切换成功，不能宣称根因已消失。

## 下一步与恢复入口

1. 在 580.178.04 上先做单次、低风险普通 CUDA 版本核验；不恢复 A3。
2. 只有单次核验通过，才按已登记的首故障停止规则逐步重跑普通 GPU/YOLO 对照。
3. 若新驱动仍复现重启，回退到 595.84 只作为软件版本对照，不把一次通过/失败解释为硬件已排除。
4. 继续前重读本文件、`reports/README_CURRENT_STATE.md`、`data/manifests/driver_switch_registration_20261011.json`。

## 保存与同步状态

本文件与登记清单在当前提交中保存；大型驱动包、apt 缓存与原始安装日志保持主机 `LOCAL_ONLY`。
