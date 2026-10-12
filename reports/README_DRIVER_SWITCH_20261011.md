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
- 用户提供的 sudo 凭据已从记录中删除；其内容不属于版本或实验参数。目标驱动仍为 580.178.04。

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
- 后续 D01 单次普通 CUDA 冒烟在 580.178.04 下完成 Python/Torch 导入、CUDA 初始化、1024 矩阵分配、一次矩阵乘及同步；持久化日志停在 `CHECK_BEGIN`，没有 `CHECK_END`/结果回执。随后确认 boot 从 `560a7f8b-e70c-45a7-b3d7-5388308339cb` 变为 `fbae4671-acc0-470e-8e39-60d8340b5da2`，因此 D01 为 `STOPPED_REBOOT`，不重试。
- 580 驱动下的 D01 仍发生新 boot，且 BIOS 现场明确提示异常供电；这使“电源/主板保护触发”成为当前首要嫌疑。D01 未出现保存的 Xid/MCE/AER 错误，不能据此定位具体器件，也不能把 `CHECK_BEGIN` 后的缺失直接归因于 allclose 或驱动。
- 用户随后提供了现场 BIOS 照片：MSI `E7D31IMS.A10` 明确显示“上次开机时检测到系统电源发生异常供电现象，已启动电源保护机制”。这是主板对供电异常的直接证据；在确认系统恢复前，不继续任何 GPU 负载、不按 F1 修改 BIOS。

## 下一步与恢复入口

1. 现场按任意键继续启动（不进入 F1 Setup）后，已确认新 boot 和持久化 D01 日志；保存前一 boot 内核只读记录，未发现 Xid/MCE/AER/panic。
2. 证据已达到首故障停止条件：停止全部后续 GPU/A3/YOLO 负载，不回退、不重复试跑；优先现场检查 PSU、主板供电、显卡辅助供电线、插接和 PCIe 插槽/供电路径。
3. 若以后需要版本对照，先完成供电链路检查，再由用户明确授权回退或换另一驱动；一次通过/失败都不能单独排除硬件。

## 保存与同步状态

本文件与登记清单在当前提交中保存；大型驱动包、apt 缓存与原始安装日志保持主机 `LOCAL_ONLY`。
