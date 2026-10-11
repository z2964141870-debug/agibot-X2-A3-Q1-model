# Ultralytics YOLOv10n 普通训练排查 E01

记录日期：2026-10-11。状态：首故障为确认新 boot，训练已停止；不调用 A3、Isaac 或强化学习代码。

## 目标

使用 Ultralytics 8.4.175 的 YOLOv10n 架构，在 RTX 3090 上以 `batch=32` 做一轮普通视觉训练，观察它是否也触发主机重启。该试验用于硬件/驱动/训练链路排查，不用于评价检测精度。

## 固定配置

- 模型：Ultralytics 8.4.175 内置 `yolov10n.yaml`，随机初始化，不依赖 A3 权重。
- 数据：1800 张训练图、200 张验证图、3 类彩色几何目标；由固定种子脚本生成。服务器没有现成检测数据，COCO128 下载地址在当前网络中返回异常，因此不把下载失败伪装成完成。
- `epochs=20`、`batch=32`、`imgsz=640`、`workers=4`、`amp=False`、`device=0`、`cache=False`、每 epoch 保存。
- 监测：独立标准库 supervisor 每5秒记录 boot、CPU/GPU温度、GPU功耗/利用率/显存、可用RAM和训练进程；CPU90°C/GPU85°C连续两次或可用RAM低于4GiB立即停止。新 boot、telemetry 不可读、worker 错误或超时均首故障停止，不自动恢复。

## 代码与产物

- `script/diagnostics/make_synthetic_yolo.py`
- `script/diagnostics/yolo_train_worker.py`
- `script/diagnostics/yolo_train_probe.py`
- 数据目录：`data/datasets/yolo_probe_20261011_E01/`
- 训练输出：`data/experiments/yolo_probe_20261011_E01/ultralytics/`
- 监测日志：`logs/yolo_probe_20261011_E01/`
- 注册清单：`data/manifests/yolo_probe_registration_20261011_E01.json`

## 解释边界

数据由合成目标组成且存在固定类别分布，不能用于宣称 YOLO 的实际效果；本轮只比较训练链路是否在普通神经网络负载下稳定。若出现新 boot，立即停止并记录最后持久化阶段，不继续训练、不恢复 A3。

## 实际结果

训练于北京时间约19:35:34启动，初始 boot 为 `c99ddfa9-8478-441a-ab73-a6495ef0feb3`。worker 依次完成 Ultralytics 导入、YOLOv10n 模型构建，并写入 `TRAIN_BEGIN`；没有开始任何 epoch，也没有生成 worker 结果。重连后确认 boot 变为 `7f9e3cd2-cfcf-4ca0-9493-527fc78a418d`，估计重启约在 UTC 11:36:09（只能作为区间估计）。

重启前两次采样的 CPU 温度为40–47°C、GPU温度30–33°C、GPU利用率均为0%、可用内存约30.8–31.3GB，因此没有观测到本次热保护或显存压力。重启后 GPU 进程为空；旧 boot 的内核过滤记录只包含启动阶段日志，没有保存到 Xid/MCE/AER/panic。由于 supervisor 随主机一起中断，最后边界只能定位在 `TRAIN_BEGIN` 之后、第一 epoch 之前，不能进一步指向具体 Ultralytics/CUDA 调用。

这次结果与之前普通矩阵循环的证据叠加后，说明问题已经不局限于 A3 训练代码；普通 YOLO 训练初始化也能触发重启。但它仍不能证明 Ultralytics 或 YOLO 模型是根因，下一步应停止计算并优先检查硬件、PCIe、供电、驱动和更完整的崩溃留存机制。

## 恢复步骤

先读取 `state.json`、`health.jsonl`、`worker_phases.jsonl` 和当前 boot；确认同 boot 且状态正常后才可汇总。首故障状态不可重置。
