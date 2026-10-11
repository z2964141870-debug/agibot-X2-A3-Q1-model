# Ultralytics YOLOv10n 普通训练排查 E01

记录日期：2026-10-11。状态：已登记，准备进行一次受保护的普通目标检测训练；不调用 A3、Isaac 或强化学习代码。

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

## 恢复步骤

先读取 `state.json`、`health.jsonl`、`worker_phases.jsonl` 和当前 boot；确认同 boot 且状态正常后才可汇总。首故障状态不可重置。
