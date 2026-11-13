# X2 FGP CANO TensorRT 加速

记录日期：2026-11-13。阶段：CANO_TRT_R1。当前为独立方案与 CPU 合同准备完成，实际 GPU 构建/推理待执行。后续实测在本文件追加。

## 目标与边界

用户要求继续检查 FGP 加速可行性，并说明第二、第三终端工作已完成。本轮只处理第一终端的人体推理；不修改已完成后端，不启动 BLE/新标定/反馈订阅/电机/MC，不恢复 Q1 训练。`publish_allowed=False`、`hardware_commands=0`。

上一阶段 ARM64 完整链已通但 S1 中位 129.324 ms，CANO 稳态中位 112.741 ms。选用作者原样 `onnx_s1/cano.onnx`，只替换 CANO 的执行器为 TensorRT；LIP、GP、物理、标定、30 帧历史、单步 Euler、EMA 与失联处理保留。原作者源码、模型和 `script/x2_fgp_realtime` 基线入口均不编辑。独立适配器只在本次进程中替换 flow model 工厂，并在 finally 恢复。

## 已核设备条件

Mac 原生有线 SSH 直达 SoC1 `10.0.1.41`；boot `8f960fd3-d5da-4630-b5de-4d61d7251901`。设备已安装 TensorRT 10.7.0.23/CUDA 12.6.68，原 `fgp-preview-py310` 环境通过本进程追加 OS TensorRT 包路径能导入 10.7.0，不更换 numpy/torch 或安装新 GPU runtime。

16:44 的实际可用内存约 13.05 GB/总 16.42 GB，swap 使用 0，GPU 约 63 C，CPU/Tj 约 65 C。`trtexec --help` 实读了 `--noTF32`、`--fp16`、workspace、builderOptimizationLevel、maxAuxStreams 及 skipInference 选项。它们是已装版本的能力，不是尚未运行的加速结果。

## 有界实验与门

1. 原样 ONNX 分别构建 FP32（关闭 TF32）和允许 FP16 的两个独立 engine。每次新输出目录，最多 600 秒，workspace 256 MiB，优化级别 2，辅助流 0。workspace 不是全部内存上限；监视真实 MemAvailable，低于 4 GiB或温度达到 80 C 时只终止本次自有子进程组。至少 10 GiB余盘，项目专用非阻塞锁只协调本项目加速任务，不冒称全局 GPU 锁。
2. 每套沿用现有 61 帧夹具，29 帧历史/32 输出，最多 32 次 GPU CANO；CPU LIP/GP/物理原样执行。计时含 CPU→GPU→CPU 同步传输和保存 CANO 输入的开销，初始化单列。外层 120 秒保护。
3. 完成计时后，才对同样的 32 个实际 CANO 输入各跑一次 CPU 参考，避免 CPU 比较污染 GPU 时间。全链与已保存同夹具 `full_s1_t4_r1` 输出比较，不重跑整个旧 CPU 物理链。
4. 数值筛查限值预先固定：FP32 residual 最大绝对差 `2e-4`、相对 L2 `1e-3`、全链最大关节角差 0.5 度、根平移最大差 5 mm；FP16 分别为 `5e-3`、0.02、1 度、10 mm。所有值有限且 32 帧完整；超阈值保留失败，不据速度选择。限值是工程一致性筛查，不是真人精度或硬件安全认证。
5. 分列全部输出与去前两次中位/p95/最大、超 33.33 ms 帧数，以及 CANO/pose/物理。30 Hz 只能按整链判断；30 帧历史是启动历史，不是持续未来等待。无 BLE/网页/网络/机器人延迟，不能冒充端到端。

本轮上限为两个 engine、64 次 GPU CANO、64 次 CPU CANO 对照，0 训练更新。失败的新根保留；不扩大预算补跑，不进行 INT8/稀疏权重改写、全局 cache 清理或系统功耗/频率变更。若构建不支持算子，先记录具体 parser 错误，不静默跳层或更改模型。

## 自有代码与验证

新增 `script/x2_fgp_accel/` 五文件：有界构建、静态 TensorRT/CUDA session、完整链 wrapper、六项 CPU 合同与包入口。使用已安装 TensorRT 核心，不自行实现神经网络算子；CUDA 内存/流只属于本进程。CPU 合同在既有 `q1-teleop-mps-20261106` 环境六项通过，Python 编译与 diff 检查通过。一次命令使用了不存在的短环境路径，未运行测试；随后使用实际环境通过，不能把路径错误算成合同失败。

准备记录：[x2_fgp_trt_prepared_20261113.json](../logs/session_records/x2_fgp_trt_prepared_20261113.json)。基线：[ARM64 移植与原始报告](README_X2_FGP_REALTIME_ARM64_20261113.md)。重要来源、自有代码先审计/发布，再部署原运行时合同及资源准入，之后依防重输出单次执行。

设备项目根 `/home/agi/x2_morph_human_20261113`，独立输出根 `data/analysis/X2_FGP_TRT_ACCEL_20261113`。构建入口：

```bash
env -u PYTHONPATH -u PYTHONHOME \
  data/environments/fgp-preview-py310/bin/python -m script.x2_fgp_accel.build \
  --onnx data/runtime/clotho_fgp_realtime_20261113/models/onnx_s1/cano.onnx \
  --precision fp32 \
  --output data/analysis/X2_FGP_TRT_ACCEL_20261113/build_fp32_r1
```

真实部署/构建/推理结果及后续发布收据待执行后登记，不以准备完成代表加速通过。
