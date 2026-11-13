# CLOTHO FGP 实时版 ARM64 移植与完整链预测试

记录日期：2026-11-13。阶段：ARM64_NATIVE_R1。状态：原生编译、数值不变量、两套模型录制输入全链、原始结果回传与来源复核通过；实时性能未达标。

## 结论与范围

x64 到 ARM64 的兼容问题可以在本项目中推进，不必等待作者提供整套 Orin 版本。本次已经在 X2 SoC1 的 ARM64/Python 3.10 上编译原生 carticulate，并运行作者原有 CANO→LIP→GP→double-tracking physics 完整链，产生 SMPL24 姿态和原生根平移。没有重写动力学方程，也没有用零平移或其他优化器替代作者的物理。

兼容通过不等于实时通过：4 线程完整链中位约 129–132 ms/帧，全部 32 个输出均超过 30 Hz 的 33.33 ms 预算。当前主要性能问题在 CANO 去噪；尚不能直接作为实时衣服遥操前端。

本轮仅使用现有 61 帧录制 IMU 夹具，29 帧建立历史后输出 32 帧。没有新 BLE 连接、真人标定、机器人反馈订阅、SDK/MC/ROS 操作或电机输出；`publish_allowed=False`、`hardware_commands=0`。原默认引擎没有切换，旧 Q1 训练保持暂停。

## 资源已到齐

用户提供 `/Users/yu/Downloads/fgp_realtime_weights.zip`，271,917,294 B，SHA256 `c5994a8cac85550539526408259ce3eb2884450a08953f75c8cc3d220d69df9d`。

Mac 与 SoC1 均完成审计，24 个文件与固定作者 `deployment_assets.json` 的逐件大小和 SHA 一致，解包总计 295,951,921 B。完整 ZIP 经 Mac 有线直连 SCP 到 SoC1，安装在独立 `data/runtime/clotho_fgp_realtime_20261113`，旧运行目录保留。此前“缺新版资源包”是历史状态，已被本次到包与审计结果取代。

两套目录 `onnx_s1`、`onnx_cano_s_ep120_fast` 的包内 manifest 均声明 **S1**、float32、opset17、30 帧历史、`cano_output=latest`。不能按第二个目录名把它称为另一套纯 S 模型。

保留作者训练来源警告：`onnx_s1` 的 GP 所记 LIP 来源哈希与当前 LIP manifest 不同；`onnx_cano_s_ep120_fast` 的 GP 所记 CANO 来源哈希不同。两目录的 LIP ONNX 二进制相同。成功执行不能证明其组合精度或训练来源完全一致。

## ARM64 原生库

| 项目 | 固定来源或实测 |
| --- | --- |
| CLOTHO | `humanplus-ai/CLOTHO-mocap`，`fgp_realtime/1e719140b43a2ec4dc22f4159014b2eb9de51ed9` |
| carticulate | `Xinyu-Yi/carticulate`，`ce5b20464f580f81033ed063fdf3fba7f7a05346` |
| Sophus 1.22.10 | `strasdat/Sophus`，`de0f8d3d92bf776271e16de56d1803940ebccab9` |
| 设备 | X2 SoC1，Orin NX，Ubuntu 22.04，aarch64，boot `39cb8e5a-9c22-451e-a3ba-23c98bebccd2` |
| 编译 | GCC 11.4 / C++17 / Eigen / fmt / pybind11 2.13.6，双编译作业，43.047 秒 |
| 模块 | `carticulate.cpython-310-aarch64-linux-gnu.so`，421,920 B |
| 模块 SHA256 | `2d75a14be1bf5a79aa09d34ce99d9540e79823adf05a33c2c6399a6eb099c08d` |

构建使用上游原 dynamics、kinematics 实现文件；从上游绑定机械选出 KinematicArmature/KinematicModel、DynamicArmature/DynamicModel 绑定，省略本链不使用的 optimizer/ESKF，避免引入 g2o。不是原库所有功能的完整移植。

原 Windows `.pyd` 的精确源码版本尚未知，**没有建立 Windows 与 ARM64 的逐值等价**。若要认证作者发布二进制的同版等价，需要作者确认其源码提交或提供 Windows 参考输出；这不阻止目前的 ARM64 运行与进一步优化。

实际数值检查使用包内 SMPL 动力学缓存、固定非零姿态/速度及五个关节，通过状态往返、质量矩阵对称/正定、正逆动力学、Jacobian 速度与短积分有限差分：

- 质量矩阵对称最大差 `2.8610e-6`，最小特征值 `1.70697e-4`。
- 逆动力学残差 `5.72447e-5`，正逆加速度最大差 `1.84216e-5`。
- Jacobian 速度最大差 `1.49012e-8`，短积分速度最大差 `7.57836e-5`。

这些是一个确定性状态的数值不变量检查，不是人体精度、接触稳定性或 Windows 参考等价验收。

## 录制输入实测

固定夹具 `data/runtime/x2-fgp-viewer/recorded_fixture.npz`，SHA256 `0b30bd434ce8d7326b1a866de7c96109a8a0a6ebcb3f8c6283477b47d6c0f34d`；`raw=(61,10,7)`，是已选择好的十路 IMU。复用已保存标定，不新标定。夹具中旧姿态/根平移不是新版本的 GT。

以下为本次原生 SSH 实际返回、随后已回传 Mac 并核 SHA 的原始报告摘要：

| 模式 | 线程 | 输出 | 全输出中位 / p95 / 最大 ms | 超 33.33 ms |
| --- | ---: | ---: | --- | ---: |
| S1 神经阶段，无物理根输出 | 1 | 32 | 390.142 / 392.215 / 393.066 | 32/32 |
| S1 神经阶段，无物理根输出 | 2 | 32 | 209.955 / 212.663 / 213.803 | 32/32 |
| S1 完整链与原生根平移 | 4 | 32 | 129.324 / 134.535 / 139.025 | 32/32 |
| cano_s_ep120_fast 完整链与原生根平移 | 4 | 32 | 132.481 / 137.488 / 155.945 | 32/32 |

两个完整链均通过有限性和 24 关节旋转有效性检查，状态输出为 tracking。去前两次后 S1 中位/p95 为 129.288/132.843 ms，另一套为 132.481/137.659 ms，仍全部超预算。GP 神经网络在各组固定 1 线程。

去前两次后，S1 作者 exclusive profiler 的 CANO/pose/外层 physics 中位分别为 112.741/2.316/12.819 ms，内部原生物理中位 6.470 ms；另一套分别为 115.711/2.326/12.768 ms、内部物理 6.344 ms。外层 physics 包含 GP 特征/网络及内部物理，内外层计时嵌套，不能相加。这定位了当前运行的主要耗时，不证明频率、功耗或线程调度的单因。

计时为不按墙钟节拍运行的短录制测试，不含 BLE、网页、网络、MorphMimic 窗口或机器人执行延迟。它证明模型与原生物理能在 ARM64 同进程运行，不证明真人重建质量、30 Hz 实时性或硬件端到端达标。

首个 `neural_s1_t2_r1` 因本项目 probe 把十路夹具误写为十一路而在模型创建前失败；保留原目录。修正输入检查后使用独立 `neural_s1_t2_r2`，不是作者模型失败。

## 代码、命令与证据

新增自有代码 `script/x2_fgp_realtime/`：资产核验/安装、原生构建、数值检查、录制输入 probe、最终来源复核 helper。Mac 六个资产边界检查在 0.003 秒内通过，Python 编译与 `git diff --check` 通过。恢复后先证 helper 未传到，才补传并真实执行 `verify_stage.py`，返回 `ARM64_STAGE_INTEGRITY_PASSED`：CLOTHO 96、carticulate 46、Sophus 105 原始文件 SHA 全同，24 个已安装模型/资产 SHA 全同，安装 `.so` 与成功构建一致。

设备项目根 `/home/agi/x2_morph_human_20261113`，独立环境 `data/environments/fgp-preview-py310`。ONNXRuntime 1.19.2 为 CPUExecutionProvider；原 numpy 1.26.4/torch 2.5.1 CPU 未替换。新增本项目环境依赖 pybind11 2.13.6、osqp 0.6.7.post3、qdldl 0.1.9.post1；没有安装全局 SDK 或修改服务。

可复现入口如下；必须使用新的输出路径，已有通过结果无需重跑：

```bash
env -u PYTHONPATH -u PYTHONHOME \
  data/environments/fgp-preview-py310/bin/python \
  -m script.x2_fgp_realtime.build_native \
  --cart-source "$PWD/data/runtime/carticulate_upstream_ce5b2046" \
  --sophus-source "$PWD/data/runtime/sophus_de0f8d3d" \
  --output data/analysis/X2_FGP_REALTIME_20261113/native_build_NEW

timeout 120 env -u PYTHONPATH -u PYTHONHOME \
  data/environments/fgp-preview-py310/bin/python \
  -m script.x2_fgp_realtime.probe \
  --source-root data/runtime/clotho_fgp_realtime_20261113 \
  --fixture data/runtime/x2-fgp-viewer/recorded_fixture.npz \
  --bundle onnx_s1 --threads 4 --full \
  --output data/analysis/X2_FGP_REALTIME_20261113/full_s1_NEW
```

实际设备证据根 `data/analysis/X2_FGP_REALTIME_20261113/`：

- `asset_audit_orin.json`；Mac 对应为 `asset_audit_mac.json`。
- `native_build_r1/build_receipt.json`、`native_invariants_r1.json`、`native_build_r1/module/`。
- `neural_s1_t2_r2/`、`neural_s1_t1_r1/`、`full_s1_t4_r1/`、`full_cano_t4_r1/` 中的 `report.json` 和 `outputs.npz`。
- 终端实录摘要：[x2_fgp_realtime_arm64_20261113.json](../logs/session_records/x2_fgp_realtime_arm64_20261113.json)。
- 原样来源/工件复核：[x2_fgp_realtime_arm64_integrity_20261113.json](../logs/session_records/x2_fgp_realtime_arm64_integrity_20261113.json)。

编译产物已安装至作者 runtime 的 `carticulate/`，是新增 Linux `.so`；没有替换 Windows `.pyd` 或原始 24 资产。测试后全套来源复核也通过。

## 连接与保存状态

完整链返回后，取证期间 SCP 报 `No route to host`，原 SSH 会话报 `Operation timed out` 并退出。Mac 当时 `en7` 的 `10.0.1.10/24` 仍在，但 `status: inactive`、`media: autoselect (none)`；到 SoC1 的路由仍指向 en7，并有 REJECT。这里只记录该次有线链路状态，不据此推断机器人关机或所有网络不可达。

一次原生 `ssh agi@10.0.1.41` 恢复成功，en7 回到 active/100baseTX，SoC1 的 boot 已变为 `8f960fd3-d5da-4630-b5de-4d61d7251901`，登录横幅显示刚重启。重启原因未经核实；本项目没有发出重启命令。此前推理结果属于原 boot，不能写作重启后再测通过。恢复后核同 SoC1 身份与全部已有原件，只补来源复核及取证，没有重复已完成模型测试。未使用 Windows、替代端口、代理或免密认证。

结果包 `data/deliveries/X2_FGP_REALTIME_ARM64_RESULTS_20261113.tgz`，321,650 B，SHA256 `e19ed1527ea5a313f328e8fe6bfd950c4442da26edc771affd0c8a4eab12934f`，13 个文件（另五个目录），包含原始 JSON/NPZ 与 ARM64 模块。原生 SCP 回传后包 SHA 同，Mac 按 receipt 复核 20 项共 610,641 B，包含八份自有源码，逐件大小 SHA 全同。原始结果已在 Mac `data/analysis/X2_FGP_REALTIME_20261113` 和设备保留，未将模型/二进制提交 Git。

本阶段 README、自有代码与摘要保存 Mac；完整原始结果与模型包为 `MAC_AND_DEVICE`，未做新的网盘备份。阶段 Git 发布以 Git 历史及后续 publication 收据为准；设备新增文档镜像待补。

## 下一步

围绕 CANO 做独立 GPU/TensorRT 或其他执行优化，保留同录制输出差及整体长尾门，不重跑本阶段已过组、不承诺未经测试的加速效果。若需 Windows 二进制同版认证，再向作者确认 carticulate 提交及两套 GP 的训练来源配对。之后才接现有 Bleak、HumanFrame 与网页的只读真人验证，当前无电机测试。
