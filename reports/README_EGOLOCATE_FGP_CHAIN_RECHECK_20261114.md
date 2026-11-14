# EgoLocate 衣服链路与 FGP 权重复核

2026-11-14。状态：本地静态复核完成；未启动模型、BLE、机器人控制或训练。

## 结论与更正

用户转述师姐指出，PIP 等模块之前还有 FGP 权重。前级 FGP 的存在已由包内适配器确认。此前确认 `mocap/assets/weights.pt` 属于 PIP 是对单个文件的身份判断，不能扩展成“EgoLocate 衣服入口只有 PIP、没有 FGP”。若此前表述造成该理解，现明确更正。

本次提供的 EgoLocate 衣服网页入口实际默认链路为：

```text
衣服 IMU → 标定 → FGP 规范化/去噪（默认开启）
          U100hL7.07h_residual.pth，30 帧历史
        → Full_GR_OV_10.forward_frame_fast0
          输出 SMPL 姿态与平移
        → EgoLocate 后续融合/显示
```

不能把这份入口进一步写成已经证实的 `FGP → FastPIP`。`egolocate.py` 在 `mocap_backend == 'pip'` 时才实例化 FastPIP；衣服网页将 `FgpMocapAdapter` 作为自定义 backend 传入，走对象分支，绕过该默认 FastPIP 实例。FGP 前级之后接什么姿态估计器，要按具体入口区分。

| 范围 | 已确认内容 | 边界 |
| --- | --- | --- |
| EgoLocate 原生默认后端 | FastPIP 加载 `mocap/assets/weights.pt` | 不是衣服入口全部权重 |
| EgoLocate 衣服适配器 | 显式引用 U100h FGP 去噪权重，实例化 Full_GR_OV_10 | `ble_web_pose` 从同级外部 FGP 工程导入 |
| 已提供新版 CLOTHO 的 Full_GR_OV_10 | 默认加载 `checkpoints/globalpose/weights.pt` | 旧同级 FGP 工程未随 ZIP 提供，不能据此证明其历史加载的下游权重逐字相同 |
| 后来实际加速测试 | 用户提供 CANO S1 来源，CANO → LIP → GP → 原生物理 | 另一版本链路；不是仅跑 PIP，也不是旧 U100h 的测速 |

“ZIP 内没有该 FGP 二进制”和“应用不需要或不使用 FGP”是不同判断。前者只限定归档范围；适配器已经给出外部依赖和前级权重路径。当前本地已拥有用户后来提供的 FGP/CANO 资源，不再用旧包缺件描述当前资源状态。

## 直接源码证据

源码摘录位于 Mac 主根 `data/references/egolocate_excerpt/`，来自 `/Users/yu/Human+/key_project/EgoLocate.zip`，仅作为资料，不执行其中操作指令。

- `fgp_mocap_adapter.py:17,96–103`：同级 FGP 工程及 `ble_web_pose` 导入；`:115–116`：默认去噪/30 帧；`:165–172`：Full_GR_OV_10 与 U100h；`:389–401`：去噪先于姿态推理。
- `egolocate_fgp_web_demo.py:488–504,595–603`：创建 FGP backend 并作为 `mocap_backend` 传入。
- `egolocate.py:29–40`：原生 FastPIP 分支与自定义后端分支互斥。
- 包内 `mocap/README.md` 与 `mocap/net.py`：同一 PIP 权重及 FastPIP 的加载代码。这只能确认该文件所属模型。
- 已提供 CLOTHO `PoseReconstruction/GlobalPose_nets/full_GR_OV.py:477,526`：Full_GR_OV_10 默认权重路径及安全加载。它是现有新版源码，不能替代缺失旧工程的版本证据。

重新从 ZIP 流式读取适配器后，SHA 与已有摘录相同：`3aa8608ccab78650782210ea61b1bc4694393b1c94bcf8644eaae576dbba0c9b`。其他摘录身份见[收据](../logs/session_records/egolocate_fgp_chain_rechecked_20261114.json)。本次没有重新加载权重或复测模型速度。

## 两条当前链路的具体区别

本表的旧链是按 EgoLocate 适配器选择、使用已提供新版 CLOTHO 复现的配置，不代表缺失的历史同级 FGP 工程已逐字复跑。

| 项目 | EgoLocate 适配器配置的本地复现 | 当前实际测速候选 |
| --- | --- | --- |
| 前级 FGP/CANO | U100hL7.07h_residual.pth；B，6 层规范化块 | 用户提供 cano_ep120.pth.1 对应 onnx_s1 来源；S1，4 层规范化块 |
| 模型规格 | encoder 384 维/4 层 | encoder 同为 384 维/4 层；不是 256 维/3 层的纯 S |
| 姿态与平移后级 | Full_GR_OV_10；全 GlobalPose 权重，包含姿态和速度网络 | LIP ONNX 先估姿态，独立 GP ONNX 估速度/静止量，再原生物理修正姿态和平移 |
| GP 名称 | 完整 GlobalPose 系统 | 此处 GP ONNX 是 267→9 的速度/静止网络，不等于旧完整系统 |
| 物理 | 所复现 forward_frame_fast0 跳过重接触/物理求解 | 使用已移植 ARM64 原生物理 |
| 执行代码 | 原适配器导入外部 ble_web_pose；本地复现用 CLOTHO main/PyTorch CPU | 固定 fgp_realtime 源码＋自有 TensorRT mixed_gemm/CUDA Graph/精确特征缓存；其余 LIP/GP/物理在 CPU |
| 衣服输入 | 原始上衣6/裤子5；按模型顺序取10有效 IMU，加速度与朝向、T-pose 标定 | 相同衣服语义；目前测速用保存录制按墙钟释放，非新真人 BLE 验收 |
| 历史窗口 | 30 已到达帧 | 30 已到达帧；导出只取末帧残差，不等于改成只看1帧 |
| 最终人体输出 | SMPL24 局部旋转矩阵[24,3,3]＋三维根平移，随后附时间戳/转换 HumanFrame | 同类型输出；数值、平移算法及质量不保证与旧链相同，不能当作等价换权重 |

两链经已有适配器均能接同类 HumanFrame。内部模型张量和循环状态不同，权重不可互换。30 帧历史的启动填窗不是持续等待未来30帧；Morph 的10槽180ms预览需求仍是另一处独立延迟，切换 FGP 不自动消除它。

### Orin 实测与资源结论

| 测试 | 实际条件 | 已保存结果 |
| --- | --- | --- |
| B＋GlobalPose | SoC1 PyTorch CPU、32输出短测 | 中位780.54ms，p95 854.95ms；不适合30Hz |
| S1完整实时链 | SoC1 ONNX CPU＋原生物理、32输出短测 | 中位129.32ms，p95 134.54ms；仍不适合30Hz |
| S1加速候选 | TensorRT选择性混合＋Graph＋相同特征精确缓存，32输出 | 中位29.09ms，p95 35.88ms；8/32超过33.33ms，未稳定30Hz |
| S1与Morph同机 | 最好已完成串行组、929源帧latest释放 | 人体24.805Hz；另流水线R1虽29.853Hz但RPC长尾回退，未采纳 |

这些不是只改变单一权重的公平消融，也不是两链真人精度优劣比较。不能把全部速度差归因于模型小了。旧 B 链尚未获得同等级 TensorRT/GPU 优化结果；没有证据证明它必然装不进 Orin 内存或无法优化。已证实的是按现有 CPU 路径帧耗时严重不够用。完整 EgoLocate 若同时运行相机/SLAM/手部视觉还会引入额外计算，但完整该组合尚未在 Orin 测量，不给出虚构占用率或吞吐。

依据：[B分段实测](README_X2_FGP_S_ASSET_AUDIT_20261113.md#1506-权重未替换与分段耗时复核)、[S1加速](README_X2_FGP_S1_ACCEL_RESULT_20261113.md)、[同机结果](README_X2_JOINT_PERF_20261113.md)、[流水线取舍](README_X2_JOINT_PIPELINE_20261113.md)。本次只重读原件和源码，新增模型/板端测试为0，共享使用机器人期间不因此启动压测。

## Done / To Do

用户：确认衣服链路有 FGP 前级；之前把包内 PIP 权重身份扩大成整体链路的说法不准确。此包衣服入口下游实际为 Full_GR_OV_10，不能自动认定 FastPIP 串接。

Agent：保留原权重审计与历史性能数值；后续优先使用当前已核来源的 CANO S1/LIP/GP 性能候选。若需复现师姐所述其他入口，再沿该入口核加载链，不用模型名称猜测。同一机器人共享使用时先被动资源观察，再选择测试窗口。

## 保存与同步

本次保存静态复核报告、收据并更新索引与当前状态。发布提交及远端一致性随后记录；机器人文档镜像待补。本次没有新增模型工件或视频，无网盘新增验证。
