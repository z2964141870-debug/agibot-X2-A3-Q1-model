# X2 PC2 RK3588S 推理与跨板试测

## Done

按用户给定地址从 Mac 有线直连进 PC2。核实为同一 X2 Ultra 的 SoC2，设备树为 Rockchip RK3588S，8 核 A55/A76，约 7.9 GiB 内存，登录时可用约 3.3 GiB。设备现有 Agent 和传感服务继续运行。原系统无 ONNX Runtime 和 RKNN 用户态环境，内核可见 rknpu 及 NPU devfreq。此次仅将已核 ARM64 轮子解进独立目录并使用 CPUExecutionProvider。测试只在两个 A76 核运行、低优先级、限定时长。未安装系统包、修改设备服务或使用 NPU。

| 试测 | P50 / P95 | 峰值 | 超预算 |
| --- | ---: | ---: | ---: |
| FGP S1 CANO，PC2 1 线程，24 次 | 464.28 / 472.84 ms | 478.67 ms | 24 / 24 超 33.33 ms |
| FGP S1 CANO，PC2 2 线程，24 次 | 283.97 / 312.89 ms | 333.32 ms | 24 / 24 超 33.33 ms |
| MorphMimic ONNX，PC2 1 线程，32 次 | 10.98 / 18.85 ms | 31.31 ms | 1 / 32 超 20 ms |
| MorphMimic ONNX，PC2 2 线程，32 次 | 10.21 / 15.54 ms | 18.01 ms | 0 / 32 超 20 ms |
| Orin→PC2 推理→Orin，160 次 | 9.85 / 17.31 ms | 25.47 ms | 3 / 160 超 20 ms |

CANO 使用已保存的 30 帧真实输入张量，对原 CPU 残差最大差 3.73e-7。只测了 CANO，开销已约是 30 Hz 帧预算的 8.5 倍。MorphMimic 根据实录人体与固定合成机器人反馈生成张量，PC2 动作最大差 7.16e-7。真正的跨板测试用 Orin 每 20 ms 计划发送 12,639 B 预计算输入，PC2 做原 human ONNX 推理并返回动作；PC2 前向 P95 为 13.31 ms，按计划发送到返回有 4/160 次超过 20 ms。PC2 测试监听只绑定开发网卡，限制 Orin 对端，完成后退出 0。测试没有新 FGP 并发、没有真实反馈和电机，没有计算在线特征及控制后端。开发网直连 ICMP RTT 平均 0.474 ms（8/8），这个数字不是策略 RPC 时延。

先保留 Orin 分阶段 FGP 与 2 线程策略。PC2 CPU 跑原版 CANO 的路线可以排除；把策略前向放到 PC2 可以工作，但这个短窗口已出现期限超时，尚无实测联跑收益。PC2 NPU 的 ONNX 转换、数值精度和实际节拍需要独立验证。

## To do

板端 FGP 输出和策略运行仍作为当前基线。若需要利用 PC2 改善实时主链，先独立验证 CANO 的 RKNPU 兼容路径，或测量分工方案在同一负载下能否降低 Orin 的出帧长尾。此次没有改变现行控制入口。

PC2 原件在 `/home/agi/x2-pc2-cano.yuqmY8`，Orin 跨板探针原件在 `/home/run/x2-pc2-rpc.rceV5J`，三份 JSON 均和 Mac 核同 SHA-256。

[CANO CPU](../logs/session_records/x2_pc2_cano_cpu_20261116.json) · [PC2 策略 CPU](../logs/session_records/x2_pc2_morph_cpu_20261116.json) · [跨板 RPC](../logs/session_records/x2_pc2_morph_rpc_20261116.json)
