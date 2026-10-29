# MorphMimic原版G1工程复现结果

2026-10-29，北京时间。阶段MM05。训练、原版ONNX导出及三固定片初末回放已执行；43小工件5600206字节Mac/25386逐大小及SHA一致，六联系图及六格曲线已目视。[实测与交付收据](../data/manifests/morphmimic_g1_results_20261029.json)，最终Git发布另见publication收据。

原版工程链路已跑通，稳定步态没有复现。600轮中前500轮仅更新世界模型，实际actor只有100次更新；初末三片均提前失败。不能把进程完成、有限loss、ONNX一致性或失败时间延后称为稳定走路，也不能将本结果作为Q1已改善的证据。

## 已执行范围

- Windows与25386转发已恢复；原boot/GPU，RTX3090/24GiB，未重启容器或旧实验。
- 上游固定`aa40fe46a7fa995ba7a7354ad065a2c8ad64a302`，183个主仓blob/140372650字节核验一致；GMR固定gitlink1938个blob已在部署阶段核过。使用源码归档及原锁文件，不是完整递归Git历史；没有安装真机SDK。
- 独立Python3.11.18环境，原`uv.lock`安装149包；既有HP环境未改。
- 128条原始G1 CSV通过原版转换与过滤，原seed0 tiny为50片/363.72秒。不是Q1重定向数据或论文全库。
- 单G1、64环境、WM/imag batch128、H32、原500轮WM预热、总600轮、compile none。原网络宽度、奖励和控制保持；未追加PPO或残差训练。

训练运行冻结`c7f049965d1841e5ee556637695ea2494fda5bb1`，启动前10文件319374字节三端SHA和独立远端已核。后续报告及交付工具提交不改变训练冻结或原版源码。[准备/来源](README_MORPHMIMIC_REPRO_20261029.md)、[启动收据](../data/manifests/morphmimic_g1_launch_20261029.json)。

## 训练与部署核验

13:47:33启动，14:08:52训练结束，14:08:55 guard正常回收0。原始iteration同步累计1249.131082秒；全进程3500秒硬限另含初始化和保存，不混作实训时间。600轮实际WM/policy/critic更新6000/100/2000，与最终checkpoint优化器状态一致；初始checkpoint三优化器状态为空。初末网络张量有限且策略张量确有变化。TensorBoard60窗口共96000训练采集转移，不含初始化预热。

Torch峰值allocated2079580672字节，非整个GPU显存峰值。训练与补交结束后的进程/GPU/OOM终态以最终交付收据现场核验为准。未恢复、重试或延长训练。

原版asset exporter的场景模型一致性通过，IMU/FK最大误差`5.16e-08`。初末ONNX分别用64个随机输入通过原版PyTorch parity，最大动作误差约`1.3e-07`和`1.1e-06`，两组SHA256SUMS均通过。使用motion接口、相同导出资产/初态及原版失败条件；checkpoint/ONNX/大数据仅留服务器。

## 三固定片结果

片名及初末版本在观察回放前固定，均来自训练tiny子集。这里没有独立验证集或重复种子结论。失败时间是失败帧号除以50；视频另含第0帧。

| 动作 | 参考帧数/时长 | 初始失败帧/时间 | 600轮失败帧/时间 | 完整走完 |
| --- | --- | --- | --- | --- |
| 前走loop A022 | 1032 / 20.64秒 | 46 / 0.92秒 | 65 / 1.30秒 | 初末均否 |
| 起走A023 | 152 / 3.04秒 | 43 / 0.86秒 | 57 / 1.14秒 | 初末均否 |
| 停走A026 | 253 / 5.06秒 | 44 / 0.88秒 | 116 / 2.32秒 | 初末均否 |

初末完整完成率均0/3。最终失败稍晚不等于自然换脚、脚底离地或承重滑移达标。原版摘要的关节/身体误差取各自不同长度的运行前缀，没有额外重跑共同前缀，不据这些均值声称净跟踪收益。

## 本地审阅

Mac根：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029`。

| 内容 | 文件 |
| --- | --- |
| 前走初始 | [MP4](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_review/it0_walk_forward_loop_001__A022.mp4) |
| 前走600轮 | [MP4](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_review/it600_walk_forward_loop_001__A022.mp4) |
| 起走初始 | [MP4](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_review/it0_walk_forward_start_002__A023.mp4) |
| 起走600轮 | [MP4](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_review/it600_walk_forward_start_002__A023.mp4) |
| 停走初始 | [MP4](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_review/it0_walk_forward_stop_005__A026.mp4) |
| 停走600轮 | [MP4](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_delivery/it600_walk_forward_stop_005__A026.mp4) |
| 六格训练曲线 | [PNG](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_delivery/curves/training.png) / [PDF](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_delivery/curves/training.pdf) |
| 原始指标 | [60行CSV](../data/supermimic_for_q1/MORPHMIMIC_G1_ORIGINAL_20261029/deployment_delivery/curves/metrics.csv) |

六段均1280x720、50fps原速，共377帧/7.54秒编码时长。短片如实保留失败，不补hold帧、不换片或拼成连续成功展示。每片有五帧联系图、原版JSON摘要及日志；逐帧解码的帧数与失败帧加一完全一致。

## 交付异常及范围

第一次最终停走回放在约第106步仍推进时，被交付脚本180秒进程上限中断，未生成完整视频或策略失败摘要。原`deployment_review/progress.json`的FAILED/TimeoutExpired及日志保留，没有改写为成功。单独在`deployment_delivery/`按原始初态重做这一片，600秒仿真/渲染硬限，14:27:21至14:30:37正常退出，实际第116帧触发原版策略失败。其他五片没有重复。还保留一次补交入口因父目录缺失而在worker启动前退出的限制；目录补齐后才实际启动独立回放。这些不是训练重启。

后续交付工具按参考帧数给CPU仿真与1280x720渲染设置时限，不改变训练预算、原版物理或策略。六份有效回放之外另有一次中断的物理回放，不能宣称总共只执行了六次。

这次完成的是小规模工程闭环，不是作者4096环境/40000轮等规模的性能复现。没有充分证据将失败归于单一算法或机器人差异。SuperMimic底座与PPO适配两线方向保留，Q1接入、追加训练及候选采纳均未在本结果中执行，`user_accepted_checkpoint=null`；自动任务、agent和旧goal未改动。
