# A3 官方预训练模型核验与训练暂停

记录日期：2026-10-10，北京时间。阶段：官方035模型基线核验。状态：训练与巡检已暂停，官方发布说明/固定版本清单已核验；模型尚未下载或评测。

## 目标与授权

用户要求“先停一下”，并提供 Hugging Face `sonic-for-a3/sonic/tree/main/035_step200000/rknn`。立即停止R03，核对已有预训练模型、格式与下一步选择。本轮不恢复训练、不执行评测或真机操作，不下载大型模型。

## 完成内容与结论

官方确实发布035 step-200000模型，RKNN只是该发布中的一种部署格式。固定vendor版本 `fe6868ba37034f89b912f0fb851bce19f120266d` 的 `a3_hf_manifest.json` 登记：

| 格式 | 发布路径 | 用途与清单大小 |
| --- | --- | --- |
| PT | `035_step200000/model_step_200000.pt` | PyTorch评测/导出和权重微调，402050931 bytes |
| ONNX | `035_step200000/onnx/model_step_200000_a3_fast.onnx` | 推理与RKNN导出，58120816 bytes |
| RKNN | `035_step200000/rknn/model_step_200000_a3_fast.rknn` | Rockchip NPU部署推理，31101485 bytes |

对应G1版本也登记在清单中。PT发布SHA `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`，HF清单revision `db1f40787d580af6d1e53df0df72242904a1dae5`。这些是官方发布清单数据，未下载不能视为本机大小/SHA验证。

官方 `docs/a3_training2sim2deploy.md` 3.2节提供从035 PT微调流程：加载policy/value权重，固定resume=false，优化器、调度器、采样器、环境状态和更新数重新开始；不是将旧实验全部状态接续。不能把官方200000计数直接交给当前目标10000的完整恢复控制器，否则会被误当作已完成。

官方README说明该模型基于内部动作数据训练，面向基本行走和简单操作；仅提供G1/A3-fast编码器，未提供SMPL/teleoperation编码器。A3-fast仍需要上游重定向；公开模型存在不代表原版SMPL遥操链已可用，也不证明我们的资产/动作已经评测通过。

此前R01/R02/R03是随机初始化实验的续训链，并非官方035 PT起点。优先顺序应改为官方PT基线验证、同条件仿真评测、按需要微调；已有从零训练工件保留作实验记录，不再自动推进至10000。

## 版本与来源

- 用户页面：`https://huggingface.co/sonic-for-a3/sonic/tree/main/035_step200000/rknn`。
- 官方README：`https://github.com/AgibotTech/sonic_for_a3/blob/main/README.md`。
- 官方流程：`https://github.com/AgibotTech/sonic_for_a3/blob/main/docs/a3_training2sim2deploy.md`。
- 本机固定版本：`script/vendor/sonic_for_a3/{README.md,a3_hf_manifest.json,docs/a3_training2sim2deploy.md}`。
- 网页已实际核验用户提供的RKNN目录，列出 `model_step_200000_a3_fast.rknn` 和 `model_step_200000_g1.rknn`，各约31.1MB及对应sidecar；官方README/指南已在线打开。服务器HF实时目录API访问超时，未获得目录JSON，也未下载模型，不能据网页可读认定服务器传输入口可用。

## 实际暂停结果与新重启证据

01:09:08停止 `yuanqi-a3-longtrain.service`，随后disable；01:11:00核验inactive/dead、MainPID0、disabled，GPU无计算进程。原巡检 `yuanqi-a3` 工具返回PAUSED，automation.toml回读PAUSED；每日研究 `yuanqi` 保持ACTIVE。服务自动启动已禁用，不因新boot自行恢复。

本轮发现01:03:13已有一次新的整机boot：从 `a95a89b9-df91-4c13-b609-86273fbd847d` 变为 `40497006-bd02-449e-931a-dac12218a7eb`，发生在用户暂停之前，根因UNKNOWN。最后训练health01:02:38 CPU73°C/GPU42°C；不据此断言温度、供电或驱动导致。新boot控制器进入冷却，01:08:07返回75等待下次恢复，尚未登记第二attempt即被用户暂停。R03账本残留running为上次持久字段，不代表实时训练；保持原账本，不将本次暂停写成训练完成或恢复成功。

最近保存2950模型独立大小/SHA/CPU重载/网络与优化器有限性/计数校验通过，校验日志mtime01:11:56：`data/training/a3_20261009/E008_cont_01_s2700/model_step_002950.pt`，402081495 bytes，SHA `7d0c1b167aad2be746360794d1b8d8f8a5d0d7f84a2464f5050cf79a87ba1536`，LOCAL_ONLY，未评测。快照/服务/journal/校验日志 `logs/a3_pretrained_20261010/pause/`。

## 保存与下一步

本README、索引/current state、R03阶段状态与巡检清单审计后scoped发布main并核验远端；提交号由Git历史提供。大型模型留服务器，网盘未验证。

下一步准备官方PT及配套配置的下载/校验与本地基线评测；先处理HF连接入口，若传输慢按用户文件迁移规范列清文件名、大小、来源与目标。未授权自动重启原从零实验。保留原生Isaac指标依赖修复事项；仿真效果验证后才讨论微调或部署。
