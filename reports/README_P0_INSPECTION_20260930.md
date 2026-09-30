# P0 静态资源核验与教学准备

日期：2026-09-30。状态：完成本轮静态检查与 CUDA 基础探测；P0 尚未通过。没有进行策略训练、checkpoint 重载或真机操作。

## 目标与实际范围

用户要求开始教学。本轮先为第一课找到真实控制代码，同时核实未来训练可用的候选环境。读取既有工程及其适用 AGENTS/README，不运行其中的历史启动指令，不修改原工程。

## 实际发现

| 候选 | 本轮证据 | 结论 |
| --- | --- | --- |
| `/home/yu/projects/IsaacLab` | Git `37ddf626871758333d6ed89cf64ad702aef127d0`，工作区干净，VERSION 2.3.2；有 RSL-RL train/play 等入口 | 官方基础任务是教学烟测候选；尚未启动仿真 |
| `/home/yu/IsaacLab` | 实际解析到上方同一目录 | 不是第二套独立框架 |
| `/home/yu/projects/a2a-x2` | Git `d2384821583007306f5feee6cffa0ed3e78ab3f8`，工作区干净；ONNX/MuJoCo 评估入口、X2 MJCF 可读 | 用于本课控制循环阅读；未核验所有网格、权重与运行参数 |
| `/home/yu/projects/gr00t-wbc-x2` | Git `70bed4539efae431621013b3df71e6f81df1ab1c`，工作区干净；本轮所查 `gear_sonic/train_agent_trl.py` 等路径不存在 | README 中的训练能力描述不足以证明当前 checkout 已有相应入口；继续作为候选 |
| `/home/yu/projects/x2-sonic-sim/x2_sonic/sonic_x2_sandbox` | Git `bc38f6d0ce6cab4589e025037ad0bfbab7ba73d8`，多处本地修改；存在 X2 config、训练主入口、PPO loss/backward/optimizer 更新 | 确有训练侧实现候选，但不能用提交号代表当前全部内容；不得覆盖或直接复制未审改动 |
| `/home/yu/projects/x2-rl-deploy` | 发现实机/站立部署与评估文件，本轮根目录未识别为独立 Git 仓库 | 作为部署资料，未连接机器人，也未认定为训练框架 |

主项目 `YUANQI` 本轮编辑基线为 `ae7c7bc3b8a292cdde8b87366f5b772d12ed15c3`，开始时干净；待编辑记录的本地/服务器哈希一致。

## 环境与一次真实计算

`/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python` 的包元数据：

- Python 3.11.15；PyTorch 2.8.0+cu128；Isaac Sim 5.1.0.0。
- `isaaclab` Python 包元数据版本 0.54.2；框架根目录 VERSION 文件为 2.3.2，分别记录，不混用。
- `rsl-rl-lib` 3.0.1；MuJoCo 3.3.7。按 `onnxruntime` 名称查询没有记录，未据此断言其他分发包或环境没有 ONNX Runtime。
- GPU 探测时为 RTX 3090，使用约 244 MiB / 24576 MiB，利用率 0%；这是当次快照。
- 实际导入 PyTorch 并在 CUDA 上创建 `[2.0]` 张量、计算平方，返回 `4.0`。确认运行时 CUDA 12.8 和 GPU 名称。

这只能证明 PyTorch/CUDA 基础计算可用，不能证明 Isaac Sim 启动、物理资产加载、PPO 训练或模型导出已经通过。

最小计算复现命令：

```bash
ssh hp3090 '/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python -' <<'PY'
import torch
x = torch.tensor([2.0], device='cuda')
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print((x * x).item())
PY
```

第一次使用 SSH 传递 `python -c` 时发生远端 shell 引号错误，Python 没有启动；改为以上 heredoc 后成功。这不是 CUDA 故障。

## 训练候选中待解决的具体事项

1. sandbox 的 `gear_sonic/envs/manager_env/robots/x2.py:20` 指定 `/home/humanplus/x2_teleop_final/assets/agibot_x2`。本轮确认该路径下的 `x2_ultra.urdf` 不存在。
2. `config/exp/manager/universal_token/all_modes/sonic_x2.yaml` 仍带候选/占位说明，并指向旧 motion 与 asset 目录。本轮未解析所有 Hydra 覆盖，不能据单个候选配置断言整个工程都无法训练。
3. sandbox 有大量未提交改动，包括观测、动作、奖励、终止和训练实现。首先需要核验其来源与当前实验用途，再选择独立、版本明确的教学副本。
4. 已找到 `train_agent_trl.py:477` 的 `trainer.train()`，以及 `ppo_trainer.py:11600` 的 loss、`:12863` 的 backward、`:12879` 的 optimizer step。存在代码不等于本轮实际执行了参数更新。
5. 尚未完成：完整配置解析、资产与 motion 依赖核验、仿真启动、最短训练、checkpoint 保存/重载和导出。因此 P0 的验收条件尚未满足。

## 选择与下一步

教学先使用干净的 X2 评估文件解释闭环。第一课的理解检查见[本节讲义](README_LESSON01_20260930.md)。学生尚未回答，不标记掌握。

下一项工程工作：为第一次 PPO 烟测准备独立工作目录和输出路径，优先检查现有 Isaac Lab 的官方简单任务与 `rsl_rl` 入口，确认资产可用后再进行短训练、保存和重载。简单任务仅用于学习训练流程，不替代浮动基座 X2 站立/动作跟踪验收。与此同时，可继续找齐 X2 候选的资产与最小配置，但不直接接手旧实验的未提交改动。

所有后续训练代码/配置放 YUANQI 的 `script/`，产物放 `data/`，终端输出放 `logs/`；不在其他既有工程里写入实验结果。需要独立 Git checkout 时按工作区规则创建，不覆盖旧工程。

## 产物与实际验证

- [资源核验清单](../data/manifests/training_resource_check_20260930.json)：提交、路径、包版本、范围与关键源码哈希。
- [检查日志](../logs/session_records/lesson01_checks_20260930.log)：只读输出、CUDA 计算结果与教学交互检查。
- 本次没有需要接续的 GPU 训练作业，没有模型工件或百度网盘上传。
- 关键文档、源码片段和小日志按规范检查、提交、推送；发布结果记录在 `logs/lesson01_publication_20260930.log`，最终提交身份通过 Git 历史核验。
