# YUANQI 当前状态与下一步

更新：2026-09-30。此文件用于恢复工作；只保留最新摘要，详细过程见阶段 README。旧机器/进程信息不是实时状态，执行前重新核实。

## 用户目标与固定约定

- 长期目标：衣服与头环自然遥操 X2 / A3 / Q1，并形成自主策略训练能力和科研课题。
- 近期目标：独立完成一轮机器人策略训练；按[训练工作包](训练工作包.md)从 P0/P1 开始。
- 训练位置：`ssh hp3090`，`/media/yu/FAFF-E9771/YUANQI`。
- 目录：`data/` 产生的数据，`logs/` 纯执行记录，`script/` 代码，`reports/` 重要 README。
- 关键记录与代码：`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`；大模型和关键大型数据：百度网盘。
- 每完成重要任务立即写阶段 README，更新此文件和[报告索引](README.md)，提交并同步 GitHub。规范见 [AGENTS.md](../AGENTS.md)。

## 已完成，可从证据继续

1. 学习路线、第一课、训练工作包、实验模板与 EgoLocate/头环静态分析已保存。尚未开始实际机器人策略训练。
2. hp3090 上四目录项目已建好，确认 RTX 3090 24GB 与可写 ext4 磁盘；初始化前可用空间约 634 GB，后续使用前重新检查。
3. 初始化提交 `c7b64a7515b551a63830d0e040a203ba543346e5` 已推送 `origin/main`，该阶段结束时本地/远端一致且工作区干净。见[初始化记录](README_SETUP_20260929.md)。
4. 17 份参考节选已同步到项目 `data/references/` 并核验 SHA-256；原始附件仍在 Mac，未复制全部 ZIP。见[清单](../data/manifests/reference_excerpts_20260929.json)。
5. 已加入工件登记、Git 暂存检查和只读环境盘点脚本；见 [script README](../script/README.md)。
6. 本轮把重要阶段落盘要求写入项目规则，建立当前状态和阶段模板。见[记录规范阶段 README](README_RECORDING_POLICY_20260929.md)。
7. 每日研究检查 `yuanqi` 已于 2026-09-30 北京时间 09:01:12 首次实际触发，完成本期检索与简报；公开公告列表仍为 9 月 29 日，已区分首次发现与首次发表。GAE 加入 H001 先行工作；未启动研究实验。见[研究跟踪说明](README_RESEARCH_WATCH.md)、[最新简报](research/2026-09-30/README.md)与[运行记录](README_RESEARCH_CHECK_20260930.md)。
8. 学员已正确回答优势 +4 应增加该动作选择概率；此前负优势方向已纠正，后续代码例子继续复核。现进入[第六小节：PPO 裁剪目标](README_LESSON01_PPO_CLIP_20260930.md)，旧概率 20%、优势 +4、epsilon=0.2 的数值题待回答。RSL-RL 3.0.1 源码位置见[第五小节](README_LESSON01_ACTOR_CRITIC_20260930.md)。奖励检查尚未实施，助手的单步实验见[第二小节](README_LESSON01_REWARD_UPDATE_20260930.md)，学员尚未独立训练。
9. P0 静态检查找到干净的 Isaac Lab 2.3.2、X2 评估代码和有未提交改动的 SONIC/X2 训练 sandbox；一个候选配置仍指向不存在的旧 URDF 路径。既有训练环境的 PyTorch/CUDA 小计算通过，尚未完成 PPO 烟测。见[核验报告](README_P0_INSPECTION_20260930.md)。

## 尚未验证 / 尚未完成

- PyTorch 2.8.0+cu128 / CUDA 12.8 在 3090 上的小计算已通过；Isaac Sim 启动、精确机器人资产、PPO 训练、checkpoint 重载与导出仍未通过本轮验证。
- 百度网盘官方 Linux 客户端 8.7.0 已发现，入口 `/opt/baidunetdisk/baidunetdisk`；登录、上传/下载未验证，未上传模型。记录状态仍是 `LOCAL_ONLY`；如现有入口不合适，再与用户讨论替代方案。
- 历史 X2 支撑站立/A3 上肢里程碑不能当作当前完整遥操验收；当前工作没有启动机器人控制。
- EgoLocate 的 Stage-A 推理实现已找到，所引用完整外部训练项目与 `best.pth` 不在已检查的 ZIP 中。

## 下一项具体工作

教学侧：收取[第六小节](README_LESSON01_PPO_CLIP_20260930.md)的解释：旧概率 20%、优势 +4、epsilon=0.2，新概率从 30% 到 40%，这个样本的裁剪目标是否继续增大。巩固目标平台区与实际概率硬上限的区别，再对照源码串起 PPO 更新；已答对的 +4 不重复提问。学员尚未独立训练，源码静态检查不能代替运行验收。

工程侧：继续 P0 的运行核验，按[已完成的静态核验](README_P0_INSPECTION_20260930.md)准备独立的最短训练、保存和重载实验。优先检查干净 Isaac Lab 的官方简单任务与 RSL-RL 入口；简单任务是流程教学，不等于完成 X2 自主站立/跟踪。X2 的原生训练候选确实存在，但旧资产路径、配置覆盖与未提交改动仍待处理，不直接接手旧实验。

训练环境候选：`/home/yu/miniconda3/envs/x2-sonic-isaaclab`；框架：`/home/yu/projects/IsaacLab`（`/home/yu/IsaacLab` 是同一位置）。X2 训练候选的准确路径、提交与缺口见核验报告。新代码/配置与产物遵守 YUANQI 四目录约定。

本聊天截至本次记录没有启动需接续的训练作业。既有项目的版本和进程状态在下一次执行前重新核实，不覆盖无关改动。

## 恢复时的最小检查

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
git status --short --branch
git log -3 --oneline
```

先确认当前文件版本，再读本文件及相关阶段 README。只复核会变化的状态和仍未解决的问题，不重复已完成的静态盘点。
