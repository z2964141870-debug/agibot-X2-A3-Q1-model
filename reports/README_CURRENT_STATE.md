# YUANQI 当前状态与下一步

更新：2026-09-29。此文件用于恢复工作；只保留最新摘要，详细过程见阶段 README。旧机器/进程信息不是实时状态，执行前重新核实。

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

## 尚未验证 / 尚未完成

- 机器人训练框架、PyTorch/CUDA 组合、精确机器人资产和完整 PPO 训练入口尚未验证；旧项目目录存在不等于训练可复现。
- 百度网盘官方 Linux 客户端 8.7.0 已发现，入口 `/opt/baidunetdisk/baidunetdisk`；登录、上传/下载未验证，未上传模型。记录状态仍是 `LOCAL_ONLY`；如现有入口不合适，再与用户讨论替代方案。
- 历史 X2 支撑站立/A3 上肢里程碑不能当作当前完整遥操验收；当前工作没有启动机器人控制。
- EgoLocate 的 Stage-A 推理实现已找到，所引用完整外部训练项目与 `best.pth` 不在已检查的 ZIP 中。

## 下一项具体工作

进入 P0：只读核验 hp3090 上已有训练资源，优先定位一个匹配 X2 的完整训练闭环；若只有部署包，则选择资产和训练配方完整的人形示例完成第一轮学习，再迁移。

候选路径（只确认过目录名）：`/home/yu/projects/IsaacLab`、`/home/yu/IsaacLab`、`/home/yu/projects/gr00t-wbc-x2`、`/home/yu/projects/a2a-x2`、`/home/yu/projects/x2-sonic-sim`、`/home/yu/projects/x2-rl-deploy`、`/home/yu/miniconda3/envs/x2-sonic-isaaclab`。

执行前先读候选项目的 `AGENTS.md` 与 README，核对 Git 工作区、版本、`train/play/export` 入口和资产。保留其他项目的已有工作，不直接复制整个目录或启动旧训练命令。

本聊天截至本次记录没有启动需接续的训练作业。下一步应生成 P0 资源核验 README，列明可用与缺失的具体证据。

## 恢复时的最小检查

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
git status --short --branch
git log -3 --oneline
```

先确认当前文件版本，再读本文件及相关阶段 README。只复核会变化的状态和仍未解决的问题，不重复已完成的静态盘点。
