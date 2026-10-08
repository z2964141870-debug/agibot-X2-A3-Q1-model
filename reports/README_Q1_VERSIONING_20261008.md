# Q1 分支建立与重要文件版本管理

记录日期：2026-10-08。阶段编号：Q1-V1。状态：分支创建与基础文件发布完成，远端提交已核验。

## 目标与范围

- 用户指定仓库为 `git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`，Q1 重要文件保存到大小写准确的 `Q1` 分支。
- 工作位置为 `ssh hp3090` 的 `/media/yu/FAFF-E977/YuanQi_Q1`，继续遵循 `data / logs / script / reports` 目录约定。
- 本阶段管理代码与资料分析的版本；没有启动训练、操作机器人或继续大型资源传输。Mac 仅保留小型恢复记录。

## 完成内容

1. 重查先前断线的提交/推送：服务器 HEAD 与远端 `main` 均为 `d0868987862307a122d193cf537b5b940e3663a5`。此前 Q1 资料评估已经成功提交和推送，原“待核验”状态已解除。
2. `git fetch origin` 更新此前滞后的 `origin/main`，从该提交建立 `Q1`，执行 `git push -u origin Q1`。工作副本已跟踪 `origin/Q1`，核验时工作区干净。
3. 从 Mac 使用 HTTPS 独立查询仓库，确认远端 `Q1` 与 `main` 均指向上述提交。分支入口：[GitHub Q1](https://github.com/z2964141870-debug/agibot-X2-A3-Q1-model/tree/Q1)。
4. 在 [AGENTS.md](../AGENTS.md) 与 [保存规范](README_STORAGE.md) 写入用户指定的 Q1 分支及显式推送/核验流程；更新报告索引、当前状态与 [SMPL 评估 README](README_Q1_SMPL_20261008.md)。合并 Mac 上补充的连接故障记录，保留未完成的资源迁移状态。

已随基础提交发布的重要文件：

| 文件 | 用途 |
| --- | --- |
| `script/q1/inspect_resources.py` | 原创 ZIP/XML 静态检查与流式 SHA-256 |
| `script/q1/check_mujoco_assets.py` | MuJoCo 模型加载检查入口，尚未实际运行 |
| `reports/README_Q1_SMPL_20261008.md` | SDK 接口限制、22DOF 资产核验与全身跟踪路线 |
| `data/manifests/q1_resources_20261008.json` | 6 个原始附件的大小、哈希、来源与迁移/备份状态 |
| `logs/session_records/q1_inspection_20261008.log` | 已执行检查与连接失败摘要 |
| `AGENTS.md`、`reports/README.md`、`reports/README_CURRENT_STATE.md` | 项目约定、报告索引及恢复入口 |

本次版本管理记录为 [核验日志](../logs/session_records/q1_versioning_20261008.log) 与本 README；当前记录的提交号由 Git 历史提供，不写入自身提交号。

## 关键决定及依据

- 从已经核验的提交创建 `Q1`，保留已有历史。首次 Q1 资料提交也已在 `main`，本阶段不改写该历史；后续 Q1 工作显式推送 `Q1`。
- 第三方 SDK、原始文档节选、URDF/MJCF 网格、ZIP、模型与训练数据继续保存在被忽略的 `data/`。目前没有确认它们的公开再分发许可，Git 只保存本项目代码/分析及工件身份清单。
- 服务器共享 README 以服务器最新内容为基线，仅合并 Q1 项。Mac 10/6–8 的其他研究记录独立待同步，不能通过覆盖共享入口将其标记为已发布。

## 版本、命令与证据

- 主机：`hp3090`；工作副本：`/media/yu/FAFF-E977/YuanQi_Q1`；远端：上述用户指定仓库。
- 分支起点：`d0868987862307a122d193cf537b5b940e3663a5`，提交标题 `docs: assess Q1 SMPL full-body control interfaces and assets`。
- 初次资料提交的审查结果为 `PASS: 8 staged files, 50731 bytes`；本阶段 8 个文本文件已通过 `git diff --cached --check` 与 payload 初审（54,192 字节）。补写本条审查记录后，提交前再运行同一检查；最终字节数以执行输出为准。
- 分支发布的实际 stdout 摘要保存在 `logs/session_records/q1_versioning_20261008.log`，包含 upstream 状态与远端完整提交号；复核命令如下。

```bash
ssh hp3090
cd /media/yu/FAFF-E977/YuanQi_Q1
git branch --show-current
git status --short --branch
git log -3 --oneline
git rev-parse HEAD
git ls-remote origin refs/heads/Q1
```

最后两条命令的提交号一致才能确认当前版本已同步。发布后独立检查可在 Mac 执行：

```bash
git ls-remote --heads https://github.com/z2964141870-debug/agibot-X2-A3-Q1-model.git Q1
```

## 结果与边界

- 远端 `Q1` 已存在，之前的重要 Q1 文件已包含在该分支，服务器工作副本跟踪关系正确。后续阶段按 [保存规范](README_STORAGE.md) 审核、提交、显式推送并核验。
- 分支创建不是模型训练或控制验证；当前仍仅完成静态资料核验。MuJoCo 加载、策略训练、动捕格式及真机反馈合同尚未验证。
- SSH 小型操作本轮成功，但不能据此宣称大型传输故障已解决；已有不完整 ZIP 仍不可作为可用资产。
- 本阶段未启动后台训练、仿真或真机控制作业。

## 保存与同步状态

- 基础资料评估提交已在远端 `Q1` 核验；本次规则与报告修改按规范审核后提交到同一分支，最新同步状态以本地 HEAD 和远端 ref 的比较为准。
- 原始大型文件仍保留在 Mac `/Users/yu/Documents/ChatGPT/Q1/`；服务器迁移尚未完成，百度网盘备份仍为 `LOCAL_ONLY`，详见资源清单。本阶段未新增 Mac 大型副本，也未删除原件。

## 下一步与恢复入口

1. 在 Q1 工作副本重查分支/远端与实际资源大小，完成开发 ZIP 的服务器迁移并逐项核对 SHA-256。
2. 对完整资产运行 `script/q1/check_mujoco_assets.py`，记录实测结果后再建立重定向/训练环境；保留 SMPL 全身跟随目标。
3. 先读 [当前状态](README_CURRENT_STATE.md)、[报告索引](README.md)、本 README 与 [资料评估](README_Q1_SMPL_20261008.md)，再按任务重查会变化的主机/作业状态。
