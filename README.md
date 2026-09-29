# YUANQI

衣服 / 头环 → FGP / EgoLocate → 人体动作参考 → 机器人策略训练 → X2 / A3 / Q1 自然遥操。

当前阶段：完成项目落盘和记录规范，下一步核验完整训练入口，独立完成第一轮机器人策略训练。

## 固定工作位置

- 训练主机：`ssh hp3090`
- 主项目目录：`/media/yu/FAFF-E9771/YUANQI`
- Git 远端：`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`
- GitHub：[agibot-X2-A3-Q1-model](https://github.com/z2964141870-debug/agibot-X2-A3-Q1-model)
- 大型工件备份：百度网盘；只有上传并完成校验后才标记为已备份。

```text
YUANQI/
├── data/       产生的数据、模型、录制、视频、评测输出；Git 仅保存索引与清单
├── logs/       终端输出和执行记录；Git 保存检查过的小型关键记录
├── script/     脚本、代码和配置
└── reports/    重要结论、计划、操作说明与 README
```

根目录仅放本入口、`AGENTS.md` 和 `.gitignore` 等项目管理文件。

## 开始阅读

1. [项目状态与报告索引](reports/README.md)
2. [当前状态与下一步](reports/README_CURRENT_STATE.md)：恢复上下文时先看这里。
3. [训练工作包](reports/训练工作包.md)
4. [第一课：机器人控制与 PPO](reports/第一课_机器人控制与PPO.md)
5. [训练服务器与存储](reports/README_INFRASTRUCTURE.md)
6. [Git 与百度网盘保存规范](reports/README_STORAGE.md)

每完成重要阶段，立即保存阶段 README，更新状态与索引，并提交到 GitHub。该要求已写入 [AGENTS.md](AGENTS.md)，记录格式见 [阶段模板](reports/README_STAGE_TEMPLATE.md)。

初始化时确认：RTX 3090 24GB；目标磁盘为可写 ext4；已安装百度网盘 Linux 客户端 8.7.0。客户端登录和上传/下载能力未验证。完整证据见 `logs/session_records/`。

## 常用入口

```bash
ssh hp3090
cd /media/yu/FAFF-E9771/YUANQI
python3 script/project_tools/inventory_host.py
python3 script/project_tools/audit_git_payload.py
```

训练、仿真或转换命令的 stdout/stderr 存入 `logs/`；它们产生的模型和数据存入 `data/`。重要实验完成后在 `reports/` 写明版本、命令、结果和证据路径，再提交关键记录。
