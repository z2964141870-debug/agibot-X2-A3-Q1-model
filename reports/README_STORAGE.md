# GitHub 与百度网盘保存规范

## 四个目录的边界

| 位置 | 内容 | Git / 网盘策略 |
| --- | --- | --- |
| `data/` | 训练数据、模型、录制、轨迹、视频、评测输出与参考资料 | Git 只保存索引/清单；关键大型工件备份到百度网盘 |
| `logs/` | 终端输出、命令运行与环境记录 | 审核后的关键小日志进 Git；大日志登记后走网盘 |
| `script/` | 代码、脚本、配置 | 保存到指定 GitHub 仓库 |
| `reports/` | README、实验结论、操作说明、阶段计划 | 保存到指定 GitHub 仓库 |

Git 远端固定为 `git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。不把模型塞进普通 Git 历史，也不默认改用 Git LFS 代替用户指定的网盘。

## 一次训练需要保存什么

- 数据和模型：`data/<用途>/<experiment_id>/`。
- 终端记录：`logs/<experiment_id>.log`。
- 代码及配置：`script/<任务>/`。
- 结论：`reports/<experiment_id>/README.md`，引用配置、数据、日志和工件清单。
- 最终 checkpoint、导出模型、归一化参数、机器人资产身份和评测结果要能对应到同一个实验。

## Git 保存流程

1. 更新相关 README/实验结论与小型工件清单。
2. 查看 `git status` 和 diff，只暂存本阶段需要的文件。
3. 运行 `python3 script/project_tools/audit_git_payload.py`。默认检查单文件不超过 10 MiB，并拒绝模型/归档格式与明显凭证；这是项目记录仓库的初始限额，特殊情况应说明用途后调整。
4. 提交并推送，核对本地 HEAD 与远端分支一致。

自动检查不能替代人工查看日志。禁止在仓库写入登录 Cookie、BDUSS/STOKEN、token、密码、SSH 私钥或本地认证配置。

## 百度网盘备份流程

当前只确认官方客户端已安装，尚未验证登录/传输。网盘建议采用 `/YUANQI/<kind>/<experiment_id>/`，该路径目前只是规划，未声称已在云端创建。

1. 生成工件后，登记大小、SHA-256、相对路径、来源和计划远端路径：

   ```bash
   python3 script/project_tools/register_artifact.py \
     data/checkpoints/EXPERIMENT/model.pt \
     --kind checkpoint \
     --experiment-id EXPERIMENT \
     --source "训练配置和代码版本见对应实验报告"
   ```

2. 确认客户端正常登录且能完成小文件上传/下载后，再上传大型工件。
3. 记录真实远端路径与上传时间，检查云端文件大小。
4. 下载到 `data/verification/` 下的独立路径，对下载文件计算 SHA-256。
5. 原文件与下载文件的字节数、SHA-256 都一致才把清单状态改为 `VERIFIED`，并保存相应证据。

状态含义：

- `LOCAL_ONLY`：只在本地；包括仅规划云端路径的情况。
- `UPLOADED_UNVERIFIED`：已报告上传，但还没有完成下载哈希验证。
- `VERIFIED`：上传后完整往返校验通过。

未校验不得删除本地原件；即使已经校验，本规范也不自动授权删除仍被训练、推理或复现实验使用的文件。只有原始压缩包路径、已有账号缓存或上传队列都不能证明备份成功。
