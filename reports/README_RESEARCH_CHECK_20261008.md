# 每日研究检查 · 2026-10-08

记录日期：2026-10-08，北京时间。阶段：research-watch-20261008。状态：**三条新候选与一条项目说明更正完成，本地保存；主仓库同步仍阻塞**。

## 目标与范围

执行 `yuanqi` 研究检查，触发 `2026-10-08T01:12:56.607Z`，实际首次来源请求 `2026-10-08T02:12:21.103166Z`。从上次成功来源检查（10/7 北京时间 09:19:15）继续，筛选 10/7 公告与近期原稿；不把公告日、首发日与发现日混为一谈，不推测触发到请求之间的间隔原因。

只在 Mac 进行研究、来源缓存与文档维护。没有安装依赖、执行下载代码、训练、仿真、模型下载或机器人操作；未核验 GPU/新训练结果。

## 完成内容

- 读取 AGENTS、报告索引、当前状态、研究规则、阶段记录、候选清单、存储规范和去重清单。两次有界 `ssh hp3090` 均超时，当前服务器 Git 状态与远端 HEAD 未知；未写服务器，也未让旧镜像覆盖远端。
- cs.RO/cs.LG/cs.CV RSS 分别为 181/526/247 条，均对应 10/7 公告；跨类别条目不当作独立论文累计。核验五篇新来源的原摘要和提交历史；选读 Beyond Retargeting、BRACE、BiGym 2.0 的方法、实验与限制。
- 另复查 OCLO 原稿及作者项目页。arXiv v1 与 10/7 缓存字节相同，项目页改为“双手目标＋行走速度”，真机摇杆接口得到澄清。明确作为项目说明更正，已回注 10/7 简报，保留原文与编辑前快照；不声称新论文版本或新实验。
- [本次简报](research/2026-10-08/README.md)三条新内容分别对应在线重定向、受力/地形参考适配、评测状态恢复。CARAT 与 From Legs to Wheels 仅摘要筛选，登记为背景。未提出新研究假设。
- 检查 BRACE 项目页，未确认完整实现/权重入口；BiGym 2.0 作者所给仓库元数据、目录、提交 API 均 404，网页连接也未成功，不能确认其开源资产可用性。没有因访问失败推断仓库不存在或论文结果无效。
- Isaac Lab 发布 JSON 只有 reactions 变化，无版本/正文变化；RSL-RL 和七项既有项目/仓库入口响应与昨天相同。OCLO 的实质说明变化单列，不重复其他未变化信息。

## 关键决定及依据

- 延迟分清算法执行与真实响应：Beyond Retargeting 的 0.604 ms 来自共享仿真运行环境；147.5 ms 来自选定 5 秒真机视频片段，两者都是作者报告，不能互相代用。
- 保留 BRACE 输入与复现成本边界：部署无高度图/实测外力，但仍有动作、模式与力指令；约 6.5 万参考和 8 张 RTX Pro 6000 的作者训练规模不作为当前工作计划。
- BiGym 2.0 只借鉴评测设计，不把冻结底层控制器之上的仿真任务学习当作从零训练人形平衡，也不据论文开源声明自动安装不可核验资产。
- H001 仍为待验证问题。新来源进一步说明缺失掩码、动作补全、可靠性门控已有先例；现有 EgoLocate 的 GRU/新鲜度门控/fallback 继续是必要对照。教学主线保持奖励权重 1→3。

## 版本、命令与产物

- 工作目录：`/Users/yu/Documents/ChatGPT/元启`（本地镜像，无 Git 仓库）。主项目：`hp3090:/media/yu/FAFF-E9771/YUANQI`。
- 上次已核验发布：10/5 的 `a82be9bb25450be6f432fb24d7b8f70e24df1d6e`；今天未核验服务器/GitHub HEAD，不将旧提交当成实时状态。
- 今日共享文件编辑前快照：`data/research/2026-10-08/local_edit_base.json`；10/7 简报更正前快照：`data/research/2026-10-08/dated_record_edit_base.json`。10/6、10/7 原有快照继续保留。
- 来源与哈希：[检查清单](../data/manifests/research_watch_check_20261008.json)、`data/research/2026-10-08/source_fetches.json`。正文提取仅供阅读，HTML 原件保留。OCLO 项目差异为 `data/research/2026-10-08/oclo_project.html.diff`。
- 日志：`logs/research_watch_fetch_20261008.log`、`logs/research_watch_ssh_20261008.log`、`logs/research_watch_review_20261008.diff`、`logs/research_watch_validation_20261008.log`、`logs/research_watch_publication_20261008.log`。

只读复查示例：

```bash
curl -L --connect-timeout 6 --max-time 25 https://arxiv.org/abs/2610.07891v1
curl -L --connect-timeout 6 --max-time 25 https://arxiv.org/html/2610.07052v1
curl -L --connect-timeout 6 --max-time 25 https://oclo-humanoid.github.io/
ssh -o BatchMode=yes -o ConnectTimeout=12 hp3090 'cd /media/yu/FAFF-E9771/YUANQI && git status --short --branch && git rev-parse HEAD'
```

## 验证与边界

29 次请求中 25 次 HTTP 200、3 次 BiGym 仓库 API 404、1 次仓库网页连接超时（curl 28，无 HTTP 响应）。本地 JSON、Markdown 链接、17 个待同步文件哈希、29 项来源缓存/失败记录以及去重键检查通过；AGENTS 与其他独立历史记录未改，10/7 简报仅插入明确更正段。共享文档 diff 已人工检查，累计 18 份记录通过暂存空白检查和载荷审计，详情见验证日志。

主仓库不可达，按既有方式在隔离临时 Git 仓库暂存累计待同步文件，运行原样复制的 `script/project_tools/audit_git_payload.py`；只作发布前预审，不产生项目提交、不配置远端。恢复连接后仍需在真实仓库检查 diff、审计、提交推送并核验。

没有算法复现、完整实现/资产核验或智元结果。未遍历所有论文、版本与厂商 SDK；缺失的研究证据按条目标明，不把有限筛选当作系统查新。

## 保存与同步

本次及 10/6–7 研究记录保存在 Mac，**尚未复制到 hp3090、未提交或推送**。当前原始缓存为 `LOCAL_ONLY`，没有百度网盘备份验证。

[10/8 累计待同步清单](../data/manifests/research_watch_pending_sync_20261008.json)是当前文件哈希入口。10/6–7 旧清单保留为历史快照，共享文件及带显式更正的 10/7 简报哈希已更新，不能据旧清单回退当前内容。相同连接故障不重复向用户告警或请求处理。

## 下一步与恢复

1. 先保留三天记录、编辑前快照与累计清单，再连接主服务器读取最新 AGENTS、Git 状态、HEAD 和远端分支。
2. 用 10/5 已发布镜像基线、逐日快照、当前本地与最新服务器做合并。日期独立文件按内容/哈希核对；10/7 简报仅新增明确的更正段，不能删除历史正文。
3. 仅提交相关研究记录，通过审计后推送并核验；发布无成功证据前维持待同步状态。
4. 教学继续 `lesson02_tilt_weight3`，先核验用户实际产物，再比较失败类型、维持时间和倾角。本轮未证明新训练已执行。
