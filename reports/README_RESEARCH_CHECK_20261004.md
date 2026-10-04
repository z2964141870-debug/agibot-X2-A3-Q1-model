# 每日具身智能研究检查 · 2026-10-04

日期：2026-10-04，北京时间。阶段：research-watch-20261004。状态：三条来源分析与简报完成；同步结果以发布日志和远端核验为准。

## 目标与恢复

执行 `yuanqi` 当前聊天的研究跟踪，接续[10/3 中断检查](README_RESEARCH_CHECK_20261003.md)。本次触发标记 `2026-10-04T03:07:15.055Z`，即北京时间 11:07:15；首批重新获取来源为 11:08:35。按实际时间记录，不将调度设定 09:00 写成实际触发时间。

近期训练已完成倒立摆基础闭环，下一项仍是用户运行倾斜惩罚权重 3 的对照并分析结果；尚未完成智元人形策略训练。本次没有检查或启动该新训练，也不因新闻调整训练主线。

## 完成内容与决定

- 重读既有恢复入口、研究规则、候选和去重；hp3090 SSH 恢复，Git 工作区干净，HEAD 为 `35f960d460be4e5c958ef048a79c997c4be6d582`。七份入口/待编辑文件与本地哈希一致，保存编辑基线后才修改。
- 10/3 和 10/4 获取的 cs.RO/cs.LG/cs.CV RSS 均为零条；cs.RO new 可见 10/2 公告，含 83 个新稿、24 个交叉投稿和 51 个替换条目。不能用空 RSS 宣称无新闻。
- 从 158 个条目筛到 45 个标题候选，核验 5 篇原摘要；选读 DexPolicy、HumanVerse-500/λ₀、FlashDexRetarget 的方法、实验和局限，形成[本期简报](research/2026-10-04/README.md)。没有声称精读了全部候选或通读全文。
- DexPolicy 指定提交的调度核心和 SB3 callback 静态核验：`std_only` 在采集批次开始前更新动作标准差，不启用优化器超参数修改；计时采用总环境转移数。只读源码，未导入或执行。README 明确是核心方法发布，缺少完整任务/资产；MIT LICENSE 已读，代码节选仅在忽略的 `data/` 保存。
- DexPolicy 源码提交固定为 `98bf66807eecd3053d0f99c36f63f89a9d84f349`，时间 `2026-10-04T03:02:33Z`，提交信息为更新论文链接；首次核验可访问仓库不等于确认该日首次发布方法代码。
- HumanVerse-500 的 SONIC→G1 与当前既有经验相关，但正文只支持 G1，完整模型/数据/训练资产未核实。FlashDexRetarget 作者页仍为 Code coming soon，未来参考和仿真特权状态是在线遥操的迁移障碍。
- HumanoidTTT 和 DITTO-X 只保留原摘要/历史核验的背景条目；前者首次提交实际为 9/18 UTC，不能称为 10/4 新稿。未提出新研究假设，H001 仍未确立新颖性。
- 对比两个训练框架前三条发布记录的标签、日期、正文，没有变化；10/3 的 CrossBFM/T²Mem/NEXUS 页面与 10/2 字节相同。DTMR 连接失败延续旧状，不重复通知同一故障。

## 版本、命令与证据

- 主项目：`hp3090:/media/yu/FAFF-E9771/YUANQI`；Mac 镜像：`/Users/yu/Documents/ChatGPT/元启`。
- 编辑基线：`data/research/2026-10-04/project_edit_base.json`；来源原件、文本提取、只读源码节选与请求结果：`data/research/2026-10-04/`。10/3 原件原地保留。
- [本次检查清单](../data/manifests/research_watch_check_20261004.json)记录 URL、HTTP 状态、实际检查时间、大小、SHA-256 和核验边界；累计去重为 `data/manifests/research_watch_state.json`。
- 文档/JSON/本地链接检查：`logs/research_watch_validation_20261004.log`；diff、Git 暂存载荷审计、原件复制哈希、提交/推送/远端核验：`logs/research_watch_publication_20261004.log`。

只读复核示例：

```bash
curl -L --connect-timeout 5 --max-time 25 https://arxiv.org/abs/2610.00360v1
curl -L --connect-timeout 5 --max-time 25 https://arxiv.org/abs/2610.00438v1
curl -L --connect-timeout 5 --max-time 25 https://arxiv.org/abs/2610.01849v1
```

发布前获取远端分支并核对 HEAD、旧文件哈希及暂存范围；只暂存本次研究记录。检查 diff、运行 `git diff --cached --check` 和 `python3 script/project_tools/audit_git_payload.py` 后提交推送，比较 HEAD 与 GitHub main；不强推、不覆盖并行工作。具体结果只写入实际日志。

## 结果与边界

本次覆盖自上次成功检查（10/2 09:52，北京时间）以来可见的 10/2 cs.RO 公告及有限项目/发布入口；没有覆盖所有厂商渠道，也没有补齐空 RSS 之外的独立 cs.LG/cs.CV 列表。公告页面只显示日期，没有单列时区；论文提交历史保留明确 UTC。作者结果不是本项目测量，也不能外推智元能力。

没有运行下载代码、安装依赖、下载大模型、训练、操作真机或修改算法。正文与源码阅读不等于完整代码审计或复现。保持当前奖励对照的单变量原则；噪声调度、缺失掩码、多示范训练只能作为已有方法候选。

## 保存、下一步与恢复

关键研究记录、小型清单和索引按既有授权同步指定 GitHub；原始来源缓存排除在 Git 外，服务器副本哈希以发布日志为证，备份状态仍为 `LOCAL_ONLY`。同步完成与否必须读实际日志，不能把本地写入当作远端完成。

下一次先读取 [当前状态](README_CURRENT_STATE.md)、本报告与去重清单；只检查已报论文的新版本或代码/资产实质变化。教学继续既定奖励对照，科研候选不自动进入实现。
