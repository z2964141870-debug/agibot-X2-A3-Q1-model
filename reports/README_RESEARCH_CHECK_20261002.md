# 每日具身智能研究检查 · 2026-10-02

日期：2026-10-02，北京时间。阶段：research-watch-20261002。状态：来源核验与三条简报完成；发布结果以执行日志和 Git 历史为准。

## 目标与范围

执行当前聊天 `yuanqi` 每日研究检查，维护去重、候选和恢复文档。近期目标仍是第一次策略训练；本轮只读取公开资料并维护研究记录，没有安装依赖、执行第三方代码、下载模型、启动训练或操作真机。

## 完成内容与决定

- Heartbeat 标记为 `2026-10-02T01:40:10.703Z`，即北京时间 09:40:10；实际来源获取始于 09:41:54。记录实际时间，不将任务设定 09:00 当成准时完成，也不猜测调度延后的原因。
- 读取规定入口，核实 hp3090 工作区干净，HEAD/GitHub main 一致为 `a2eaf581e904391d12b399df226a437049b77460`。七份恢复/待编辑文件与本地哈希一致后建立编辑基线。
- 获取 cs.RO/cs.LG/cs.CV RSS、cs.RO new 和两项官方框架发布 API。三个 feed 分别含 203、638、353 条；标题关键词筛到 122 条、109 个唯一 ID，不等于精读 109 篇。
- 核验六篇新候选原摘要。选读 DTMR、NEXUS、LBDU-VIO 的方法、实验或局限相关段落，形成[今日简报](research/2026-10-02/README.md)。所有算法均未复现，没有完整代码审计。
- NEXUS 项目页代码待发布；DTMR 作者项目页两次连接重置，代码/模型/数据状态未核实；LBDU-VIO 已查原文没有识别到官方资产入口，不据此断言未开源。
- 比较已知项目：CrossBFM 仅 BibTeX 更改，T²Mem 页面文字不变，代码仍待发布。Think Fast, Plan Selectively v1/v2 正文文本差异仅发现版本日期与公式输入符号修正，没有确认方法或实验实质更新，不重报。两个框架已查的前三条发布记录标签、时间与说明不变，不升级环境。
- 更新候选清单，补充 H001 已有方法边界，没有新增科研假设。保留教学当前“网络参数/经历/输出”卡点及训练工作包，不因资讯改变主线。

## 版本、命令、产物与日志

- 主项目：`hp3090:/media/yu/FAFF-E9771/YUANQI`；本地：`/Users/yu/Documents/ChatGPT/元启`。
- 编辑基线：`data/research/2026-10-02/project_edit_base.json`。
- 原始 HTML/XML/JSON、正文提取和版本差异：`data/research/2026-10-02/`，排除在 Git 外；文件大小与 SHA-256 登记在[小型检查清单](../data/manifests/research_watch_check_20261002.json)。
- 累计去重：`data/manifests/research_watch_state.json`；本轮简报、报告索引、候选和当前状态一并维护。
- 本地文档检查：`logs/research_watch_validation_20261002.log`；发布、暂存审计、推送与远端核验：`logs/research_watch_publication_20261002.log`。

只读复核示例：

```bash
curl -L --connect-timeout 5 --max-time 22 https://arxiv.org/abs/2609.38617v1
curl -L --connect-timeout 5 --max-time 22 https://arxiv.org/abs/2609.39000v1
curl -L --connect-timeout 5 --max-time 22 https://arxiv.org/abs/2609.39125v1
```

发布时核验工作区、旧文件哈希、远端分支和待提交范围；审阅 diff，运行 `git diff --cached --check` 与 `python3 script/project_tools/audit_git_payload.py`，提交后推送并比较 HEAD 与 GitHub main。实际结果写入日志，不能用阅读完成代替发布完成。

## 结果与局限

- 可见来源覆盖 10/1 公告，三个 v1 首次提交在 9/29–9/30 UTC，10/2 首次发现。10/2 后续公告、全部厂商渠道、旧仓库的每条提交没有完整覆盖。
- 三项入选工作分别针对参考动作时序、地形差异、视觉中断定位；都不能证明智元 X2/A3/Q1 已有可用实现。NEXUS 用深度感知适配本地地形，与操作者头环提供动作意图是不同信息来源。
- 未使用作者数字推算本项目可达到的性能或训练耗时；没有将预测回报、定位误差或跨机器人论文结果当成本项目实测。
- 没有完整系统查新，没有确立算法创新。H001 保留为待验证候选。

## 保存状态与下一步

关键文本与小型来源清单按授权同步到指定仓库；原始证据缓存的服务器副本核验结果与最终提交号见发布日志。原始缓存为 `LOCAL_ONLY`，不代表已备份百度网盘；本轮没有大型模型产物，也没有在跑的新训练作业。

恢复时先核对 Git 与[当前状态](README_CURRENT_STATE.md)，读取[今日简报](research/2026-10-02/README.md)及[研究去重状态](../data/manifests/research_watch_state.json)。教学继续确认网络参数与训练数据的关系；工程继续既定 P0/首次训练闭环，不因新论文自动安装或训练。
