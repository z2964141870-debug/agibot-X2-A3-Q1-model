# YUANQI 项目状态与重要文件

更新日期：2026-09-30。

## 恢复上下文先读

1. [当前状态与下一步](README_CURRENT_STATE.md)：目标、完成项、未验证项和下一条工作步骤。
2. 下方阶段索引中与当前工作对应的 README，再查看其中引用的日志与版本。
3. [训练工作包](训练工作包.md)及任务所需的服务器、存储说明。

用户明确要求：每完成重要部分立即写 README，不能只保留聊天。执行规则见根目录 [AGENTS.md](../AGENTS.md)，新记录使用[阶段模板](README_STAGE_TEMPLATE.md)。

## 阶段记录索引

| 日期 | 阶段 | 状态与证据 |
| --- | --- | --- |
| 2026-09-29 | 学习路线与头环资料分析 | 已完成规划及静态分析；见[训练工作包](训练工作包.md)、[EgoLocate 分析](EgoLocate与头环分析.md)；未启动训练 |
| 2026-09-29 | hp3090 项目初始化与存储规范 | 已完成，初始化提交 `c7b64a7`；见[初始化 README](README_SETUP_20260929.md) |
| 2026-09-29 | 重要阶段记录与上下文恢复约定 | 已写入规则、状态入口和模板；见[本阶段 README](README_RECORDING_POLICY_20260929.md) |
| 2026-09-29 | 每日研究检查与首份简报 | 每天北京时间 09:00 的当前聊天任务已创建，首次定时触发待验证；见[阶段 README](README_RESEARCH_WATCH_SETUP_20260929.md) |
| 2026-09-30 | 首次定时研究检查 | 已实际触发，完成来源核验与三条简报；见[运行 README](README_RESEARCH_CHECK_20260930.md) |
| 2026-09-30 | 教学第一课·控制与学习 | 学员已答对力矩变小和不能据此判断学习；见[第一小节](README_LESSON01_20260930.md) |
| 2026-09-30 | P0 资源静态核验 | 找到训练候选与具体资产路径问题，CUDA 小计算通过；未运行训练/重载，P0 未通过；见[核验 README](README_P0_INSPECTION_20260930.md) |
| 2026-09-30 | 教学第一课·奖励与参数更新 | 单步策略梯度实验通过；学员答对需要更新参数及补充运动信息；见[第二小节及跟进](README_LESSON01_REWARD_UPDATE_20260930.md) |
| 2026-09-30 | 教学第一课·角速度与累计回报 | 学员选择 B 正确；clip 抑制更新与纠正目标的区别仍需巩固；见[第三小节及反馈](README_LESSON01_RETURN_20260930.md) |
| 2026-09-30 | 教学第一课·奖励设计检查 | 已提供反例检查方法；学员“失败不能进入 clip”的误解转入第五小节纠正，未实施机器人实验；见[第四小节](README_LESSON01_REWARD_DESIGN_20260930.md) |
| 2026-09-30 | 教学第一课·价值网络与优势 | RSL-RL 源码已定位；负优势方向已纠正，学员随后答对 +4 应增加动作概率；见[第五小节](README_LESSON01_ACTOR_CRITIC_20260930.md) |
| 2026-09-30 | 教学第一课·PPO 裁剪目标 | 学员答对 4.8 平台与同批分母 20%；正说明单样本裁剪不等于 Critic 或整批停止学习；见[第六小节](README_LESSON01_PPO_CLIP_20260930.md) |

## 当前状态

- 已确认训练主机 `hp3090`、RTX 3090 24GB 和 `/media/yu/FAFF-E9771` 的可写存储。
- 主项目位置固定为 `/media/yu/FAFF-E9771/YUANQI`。
- 目录按 `data / logs / script / reports` 组织；关键文本与代码提交到用户指定的 GitHub 仓库。
- 已发现百度网盘 Linux 官方客户端 8.7.0；未验证登录或上传/下载，未上传模型。
- 已形成训练路线、第一课、头环分析和实验模板；尚未启动机器人策略训练。

## 阅读顺序

1. [训练工作包](训练工作包.md)：按 P0–P8 推进，近期目标是第一次完整策略训练。
2. 第一课按小节推进：[控制循环](README_LESSON01_20260930.md)、[奖励与更新](README_LESSON01_REWARD_UPDATE_20260930.md)、[累计回报与 clip](README_LESSON01_RETURN_20260930.md)、[奖励设计](README_LESSON01_REWARD_DESIGN_20260930.md)、[价值网络与优势](README_LESSON01_ACTOR_CRITIC_20260930.md)、[PPO 裁剪目标](README_LESSON01_PPO_CLIP_20260930.md)；[完整参考讲义](第一课_机器人控制与PPO.md)按需阅读。
3. [学习路线与 OKR](学习路线与OKR.md)：三平台交付与科研方向。
4. [EgoLocate 与头环分析](EgoLocate与头环分析.md)：手部、SLAM、双目及已有融合方案。
5. [实验记录模板](实验记录模板.md)：每次实验复制后填写。

## 项目运行与存储

- [训练服务器与存储](README_INFRASTRUCTURE.md)
- [Git 与百度网盘保存规范](README_STORAGE.md)
- [既有资料来源](README_SOURCES.md)
- [初始化结果](README_SETUP_20260929.md)
- [每日研究跟踪规则](README_RESEARCH_WATCH.md)、[简报索引](research/README.md)、[候选研究想法](research/README_IDEAS.md)

下一项工作：巩固单样本的裁剪平台与 Critic、整批学习的区别，再把策略损失、价值损失与参数更新对应到代码和日志；工程侧准备独立的最短 PPO 训练/保存/重载实验。当前已完成静态盘点，具体入口与缺口见 P0 核验 README，不重复从目录名开始盘点。
