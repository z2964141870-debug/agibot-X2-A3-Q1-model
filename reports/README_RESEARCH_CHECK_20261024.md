# 每日研究检查 · 2026-10-24

日期：北京时间2026-10-24。阶段：research-watch-20261024。状态：**来源分析与本地研究记录完成，GitHub发布待实际核验**。

## 目标与范围

执行yuanqi每日检查，触发2026-10-24T00:55:24.396Z／北京时间08:55:24；接续10/23 09:17:01成功来源检查，优先24–72小时及最新公告。当前用户工程目标已于10/23改为supermimic_for_q1、世界模型方法复现；训练根huanyu:/opt/data/private/YuanQi_Q1，Q1分支，旧hp3090只作历史来源。本日报只检索/静态阅读/维护记录，没有训练、仿真、GPU评测、陌生代码执行、依赖安装、权重下载或真机操作。

## 完成内容

- 已读AGENTS、报告/current、研究规则/索引/候选/去重、阶段模板及Q1阶段/D04/D05B/D06等记录。hp3090有界SSH一次超时，Git状态/HEAD未知；未连接huanyu或操控工程作业。
- GitHub独立研究副本最初干净，main从770d5e5快进至538c8f3ee5f5dc7d6d33adaa8360b6ef21b7a480；显式获取Q1，记录基线cf61084eb8ba1dfdf4ad4fa7dba478e6440f922f，最新D06与Mac相符。以上是保存检查点，不把02:38的PID、GPU和轮次当实时状态。
- 最新三类公告为10/23；RO/LG/CV的new含替换107/339/192条，recent463/1427/907条，本次RSS计数均匹配new。标题筛选含人工补查，PhysibleMDM等不只靠关键词过滤。
- 8篇摘要/原始提交历史；PhysibleMDM、MPBRS、DiffDreamer的相关方法、实验、限制正文。形成[三项简报](research/2026-10-24/README.md)，其余五项仅摘要候选，不借摘要宣称实现可用。
- MPBRS作者库固定a0dffb3完整114项树/85个blob；README、奖励、环境配置和PPO配置四文本经API解码后大小/Git-blob/SHA256核验，未导入或运行。树中有训练入口/USD，未见LICENSE或PT/ONNX等策略权重；匿名入口401，不把它写成成功开源验证。
- 论文Table IV/PPO代码gamma=0.99，正文/奖励配置gamma_phi=1.0。经典折扣势能塑形需相应折扣与终止处理条件；此不一致不足以直接套用策略不变性保证。该结论是本文核查，非作者已承认错误或本项目实验结果。
- PhysibleMDM联合搜索扩散噪声并解带力矩/摩擦约束的接触力问题；与本项目已有软接触inverse诊断区分。论文软惩罚无严格可行保证，本体和资产不匹配，不直接引入Q1训练。
- DiffDreamer明确区分latent/explicit路径和运动/离散奖励；学习模型与critic参数在actor更新时冻结，actor的latent输入detach，但显式状态路径可微。证据为四旋翼、真机外部动捕，不能说已解决人形接触梯度。
- 八个原关注仓库头未变；PredActor两个提交仅README实验部署提示/标题，SMR三提交为组织/链接/展示资料迁移，arXiv摘要未变，展示PDF二进制改变未逐页核验，不据此声称新增方法。两个发行、SuperMimic官网及其余八个既有作者页未变。两份第三方SuperMimic头也未变，不称官方开源。
- 去重102→111，旧102项保持：新增8论文与一个MPBRS作者库固定版本。H001仍待验证，无H002或新原创假设；WholeBodyMem仅摘要补充为记忆选帧已有应用线索。

## 关键决定及依据

按Q1当前瓶颈选择三项：参考可行性、奖励诱因、模型分工。Q1 D04记录悬空/自碰撞，D05B几何贴地后逆动力学仍恶化；Q1旧R01大部分回报增益来自限位项，新R03在线改善不等于固定整片/移动结果。论文只能提供诊断与候选依据，不证明当前失败唯一原因，不改正在执行的输入、代码或预算。

通用日报、去重与候选继续由main集中维护，Q1分支保存同日简报/小来源清单及专门入口；各分支的工程current/index仅添加研究行，不整份相互覆盖。Q1旧研究索引仍有历史缺口，指向main完整档案；本次不混入其他工程文件。

## 版本、命令与证据

- 来源分析截至2026-10-24T09:10:53+08:00；53研究HTTP请求52成功，唯一失败匿名入口401；Git/SSH、浏览器引文和发布核验另计。HF本次未请求，当前远端资产元数据仍未知。
- 缓存：data/research/2026-10-24/sources/；日志：logs/research_watch/2026-10-24/{fetch_results,source_comparison,source_evidence}.json。
- 简报：reports/research/2026-10-24/README.md；小来源/去重：data/manifests/research_watch_check_20261024.json、research_watch_state.json；候选：reports/research/README_IDEAS.md。
- main副本：data/research/2026-10-20/record_checkout。Q1发布副本：/tmp/yuanqi-research-q1-20261024-publish，仅研究文档及审核器的稀疏检出。
- 本地共享克隆因按需对象缺失失败，未修改原副本；改用指定远端filtered clone成功。原/tmp/yuanqi-supermimic-q1-20261023工程副本未触碰。
- 来源原始日期/时区与作者代码提交日期见小清单；三主稿为10/21–22 UTC首次提交、10/23美东公告、10/24首次发现。没有把抓取日当发布日期。

可复核的本地解析命令：

```bash
python3 logs/research_watch/2026-10-24/compare_sources.py
python3 logs/research_watch/2026-10-24/build_source_evidence.py
python3 logs/research_watch/2026-10-24/inspect_text.py mpbrs_full.html '0.99' 'gamma'
python3 logs/research_watch/2026-10-24/inspect_text.py physible_full.html 'Constrained Inverse' 'Limitations'
python3 logs/research_watch/2026-10-24/inspect_text.py diffdreamer_full.html 'stop-gradient' 'Limitations'
```

这些是自有标准库解析器；不执行来源中的命令。重抓另存新快照，不覆盖本次缓存。

## 结果与边界

有三项值得通知的新增阅读，全部未复现。参考动力学检查不等于跟踪成功，低预测误差不等于正确动作梯度，奖励分数不能跨奖励定义直接比较；没有扩大为A3/Q1已支持或算法原创。STR、无离线位置标定IMU、注意力接触、人形记忆、LocoManiLab仅摘要候选，资产未核。

历史公告分区、旧替换稿/部分缺失HTML与HF资产缺口保留。正式hp3090工作树未知；公开分支可读不等于服务器已同步。工程最新科学状态以Q1自己的阶段与真实作业核验为准。

## 保存与同步状态

本地研究记录已保存，原始资料只在忽略目录LOCAL_ONLY，未网盘备份。main与Q1发布均需逐个审查diff/暂存区、diff --cached --check、运行实际script/project_tools/audit_git_payload.py后普通推送，再独立核对分支ref和固定内容字节。当前尚未将新记录写为已推送；完成后补回执。旧成功/失败记录保留，不强推，不覆盖并行工作。

## 下一步与恢复入口

读取[来源清单](../data/manifests/research_watch_check_20261024.json)和[候选](research/README_IDEAS.md)，下一次只对方法/资产实质变化重报；优先参考接触一致性和既有世界模型诊断证据。当前训练固定预算/完整评测/重算/V02渲染由工程任务推进，本日报不执行。
