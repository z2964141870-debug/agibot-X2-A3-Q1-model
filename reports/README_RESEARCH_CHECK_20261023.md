# 每日研究检查 · 2026-10-23

日期：北京时间2026-10-23。阶段：research-watch-20261023。状态：**来源分析、研究内容发布与独立核验完成；正式服务器工作树未知**。

## 目标与范围

执行yuanqi每日检查，触发2026-10-23T01:03:20.444Z／北京时间09:03:20；接续10/22 09:13:36的成功来源检查，优先过去24–72小时及最新可见公告。本轮只检索、静态分析和研究文档发布；不训练/仿真、不运行第三方代码、不安装依赖、不下载/加载模型、不操作机器人。

## 完成内容

- 重读AGENTS、报告/current、研究约定/索引/候选、96条去重及阶段模板。正式hp3090一次有界Git状态连接超时，工作树/HEAD未知，不重报未变化故障。
- 独立发布副本初始干净，8ff69c6快进到15a9bfb，再快进至ed8f85f65802c7d21ddd4ce01a5029ab6a0a52bd；首次推送遇并行工程提交aaccc2208959a0da55cff2b2c7dc07931229d50b，已将本次未发布研究提交rebase到该版本。保留全部较新R04、D03、渲染及J01工程记录；Mac工程入口也有更新，研究行之外分别保留，不整份互相覆盖。
- 当前工程已转入独立SuperMimic for A3，不沿用10/18“全工程暂停”的旧摘要。R04八段全早倒、三个可用门false；D03根H32差于CV但关节/整体状态有学习收益，不能推出模型完全没学到或动作梯度错误。aaccc22为J01实现/冻结快照；后续f62b9e8工程记录已完成J01：方向多数正确、长期幅度不准，未证明唯一早倒原因或完整actor梯度正确。下一为匹配预算H8/H32方案；本研究不实施/调度它。
- RO/LG/CV最新可见公告10/22，new含替换120/379/212条、recent496/1528/958条。三个RSS数量本次分别匹配new；昨天LG数量不一致的历史事实保留，未补历史全部分区/替换版本。
- 三篇新摘要/提交历史及三篇方法/实验/限制相关正文核验，形成[三项简报](research/2026-10-23/README.md)：SMR参考生成、A2I-Calib++标定信息条件、VLA核方差诊断边界。
- SMR官方固定树/README仅展示站/论文/媒体，未见训练代码或权重。明确专家未来信息、学生严格因果和下游80ms缓冲不同；专家在学生划分之前训练，学生留出不能证明专家跨新动作泛化。G1指标不迁作A3。
- VLA不确定性作者库f658266已发布，固定README/LICENSE/两个实现文件，解码后大小、Git-blob与SHA256核验；未安装或执行。核方差的几何覆盖不等于动作正确率/跌倒概率，代码还有可选目标辅助投影分支，本次不把所有配置称为无标签或完整论文复现。
- A2I-Calib++的命题理想条件、预标定小腿IMU/编码器/刚体运动链/接触参考前提分别记录。真实重复性与里程计对LIO参考不能替代绝对标定真值，已查原文未定位代码入口，不断言全网未开源。
- 六原关注仓库/两组发行/七旧作者页及PredActor摘要未变，SuperMimic作者页与昨日专题缓存正文/链接未变。CuboiLeo第三方库当前d9be385已被工程选用，只补新版本去重；不宣布官方开源，也不重复播报工程已知变化。
- 去重96→102：三篇论文、两个作者仓库与一个工程已知复现提交；旧96条原样保留。只补已有方法候选，H001仍待验证，无H002/新原创假设。

## 关键决定及依据

今天通知三项与参考可执行性、标定可辨识性和诊断可信度直接相关的阅读。SMR最接近衣服→SMPL→重定向瓶颈，但其专家库/本体和下游控制器都有成本；不为追新闻重造A3方案。A2I不把机器人已知运动链当传感衣条件；核方差不替代J01。官方035保留，实验路线以最新工程证据和授权为准。

## 版本、命令与产物

- 来源分析截至2026-10-23T09:17:01+08:00；48个研究HTTP请求45成功。三个raw.githubusercontent.com文本请求连接重置，后续均以固定GitHub内容API成功取得；Git/SSH、web引用抓取和发布核验不计入研究请求数。HF今天未请求，不能写成今天又超时。
- 根：/Users/yu/Documents/ChatGPT/元启；通用研究服务器：hp3090:/media/yu/FAFF-E9771/YUANQI，本轮未能核验；不连接新A3科学执行主机。
- 原始来源：data/research/2026-10-23/sources/。执行：logs/research_watch/2026-10-23/fetch_results.json、source_comparison.json、source_evidence.json。
- 简报：reports/research/2026-10-23/README.md；小型来源清单：data/manifests/research_watch_check_20261023.json；去重状态：data/manifests/research_watch_state.json；候选：reports/research/README_IDEAS.md。
- 发布副本：data/research/2026-10-20/record_checkout，main；GitHub沿用用户指定远端，通过ssh.github.com:443普通推送，不改remote、不强推。
- 自有标准库解析脚本与验证日志只留logs/；未导入/执行来源代码。完整源文件/权重不提交Git。

可复核命令：

```bash
python3 logs/research_watch/2026-10-23/compare_sources.py
python3 logs/research_watch/2026-10-23/build_source_evidence.py
python3 logs/research_watch/2026-10-23/inspect_text.py smr_full.html '80-ms' 'before this split'
python3 logs/research_watch/2026-10-23/inspect_text.py a2i_full.html 'pre-calibrated' 'consistency alone'
python3 logs/research_watch/2026-10-23/inspect_text.py uq_full.html 'Unfamiliarity is not harm' 'Limitations'
```

网络重抓是新快照，不覆盖本次缓存；论文安装/部署命令只作资料。解析器输出不等于全文所有公式/实验已复核。

## 结果与边界

三条均为10/21首稿、10/22公告、10/23首次核验。SMR没有A3验证；其下游训练8GPU配置不是最低算力要求，模型单步速度不是全链路延迟。A2I在四足和已知运动链成立，不能等同衣服或头环SLAM。VLA操纵实验不证明模型敏感性正确，陌生检测不等于失败因果归因，支援策略无优势时接管无益。

较早公告分区、历史替换稿、部分缺失HTML和HF资产依旧有缺口。没有声称搜尽、代码可运行、独立复现或新方法原创，也没有新增工程结果/作业。

## 保存与同步状态

本地简报、候选、来源/去重和入口已保存。研究内容136fa58fbf74bb4ada1cad8cc707ec5571c067d4已普通推送main并于2026-10-23T09:37:57+08:00独立核验：GitHub分支API一致，五份固定文件解码、Git-blob及SHA256通过，九文件与本地或研究增量相符。diff/暂存区审阅、diff --cached --check和audit_git_payload.py通过；45来源SHA、五固定源文本及旧96条保持校验通过。两次推送遇并行工程提交被正常拒绝，先后rebase到aaccc22和f62b9e8后成功，所有工程文件/最新J01完成记录保留，不强推。随本回执提交把简报背景中过时的“J01待执行”改为已有诊断，不改论文正文/日期；最终提交号由Git历史提供。见[回执](../data/manifests/research_watch_sync_20261023.json)及logs/research_watch/2026-10-23/documents_verification.json。正式服务器工作树未更新/未核验，原始缓存LOCAL_ONLY，未网盘备份。

## 下一步与恢复入口

从本次清单接续，仅对实质变化重报；优先SMR实际训练资产、现有工程动作响应诊断及传感衣数据/时钟合同。工程计算依工程任务自己的授权和README，不由本日报调度。正式服务器可达时先核Git与并行修改，再合并研究增量；公开main成功不等于服务器工作树已更新。
