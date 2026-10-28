# 2026-10-28 研究检查与版本更正

记录日期：2026-10-28，北京时间。阶段：RW20261028。状态：检索、分析与研究记录完成；发布核验另见同步回执。

## 目标与范围

接续 10/27 09:24:30 成功检查，围绕 Q1 上肢/根轨迹跟随、模型动作响应与策略更新筛选原始来源。触发时间 10/28 08:59:08.805 北京时间；本次来源检查结算时间见[清单](../data/manifests/research_watch_check_20261028.json)。

本地根 `/Users/yu/Documents/ChatGPT/元启`。工程权威 Git 是持久副本 `data/repositories/q1-rf12-20261028` 的 Q1；完整通用研究历史/去重由 `data/research/2026-10-20/record_checkout` 的 main 维护。未使用缺损的旧 /tmp 工程 checkout。

## 完成内容

形成[五项简报](research/2026-10-28/README.md)：OpTTA 的模型任务目标、HumanoidNDM 的真实数据适配、CATS 的时序正则、MimicAll 的迁移边界，以及 HoloBoxing v2 对手协议更正。四篇相关正文为选择性阅读，不称完整代码审计或复现。

缓存 87 次请求，83 次 HTTP200：HumanoidNDM 作者项目无/有尾斜线均 404；两次 raw GitHub TLS reset 已改用固定 contents API 成功取得同路径文本。没有读取失败内容后声称资产已发布。

完整 RO/LG/CV new 列表分别 220/697/483 条，recent 分别 437/1318/858 条；可见公告到 10/27。LG RSS 只有 321 条，与 new 不一致，因此同时筛选完整 HTML；历史覆盖缺口保留，不称全领域或历史替换穷尽。

14 个已有核查仓库头均未变，包括昨日 HoloMotion/ViT-H 与既有 MorphMimic 专题。OpTTA/CATS 为本次新读实现，未据此建立新常驻监控。Isaac Lab interactive-demos 仅新增 55002191 字节教学网页压缩包的元数据，无新编号框架版本；没有下载。PointHand 只有网页构建 hash 变化。LocoManiLab v2/v3 仅所查作者/版本/日期差异，未重复简报。

## 关键判断

- OpTTA 更新模型预测器，并以预测梯度范数平衡规划梯度；所查 LeWM 代码的 encoder 只接受预测损失。可借鉴任务效果与预测误差分开检查，不能归因 Q1 的唯一瓶颈。
- HumanoidNDM 依赖已有可部署策略、实测关节/IMU 历史和机体模型，G1 的静态扰动适配不是当前 Q1 从零训练或新增真机试验授权。
- CATS 正则位于策略损失，系数从零线性增加；它与奖励中的动作变化罚项需要分开理解。公开代码未定位顶层代码许可，不能只因可访问就认定授权清楚。
- MimicAll 的 4096 是身体变体数，固定训练为 8×8192 环境；陌生家族成功率低和已见 G1 专项收益限定了跨本体结论。
- HoloBoxing v2 明确各组使用自己的冻结预热对手。已在昨日简报追加日期明确的更正，保留 v1 历史和底层预训练成本边界；未声称代码或实验重新运行。

MOTIO、FLORE、Shooting-for-Contact v2、X-WBC v2 仅摘要/提交历史核验，留为后续接口、力控制、动态参考与跨本体背景；不增加训练数据门。MorphMimic 旧专题只补入中央去重。原 122 条去重完整保留，新增 12 条后为 134；其中含版本观察、背景与旧专题归档，不代表 12 条新科研进展。没有新假设，H001 不变。

## 环境与证据

09:12 左右只读登录用户25386的 `/opt/data/private/ZHY`；AGENTS、CURRENT_STATE、RF12结果及MorphMimic专题四文件与Mac大小/SHA一致。容器没有 git 命令，服务器 Git 状态不可确认，未安装或修改服务器 Git。持久 Q1 及 main 两个发布副本 fetch 后干净，原始基点分别 `abf633fa55e4174e29430a609ef43a5e4ad65845` / `8803044a095f6c217e739b835082eb6b2a1d9e1c`。现场凭据、进程/GPU和训练不在本次核验范围。

工程状态取自最新 RF12 完成记录：首门负面结果后停止，监督已暂停。没有因旧 RUNNING/ACTIVE 记录恢复作业。没有安装依赖、执行第三方代码、启动仿真/训练、操作真机、下载二进制/数据或启用子agent；未改另两项自动任务。

`logs/research_watch/2026-10-28/` 保存 fetch、source_comparison、context_check、static_text_verification、version_differences 与发布日志。`data/research/2026-10-28/sources/` 保存忽略的原始来源；缓存为 MAC_LOCAL_ONLY，未声称网盘备份。

9 份固定文本均完成 API base64 解码、大小、Git blob SHA1 与完整固定 tree 校验（OpTTA 954 条，CATS 320 条，均未截断）。本次只静态阅读：
- OpTTA README/LICENSE/LeWM objective/adapter/config；
- CATS README/PPO/线性调度/机器人说明。

可复核本地收据的命令：

```sh
python3 logs/research_watch/2026-10-28/finish_review.py
python3 script/project_tools/audit_git_payload.py
```

前者只重新校验已缓存内容，不联网、不执行上游实现；后者在各发布 checkout 暂存本次文件后执行。

## 保存与同步

[检查与来源清单](../data/manifests/research_watch_check_20261028.json)、日报、索引和候选已保存。main 接收完整去重，Q1 接收本次检查、简报、更正和指针，保留各自工程入口和 Q1 原中央状态。

显式审核 diff/暂存区、运行实际 payload audit 后普通 push main/Q1，并核远端和固定文件内容；实际提交、服务器镜像与未完成项以[同步回执](../data/manifests/research_watch_sync_20261028.json)为准。回执自身提交由后续 Git 历史识别，不递归嵌入未来自身 SHA。

## 下一步

研究候选留待已有工程瓶颈明确后选择；不自动调整奖励、采样、策略或训练预算。以后检查从 main 去重与 Q1 最新 CURRENT_STATE 恢复，并优先看 HoloBoxing 更正和本次资产边界。无需用户处理持续未变的 git/原始缓存覆盖缺口。
