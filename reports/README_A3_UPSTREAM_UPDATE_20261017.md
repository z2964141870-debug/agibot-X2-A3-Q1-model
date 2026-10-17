# SONIC for A3 上游更新核验

后续已并入[humanplus HP01](README_A3_HUMANPLUS_20261017.md)保存，SSH入口与发布状态以该阶段为准；下文Mac-only为核验完成当时的状态。

记录日期：2026-10-17，北京时间22:21起。状态：GitHub公开版本/差异及发布清单核验完成；Hugging Face在线元数据UNKNOWN，服务器入口SSH_UNKNOWN。仅只读查询与小型记录，无vendor升级、依赖安装、模型下载或计算。

## 目标与结果

用户要求先查sonic_for_a3是否更新。官方来源为AgibotTech/sonic_for_a3；项目登记固定vendor为fe6868ba37034f89b912f0fb851bce19f120266d，2026-09-25提交。服务器当前不可达，本次不能重新证明远端实际vendor或工作树状态。

2026-10-17 GitHub API核验默认main仍为c9df753d31ef63c930b825b1b6767cabd37c1549，提交时间2026-10-09T03:33:38Z，即北京时间10月9日11:33:38。与固定版本比较ahead=1、behind=0、total_commits=1，和10/12已核验记录相同，没有10/12之后的新main提交。分支列表仅main，tags和GitHub releases均空。

唯一提交主题为Enable native Hugging Face download statistics through release metadata。完整差异只有5文件：.gitignore、a3_hf_manifest.json、docs/a3_training2sim2deploy.md、download_from_hf.py、tests/test_download_from_hf.py。下载器在选择PT/ONNX/RKNN时同时获取共享config.json，以支持下载统计；不属于Actor/PPO、输入编码、机器人资产、接触、奖励或运动跟踪修改。

## 模型清单核验

通过GitHub contents API分别取固定版本与当前main的a3_hf_manifest.json，结构化对比所有条目：旧14项、新15项，原有14项全部字段相同、无删除。新增609字节config.json，保存为checkpoints/035_step200000/hub_config.json。清单HF revision由db1f40787d580af6d1e53df0df72242904a1dae5变为9bbddf8730389c2a1980e69c8b410a8dbcc3e28e；revision变化不等于权重变化。

035 PT仍登记402050931字节、SHA256 9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d；配置、ONNX/RKNN等既有条目也未变。没有证据表明升级该提交可改善当前策略表现。

Hugging Face models API查询20秒超时，curl退出28；未核验其当前实际HEAD、文件或是否有未回写GitHub清单的资产更新。结论限于已取到的GitHub公开main/分支/发行与固定发布清单，不宣称模型站已全面核验。

## 决定与边界

保持固定vendor与官方035基线。本次仅查更新，不能把下载统计提交当作行走/接触/训练修复，也没有运行作者新增测试。E51计划、预算、效果门不变；现有SSH_UNKNOWN与待同步记录继续保留。原hp3090只尝试只读Git查询，连接100.70.125.121:22超时，未到达ZHY，不推断重启或机器故障。

## 来源与复查

- 官方仓库：https://github.com/AgibotTech/sonic_for_a3
- 当前提交：https://github.com/AgibotTech/sonic_for_a3/commit/c9df753d31ef63c930b825b1b6767cabd37c1549
- 固定比较：https://github.com/AgibotTech/sonic_for_a3/compare/fe6868ba37034f89b912f0fb851bce19f120266d...c9df753d31ef63c930b825b1b6767cabd37c1549
- API复查：repos/AgibotTech/sonic_for_a3下commits/main、compare/固定SHA...main、branches、tags、releases、contents/a3_hf_manifest.json?ref=固定SHA。
- 本次来源/版本/SHA及缺口小清单：data/manifests/a3_upstream_update_20261017.json。
- 10/12同一提交的既有结论：README_A3_UPSTREAM_LIKELIHOOD_20261012.md。

## 保存与下一步

本报告、索引/current state和小清单仅Mac保存，尚未提交/推送或同步服务器：Git工作副本位于当前不可达的hp3090。连接恢复后连同20:31待同步记录审diff/暂存、audit_git_payload.py、scoped提交推送main并核对，保留并行修改；不能称本次记录已在GitHub备份。无需为这次元数据更新升级vendor或下载模型。之后按原E51恢复入口核验实际现场；模型站状态只在可访问时补核验，不阻塞已固定输入的实验。
