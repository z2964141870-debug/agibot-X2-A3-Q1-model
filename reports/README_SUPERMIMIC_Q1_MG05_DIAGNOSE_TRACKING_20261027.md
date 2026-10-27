# MG05：Q1诊断、tracking修正与一小时验证

记录日期：2026-10-27，北京时间。状态：A–E内容和证据齐备；唯一RF10试验正常完成，但C01明显退步。正式工具结算在发布/镜像核验后执行，以get_goal实际返回为准。结论见[RF10结果](README_SUPERMIMIC_Q1_RF10_RESULTS_20261027.md)。

## 用户目标与范围

1. 解析上肢指令偏离参考、走路真实接触与滑移，以及世界模型动作改进方向是否可靠。
2. 按证据改善观测、优化真实tracking，可适当调整reward比例与数值，不以reward好看代替效果。
3. 使用唯一subagent检查师兄SuperMimicHP更新，双向核查，不默认上游或本项目正确。
4. 综合选取测试方案后，独立4096环境训练一轮，实际训练最多3600秒，验证改动是否有效。

原25245停服后，用户新25386入口恢复；先诊断/契约再执行唯一RF10。RF09及所有旧冻结来源、失败记录保留。没有第二训练、续训或新agent；最多一小时预算满足，六格图和≥30秒多类本地MP4交付。目标完成不代表训练方案取得改善。

## 阶段与证据

| 阶段 | 状态与实际结果 | 证据 |
| --- | --- | --- |
| A | 完成。上肢持续指令bias、原生接触/滑速、H1/4/8/32和唯一上游核查；未锁定唯一根因 | [离线信号](README_SUPERMIMIC_Q1_MG05_SIGNAL_ANALYSIS_20261027.md)、[BD03](README_SUPERMIMIC_Q1_MG05_NATIVE_CANDIDATE_20261027.md)、[上游](README_SUPERMIMICHP_Q1_UPDATE_20261027.md) |
| B | 完成。独立C01/obs102/合法转移bins/关节tracking，CPU5组12状态及完整4096 profile通过；未复用profile模型 | [C01契约](README_SUPERMIMIC_Q1_MG05_NATIVE_CANDIDATE_20261027.md) |
| C | 完成。唯一RF10从零3500.992597秒/3284轮33890304转移，WM32840/actor2784，worker正常reap，36项审计通过 | [实际运行](README_SUPERMIMIC_Q1_RF10_C01_20261027.md)、[训练收据](../data/manifests/supermimic_q1_RF10_training_receipt_20261027.json) |
| D | 完成。330片初末评测、RF09近一小时/五小时共同前缀、双tracking头、两次复评和动作分解；候选明显退步 | [结果](README_SUPERMIMIC_Q1_RF10_RESULTS_20261027.md)、[汇总](../data/manifests/supermimic_q1_RF10_results_20261027.json) |
| E | 完成。六格PNG/PDF/CSV、初末各54秒MP4，30小文件逐大小/SHA；12媒体检查和19结果/CSV检查通过 | [交付](../data/manifests/supermimic_q1_RF10_local_delivery_20261027.json)、[完整验收](../data/manifests/supermimic_q1_MG05_completion_audit_20261027.json) |

## 关键结论与边界

RF09的手势误差主要表现为策略持续给错目标，而不是关节无法跟随其目标；普通走已有真实单支撑，末子步接触点滑速较大。六状态闭环动作方向随horizon变弱，走路H32尤其弱；不等同完整actor目标已被证明错误。

上游/本项目共享尾bin末帧跨片bug实际复现；C01修复不删数据，也不设额外数据门。上游新提交主要是工具/资产归档，未提供新的学习公式或已独立复现的Q1改善。详情按固定提交和代码行保存在上游报告。

C01同时改变观测、tracking权重/带宽和采样，不能单因素归因。其验证提前终止29/30（近一小时RF09为2/30），完成42.88%（原95.70%）；根/腿/腕轨迹大幅退化。四条原生片段PD目标变化RMS1.91–2.36rad/20ms、动作变化罚压过正tracking，说明明显指令振荡。两次独立复评仍退步。少量上肢均误差下降不支持全身跟踪改善，不能靠更换reward口径或延长训练掩盖。

## 版本与恢复

权威Git`/tmp/yuanqi-supermimic-q1-20261023`的Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。RF10冻结`83dcc76bf1cd02e4928493e17b4522cb9d5ee5a5`，同300/30参考清单SHA`ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337`。后续报告提交不改变运行来源。

唯一服务器实体根`/opt/data/private/ZHY`；连接与Python见[CURRENT_STATE](README_CURRENT_STATE.md)。历史paused、待入口、RUNNING和“待评测”都已被实际收据覆盖；不重复旧命令/profile/训练。文档/小收据发布与镜像必须核SHA，权重/大轨迹不拉Mac。

## 保存与后续

本轮结果、限制和失败证据完成后统一发布Q1并独立核远端，再镜像Mac与25386；本次提交身份由Git历史提供，外部同步收据保存三端核验结果。既有仓库/每日研究任务保持，不新增RF10自动任务。

后续优先拆开已证采样bug修复与观测/奖励变化，单独确认动作幅度、reset和模型长程方向。当前停止C01推进、保留RF09基线；不自动开始第二轮，不承诺加时有效。
