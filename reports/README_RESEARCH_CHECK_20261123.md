# 2026-11-23 研究检查

触发 `2026-11-23T00:59:30.071Z`，北京时间 08:59:30。上次成功检查 `2026-11-22T01:10:12.651+00:00`。公开研究没有确认的新条目，但 Q1 远端从 `1cf8966` 快进到 `f871b80`，出现 FGP Control V2、真机采集与策略训练交接，因此本次决定 `NOTIFY`。见[简报](research/2026-11-23/README.md)与[清单](../data/manifests/research_watch_check_20261123.json)。

## 工程状态核对

- Q1 新增 21 个工程提交、38 个变更文件；最新交接记录为 `reports/DEPLOY_HANDOFF_Q1_FGP_POLICY_TRAINING_V1_20261122.md`。
- 保存的真机记录报告 FGP 6603 有效帧、human_smpl policy 9828 帧，FGP 推理 P50/P95 27.4/40.3 ms，FGP 年龄 P50/P95 31.9/53.0 ms，target-joint error P50/P95 0.207/0.474 rad。
- A2 FGP V2 + CGFA 的同段离线回放完成 119.72 s，最大倾角 13.50°、target error P95 0.515 rad；旧 adapter、root 置零对照失稳。现场退出摔倒按退出流程问题记录，不算 policy 失败。
- 训练交接建议冻结上肢、诊断下肢相位并补接触语义；本轮不启动训练、不发真机命令、不替换默认部署包。SONIC kplanner 现场站立基线保持。

## 检索覆盖

官方 cs.RO/cs.LG/cs.CV new 页面仍只显示 2026-11-20 为最新可见批次；SuperMimic 与 RSL-RL 监测头无实质变化。11 月 21–23 分区未暴露，不能声称完整无新稿。新增公开去重条目 0，新增研究假设 0。

来源核验保存在 `data/research/2026-11-23/source_check.json`；服务器 Git/镜像本轮未独立观察。