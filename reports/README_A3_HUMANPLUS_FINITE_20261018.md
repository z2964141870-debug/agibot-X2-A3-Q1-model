# humanplus 固定有限更新与匹配效果 HP12

记录日期：2026-10-18。实验：HP12 / R21_001_s0。状态：实现完成，尚未登记或执行训练。

## 目标与范围

在 humanplus 新根运行已授权的原固定8更新方案，以 HP11 新机匹配基线为唯一数值对照。全部七段衣服仍为开发参考；本实验不能证明真正未见会话泛化。旧 E44/E45、E46–E50 和 HP02–HP11 终态保持。

## 完成内容

- 新增独立 `humanplus_finite.py` 登记器、监督入口与最终效果汇总，不导入或运行旧 ZHY `finite_retention_trial` 登记器。
- 复用原 `FiniteRetentionTrainer` 和 `FiniteAdamW`，原035 Actor/Critic、空Adam、seed0、16环境×24步、E45四段（含zzx）、原reset/奖励/路由/cached实际token/teacher系数1/fullclip0.1/KL名义LR保持。
- 8更新、32原生Adam提案及接受提交、3072交互；固定实际位移比例1至1/128，384帧teacher RMSE≤0.05/maxabs≤0.5，同batch固定Critic值Actor目标不增超过1e-7。最多256候选、1728额外forward；保存0/2/4/8，仅最终8评测六衣服四官方完整8769步。
- 新增标准库 AST 适配，仅替换环境构建的存储代理、Kit文件日志级别和末尾直接退出：USD写本实验data；原生训练结束须有8更新/32提交/checkpoint/同步收据，再公开clear_instance和app.close；父进程核验实际退出码及正常关闭标记。vendor/共享环境源码不改。
- 新增独立checkpoint/事务核验、阶段健康审计和最终效果审计。每阶段前要求上一阶段独立审计通过，不恢复STOPPED或阶段内部分回放。

## 关键决定及依据

vendor `train_agent_trl.py` 原结尾 `os._exit(0)` 跳过正常关闭；HP09已证明Kit正常关闭可能直接原生退出而不返回Python，故使用同组标准库broker收取真实退出码。AST按固定结构核验，变形即拒绝，不做文本搜索替换。URDF转换只改变usd_dir/usd_file_name，比较其余cfg保持一致；工作cwd为本实验workspace，仅gear_sonic只读链接指向vendor，registry-cache落新目录。Isaac默认Kit运行日志仍可能写候选环境默认logs目录，如实登记，未改环境依赖。

## 版本、命令与证据

- 主机根：`humanplus@100.101.30.68:/home/humanplus/YUANQI_A3`。
- 实现父HEAD：`f53dacc5a1385f3d67af9a243a0c91b519530b83`；精确执行版本以登记清单source SHA为准。
- vendor：`fe6868ba37034f89b912f0fb851bce19f120266d`；环境只读 `/home/humanplus/anaconda3/envs/x2-sonic-isaaclab`。
- 新适配4项标准库控制Mac/新机各通过，固定vendor AST静态核验通过；不复跑旧张量/PhysX/模型控制。仅元数据确认torch2.8.0+cu128、trl0.28.0、Transformers4.57.6、Accelerate1.14.0，不据此称训练运行通过。
- 计划/结果/审计：`data/manifests/a3_humanplus_finite_20261018_HP12_{plan,result,train_audit,verify-training_audit,cloth_audit,official_audit,effect_audit}.json`，仅已实际生成文件可作证据。
- 新data：`data/experiments/a3_humanplus_finite_20261018_HP12/`；训练：`data/training/a3_humanplus_finite_20261018_HP12/R21_001_s0/`；日志：`logs/a3_humanplus_finite_20261018_HP12/`。

仅首次、实际SHA/资源门满足且登记尚不存在时执行：

```bash
/usr/bin/python3 -B -m script.a3.humanplus_finite register
/usr/bin/python3 -B -m script.a3.humanplus_finite run --stage train
/usr/bin/python3 -B -m script.project_tools.humanplus_finite_audit --stage train
```

后续仅上一阶段完成、独立审计及README保存后依次 `verify-training`、`cloth`、`official`、`effect`。不得批量盲跑、复跑、评测早期点或扩预算。

## 结果与边界

目前只完成新适配实现与标准库控制；未登记、0新增策略/rollout/训练，完整16环境与训练模型仍待实际执行。HP11六衣服5/6不倒、算术均RMSE0.1393274808rad、四官方4/4不倒保持，035没有新改善证据。

启动与运行使用已验humanplus_health/supervisor：三次5秒启动门（≥8GiB/无其他计算/GPU空闲/遥测可靠），运行保留≥4GiB，CPU90/GPU85连续两次5秒、每阶段900秒；全boot journal与cgroupv2保留原门。首新增boot/温度/OOM/内核/计算异常停止自有组、留证、暂停heartbeat，不自动重跑。阶段之间不是连续健康证明。

效果门保留：六衣服全部不倒且算术均RMSE改善≥1%；原不倒衣服及四官方逐项关节退化≤2%；世界根/腕≤5%；官方无新增跌倒。通过也仅开发适配，独立seed和真正未见会话仍待验证。

## 保存与同步状态

代码/文档/小清单按scoped审diff、暂存及audit_git_payload后本地提交。GitHub SSH写权限仍未恢复，记录为 `LOCAL_COMMITTED_UNPUSHED`，不重复同一已知失败push。原始模型、checkpoint、轨迹、日志全部 `LOCAL_ONLY`，未核验网盘备份。

## 下一步与恢复入口

先核实际HEAD、代码/输入SHA、HP11合同和实时资源，再单次HP12登记与train。每个重要阶段立即更新本README、索引和current state后才能继续下一阶段；任何STOPPED不续跑。
