# humanplus 固定有限更新与匹配效果 HP12

记录日期：2026-10-18。实验：HP12 / R21_001_s0。状态：STOPPED，有限候选全部触及teacher RMSE保护门；终态保存，不恢复或评测早期点。

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

北京时间13:56单次登记完成，执行源码父HEAD为34ac253，登记记录提交a42229f；计划SHA d851d1023169c8ecc307370918a7dfdb73dac1d87bf309e1d3f8ec7a23d97210。13:56:39开始监督，13:56:49通过三采样后启动；URDF转换写新data成功，16环境cuda:0创建，warm_start_loaded.json实证035 Actor/Critic完全匹配、global_step0/空Adam。13:57:29原生训练退出1，阶段50.163秒后STOPPED，未进入正常完成关闭协议。

实际6次整更新+第7次3batch：28次原生Adam提案，27次提交，7组rollout/2688交互，71候选检查、28基线检查（合计99次候选/基线评估）。step0/2/4已耐久保存；step8不存在，0最终效果回放、0停止后张量反序列化/模型验证。该结果不是完整8更新训练通过，不能比较早期模型或宣称改善。

第28次提案前teacher RMSE为0.049996716421312044。1至1/128八个固定比例均降低同batch固定Critic值Actor目标、均满足maxabs≤0.5，但RMSE均超过0.05；最小比例仍为0.050000153109196845，超门1.5310919684e-7。没有放宽门槛、增加比例或跳过batch。拒绝前后参数/Adam摘要均70d73b3b271c74f8f4cda03416bd1e247d9ec4b181c0dc0d8a094514b872a32a，未提交batch的buffer已恢复。该证据支持“固定有限缩步仍在累计teacher漂移边界耗尽可接受候选”，不能证明不存在其他方向，也不是新机性能或硬件故障。

独立标准库train审计：11健康同boot afb6796b-3883-4df4-99ea-bc381c52ea13、CPU最高81/GPU48°C、可用最小18,439,405,568字节、0FATAL/新增OOM/停止后问题/自有组残留；verified=false对应科学STOPPED，不能改为成功。另新增一次只读stop_verification，仅核已有JSON/大小/SHA，核对27提交、拒绝事务未写活状态及step0/2/4身份，未反序列化模型、未调用verify-training或效果阶段。两项审计均禁止复跑。

HP11六衣服5/6不倒、算术均RMSE0.1393274808rad、四官方4/4不倒仍是当前匹配035基线；HP12没有新的策略效果证据。

启动与运行使用已验humanplus_health/supervisor：三次5秒启动门（≥8GiB/无其他计算/GPU空闲/遥测可靠），运行保留≥4GiB，CPU90/GPU85连续两次5秒、每阶段900秒；全boot journal与cgroupv2保留原门。首新增boot/温度/OOM/内核/计算异常停止自有组、留证、暂停heartbeat，不自动重跑。阶段之间不是连续健康证明。

效果门保留：六衣服全部不倒且算术均RMSE改善≥1%；原不倒衣服及四官方逐项关节退化≤2%；世界根/腕≤5%；官方无新增跌倒。通过也仅开发适配，独立seed和真正未见会话仍待验证。

## 保存与同步状态

代码/文档/小清单按scoped审diff、暂存及audit_git_payload后本地提交。GitHub SSH写权限仍未恢复，记录为 `LOCAL_COMMITTED_UNPUSHED`，不重复同一已知失败push。原始模型、checkpoint、轨迹、日志全部 `LOCAL_ONLY`，未核验网盘备份。

## 下一步与恢复入口

恢复先读本README与HP12 plan/result/train_audit/stop_verification。register、train、train审计、stop审计均已单次执行；禁止复跑、执行verify-training、最终效果或早期checkpoint评测，不扩预算，也不换编号重做相同有限缩步。HP12没有活动计算。

下一步为方法选择：仅由现有失败证据与本地既有研究论证实质不同的短训练方案，不重复E46–E50投影/零更新诊断，不直接改teacher门、扫描LR/权重/batch或追加小比例。新方法须独立固定唯一变量、输入/实现SHA、预算和匹配效果门，再按授权范围审查；在形成可审查新计划前无新科学执行。此次为算法保护停止，无机器异常证据，zhy-a3保留原30分钟ACTIVE、状态不变静默，旧hp3090与每日研究任务不变。
