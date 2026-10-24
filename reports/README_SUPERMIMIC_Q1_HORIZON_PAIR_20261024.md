# Q1 R04想象时域匹配训练

记录日期：2026-10-24，北京时间。阶段：SM-Q1-R04。状态：独立入口与本地7控制通过、1项PyTorch继承控制跳过；服务器控制、真实计划登记与训练尚未进行。

## 目标与单变量

继续SuperMimic for Q1世界模型复现。D07发现走路长时根预测偏差，D08在4个已访问状态验证局部动作响应：走路根H8为16/16正向、误差中位0.4412；H32为11/16/0.7595。不是完整actor梯度证明，也不保证短时域更好。本次检验将policy imagination horizon从32改为8能否改善完整动力学跟随。

两个独立臂共同从已完成R03 final加载三网络、完整Adam和return bins，父checkpoint SHA `10d0838663b5d24215f493ce0738228597452e899b1a378df6d6b453b7e10ea7`。A_H32为匹配基线，B_H8为候选；仅policy_trainer.horizon不同。两臂使用相同fresh replay种子2026102404、相同输入/新接触/奖励/WM k8/网络/优化器/学习率日程、相同3000采集轮及更新频率，不与原R03不同预算直接作为因果对照。

每臂16环境、48000真实控制转移、1149组WM及1099次actor更新，训练后累计18000采集轮；以真实receipt为准。保留128采集轮和50组模型预热，学习率日程继承offset15000，不重置Adam。新replay与episodes不等于原进程精确恢复。H8每次想象计算与critic样本量自然更少，本试验匹配真实交互/更新次数，不匹配想象帧数、算力或墙钟时间。

## 实现与继承

新增独立`script/supermimic_q1_horizon/`，不编辑已冻结`supermimic_for_q1/`或`supermimic_q1_extension/`。旧续训入口只接受已完成R02 pair，且其父args结构不适用于R03；新入口明确验证R03的原终态、8个成功事件、完整summary重算、保存及匹配评测审计，再登记共同父模型。

`horizon_contract.py`复用原严格张量/Adam加载器，只对已登记H32/H8规范化horizon后核对完整合同，其余字段仍拒绝变化。进程内临时绑定校验器、finally恢复；新绑定/源码身份写manifest。证据字段兼容别名明确记录`pair_receipt=R03终态`、`comparison=R03 summary`，不称其为R02批量结果。

`train_horizon.py`在独立目录沿用旧采集、2/5更新、预热、学习率、检查点与健康保护循环；读取R03顶层warmup字段。三网络/Adam/bin从原权重加载；新入口/全部源码进入manifest。`run_horizon_pair.py`复用原controller的子进程运行和健康保护，串行执行两臂共16阶段；新保存审计复用原13项数值审计，并替换为明确的R03谱系与单变量合同。

每臂训练→保存审计→initial三片整段动力学评测/独立重算/位移分析→final相同流程。最后检查两臂设置仅H/输出不同、各自初始权重精确继承同父、三段初始轨迹SHA相同；自动输出完整匹配比较，不自动选赢家或追加预算。大权重原件留服务器LOCAL_ONLY。

## 控制与评判

本地`python3 -m unittest discover -s script/supermimic_q1_horizon -p test_horizon_pair.py -v`原生退出0，7通过、1跳过（Mac无PyTorch）。已覆盖仅H变化接受、额外gamma/资产/H16变化拒绝、匹配种子保留、加载抛错恢复、共享锁、失败阻止下一臂及禁止重跑、三片完整评测安排。服务器必须8项全通过，含3个微型Linear/Adam继承fixture；fixture不是Q1训练或科学结果。

主要判断：走路整片完成/早倒、腿部关节RMSE、水平根跟踪RMSE和离起点最大水平距离；同时报告全身、浅蹲和复用开发出拳，避免只按奖励选方案。未达原0.15rad或实际移动要求时仍如实报告失败；不因某项改进自动称可用。单种子/固定A后B顺序与反复使用开发片段，不能支持统计显著性或独立泛化结论。

最多新增96000训练控制转移；四次三片评测的完整上限13948控制步，共109948控制步/2198960物理子步。真实早倒保留，不重置补齐或替换。A保存/评测任一阶段失败则整次终态失败、B不启动；不覆盖失败、不自动重启。

## 版本、入口与产物

代码副本`/tmp/yuanqi-supermimic-q1-20261023`、分支Q1。D08结果已发布并核对`00936b81b9a69e0ea866ab517352a97a78e2c793`。服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。冻结输入科学发布`d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429`，新增代码由release/Git历史/执行SHA标识。

先在已推送Q1的Mac副本生成小release，再部署全部新增源码/两个analysis依赖并核对release。服务器项目根运行：

```bash
"$PY" -m unittest discover -s script/supermimic_q1_horizon -p test_horizon_pair.py -v
"$PY" script/supermimic_q1_horizon/run_horizon_pair.py register --root "$PWD" --output data/supermimic_for_q1/R04_H32_H8_matched3000_20261024 --release data/manifests/supermimic_q1_R04_release_20261024.json --parent-experiment data/supermimic_for_q1/R03_B128_extend12000_20261023 --diagnosis data/supermimic_for_q1/D08_R03_action_response_20261024
"$PY" -u script/supermimic_q1_horizon/run_horizon_pair.py execute --plan data/supermimic_for_q1/R04_H32_H8_matched3000_20261024/plan.json
```

计划目录`data/supermimic_for_q1/R04_H32_H8_matched3000_20261024/`；A/B各自`run/`保存权重/metrics、各自评测/审计/位移结果；顶层保存plan、started、status、events、comparison、receipt。日志目录`logs/supermimic_q1_R04_H32_H8_matched3000_20261024/`。重启/失败保护沿用容器30GiB、GPU低于86摄氏度/空闲至少2GiB、OOM不增、磁盘至少10GiB及15秒controller采样。

## 保存与下一步

先审核/提交/推送并独立核对Q1；服务器8项控制通过后register，检查真实输入/16阶段/预算与SHA，再启动一次有限流水线，确认起始继承/真实采集与原进程存活。立即记录启动证据并同步Q1。尚无R04最终结果，原R03仍保留，A3停止；全部模型/视频/数组不下载到Mac，网盘未验证。
