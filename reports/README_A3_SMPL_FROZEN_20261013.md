# A3六段合格参考的冻结官方闭环E24

记录日期：2026-10-13，北京时间。实验A3_ZHY_SMPL_FROZEN_20261013_E24。11:08:52停止，ENTRY_ERROR_BEFORE_POLICY；六段ABI合格、0策略回放。旧代码/登记/账本保留，不重复E24。

## 目标与范围

E23有6/7开发参考通过原运动学门。本阶段核验这六段参考经官方CSV与loop模型加载后，冻结035策略是否能够完整跟随并保持平衡。拒绝的下蹲片段不纳入本阶段。训练、真机均不执行。

## 固定协议与判据

独立编号E24；主变量为首次使用E23合格参考测冻结官方035，不与不同动作的历史误差比较。各段原300帧/30Hz、stride1，完整回放、零延迟、无缓冲、seed0、a3_fast；预期策略步数由固定官方重采样器确定。官方PT SHA `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`，vendor `fe6868ba37034f89b912f0fb851bce19f120266d`，0optimizer。

先验证CSV根位置/旋转/关节/时长roundtrip最大误差1e-6，再核验同名身体在两MJCF中的FK最大差2mm（沿用原source FK预算），重采样后实际脚mesh不得穿地超过5mm。全部通过后才进行六段策略回放；不降低判据。动力学判据为每段完整、有限、0跌倒；沿用官方高度<0.45m或绝对roll/pitch>60度的跌倒判据。跟踪误差和动作变化完整公开，不用退出码替代效果。

保存50Hz实际qpos/qvel、脚与floor的contact数量/法向力、实际脚mesh最低点；这是仿真采样，不是现场人体接触真值。原策略控制、模型和物理步进保持官方行为，捕获在指标记录边界只读。独立复算覆盖、跌倒、关节/root/body误差、动作一致性及实际脚几何。

预算为六段一次回放，计算/验收各900秒上限。5秒采样，CPU90/GPU85摄氏度连续两次保护、4GiB保留、首boot变化/OOM/内核/计算异常停止，无重跑或开机恢复。所有计算仅ZHY。worker与仿真子进程共享受保护进程组，确保保护停止覆盖全部子进程。

## 代码与恢复命令

新增`script/a3/smpl_frozen_probe.py`及`script/a3/smpl_frozen_entry.py`，复用既有评测/保护/独立指标核验。六段引用使用完整sequence唯一文件名，防止原文件均名reference.csv造成覆盖。运行前登记源码、输入、官方配置和资产SHA，不改E23及旧账本。

```bash
cd /opt/data/private/hechengzhu/ZHY
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_probe register
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_probe run
data/environments/a3-sonic/bin/python -m script.a3.smpl_frozen_probe verify
```

上述run/verify使用既有环境的监控入口启动受保护worker，不能直接绕过监控调用worker。恢复先读state，已运行状态不得再run。

## 产物与当前证据

- 实验：`data/experiments/a3_zhy_smpl_frozen_20261013_E24/`。
- 日志：`logs/a3_zhy_smpl_frozen_20261013_E24/`。
- 小型核验清单：`data/manifests/a3_zhy_smpl_frozen_20261013_E24.json`。
- 恢复过程中hp3090入口两次SSH超时，后恢复；ZHY仍为boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`。超时只记UNKNOWN，不认定重启；恢复前未启动计算。
- 恢复时source/target HEAD `26a3b4cb441093b67a036342cd78db028bda5d52`且worktree干净；容器CPU额度28核，GPU进程查询空。启动前另执行可靠遥测preflight。

登记11:06:56，运行11:08:36–11:08:52，共15.64秒。六段各300源帧→499策略参考帧，根/关节/旋转roundtrip最大约1.34e-15，原IK模型与loop模型14个同名身体FK最大约1.65e-15m；这些不是策略误差。重采样后脚mesh最低点最差-3.719mm，六段均通过原5mm门槛。

失败为评测包装器的`SimpleNamespace`漏传`summarize_only`，在`evaluate_explicit`进入循环前AttributeError退出；尚未加载官方策略，0回放/0optimizer，没有把退出0或运动学通过当作策略成功。保留源码及console SHA。下一独立E25补全评测参数并复用本次已核验ABI，不重复这六段接口计算。

5健康样本同boot，CPU最高42/GPU26摄氏度、GPU功率最高33.48W，最小容器可用8,572,231,680字节、OOM0、无新关键内核或遥测错误。最大100.10秒间隔是登记到启动空档，不宣称全过程5秒连续覆盖；计算时保护约5秒。退出后GPU进程查询为空。此为入口代码错误，没有确认机器故障。

## 结果边界与保存

ZHY入口py_compile及六段接口/身体FK/重采样脚几何验证通过；E24停止于入口参数错误，尚无策略实测结果。六段都参与过参考开发，旧partition为test也不代表未见最终验证；成功仅支持本开发集冻结官方闭环，不能证明策略改善或真机可用。任何后续回放用新编号，不能边改边重复E24。

本次失败和ABI结果已落盘README/索引/current state及小型SHA清单，待完成本阶段scoped审计提交推送核验。原始数据/模型/执行日志LOCAL_ONLY，无网盘备份完成证据。持续工作入口[长期工程](README_A3_LONGTERM_20261013.md)。
