# E54 — 通道校准双探针 (A3_ZHY_CHANNEL_PROBE_20261019_E54)

**状态: COMPLETE 2026-10-19** — 双判据通过, winner = P (逐动作标准化通道)。
合同: analysis/E54_CONTRACT.md (含负责人修订: eff_max_bound=1.1608 钉死, t=0 锚恒等沿用 E53 anchor receipt ≤1e-5)。

## 判定 (effect_verification.json, data/ 侧工件)

| 候选 | official4 WDR max (主键 ≤1.1608) | cloth6 (次键 ≤0.183447) | 双过 |
|---|---|---|---|
| P 逐动作标准化 | **1.0367** (029_rotate_place) | 0.181763 | ✓ |
| Q 误差形 | **1.0367** (029_rotate_place) | 0.181763 | ✓ |

选择规则: 双过取 WDR max 更小者, 差 ≤2% 取 P (更简, 仅依赖参考统计) → **winner = P**。
逐动作 WDR (两臂位级相同): 001_walk 0.9888 / 029_rotate_place 1.0367 / 048_balance 1.0049 / 050_march 0.9950。
对照: E53-I8 异常读数 1.294 (march) — E54 的 march = 0.9950, 与 N8 (0.993) 同带, I8 域外异常在读数带内消失。

## 双臂读数位级相同的说明

cloth-constrained 训练域的 trainable scope = g1_dyn decoders (14 层, 见 arm_p_tier16/cloth_contract.json), 动作策略冻结; 两臂注入通道均被 035 零列门控静默 → 两臂 rollout 轨迹位级一致 → WDR/cloth6 相同。通道标准化差异因此未体现在本探针的 rollout 读数中; 判据与选择规则按合同钉死执行 (tie → P)。此解读供负责人裁定后续 (如需区分两臂, 需解冻策略或改读 decoder 漂移量)。

## 执行台账 (5 次注册, 3 次 pre-training 修复, 全部归档)

1. register#1: stats manifest 键带 .csv 后缀 vs 训练词干 → stats builder 词干归一化 (archive: 1st/2nd registration + 1st_stats.csvkeys.json)。
2. train P#1: FiniteCheckpointCallback 硬校验 'Registered E51 update budget changed' (E51 UPDATES=8, 不容 16) → 自建 e54_callback.E54MilestoneCallback (DurableModelSaveCallback + milestone_{step} 包裹, 无预算死锁; checkpoint 路径改 protocol 钉死 checkpoints/milestone_16/model_step_000016.pt)。
3. train Q#1: 'Official saved teacher anchors do not match warm start' (t=0 恒等守卫)。诊断子类 E54InjectedTrainer 捕获真实差值 max_abs=2.918 + pad 打印 **2402→2432** 定位根因: 误差形 func reset 分支返回 zeros_like(sim) (3 宽), env 建场 spec 探测取 3 通道/槽 → 网宽 2432, 槽几何全错。修复: reset 返回 6 宽零 (与稳态 cat[disp,vel] 同形)。
4. 修复后 Q: pad 2402→2462 ✓, 守卫通过 (无 DIAG), 训练+评测+effect 一次通过。
5. 注册#5 amendment_note: P 在注册#4 代码态下训练+评测; reset 宽度修复与诊断子类仅影响 Q; Q 历次失败启动均零更新消耗。archive: 3rd/4th registration, failed_launch_archive/arm_q_tier16_3ch_spec.log。

代码: yuanqi_ops e54_trial.py / e54_protocol.py / e54_obs_terms.py / e54_channel_stats.py / e54_callback.py @ d34df4e (controls 5 项全过, registration 内 stats sha 钉死)。
