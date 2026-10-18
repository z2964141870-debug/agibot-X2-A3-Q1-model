# A3 HUMANPLUS 根位移可辨识性上界探针（Exp-2a / HP24–HP25）— STOPPED

2026-10-18。合同：analysis/EXP_CONTRACTS.md §2（判据 2.4 预登记）。假设 H2：允许输入（到达参考关节差分 / 关节 q-v 历史 / 上一动作 / 相对 anchor 朝向 / 重力方向）不能统计恢复未来世界根位移。纯 CPU、零 GPU 零仿真、单次运行；数据只读（HP11 基线 18 个轨迹文件 + 6 参考契约 CSV，28 项输入 SHA 每轮全 PASS）。执行机 Legion-Y9000P，git d87e05a，环境 x2-sonic-isaaclab（py3.11.15 / numpy 1.26.0）。

## 执行史

### HP24（世界系标签，无 clip）— STOPPED_FAILED_POSITIVE_CONTROL
8 项标准库控制测试通过（其中修正 2 处测试自身 bug：horizon-tick 断言方向、ridge 真值测试广播形状；均为注册前修正）。trial exit 3：阳性对照全部门未过（pc ridge / CV = 3.647/0.131 @H0.5，4.807/0.823 @H2.0，1.007/1.997 @H4.9）。main true：H4.9 improvement +36.5% 但 R²=−1.40；H0.1 标签尺度 ~0.02 m 而 ridge RMSE 1.08 m。
根因（证据：identifiability_summary.csv 分段行）：zzx 摔倒段（tick 283 起，roll 161°）特征远出训练分布，无界线性外推产生米级预测（zzx H0.5 ridge 8.50 m）；世界系标签叠加跨被试朝向/速度偏置。

### HP25（合同 2.4 唯一一次 bug 修复，换号重登记）— STOPPED_FAILED_POSITIVE_CONTROL
v2 改动 = anchor（根朝向）系标签 / CV 基线 / 阳性对照特征（改为位移-so-far，anchor 体即 root，绝对位置在该系退化）+ train-only z-score ±5 截断；其余规格不变。8/8 控制测试通过（新增 anchor 旋转、clip 有界两条；其中 anchor 旋转测试首版期望方向写反已修正——代码正确）。28 输入 SHA 全 PASS。trial exit 3：阳性对照仍未过门（pc/CV = 1.824 @H0.5，0.913 @H2.0，0.535 @H4.9）。main true（val 池化，anchor 系）：hold/CV/ridge RMSE = 0.0411/0.0168/0.0517 @H0.1，0.1905/0.1313/0.2293 @H0.5，0.6087/0.8227/0.8080 @H2.0，1.0428/1.9972/1.2008 @H4.9；improvement = −207%/−74.7%/+1.8%/+39.9%；max R² = −0.35（全 H 为负）。

### 分段事实（供决策）
zls/yy 两被试上线性读出一致优于基线（H0.1 ridge 0.0170/0.0210 vs 池化 hold 0.0411；H4.9 1.010/0.805），而 zzx 摔倒段在全部 H 主导池化误差（H0.1 0.0853，H4.9 1.630）。阳性对照特征对 ridge 增益几乎为零（H4.9：pc 1.069 vs main 1.201）——线性读出下，已知当前真实根位移-so-far 的增量信息 ≈ 0；这本身是支持 H2 的观察，但不足以按预登记判据定论。

## 终态与判据适用性
两轮阳性对照均未过门 → 按合同 2.4（单次修复预算已耗尽）**STOPPED：保存证据、不第三次修复、上报 lead**。预登记判据（<5% 且 R²<0.2 → H2 坐实；≥10% → H2 降级）**均不适用**：其前提是阳性对照通过（管道自证有效）。闸门按「跨被试池化 + 线性模型」设计，对摔倒段 OOD 过严；若改分段评估 / 剔除摔倒帧 / 非线性读出，属合同修订，须 lead 与用户批准后另立编号，本轮不擅自扩scope。

## 产物（路径 + SHA256 见 data/manifests/…_result.json）
- data/experiments/a3_humanplus_root_identifiability_20261018_HP24/：registration.json、result.json（FAILED_POSITIVE_CONTROL + 全部分段数字）、identifiability_summary.csv、2×PNG（世界系）
- data/experiments/a3_humanplus_root_identifiability_20261018_HP25/：同结构（anchor 系，终版）
- data/manifests/a3_humanplus_root_identifiability_20261018_HP2{4,5}_plan.json、…_result.json
- script/a3/root_identifiability_trial.py（v2）、script/a3/test_root_identifiability_trial.py（8 控制）
- logs/a3_humanplus_root_identifiability_20261018_HP2{4,5}/console.log（含 3×5 s 预检、内存门 23 GiB、trial 退出码）

## 关联
HP11 数据合同 data/manifests/a3_humanplus_baseline_reference_20261018_HP11_contract.json；E27（根独立反事实边界）；analysis/BOTTLENECK_HYPOTHESES.md H2；zzx tick 283 摔倒按原样纳入（未剔除）。
