# E52 — ZHY Adam 位移投影三臂对照（第 0 步机制判别）

状态: **COMPLETE**（单次运行纪律保持；候选从未部署；hechengzhu 035 与全部既有产物零接触）

## 设计
- 固定起点: E45 step4 `model_step_000004.pt`（3e679581…）；Adam 矩 step=16 还原（时刻摘要回执校验）；16env×24steps×seed0；FP32；编解码器/标准差冻结；奖励/路由/KL 调学习率原样。
- **臂 A** 原 Adam + 原教师 0.05 硬门（期望读数=第三复现，不算失败）。
- **臂 B** ProjectionAdamW：影子步原生 AdamW 提案 → E50 算子投影（d' = d − max(0,⟨g,d⟩)/‖g‖²·g）→ 无条件提交；门只记录不拦截。
- **臂 C** 原 Adam LR×0.1（2e-7，自适应窗 1e-7~2e-6）。
- 8 更新预算；里程碑 0/2/4(8)；训练验证 §1.4 全绿（改动张量 ⊆ g1_dyn 14 张，critic 改动，还原矩与起点一致）。

## 结果（六布料平均 RMSE，基线 E26 = 0.1413798）
| 臂 | six-cloth mean | vs E26 | 布料跌倒 | 官方 world 比值(root/wrist) | 结论位 |
|----|----------------|--------|----------|---------------------------|--------|
| A | 0.197386 | +39.6% | 2/6 | 1.707 / 1.704 | step20 撞门（~19-20 现象第三次复现，记录性） |
| B | 0.220072 | +55.6% | 1/6 | 1.695 / 1.714 | 比 A 差 11.5pp → **投影机制不支持** |
| C | 0.167798 | +18.7% | 1/6 | 1.606 / 1.642 | 三臂最优，比 A 好 15.0pp → 指向步长（H5 方向） |

## 判据结论（§1.5 预注册）
1. **机制判别: NOT_SUPPORTED** — B−A = −11.5pp（需 ≥+3pp 于劣化更轻方向）。Adam 位移方向不是主因；**转移 H5（步长/更新幅度假设）**。
2. A 臂撞门步 20 vs E44/E45/HP12 的 ~19-20：**第三复现成立**（记录性判据）。
3. 官方门：三臂全部超出 +5%/+2% 预算（自 E45-step4 起点继续更新本身就在退化；无臂可部署）。
4. B 臂首步教师 RMSE 变化 0.00156 > 0 @E50 量级 → 记录为 **FP32 舍入下限读数**。
5. B 投影全量活跃：8 更新 × 112 参数步 = 896 步投影 >0；教师门全程未触（max 0.038 < 0.05，按设计只记录）。

## 产物（manifest: data/manifests/a3_zhy_projection_finetune_20261018_E52.json，32 项含 SHA）
- 代码: yuanqi_ops/{e52_protocol,e52_trainer,e52_trial,e52_anchor_probe,e52_verify_training,test_e52_controls}.py（10/10 控制测试通过）
- 训练: yuanqi_ops/data/training/…E52/arm_{a,b,c}/milestone_*
- 训练侧评估收据: yuanqi_ops/.../eval_arm_*_{cloth,official}.json
- 模拟评估产出（修正案: 仅新命名空间）: repo data/experiments|logs/a3_zhy_projection_finetune_20261018_E52/eval_arm_*
- 对照表: OUT/unified_table.json；效果验证: OUT/effect_verification.json
- 锚点: OUT/anchors_e45_step4.pt（bd96fceb…）
