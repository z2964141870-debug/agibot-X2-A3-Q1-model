# P2-M9L 千步长跑——拒绝 + 全剂量-崩塌曲线报告

- **日期**:2026-10-20 13:20
- **动机**:用户指令"训练到千步级别"——检验长训下血统包络
- **配方**:M9@32 底座 + m9_motionlib(20A+15C+伪金标×5) + 1024 updates(62.7 分钟全程)
- **产物**:m9l/checkpoints/milestone_1024/(已冻结 REJECTED)

## 1. 剂量-崩塌全曲线(orientation_max_deg)

| 步数 | zj_warmUp | zzx | 均值 | 状态 |
| :--- | :--- | :--- | :--- | :--- |
| 32 (M9) | 23.43 | 17.66 | 16.30 | 过线 |
| **128 (M9C)** | **22.97** | **17.11** | **16.54** | **过线=现役甜点** |
| 256 | 24.41 | 23.32 | 18.84 | 双破线 |
| 512 | 21.34 | 22.45 | 18.09 | zzx 破线 |
| 768 | 20.60 | **178.67 摔倒** | 45.57 | 灾难 |
| 1024 | 20.73 | 26.83 | 17.99 | zzx 破线 |

## 2. 结论
1. **包络破裂点 128-256 步**:zzx 是长训最先崩的动作;768 步处直接摔倒
2. warmUp 单调受益于长训(坐姿轻动作数据的长训红利),但救不了均值
3. **M9C@128 是本配方的验证甜点**——用户的千步之问有了定量答案
4. 下一程 M11 换真 warm-up 窗数据(见 smb_triage/ranking.json)后,甜点位置需重测

## 3. 台账
- 权重与曲线:m9l/checkpoints/* 与 experiments/p2_main/m9l_{eval,diag*}/
- 狩猎交付:data/smb_triage/{ranking.json,TRIAGE_REPORT.md,hunt_features.jsonl,preview_labels.json}
- 入库工位:data/demo100_ingest/(89 场 stage A 毕,stage B 进行中)
