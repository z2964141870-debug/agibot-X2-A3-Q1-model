# P2-M11 真·靶向程——门禁全过,新现役最佳

- **日期**:2026-10-20 13:55
- **配方**:M9C@128 底座 + m11_motionlib(20A+15C+真 warm-up 窗 3 场×2,剔除伪金标 175248×5)+ 32 步微剂量
- **判据**:DESIGN_PREDECLARED.md 预声明,四项全过
- **产物**:m11/checkpoints/milestone_32/model_step_000032.pt(指纹 $FP)

## 1. 门禁

| 动作 | M9C(原现役) | **M11** | 判据 | 结果 |
| :--- | :--- | :--- | :--- | :--- |
| zj_walk | 15.92 | **14.48** | — | 优 |
| **zj_warmUp** | 22.97 | **22.72** | ≤23.81 | **✓ 新低** |
| fjw | 10.20 | 10.77 | — | 噪声带内 |
| zls | 15.90 | 16.29 | — | 噪声带内 |
| yy | 17.15 | **13.37** | — | **大幅改善** |
| **zzx** | 17.11 | 18.29 | ≤20.20 | **✓** |
| **均值** | 16.54 | **15.99** | ≤17.44 | **✓ 全血统最佳** |
| 跌倒 | 0 | **0** | 0 | **✓** |

## 2. 里程碑意义
1. **真 warm-up 数据首胜**:smb 狩猎(722 场粗筛→top-3 窗)→G0 审计→靶向,全链路首次闭环兑现
2. 伪金标真相(坐姿轻动作)下的 M9 系机理=正则化;真窗不仅保住还略优 warmUp,且 yy 大幅受益
3. 血统:M3→M9→M9C→**M11(现役)**;M11C 128 步巩固为下一步

## 3. 台账
- 权重:m11/checkpoints/milestone_32/;库:m11_motionlib(41 文件+manifest)
- 评估:experiments/p2_main/m11_eval/;设计预声明:m11_motionlib/DESIGN_PREDECLARED.md
- 窗口审计:data/demo100_ingest/{smpl_pool,win_g0_audit_rows.json}
- 狩猎链:data/smb_triage/{ranking.json,window_pull/}
