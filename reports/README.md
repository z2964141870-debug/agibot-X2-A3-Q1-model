# Yuanqi Ops 战役总结与阶段报告索引 (reports/)

本目录遵循标准四分法规范（data/, logs/, script/, reports/），集中收录所有战役的最终结论、复盘报告与交付文档。

## 目录清单

| 文档名称 | 所属阶段 | 核心结论摘要 |
| :--- | :--- | :--- |
| **README_E55_P12_FINAL_VERDICT.md** | E55 残差线 (P1.2 结案) | **残差头假说证伪**（头净贡献边际甚至有害）；**功臣为躯干位移**；画出安全剂量地图（2e-5 金区 / 4e-5 悬崖）；三臂排序 A0 > P > Q（Q 有害实锤）。残差线光荣关闭，成果转入 P2 主格训练。 |
| **README_E54.md** | E54 通道探针 | 根通道有效性排查、通道敏感度测试记录。 |
| **README_E53.md** | E53 根通道分析 | 根通道重定向与观测流一致性校验。 |
| **README_E52_ZHY_PROJECTION_FINETUNE_20261018.md** | E52 投影微调 | 早期 035 策略投影与微调可行性验证。 |
| **README_G0.md** | G0 可达性审计 | 动作重定向可达性（Reachability）门禁与关节越界检查基线。 |
| **README_P2_M2_REPORT.md** | P2 主格线 (M2 扩编) | 20 场扩编：0/6 跌倒保持，zzx 治愈保持（20.53°），zj_walk 漂移 +7.08° 标记观察。PASS。 |
| **README_P2_M3_REPORT.md** | P2 主格线 (M3 布料入训) | 90 场布料入池 G0 全过，40 场混训：0/6 跌倒，zj_walk 漂移回落（14.84°监视解除），zzx 治愈保持（19.60°）。PASS。 |
| **README_P2_M4_REPORT.md** | P2 主格线 (M4 布料加料) | 67% 布料配比触发 zzx 跌倒回退+zj_warmUp 超阈。**REJECTED,冻结归档**,现役最佳回退 M3。 |
| **README_P2_M4B_REPORT.md** | P2 主格线 (M4b 换血重混) | 50% 配比换血验证安全：0/6 跌倒，zzx 治愈保持（20.20°），布料覆盖 50/90。PASS，现役最佳 M3→M4b。 |
| **README_P2_M5_REPORT.md** | P2 主格线 (M5 换血第二程) | zj_warmUp 25.69° 触发预声明监视阈+zzx 侵蚀。MONITOR-TRIGGERED，冻结隔离，现役最佳回退 M4b。 |
| **README_P2_M6_REPORT.md** | P2 主格线 (M6 A锚回拉) | zj_warmUp 26.90° 判据FAIL;但 zj_walk 史上最佳+zzx 回拉成功,漂移曲线定位 zj_warmUp 为配方无关约束动作。REJECTED 冻结,现役最佳仍 M4b。 |
| **README_P2_M78_REPORT.md** | P2 主格线 (M7/M8 靶向过采样) | 剂量-响应定谳:12.5% 最优(+0.20°),22% 反噬;机制有效但池内 warm-up 数据枯竭。双双 REJECTED 冻结,现役最佳 M4b。 |
| **README_P2_M9_REPORT.md** | P2 主格线 (M9 靶向变阵) | **门禁四项全过,新现役最佳**!M3 底座余量是胜负手;zzx 17.66°/均值 16.30° 均史上最佳。布料在库 15+warmUp 靶向。 |
| **README_P2_M9_REPORT.md 附:M10** | P2 主格线 (M10 覆盖推进) | REJECTED:布料+5 场换 warmUp+1.04°(24.47>23.81);均值/zzx 仍优。冻结,现役最佳 M9。 |
| **README_P2_M9C_REPORT.md** | P2 主格线 (M9C 巩固跑) | **128 步 4 倍剂量全过,现役最佳接替 M9**;回应小剂量假象质疑:warmUp 余量翻倍/zzx 再破纪录/fjw 史上最佳。 |
| **README_P2_M9L_REPORT.md** | P2 主格线 (M9L 千步长跑) | REJECTED+全剂量曲线:包络破裂点 128-256 步,768 步 zzx 摔倒;warmUp 长训红利确认;M9C@128 是本配方甜点。 |
| **README_P2_NOISE_AUDIT.md** | P2 门禁定标 (seed=1 复刻) | warmUp 跨种子仅 0.06°(最可信指标);zj_walk ±2.7°(高噪声);均值带 ±0.6°。历次贴线判罚据此重审。 |
| **README_P2_M11_REPORT.md** | P2 主格线 (M11 真·靶向) | **PASS 新现役最佳**:真 warm-up 窗×6 替换伪金标,warmUp 22.72 新低,yy 13.37 大优,均值 15.99 全血统最佳。狩猎→审计→靶向全链路首闭环。 |
| **README_P2_M11_REPORT.md 附:M11C** | P2 主格线 (M11C 巩固) | REJECTED:128 步巩固侵蚀 warmUp(25.52),zzx 15.95 史上最佳;与 M9L 同构的剂量模式。现役仍 M11@32。 |
| **M11D(64 剂量)** | P2 主格线 | **PASS 全史最佳**:均值 15.21/zzx 14.95/fjw 9.56 三史 best,warmUp 22.75 稳;真窗系甜点=64。 |
| **M12(AMASS 首混)** | P2 主格线 | REJECTED 0.07 度贴线:warmUp 23.88;AMASS 管线全链贯通;泛化判据缺位教训入册 |
| **README_GEN_AUDIT_VERDICT.md** | 泛化审计 | **GT 级开卷:全体近乎全摔(含 035),泛化墙普遍**;血统持平未破坏;AMASS 升级为主力弹药。v1-v7 方法学链全部留档。 |
| **README_P2_M1_REPORT.md** | P2 主格线 (M1 微调) | 金区微调 64 步：跌倒率 1/6→0/6，zzx_walk 治愈（178°→21.4°）。STRONG PASS。 |
| **README_P2_M0_BASELINE.md** | P2 主格线 (M0 基线) | 035 本地基线复现：5/6 通过 + zzx 缺陷精确复现。PASS。 |

---

## 全局规范索引

- **数据产物**：全部移至 ../data/
  - data/experiments/：历次训练的正式权重与指标
  - data/motion_pool/：73 场 A 档动作池
  - data/p15c_intermediate/：中间合成与临时特征数据
- **纯运行日志**：全部归档至 ../logs/
  - logs/p12_runs/：P12 探针与正跑完整输出日志
  - logs/p15c_synth/：布料特征合成与转换日志
- **代码与脚本**：全部归类至 ../script/
  - script/residual_e52_e55/：历史残差线算法与训练脚本（已封存）
  - script/p12_launches/：标准化启动脚本
  - script/p15_cloth_pipeline/：FGP/布料重采样与对齐流水线
  - script/g0_audit/：G0 可达性审计工具
  - script/tools/：通用提取与数据拉取工具
