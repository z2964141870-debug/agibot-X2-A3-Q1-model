# E55 残差线结论文档（P1 全战役收官件）

日期: 2026-10-20 | 目录: a3_zhy_residual_p1_20261019_E55 | 执行: exp-operator | 裁决: lead
本文档按"给半年后的自己看"标准写。所有数字可由 receipt 复算。

## 三结论（lead 定稿 ACCEPTED）

### 结论一：残差头假说证伪（marginal-to-negative）
- **假说**：在冻结 trunk（035_step200000 热启）上加 89,885 参数 zero-init 残差头（e55_head 六键），微调头可改善行为。
- **实况**：配方实为 head+trunk 可训（"only head trainable"注释与实况不符,已入档）。
- **行为面判定**（orientation_max_deg,三臂评测族,cloth6,基线=zero_head_p1 裸 trunk）：
  - A0 头满配置 vs 基线：6/6 IMPROVE,但
  - **置零对照（fb_a0 ckpt 头置零重评）**：fjw 置零 7.14 < 满头 7.42;zzx 置零 18.15 < 满头 19.44 —— **头在这两个 motion 是净负贡献**。
  - 头的边际贡献量级 0.1–2.3°,方向不稳定。
- **有效载体 = trunk 位移 @ 金区剂量**：zzx 的戏剧性改善（基线 179.67° 翻滚 → 19.44°）在头置零后依然成立（18.15°）,即该改善由 trunk 学得,与头无关。
- digest 面旁证：A0 头参数 L1 位移非零但小（body_max 9.9e-4 @A0 剂量）——头"动了但没用,甚至轻微有害"。

### 结论二：剂量地图（金区/悬崖/拆阀）
| 域 | 配方 | 行为 |
|---|---|---|
| **金区** | nominal 2e-4, adaptive_lr_min=1e-6, max=2e-5（KL 阀在）, 64 步 | 有效剂量 ≈2e-5×64;6/6 no-fall 全面改善 |
| **悬崖** | min=max=4e-5 钳死（KL 零自由度）, 64 步 | 三臂 14/18 motion fall（翻滚 95–180°） |
| **拆阀混淆** | Stage1→P1.2 一次动了两个变量（剂量×2 **和**拆 KL 阀） | 崩因复合,归因不可分;教训入册 |

- **教训 A（入册）**：×2 启发式假设剂量-响应单调;近悬崖区先验证单调性再用倍数。一次只动一个变量。
- **教训 B（入册）**：KL 自适应阀是承重墙不是官僚主义;min=max 钳死只有在阈值内侧才安全。
- 阈值注记：金区/悬崖间存在断崖且可能很窄（2e-5 与 4e-5 之间未细探,预算未批不再探低）。
- 有效剂量条款（A1 增补）：**有效剂量 = min(名义 lr, adaptive_lr_max)×steps**,名义 lr 单独不构成剂量声明。

### 结论三：Q 通道有害（金区复验）
- 4e-5 悬崖域：Q 4/6 fall。
- 金区回退域：**Q 仍 2/6 fall**（zj_warmUp 179.36°、zzx 179.97°）+5/6 WORSE,rank 0 胜。
- P1 冻结论（Q 有害）在金区被独立复验。臂序定稿：**A0 > P > Q**（A0 唯一可交付;P 中性偏负 3 WORSE）。

## 确定性复现注记（单跑纪律方法论背书）
金区回退 A0 臂（fb_a0）6-motion omax 与 Stage 1 探针 A0 **逐位一致**（10.6153/14.9202/7.4208/12.2606/14.4100/19.4376,六位小数级）——同配方同 seed 必然同结果,全战役第一次闭环验证。单跑（seed=0,不设多 seed）作为纪律被数据背书。

## 置零对照全表（omax 度,低=好）
| motion | 基线(裸trunk) | fb_a0 满头 | fb_a0 置零 | 头贡献 |
|---|---|---|---|---|
| zj_walk | 13.32 | 10.62 | 11.28 | -0.66 |
| zj_warmUp | 18.66 | 14.92 | 14.35 | +0.57 |
| fjw | 10.85 | 7.42 | 7.14 | **-0.28(净负)** |
| zls | 15.87 | 12.26 | 12.36 | +0.10 |
| yy | 14.64 | 14.41 | 14.74 | +0.33 |
| zzx | 179.67 | 19.44 | 18.15 | **-1.29(净负)** |

## 金区三臂全表（回退跑）
| motion | base | fb_a0 | fb_p | fb_q |
|---|---|---|---|---|
| zj_walk | 13.32 | 10.62 | 13.96 | 16.69 |
| zj_warmUp | 18.66 | 14.92 | 22.86 | **F** 179.36 |
| fjw | 10.85 | 7.42 | 9.33 | 9.30 |
| zls | 15.87 | 12.26 | 12.93 | 14.04 |
| yy | 14.64 | 14.41 | 14.28 | 17.81 |
| zzx | 179.67 | 19.44 | 27.59 | **F** 179.97 |

rank（(fall,omax) 字典序,1=最优）: fb_a0 5/6 第一,fb_p 1,fb_q 0。

## 4e-5 悬崖崩坏表（三臂 fall 计）
| 臂 | fall 数 | 幸存 motion |
|---|---|---|
| A0 | 5/6 | fjw(40.09 no-fall 但 40° 已劣化) |
| P | 5/6 | zj_walk(18.72,独稳但劣于基线 13.32——臂特异脆弱性签名) |
| Q | 4/6 | zj_walk 27.07、fjw 17.03 |

## receipt 索引
- 金区三臂评测: EVAL_ROOT/eval_p12fb_{a0,p,q}_m64_cloth/*.metrics.json（6 motion×3）
- 置零对照: EVAL_ROOT/eval_p12fb_a0_m64_cloth_zeroh/
- 悬崖三臂评测: EVAL_ROOT/eval_p12run_{a0,p,q}_m64_cloth/
- Stage 1 探针: EVAL_ROOT/eval_p12_probe_m64_cloth/ + 日志 p12_probe_stage1_run4.log
- 训练: data/training/p12_p12run/{a0,p,q}/（悬崖）、data/training/p12_fb/{a0,p,q}/（金区）,各含 checkpoints/milestone_{0..64}+expdir
- gate_log: experiments 侧 gate_log.jsonl（58 行,append-only）
- 评测副本: EVAL_ROOT/checkpoints/p12run_{a0,p,q}_m64、p12fb_{a0,p,q}_m64、p12fb_a0_m64_zeroh（含 sidecar+config）

## 线处置
E55 残差线光荣关闭（假说证伪+剂量地图+Q 有害三结论）。处置去向（残差→整网微调叙事、并入 P2）为 lead 向用户的门。本侧不再启新训练。

技术备忘（新开线时必读）：
1. P/Q 分支训练需 E54_CHANNEL_STATS env（e54_obs_terms._manifest 强制）,A0 分支不需要——launch 模板已带。
2. 评测 ckpt 信任门需 sidecar json（model_step_XXXXXX.json,含 sha256）+config.yaml 同目录,缺一即拒。
3. 评测与训练不可并行（GPU 冲突 RuntimeError GPU occupied or unreadable）——串行纪律含评测。
4. vendor sh 内部合成键（checkpoint/exp_var/experiment_dir）argv 看不见,显式 argv 键 hydra 后写覆盖生效——launch 记录必须含合成键清单。
