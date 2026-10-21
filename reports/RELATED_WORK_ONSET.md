# Onset 脆弱性:相关工作对照表 v1 — 2026-10-21 15:00(文献定谳)

## 检索面
arxiv/web 三轮定向(reset-free/episode init/eval protocol/onset/warm start/fall at start)+ 近邻摘要深读。

## 直接近邻(同一问题)
**未发现专文**。无一论文将"参考起始瞬态作为评估伪影"系统隔离刻画。

## 近邻族分层
| 工作 | 做什么 | 与我们的差异 |
|---|---|---|
| Fall-Safety(2511.07407,ICRA26) | 摔倒预防/缓冲/恢复统一技能 | 治症状(摔倒),不问成因(onset 伪影);互补可引用 |
| FAST(2602.11929) | 残差策略快速适配 OOD 动作 | 适配期鲁棒,非评估起始 |
| BeyondMimic | 扩散引导的通用控制 | 泛化方法,非评估方法学 |
| MimicX(2610.09055) | 视频跟踪的策略在环精修 | 改跟踪质量,非 onset |
| InterMimicGen(2610.06850) | 数据扩产 | 数据侧 |
| PHC/H2O/Humanoid-X/PHUMA/SONIC 协议族 | **init-at-frame-0 + completion-rate** | 全体默认把机器人初始化在参考首帧——正是 onset 敏感性被系统性混淆/掩盖的机制;无一家隔离 onset vs 内容 |
| HuMiT 等 teleop 线 | 低延迟遥操作 | 不同轴 |

## 结论(v1)
**切口开放**:onset-伪影基准+失败可预测性+轻量缓解 无直接占位者。
最大风险:(a)审稿人视其为"工程常识"——对策=跨策略跨机器人定量矩阵+可预测性分析做深;(b)Fall-Safety 一族可能顺带处理——对策=明确分层(他们学摔后恢复,我们消除评估混淆+预防起始失败)。

## 待深查(v2,动工前)
BeyondMimic 全文 / PHUMA 与 PHC 论文的 eval 章节(逐句核 init 协议描述) / OmniTrack
