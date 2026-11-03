# Primebot Q1 首轮训练与最终评估

2026-11-04北京时间。Primebot首轮已按用户批准的连续三次观测规则，在10000轮正常早停；初末评估和Mac交付已完成。服务端门通过，条件共享训练仍须在收到本地验收收据后重新计算总门。整个五阶段goal保持active，q1每10分钟ACTIVE，用户采纳null。

## 训练实际完成量

执行冻结b8f920cbba9be3f7247bddbe6535febfbc06186b，原Primebot118338b训练器的V3与监控独立适配。4096环境、WM2048、policy2048、393条共同参考、从零初始化；没有恢复profile权重或缩减网络。250000轮为批准的上限，实际只完成10000轮，因此不能称跑满原配方。

训练2026-11-03 20:23:40至23:51:41正常回收，100000次WM、10000次actor、200000次critic更新，102563840真实环境转移。终轮日志elapsed12352.401秒，含采集及已有观察等待，不能全部当纯优化时间。1000、5000、10000三次临时门连续通过，保存后由原流程进入最终交付，没有重启或追加训练。

2026-11-04 00:15:18独立收尾核验：104份原输入大小SHA全部未变；controller29435、train guard29965/worker29970、delivery guard44301/worker44306均不存在，两个guard均WORKER_COMPLETED/reaped0。原boot/GPU一致，GPU1MiB/25度/recovery=None、cgroup OOMkill0。[原始终态收据](../logs/session_records/primebot_q1_early_stop_20261104.json)。

## 效果与边界

固定前走、起走、停走各seed0/1/2，初始0/9完整，最终9/9完整。共同有效前缀关节RMSE .553057→.119617rad，骨盆相对身体位置RMSE .0583254→.0103590m；排除初始化和终止帧、逐片等权。由于零更新早跌倒，前缀仅70/51/51步，即1.40/1.02/1.02秒，不能用此指标代表全长误差。三种子为名义无扰动的确定性回放，不是三次独立扰动试验。

| 最终seed0动作 | 完整关节RMSE rad | 完整body RMSE m | 脚底p95左/右 mm | >10mm抬脚/交替事件 | 承重滑速左/右 mm/s | 上肢RMSE rad |
| --- | ---: | ---: | --- | --- | --- | ---: |
| 前走 | .138147 | .025574 | .303/1.156 | 9/4 | 61.79/37.11 | .154571 |
| 起走 | .120079 | .013623 | 2.271/.399 | 1/0 | 62.45/128.10 | .126943 |
| 停走 | .237651 | .025891 | 1.898/.324 | 1/0 | 55.17/43.84 | .312272 |

对应参考脚底p95约12.74–25.73mm。前走开始出现交替抬脚，但大多数时刻仍低离地、承重时滑动；起走和停走没有交替抬脚事件。最终躯干倾角RMS分别.10493/.09139/.10929rad。停走上肢幅度仍明显偏大，例如一个肩部角度峰峰值1.255rad而参考.348rad。根XY末误差仅作诊断，不代替足部质量。不能称自然步态合格，也未隔离奖励、控制、WM或策略的单一原因。

全393条训练参考均完成初末逐片评测：初始0/393完整，最终393/393完整。最终逐片等权完整关节RMSE均值.142626rad、body相对RMSE均值.018626m。它们都是训练集名义回放结果，不是未见动作泛化或真机结果。

## 本地交付

2444份数值与媒体工件81635329字节，含全部逐片JSON、脚底姿态与上肢CSV、初末检查点审计、配对误差和等曝光缺失记录。六段初末原速MP4为1280x392/50fps，共3174帧，Mac逐文件大小SHA、完整RGB解码及十八帧目视均通过。原始NPZ、replay和权重仍只在服务器；本轮没有新增物理回放，复用原交付流程结果。

原速视频各20.64/3.04/8.06秒，初始跌倒后保留FAILED/HOLD至原参考结束，非跌倒后继续存活。视频实际与参考分别位于左右：

- [最终前走](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_FINAL_DELIVERY_20261104/review/final_fixed/walk_forward_loop_001__A022_s0/tracking.mp4)
- [最终起走](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_FINAL_DELIVERY_20261104/review/final_fixed/walk_forward_start_002__A023_s0/tracking.mp4)
- [最终停走](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/PRIMEBOT_Q1_FINAL_DELIVERY_20261104/review/final_fixed/walk_forward_stop_002__A024_s0/tracking.mp4)

Mac交付收据沿用条件配置已固定的文件名primebot_q1_final_mac_delivery_20261103.json，内部实际时间为2026-11-04，不冒充前一日已验收。[总收据](../logs/session_records/primebot_q1_final_delivery_summary_20261104.json)、[目视](../logs/session_records/primebot_q1_final_visual_20261104.json)、[检查点审计](../data/manifests/primebot_q1_final_checkpoint_audit_20261104.json)、[完整交付身份](../data/manifests/primebot_q1_final_delivery_manifest_20261104.json)。

## 条件推进

早停只采集102563840，未达到等曝光目标410009600；没有±1%可比检查点，等采样量对照必须标记UNAVAILABLE_EARLY_STOP_BEFORE_TARGET，不为凑比较补训。后续将早停Primebot配方与完整共享配方分开解释，不能宣称等算力优劣。

本阶段发布并将Mac收据镜像服务器后，重新计算完整总门，全部通过才自动新建G1+Q1共享各1024环境八全更新/保存审计，之后从零160000轮；仅健康设备的明确容量OOM才各512/320000。后续Q1 motion/KP、G1原生评估及综合比较仍未完成，目标不结算。原生推理和共同Q1物理比较代码已在独立本地准备，4项标准库测试通过，真实Torch CPU补验及原生场景接续仍待；它们尚未进入本次训练的冻结输入。

后续实际共享启动和执行提交、曲线、发布核验以新收据为准，不能把当前“允许进入”写成已正式训练。保持唯一Windows原生25362入口、原身份、GPU锁、24/30GiB内存、86度及100GiB磁盘保留，不改其他自动任务或开启agent。
