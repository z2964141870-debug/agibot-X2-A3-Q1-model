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

## 全库足部与上肢诊断

后续只读汇总复核原交付清单中的788份JSON（初末各一份全库及393份逐片），逐文件大小/SHA与逐片内容均相符；未加载权重或NPZ、未新增策略回放、物理积分或优化。低离地并非只出现在三固定片上：

| 最终完整动作集合 | 条数 | 无>10mm抬脚事件 | 无左右交替事件 | 双脚脚底高度p95均<=5mm | 参考双脚p95均>10mm、实际均<=5mm |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全冻结训练集 | 393 | 164 | 306 | 308 | 289 |
| 名称以walk_forward_开头 | 21 | 7 | 15 | 18 | 17 |

全393片逐片等权：实际脚底高度p95的跨片中位数左1.158/右1.100mm，参考左23.631/右22.405mm；承重滑速的跨片中位数左106.525/右104.662mm/s、p95为204.144/166.961mm/s。此处“p95的跨片中位数”先取每片时序p95，再跨片取中位数，不是把所有帧混池。躯干倾角RMS的跨片中位数.081067rad，上肢RMSE中位数.143641rad、跨片p95 .213103rad；每片最大单上肢关节峰峰值绝对幅度误差的中位数.354226rad。

名称前缀21片只是字面分组，不能代表所有普通前走或独立测试集。全部数字来自最终完整片；零更新393片全提前结束，因此没有可比较的初始全长步态分布，不把初始短暂低滑速与最终完整滑速直接相减。抬脚事件沿用原>10mm上穿阈值，无持续时间过滤，交替事件不是已经确认的自然步数；载荷沿用控制末端>=5N采样，也不证明整段连续承重。

新增`script/q1_five_stage/diagnostic_summary.py`仅汇总保存JSON。初末、Q1 motion/KP与G1原生比较均接入相同的完整片诊断，重复seed先片内平均、再逐片等权；缺载荷样本保留missing，不用零滑速代替。缺失完整片诊断会拒绝发布比较，失败片单独计数，综合README新增足部/滑移/上肢诊断表。本地81项检查74过、7项依赖跳过；04:43服务器15项纯CPU检查全部通过，连同原788份JSON独立重汇总耗时.989702秒，Mac/服务器所有分组数字逐值相同、八份被观测源码/清单前后身份未变。

执行源码50afcfd0334a06d972f2e6e41f7b35dd01590cf2已独立40位远端核同，12份559976字节在权威Git/Mac主根/25362逐大小SHA相同后才执行服务器补验。收尾原boot/GPU保持，GPU1MiB/25度/recovery=None/无计算进程；cgroup总7886368768B/cache6698086400B，低默认缓存维护触发线，未强制清缓存。原V03仍FAILED、五旧PID不存在；追加V04决定仍待，真实共享训练与物理评估未开始，不能据此称第二轮有收益。[汇总与输入身份](../data/manifests/primebot_q1_library_diagnostics_20261104.json)、[服务器独立汇总](../logs/session_records/primebot_q1_library_diagnostics_server_v01_20261104.json)、[CPU收据](../logs/session_records/q1_library_diagnostics_cpu_v01_20261104.json)、[阶段发布](../logs/session_records/q1_library_diagnostics_publication_20261104.json)。

复现命令（在权威Git或已同步服务器根，`REVIEW_ROOT`指向已有交付，不读取大轨迹）：

```sh
PYTHONPATH=script/q1_five_stage:script/primebot_q1 python3 script/q1_five_stage/diagnostic_summary.py --manifest data/manifests/primebot_q1_final_delivery_manifest_20261104.json --review-root REVIEW_ROOT --output NEW_SUMMARY_JSON
```

该结果把“训通/跟踪改善”和“自然步态仍差”进一步区分；不能由这些诊断隔离奖励、控制或模型单因，也不改变原早停/转接门或授权新的训练重试。

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
