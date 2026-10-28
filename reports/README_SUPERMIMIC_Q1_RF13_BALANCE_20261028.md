# Q1 RF13独立腿腰补偿

记录日期：2026-10-28，北京时间。状态：唯一正式RF13已于15:21:12启动；15:28:18核实已完成初始三重复基线并实际学习176.116秒/55轮，尚无阶段物理改善结论。实时状态以现场文件/进程为准。

## 实际启动与监督

训练代码提交`51d4f00a5c9fb6f8bbdef29e11142a6bf822d41a`已push并独立核40位远端；17项代码/准备记录301773字节在权威Git、Mac主根与25386逐大小/SHA一致后启动。后续纯记录提交不改变冻结来源。plan绑定1137项，含原1123项与本次head/标定/固定展示来源。

2026-10-28 15:21:12 controller93989/start_ticks8571724；worker93994/start_ticks8571852，boot`4d859008-4c83-4a6d-814d-cbb9dfe885c9`、GPU`GPU-26028785-9fb9-45af-e614-4cc53af15d08`。15:21:55核controller/guard均RUNNING、worker INITIALIZING/evaluation，完整命令、真实start_ticks与文件mtime相符；RF11_fixedblock repeat0/frame300，实训0/PAUSED_FOR_NONTRAINING正常。GPU2871MiB/35°C，共享11030843392字节、OOM0、盘余1673493258240字节。

原`q1-rf08-4096`已实际工具更新为“Q1 RF13腿腰补偿4096环境监督”，配置回读ACTIVE/每10分钟；其他两研究任务保持。完整启动命令、状态/时钟/进程/来源及监督回执见[launch_observed](../data/manifests/supermimic_q1_RF13_launch_observed_20261028.json)。普通进度静默，异常、物理阶段结果或完成通知。

15:28:18二次核身份未变，controller/guard/worker均RUNNING、phase collecting，clock ACTIVE；累计176.116397秒/55学习轮/5406720真实转移，optimizer1100/actor900，前10轮critic-only已结束，0新增WM。课程cap0.11007；GPU8267MiB/50°C，共享11621736448字节/OOM0，盘余1672784736256字节。状态里保留的static075/repeat2/frame450为上一次已结束评测字段，当前阶段按collecting解释。首个学习后物理门仍为累计约1200秒，不能把正常更新或初始对照当改善。

## 目标与已完成准备

用户批准UR03提出的分离腿腰适应、渐进手臂干预和回退验证，并要求继续。仅执行一次独立4096环境、实际训练最多3600秒；不恢复旧实验、故障不重启、不追加五小时、不开agent。实现/负面对照/完整配方见[UR04](README_Q1_BALANCE_ADAPT_UR04_20261028.md)及[准备收据](../data/manifests/supermimic_q1_UR04_balance_prep_20261028.json)。

五单元测试、16/4096真实零optimizer转移/奖励/终止与人工截断/全小批反向通过；最终4030截断子集、24576小批似然差0，独立物理评测保留训练环境/门控/接触和CPU/CUDA RNG。旧1123项冻结来源未变。启发式回退只换得约3%根改善却损失大量上肢精度，故本轮关闭，最终作为独立对照报告。

## 冻结配方与预算

RF11冻结教师/归一化，检查点SHA`6c4de3171bd45359691a63ffe1712729c14fdb2080f8066e4c25632f4bbc5146`；独立256/128 ELU head输出14腿腰潜变量，末层零，物理残差`0.15*tanh(z)`rad，固定潜变量std0.1。8手臂执行参考/教师混合；实际历史/已执行动作均真实闭环。

4096真实MuJoCo Warp/8线程，原AP02 300/30/SHA`ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337`，不混AP03。上肢cap前1200实训秒0→0.75、每次重置25%概率alpha0。PPO复用安装RSL-RL，24步/5epoch/4minibatch、每轮98304转移/小批24576、lr1e-4、clip.2、gamma.99、lambda.95、grad1，前10轮仅critic。只有head/critic优化，0新增WM/想象更新。

原reward加根XY奖励0.02（尺度0.15m）与腿奖励0.01（0.3rad），减残差平方代价0.002；分别记录。新step最大0.13。结构、课程和奖励同时变化，不能单因素归因。

软3500、外部3540+45退出宽限、实际硬上限3600秒；只累计同步采集/GAE/更新/逐轮诊断，初始化、评测、保存、媒体不计。100GiB余盘/12GiB试验/30GiB共享cgroup/OOM0/86度/900秒无进展。无额外训练/自动重试/自动扩时。

## 评测与交付

首次新协议330片三重复RF11/静态0.75基线；固定4096前向块，不混旧330块BF16算术。零head初始等于静态0.75，复用其已存轨迹。约1200/2400/3500实训秒同进程暂停更新评测三重复，保存JSON/NPZ后决策。原30验证及两普通走/抬手固定，不换片；配对共同前缀与完整失败/完成同时报告。

三次至少两次相对初始候选根或上肢RMSE增加>=25%，或验证失败>=6/30，结束本轮。原RF11的根/腿<=1.10、上肢/腕<=0.85、抬手峰误差<=0.75、完成>=98%/失败<=1、振荡/滑移<=1.20和固定片三次全完成，仍独立列明；即便满足也不扩时。最终另用保存head评估带回退版本，0额外优化。

交付六格PNG/PDF/CSV原值+500iteration均值、PPO/残差/奖励分解诊断，WM/想象N/A；初始（零残差静态75%）/最终各54秒8段6类原速MP4，固定片和失败hold。Mac直接复制并逐大小SHA核，Mac解码/目视，不建网站；权重/大NPZ只留服务器SERVER_LOCAL_ONLY。

## 路径与恢复

唯一25386实体根`/opt/data/private/ZHY`；本轮根`data/supermimic_for_q1/BONES_RF13_LOWER_RESIDUAL_SOMA300_E4096_MAX01H_20261028`。源码`script/supermimic_q1_balance/`、runtime沿用C02 sampler；标定`data/supermimic_for_q1/Q1_UR04_BALANCE_PREP_20261028/calibration_v02.json`。主机/boot/GPU必须核现场，训练代码提交由本轮plan/launch收据绑定，后续文档提交不改变它。

复现入口`script/supermimic_q1_balance/supervise.py --mode train`，必须传固定runtime/library/checkpoint/calibration/frozen-plan/video-suite及已发布40位code-commit、预期boot/GPU；实际完整命令保存在plan。Python沿用`data/environments/supermimic_q1_hp_20261026/bin/python`，CUDA/EGL经activate，不改旧冻结来源。终态需run TRAINING_COMPLETED、train/delivery guard均正常reap0、pipeline COMPLETED、实际进程退出与GPU释放；否则保留真实失败。

权威Git为Mac持久`data/repositories/q1-rf12-20261028`的Q1。发布前显式暂存/payload audit，push origin Q1/独立40位远端，再镜像Mac主根与25386核大小SHA。启动后只复用原`q1-rf08-4096`每10分钟监督，普通进度保持静默、异常/物理结果/完成通知；交付齐备后暂停，其他两个研究任务保持。
