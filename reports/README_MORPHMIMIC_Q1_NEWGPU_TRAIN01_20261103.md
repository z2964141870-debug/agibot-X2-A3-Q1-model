# MorphMimic G1/Q1 新卡验证与正式训练计划

最终状态更正：2026-11-03 15:59:57本pipeline已FAILED，共享首motion CUDA OOM，各WM10/其余0/完整轮0，guard1015回收1022/return1，未启动正式训练。原失败目录保留，不恢复。用户现已明确改为[Primebot先复现与评估的五阶段路线](README_PRIMEBOT_Q1_STAGED_20261103.md)；下文运行中和自动80000轮入口均是已停止的历史。

记录日期：2026-11-03。阶段：NEWGPU_TRAIN01。当前状态：计划和自动串行入口已完成，本地6项准入测试通过；服务器CPU检查、原生CUDA验证、共享完整更新和正式训练尚未启动。用户已明确授权“验证好了以后直接开始训练”，全部验证通过后自动开训，无须再次确认。

## 15:50 实际启动

执行冻结 `3a43f7ee8aaaf2d9e39494eabae03a7114bdce5a` 已独立40位远端核验，9记录461706B在Git/Mac主根/25362逐大小SHA一致；服务器6项准入测试全过。15:44:41总controller94/start479951、原新boot启动，1820输入逐SHA通过。15:46:09 G1原生检查COMPLETED/guard回收0，奖励最大差1.67638e-8、观察/实际目标差0、384转移、0optimizer。15:50 Q1仍在新设备首次CCD内核编译，worker459/guard452，GPU346MiB/36C/recoveryNone、cgroup5310836736B/OOMkill0；日志持续新增compiled记录，尚未Q1通过或正式开训。启动和初始观察见[原始记录](../logs/session_records/morphmimic_q1_newgpu_train01_launch_observed_20261103.json)，[冻结发布](../data/manifests/morphmimic_q1_newgpu_train01_preparation_publication_20261103.json)。上方“尚未启动”为准备历史。

15:50:46 Q1检查也COMPLETED/guard回收0：384转移/240批物理子步/243次torque callback，最大力矩差2.01166e-6Nm、reward1.86265e-8、观察及实际目标差0，0optimizer。两型各1823冻结输入由guard前后逐SHA核对，未重复旧设备检查。共享profile于15:50:47自动启动guard1015/start516555、parent1022、rank G11023/Q11024；15:52各409600预热已完成，首学习更新仍在编译，当前落盘完整轮0不推断内存更新为0。GPU12803MiB/40C/recoveryNone、容器11392753664B/OOMkill0；完整8轮及保存审计待，正式训练尚未启动。

## 目标与配置

在替换服务器25362的3090上，继续原MorphMimic v0.1 Trainer与Primebot Q1控制契约适配，训练G1/Q1共同世界模型、motion policy和return-KP student。两个本体各2048环境、合计4096；真实559条配对SOMA、3604.72秒，人体侧同片同帧。此处是G1+Q1新本体接入，不等于原发布G1/X2/T2三本体配方的完全复现。

固定上游41c8382f127a6f5109bfe047445a9180f71f1fc0，既有Q1分组软件PD、2ms积分、转速限矩、量化延迟与缺失关节mask保持。沿用同GPU两个rank、HostCollectives平均梯度后clip-Adam、独立本体buffer与同帧latent抽样，不引入新奖励或PPO。

正式每机器人80000轮、WM-only1000轮、warmup200控制步；WM batch1024×10、motion/KP batch2048、H32、window8、UTD0.4、三学习率2e-4、BF16、compile=default、完整原网络、lr_decay500000。每8000轮保存原检查点，同时保留0/40000/80000与latest；每50轮日志和进度收据。历史共享检查点每份342751179B，据此本轮保存量预计低于8GiB，不触碰原12GiB阶段磁盘保护线。

原Trainer `_collect_steps()` 是2、3交替，正式每型采样量为 `(200 + 80000/0.4) × 2048 = 410009600`。WM批次样本量819200000，motion与KP各161792000，与4096环境×40000轮、WM-only500、warmup100的对应量相同；Adam更新次数翻倍、batch减半，不声称相同优化轨迹。每型预期WM800000、motion79000、critic1580000、KP79000、KPcritic1580000次更新。

## 自动执行次序与通过条件

1. 用原V04失败记录的1820项输入身份重新核对保留数据/代码；旧失败收据只作不可变来源清单，不用作任何验证成功证据，也不加载旧权重。
2. G1与Q1各一次原生16环境×24控制步，零optimizer；分别96/240批物理子步。核实际动作/延迟/缺失mask、观察和奖励，Q1逐子步软件PD与转速限矩回调；这同时验证新卡Torch/Warp真实CUDA执行，不另跑无关基准。
3. 新目录共享E2048各、完整559集、8轮原五optimizer更新。每型450560实际转移，更新和Adam计数必须80/8/160/8/160，非有限0、原生rank核对通过、完整共享参数SHA与实际同帧draw一致、独立presence buffer保留。外层guard也必须正常完成并回收，不能只认worker成功。
4. 独立CPU读取两份新ckpt_8，核保存iteration/schedule/cfg/代码身份、五个Adam计数及模型/归一化有限值；记录大小和SHA，无CUDAcontext。
5. 以上全部通过后，自动在另一个全新train目录从零初始化，先保存原方法零轮检查点，再运行80000轮。profile权重绝不续入正式训练。

任一步失败就保存该阶段日志并停止后续，不恢复失败根或无诊断重跑。正式训练完成只登记TRAINING_COMPLETED_PENDING_PHYSICAL_EVALUATION，不以reward、loss、存活或启动成功声称自然步态；后续用同片普通走/起走/停走及上肢动作，核真实换脚、脚底离地、承重滑移和姿态，并交付本地原速视频。

## 设备、保护与执行入口

唯一根 `/opt/data/private/ZHY`，仅hp_han Windows内原生SSH25362；boot `90395f47-4035-4785-858e-b3d05c44a738`、GPU `GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4`。启动前只读设备查询仍recovery=None/23C/1MiB/无计算进程，容器约60MB、limit34292629504B/OOMkill0、磁盘余903712456704B。驱动查询不是CUDA已验证。

每阶段显式传新boot/GPU，沿用原GPU独占锁、30GiB运行内存、初始12GiB准入、86C温度、OOM基线0、源身份和磁盘保护。原生检查各600秒、8轮profile10800秒；正式7天仅进程保护上限，训练完成按80000轮和实际采样量，不以旧300/1200秒门或固定小时数截短。显存低于3GiB时现有物理/学习阶段入口只交还未占用Torch缓存，不改活张量与算法数学。

代码：[validate_and_train.py](../script/morphmimic_q1/validate_and_train.py)，配置：[newgpu_train01_20261103.json](../script/morphmimic_q1/newgpu_train01_20261103.json)。在三端发布核验、服务器6项CPU准入测试后运行：

```sh
data/environments/morphmimic_original_20261029/bin/python -u script/morphmimic_q1/validate_and_train.py --config script/morphmimic_q1/newgpu_train01_20261103.json --code-commit <独立远端已核40位执行提交> --detach
```

总输出根为 `data/supermimic_for_q1/MORPHMIMIC_Q1_NEWGPU_TRAIN01_20261103`；`pipeline.json`为总状态，`validation.json`是正式开训准入证据，`g1_preflight/`、`q1_preflight/`、`profile/`、`train/`各有独立status/guard/stdout，训练rank进度在 `train/workers/{g1,q1}.json`。pipeline_controller.log和launch.json记录监督进程，正式guard收据保留PID/start/boot/GPU。源码冻结后不热改。

## 保存与边界

准备清单：[morphmimic_q1_newgpu_train01_preparation_20261103.json](../data/manifests/morphmimic_q1_newgpu_train01_preparation_20261103.json)。本地准入6项通过；真实服务器及GPU结果以之后的执行收据为准。README/索引/当前状态/AGENTS/代码显式stage、payload audit、commit/push origin Q1、独立40位远端，然后Mac主根/25362逐大小SHA；发布完成以publication旁收据为准。旧失败根与错误均保留；权重/NPZ/BVH保持SERVER_LOCAL_ONLY，用户采纳null，不新建goal/agent、不修改其他自动任务，也未运行固定旧身份的缓存维护工具。
