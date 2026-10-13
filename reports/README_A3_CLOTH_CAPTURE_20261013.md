# A3衣服首批一致性持久诊断E39

记录日期：2026-10-13，北京时间。状态：E39于17:24:01 COMPLETE/VERIFIED_ZERO_OPTIMIZER_CAPTURE，实际失配复现且已缩小批形状范围，0optimizer。

## 目标与依据

主线仍是衣服完整跟随和官方035可验证提升。E38资源入口与13控制/900帧ABI通过，但第一实际batch更新前1帧动作失配，原门拒绝，0optimizer；失败时未保存raw batch。E39独立增加先保存后判断的首批诊断，不解锁或复跑E38。当前不能归因于输入改写、FSQ边界、模式选择或硬件。

## 固定输入与预算

复用E38三段已经通过的训练PKL与相同官方暖启动/seed0/16env/FP32/冻结范围，只采24步的一次新rollout、第一实际4环境batch；禁止loss/backward及所有optimizer.step。先持久化sampling与rollout、首实际batch、Actor实际输入、量化前latent与量化后token、精度回执，再固定输入49次forward对照：一遍4x24无aux、24遍4x1无aux、24遍16x1无aux（切回同一microbatch索引）。只改诊断观测，不新增基线效果回放或训练更新。

启用现有cache_encoded_cpu只为保存真实编码结果，不改数值路径；CPU复制与持久化会改变时序，不能据未复现直接否定E38。登记SHA固定原E38 state/prepare/console、原prepared PKL与代码。准备和13控制不重做。独立CPU核验step0等于原官方Actor/Critic（仅std官方clamp边界允许）、optimizer空、采样/保存/Actor输入逐值一致、变体误差和token/latent差异。无法映射compact encoder mask就保留未判定，不伪造完整token覆盖。

计算只ZHY既有环境，系统Python监督；沿用3被动快照/8GiB额外起始余量、4GiB保留/CPU90/GPU85两次5秒/900秒/首故障停止，不并行计算或自动恢复。原机器、硬件配置、旧停止台账不改。

## 命令与工件

```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 -m script.a3.cloth_capture_trial register
/usr/bin/python3 -m script.a3.cloth_capture_trial run --stage train
/usr/bin/python3 -m script.a3.cloth_capture_trial run --stage verify-training
```

这里train仅调用已拦截的首批采集入口，不执行任何参数优化。独立目录data/experiments/a3_zhy_cloth_capture_20261013_E39/、logs/a3_zhy_cloth_capture_20261013_E39/，step0在data/training/a3_zhy_cloth_capture_20261013_E39/R16_001_s0/，小清单data/manifests/a3_zhy_cloth_capture_20261013_E39.json。raw PT/日志LOCAL_ONLY，Mac仅代码/README/小清单。

## 当前结果与下一步

代码dd0dfc1推送并ZHY同步；17:22:49–17:23:43完成一次采集，17:23:45–17:24:01完成独立CPU核验。0optimizer/global_step/backward，官方step0 Actor（仅std官方运行clamp）与Critic精确相同、optimizer空。24采样调用的actor_obs[16,24,930]/critic_obs[16,24,1645]/tokenizer[16,24,2402]与保存rollout逐值完全相同、sampled mean与old_mu完全相同，第一真实Actor输入亦与mini-batch完全相同；支持本次不是输入存储改写。

第一batch环境索引[5,2,0,7]，96有效帧仅环境7/tick2（batch[3,2]）动作差>1e-4。实际训练RMSE0.0006889391806907952、max0.017822980880737305、ratio最大偏差0.21692800521850586，复现E38同量级/同最大动作差；不同采样observer时序不据此称整条仿真逐值重跑。三个保存对照：4x24无aux仍同误差；4x1逐帧无aux仍RMSE0.0006889356/max0.0178231597/ratio偏差0.2169258；16x1逐帧无aux切回同四环境动作误差严格0、ratio偏差仅3.8147e-6。这把本次差异缩到环境批形状相关路径，aux/序列长度不能单独解释；不直接认定某CUDA kernel或量化器根因。

保存g1编码路径token逐坐标无变化、latent最大差4.7684e-7。a3_fast编码采用compact mask，原直接16行假设不满足，状态保留未判定；尚未比较该路径对应token，不能称确认FSQ放大。采样/首实际batch/49forward/metadata/精度全已持久化且SHA独立核验，见[E39小清单](../data/manifests/a3_zhy_cloth_capture_20261013_E39.json)。原始data/training及logs/server LOCAL_ONLY；没有微调收益、最终未见会话验收仍待。

16健康样本17:22:44–17:24:01、最大间隔5.56秒，同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c/OOM0、CPU41/GPU34摄氏度、最高GPU115.53W，容器最低可用5,990,244,352字节(5.58GiB)，无关键内核/遥测错误。训练采集进程采样RSS5.87GB/VmHWM5.97GB，CPU独立验收VmHWM1.43GB；全进程PSS权限/已退出记录保留，不能称完整。作业已退出，无自动恢复。

下一独立保存工件审计：依据真实tokenizer encoder mask将a3_fast compact输出映射回环境/tick，再核对失配帧token与latent；不新采样/加载模型/forward/优化。能定位则对已证实边界做新单变量修正，仍需初始概率门和能力保留，再回到小微调/六衣服四官方动态门。禁止跳过首batch概率检查。
