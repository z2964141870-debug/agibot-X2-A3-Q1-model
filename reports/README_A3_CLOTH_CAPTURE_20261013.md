# A3衣服首批一致性持久诊断E39

记录日期：2026-10-13，北京时间。状态：协议与代码已准备，尚未登记或计算。

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

尚未执行，不称已定位根因或改善。先验系统Python入口无tensor导入、审查diff/payload后scoped提交推送并核对远端，再新登记与一次采集和独立核验。能定位则只针对已证实的执行边界做新单变量修正；修正后才回到小微调与六衣服/四官方动态开发门，最终仍需未见会话。禁止以跳过首batch概率门获得训练次数。
