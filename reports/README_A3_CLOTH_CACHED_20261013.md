# A3冻结codec采样token复用微调E42/R17

记录日期：2026-10-13，北京时间。状态：18:25:57 TRAINING_VERIFIED，两轮微调/8次真实optimizer及独立CPU验收通过；动态效果尚未评测，官方仍为基线。

## 目标与单变量

E41单token干预证明本批离散token翻转足以解释零更新动作/概率失配。E42仍用E36三段train、官方035暖启动、冻结encoder/kin/std/buffer、只dyn/Critic更新及384teacher约束配方；唯一主修改是PPO动作分布复用实际采样assembled token，避免冻结编码器因16到4批形状重新量化翻转。此规则仅对冻结codec适用，不扩展到端到端训练或当前执行接口。

## 实现与固定判据

采样的Actor module现有_last_full_latent_flat在hook中detach/clone保存[16,1,64]，24步拼接，mini-batch用同一环境indices选取。官方aux与Critic原forward保持，使用官方外部token decoder形成可求梯度mean/Normal/logprob/entropy，decoder梯度不断。编码器全部参数与buffer必须冻结，任何漂移/非法token shape/grad/非有限值拒绝。初始真实batch修正前后均保存，原概率门RMSE<=1e-5/ratio偏差<=.001保留；保存采样token/rollout供CPU独立验identity。

3CPU控制测试decoder非零梯度更新且codec/std不变、unfrozen拒绝、token形状/grad/NaN拒绝。复用E38已通过13控制/900帧ABI的三PKL，不重复转换/旧测试。一次2更新/8optimizer、16env、1epoch/4mini-batch、seed0、LR2e-6及adaptive原范围、FP32不变，384锚点逐步RMSE<=.05/max<=.5。训练独立核验0/1/2保存点、仅dyn变化、8次真实LR/梯度/token身份、首批CPU概率。

合格后6衣服开发499步/4官方诊断动作，复用E26/E02基线；六衣服0跌倒/平均误差改善>=1%、旧不倒单段和官方误差<=+2%、世界根/手腕<=+5%，原门不放宽。训练合同通过也不等于效果或未见泛化；最终仍须真正未参与开发会话。

## 版本、预算和入口

独立E42/R17，代码cloth_cached_tokens.py/cloth_cached_trial.py/test_cloth_cached_tokens.py。GPU计算仅ZHY既有环境，原hp3090仅Git/传输，Mac只小型代码/README/清单。3被动起始快照/8GiB额外余量、CPU90/GPU85两次5秒/4GiB保留/阶段900秒/首异常停，不自动恢复或复跑旧E36–E41。

```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 -m script.a3.cloth_cached_trial register
/usr/bin/python3 -m script.a3.cloth_cached_trial run --stage prepare
/usr/bin/python3 -m script.a3.cloth_cached_trial run --stage train
/usr/bin/python3 -m script.a3.cloth_cached_trial run --stage verify-training
/usr/bin/python3 -m script.a3.cloth_cached_trial run --stage evaluate-cloth
/usr/bin/python3 -m script.a3.cloth_cached_trial run --stage evaluate-official
/usr/bin/python3 -m script.a3.cloth_cached_trial run --stage verify-effect
```

data/experiments/a3_zhy_cloth_cached_20261013_E42/、logs/a3_zhy_cloth_cached_20261013_E42/、data/training/a3_zhy_cloth_cached_20261013_E42/R17_001_s0/及小清单data/manifests/a3_zhy_cloth_cached_20261013_E42.json独立；raw LOCAL_ONLY。

## 当前结果与下一步

代码5a76b56、准备记录ba6ccfb推送核验并同步ZHY。18:21:08–18:21:24一次准备15.97秒通过，3CPU控制0.159秒成功，复用的三900帧PKL逐文件SHA与E38准备回执一致，原13控制/ABI不重做。18:24:33–18:25:38一次train 64.67秒完成两更新；18:25:41–18:25:57独立CPU核验16.08秒通过，无复跑。

实际8次optimizer/8个token批均保留decoder梯度，首批token与采样identity逐值一致。修正前重新编码动作max仍为0.01782298；复用采样token后独立CPU动作RMSE 4.384823e-7、max 2.861023e-6、ratio最大偏差1.716614e-5，原门通过。step1/2均仅14个g1_dyn Actor张量改变，encoder/kin/std/buffer全部精确冻结；官方step0精确核验（仅采用官方运行std clamp）、optimizer初态为空。第8次teacher动作漂移RMSE 0.02823073、max 0.12585819，未超过0.05/0.5预算；这不是能力保留或策略效果证明。

| 保存点 | 字节 | SHA-256 |
| --- | ---: | --- |
| step0 | 134048471 | d245a7cc7d0b3faf39783399635504c541cc8f7fe0efa941820ff4a8fcc249db |
| step1 | 307570767 | e23433ee5a23ffafab021a6d0919188b03b27049e520abf34b8aa39e8ef03dd5 |
| step2 | 307570831 | 8429914489ea0c2ecd58623dfafb2c33ca23898282df0c8db2ae0b9457b5da59 |

22累计健康样本同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c/OOM0、CPU最高40/GPU34摄氏度，GPU最高122.53W；可用最低4,365,545,472字节（约4.066GiB），接近但未触发4GiB保护。最长189.40秒采样间隔在阶段之间，不声称连续监测覆盖。无关键内核/遥测异常，controller未导入tensor；进程扫描的权限/消失错误另存资源清单，不声称整个容器PSS完整覆盖。本轮恢复核验作业已退出、GPU空闲、boot不变；三次相隔5秒额外预检均保留至少8GiB余量。

[E42小清单](../data/manifests/a3_zhy_cloth_cached_20261013_E42.json)保存训练/独立验收、健康及输入/代码SHA，raw LOCAL_ONLY。下一仅执行尚未进行的evaluate-cloth/evaluate-official/verify-effect，step2与E26/E02匹配，原初始化不改、原门不放宽。无需再register/prepare/train；训练合同通过不能证明改善，真正未见会话仍待验。
