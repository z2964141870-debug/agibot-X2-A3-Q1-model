# A3冻结codec采样token复用微调E42/R17

记录日期：2026-10-13，北京时间。状态：已实现，尚未登记或运行。

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

代码尚未实际验收，不称修正训练已通过或收益已取得。先scoped审计/推送并核对远端，再登记和3项准备测试，通过后一次两更新/独立核验/匹配效果。拒绝及时保存，不跳过概率门；官方仍基线。
