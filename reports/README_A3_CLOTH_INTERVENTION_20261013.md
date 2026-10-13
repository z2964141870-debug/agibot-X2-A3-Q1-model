# A3单token固定解码干预E41

记录日期：2026-10-13，北京时间。状态：18:10:43 COMPLETE/VERIFIED_TOKEN_INTERVENTION，两次解码控制通过，0优化。

## 目标与依据

E39确认输入/权重稳定与批形状相关差异，E40确认全局环境7/tick2的A3 latent差1.79e-7、token1/channel24从-.375到-.4375，恰与动作差帧重合。E41不新增采样或训练，只把同一4x24输入、官方step0和decoder固定，单独恢复一个token坐标，验证动作/初始概率恢复的因果关系。

## 固定预算和判据

一份已独立核验的E39官方step0 Actor载入一次、所有参数冻结/eval/FP32，官方forward_with_external_tokens接口仅解码两次：原训练tokens、只把[3,2,1,24]的-.4375恢复到-.375。必须确认两token矩阵只差此一坐标、恢复后等于采样assembled tokens；native按E40已验mask及官方覆盖顺序拼合，其他输入不变。0encoder/quantizer forward、0optimizer/backward/新rollout。

原tokens重算必须匹配实际训练mean最大差<=1e-5，且仍动作差>1e-4/ratio偏差>.001；单坐标修正后相对原采样动作最大差<=1e-5、ratio偏差<=.001，模型完整state前后SHA相同。保存两组decoder输出，CPU重载独立复算概率。即使通过也只定位这一批数值一致性问题，不证明跌倒原因/训练收益或全部输入域稳定。

## 版本与入口

只ZHY既有环境/GPU，系统Python监督，3被动起始快照/8GiB额外起始余量、CPU90/GPU85两次5秒/4GiB保留/900秒/首异常停止，原保护不放宽，无自动恢复。先scoped diff/payload审计提交推送核验，再新登记。E38/E39/E40不重置。

```bash
cd /opt/data/private/hechengzhu/ZHY
/usr/bin/python3 -m script.a3.cloth_token_intervention register
/usr/bin/python3 -m script.a3.cloth_token_intervention run
```

data/experiments/a3_zhy_cloth_intervention_20261013_E41/、logs/a3_zhy_cloth_intervention_20261013_E41/与小清单data/manifests/a3_zhy_cloth_intervention_20261013_E41.json独立；raw PT仅ZHY LOCAL_ONLY，Mac只代码/报告/小清单。

## 当前结果与下一步

代码60d647b已推送核验并同步ZHY。18:10:27–18:10:43一次独立干预15.89秒完成，官方step0只载一次，模型完整state前后SHA3545909b9fa2bbc6ee020c86857cfc1d51add59974abc24917f748bf789d8d62一致，0encoder/quantizer forward、0新rollout/backward/optimizer。

原训练token通过官方外部token decoder重算与真实训练mean最大差严格0，复现动作RMSE0.0006889392/max0.0178229809、ratio最大偏差0.2169280052。固定4x24输入及权重、仅[3,2,1,24]由-.4375恢复-.375后，动作RMSE4.384823e-7/max2.861023e-6、ratio偏差1.716614e-5，均满足原1e-5/0.001门。重载保存输出后CPU概率独立复算逐项相等。这证明此一批动作/概率偏差由该离散token翻转足以解释；并未定位底层具体GEMM kernel或证明机器人跌倒/旧所有批次根因。

5健康样本18:10:22–18:10:42、最大间隔5.22秒，同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c/OOM0、CPU最高44/GPU28摄氏度、GPU最高47.26W，可用最低8,313,692,160字节，无新关键内核/遥测异常；已退出。[E41小清单](../data/manifests/a3_zhy_cloth_intervention_20261013_E41.json)登记两组数值、state SHA与输出SHA2fb4cd465928eeedcf0f5f7352d9cc1f71175538d66dec2c4a9216bd89b787ba，原始输出ZHY LOCAL_ONLY。

下一独立修正仅适用于冻结codec：采样时直接保存实际assembled token，PPO小批使用该token构造动作分布，encoder/kin/std/buffer必须不变；官方aux及Critic仍保留，decoder梯度不截断。先验首实际batch概率及teacher约束，再两更新/8optimizer与独立验收。最终仍需六衣服/四官方匹配开发门和真正未见会话，官方仍基线，没有微调收益。
