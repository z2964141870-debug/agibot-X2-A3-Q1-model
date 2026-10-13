# A3单token固定解码干预E41

记录日期：2026-10-13，北京时间。状态：代码与固定协议已准备，尚未登记或运行。

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

尚未执行，不称已证明干预。通过后再把冻结codec的采样token复用等候选写成独立兼容修正，测试首实际PPO概率、参数冻结与锚点保持后才微调；不通过则保留失败并查decoder/拼合接口。最终仍需六衣服/四官方匹配开发门和真正未见会话，官方仍基线。
