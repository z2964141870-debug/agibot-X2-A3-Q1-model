# A3冻结codec采样token复用微调E42/R17

记录日期：2026-10-13，北京时间。状态：18:40:46执行COMPLETE、开发效果REJECTED。两轮微调/8次真实optimizer及独立CPU训练验收通过，但六衣服/四官方效果门失败；官方仍为基线。

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

[E42小清单](../data/manifests/a3_zhy_cloth_cached_20261013_E42.json)保存训练/独立验收、健康及输入/代码SHA，raw LOCAL_ONLY。18:34:35–18:36:21六衣服回放105.87秒完成，18:37:48–18:40:00四官方回放132.30秒完成，18:40:36–18:40:46独立效果复算10.74秒完成。共六衣服2994步、四官方5775步；逐轨迹关节RMSE/跌倒重建及保存证据SHA核验通过，执行COMPLETE不代表效果通过。

| 开发衣服动作 | 关节误差变化 | 跌倒 | 世界根/腕误差变化 |
| --- | ---: | --- | --- |
| zj_walk | -1.760% | 无 | -1.11%/-1.39% |
| zj_warmUp | +4.822% | 无 | +9.96%/+10.00% |
| fjw_walk | +2.773% | 无 | -0.69%/-0.95% |
| zls_walk | +2.872% | 无 | +5.53%/+5.40% |
| yy_walk | -7.329% | 无 | +4.05%/+4.08% |
| zzx_walk | +24.913% | 5.86秒（基线5.66秒） | +23.78%/+25.34% |

六段平均关节RMSE从0.1413798054到0.1549924643rad，退化9.63%，仍5/6不倒；warmUp/fjw/zls超过原不倒段2%预算，warmUp/zls/zzx超过世界根/腕5%预算。官方四段均不倒，但前向慢走+0.956%、原地转身+4.935%、右单腿平衡+14.534%、原地踏步+5.850%，后三段超过2%预算。development_gate_passed=false，候选不晋级，未进行真正未见最终测试或部署。

72累计健康样本同boot/OOM0、CPU最高43/GPU35摄氏度，无关键内核/遥测异常；最低可用仍为训练阶段4.066GiB，最长518.72秒间隔在阶段之间，不声称全天或连续稳定性。计算已经退出，温度/内存保护未放宽。结果与输入清单按scoped audit提交推送，raw仍LOCAL_ONLY。

下一新编号只分析已保存的E42/E26/E02同条件轨迹：验证初始输入与参考一致，比较双方共同跌倒前区间、关节/身体分组及官方能力退化，区分跌倒后轨迹对全程误差的影响，再选择保留能力约束/更新强度或参考层的单变量。E42不再注册/训练/评测；修正训练概率不保证性能上升，两个局部改善不能替代整体效果门。
