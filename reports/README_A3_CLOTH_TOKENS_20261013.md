# A3保存编码器compact token审计E40

记录日期：2026-10-13，北京时间。状态：协议/代码已准备，尚未登记或运行。

## 目标与固定范围

E39已确认输入与权重稳定，16环境逐帧mean完全复现、4环境一帧失配，g1 token无变化；a3_fast compact输出尚未按encoder_index映射，不能直接认定量化根因。本E40只读取E39已保存sampling/first_batch/variants/runtime_metadata，按官方parse_tokenizer_obs/create_encoder_masks的已核源码映射native行，比较采样/训练token和latent，定位同一个环境/tick。

不加载模型/checkpoint，不调用编码器/量化器/decoder/forward、IK或仿真，不新增rollout/backward/optimizer。两个CPU映射控制覆盖非连续native行和错误行数拒绝；全384采样输出只映射一次，比较第一96帧及已保存49forward的token。compact mask与输出行数必须精确一致，错则停止，不猜坐标。

## 版本、命令与工件

登记固定E39停止后状态、独立核验与四保存输入SHA，vendor保持fe6868ba；每阶段900秒、原温度/4GiB保留/首异常保护，系统Python监督和3被动起始快照/8GiB余量，CPU单线程/无CUDA。代码script/a3/cloth_token_audit.py，项目/文档/scoped payload审计推送后才登记。

```bash
cd /opt/data/private/hechengzhu/ZHY
CUDA_VISIBLE_DEVICES= /usr/bin/python3 -m script.a3.cloth_token_audit register
CUDA_VISIBLE_DEVICES= /usr/bin/python3 -m script.a3.cloth_token_audit run
```

独立data/experiments/a3_zhy_cloth_tokens_20261013_E40/、logs/a3_zhy_cloth_tokens_20261013_E40/，小清单data/manifests/a3_zhy_cloth_tokens_20261013_E40.json；原E38/E39不修改，raw tensor继续ZHY LOCAL_ONLY，Mac只有小型代码/README/清单。

## 当前结果与下一步

尚未运行，不称已发现token翻转。若同一帧存在微小latent变化、离散token翻转和动作失配，将进一步固定token/decoder独立因果控制后才采用冻结codec采样token复用等修正。保存相关性本身不代替干预；仍保留初始概率门、能力保留、同条件动态门和真正未见会话要求，官方仍基线。
