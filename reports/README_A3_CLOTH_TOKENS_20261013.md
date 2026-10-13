# A3保存编码器compact token审计E40

记录日期：2026-10-13，北京时间。状态：17:55:58 COMPLETE/VERIFIED_SAVED_TOKEN_MAPPING，0模型加载/forward/optimizer。

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

源码09dd52d推送核验并同步ZHY；17:55:52–17:55:58一次CPU保存工件审计完成，2映射控制通过。96个g1 native帧无token差；48个a3_fast native帧仅1帧/1坐标变化，正是microbatch[3,2]、全局环境7/tick2、第1号token/channel24。采样latent=-0.4401627480983734，训练latent=-0.44016292691230774，绝对差1.7881393432617188e-7；采样token=-0.375，训练token=-0.4375，跳变0.0625。全局latent最大差4.76837158203125e-7。

4x24无aux与4x1无aux保存结果都有同一1坐标/1帧token差，16x1原批形状无差。g1全部变体token都一致。a3_fast的token差帧与动作差帧集合完全相同，支持量化边界放大的具体机制候选；尚未单独干预token再解码，所以不称完成decoder因果证明、不认定机器人跌倒的根因，也不等于参数变化。

3健康样本17:55:47–17:55:57，最大间隔5.23秒，同boot d994e71f-bae0-46ba-a5f7-6c2163771c6c/OOM0、CPU40/GPU26摄氏度、GPU最高33.44W，可用最低9,134,800,896字节，无关键内核/遥测异常；GPU仅只读观测，实际0模型加载/forward/新rollout/optimizer，作业已退出。3次登记前额外资源快照另存，不将区间空档说成连续覆盖。

[E40小清单](../data/manifests/a3_zhy_cloth_tokens_20261013_E40.json)保存SHA、坐标和值以及控制结果；原始E39工件仍ZHY LOCAL_ONLY，Mac不下载张量。下一新编号固定官方decoder/首批输入，在相同4x24形状只替换这一token坐标，检验动作和概率恢复；证实后才实现冻结codec采样token复用或其他兼容修正。原初始概率门、能力保留、同条件动态门和未见会话要求不变，官方仍基线。
