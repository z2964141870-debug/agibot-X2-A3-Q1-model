# 低延迟遥操流式时序与短预览接口

记录时间：2026-11-06 00:58北京时间。长期目标保持active，最低执行至今日10:00；本页是实现阶段，不是目标完成、真实遥操延迟或自然性验收。

## 已实现

新增独立stream_contract.py、preview_policy.py和preview_worker.py，没有修改原V04执行输入、inference.py或final评估CLI。

输入显式记录source/session/clock、SMPL24世界坐标米制位置及wxyz四元数、capture/dependency/ready/arrival时间。所有时间属于同一个接收端或回放时钟；本模块不代替传感器时钟同步。离线sidecar无法证明预处理因果性时标为offline_unverified，dependency保持null，不能默认其重定向或平滑是因果的。

PoseTimeline默认保留256帧、上限4096；按到达事件推进，不能预装未来到达包再回退控制时间。采集乱序可接受，相同序号相同内容忽略，冲突拒绝，容量不足淘汰较早采集帧。可见性同时受control时间和指定H个20ms目标槽限制；即使更远帧已经到达，H4仍不能取用H4以后的采集值。

掩码只保留完整连续目标前缀。第一槽缺失或已使用前缀过期时can_step=false，调用者须执行明确回退；后面某槽丢失时只启用此前连续前缀。填充尾部复制最后可见姿态，但姿态和heading通过上游原生掩码共同隐藏。较新的非连续包不能刷新旧前缀的新鲜度。

离线片尾用显式terminal事件复现原评估的末帧保持，只有末帧到达后生效。保持槽标为declared_clip_terminal_hold且measured=false；不是实际采集到的未来。该模式限offline_unverified源，不能据文件长度提前决定实时人体已停下。

## 时间语义

| 字段 | 含义 |
| --- | --- |
| requested_preview_ns | 请求H个20ms槽，H1仍为+20ms |
| scheduled_origin_to_control_ns | 回放origin到控制决策的计划等待 |
| latest_capture_age_ns | 预算内最新已到达采集帧年龄 |
| latest_used_capture_age_ns | 实际启用前缀的最后采集帧年龄；过期判断依据 |
| policy_queue/compute/send | 同一软件时钟的排队、策略和下发分段 |
| measured_end_to_end_latency_ns | 始终null，没有执行器或外部动作测量 |

worker的50Hz回放时间和perf_counter策略计算耗时属于不同的时钟，分别记录，不相加冒充实测端到端延迟。+20..+200ms目标语义保持，没有将H1悄悄解释为当前帧。

## 原生策略适配

HumanPreviewPolicy复用native actor、冻结normalizers、原pack_human_goal/act_human和fixed_human_masks，仅改变全24点条件下的时间前缀。没有改为mixed并连带改变空间点子集，也不对motion接口截帧。

新CPU策略worker复用NativeSession的权重身份、同片同帧、物理状态转换和历史构造，只按当前到达计划索引人体帧，提交十个原时间槽与前缀掩码。沿用原共同物理RPC协议，供新独立controller调用。

当前checkpoint训练为full/h10，H5/H4/H2/H1是待检验的分布变化。必须先在服务器原运行时验证H10与旧human动作等价，再做新物理对照；此前motion三片结果不能当作本轮human/H10结果。

## 本地验证

12项本地检查通过，1项原运行时检查因缺原Torch/Q1资产明确跳过；三份新增执行模块语法编译通过。0新物理积分、optimizer或CUDA工作，没有重复96组离线参考实验。

覆盖H10/H5/H4/H2/H1前缀、未来预装/时间回退、远帧毒化、乱序/重复/冲突/丢帧/容量、过期边界、不可变有限姿态和四元数、40个随机到达场景与独立oracle、显式片尾、软件延迟及策略输入边界。原运行时H10等价、隐藏pose/heading不变、冻结normalizer/RNG及absent-joint检查已写，待服务器执行。

```sh
python3 -m unittest discover -s script/supermimic_teleop -p test_stream_contract.py -v
python3 -m unittest discover -s script/supermimic_teleop -p 'test_preview_policy*.py' -v
```

第二条在普通Mac上明确跳过服务器测试，不能把零退出当作全部原生验证完成。[阶段记录与代码SHA](../logs/session_records/q1_teleop_stream_contract_20261106.json)。

## 下一步

继续实现独立有界controller：复用已完成16000双型CPU保存审计，同human接口H10/H5/H4/H2/H1、三固定片seed0。原运行时等价、真实训练身份和资源门通过后，再逐阶段物理及GPU EGL视频；真实指标和媒体齐后判断是否值得训练因果/短预览学生。预测尾部与滤波另设对照。

本轮一次原Windows桥接恢复在100.65.183.91:22超时，未到认证/Linux，实时观察仍UNKNOWN。继续独立实现，不重复认证或换入口，不据失联停止/重启训练。服务器镜像、原媒体补传及原运行时测试待连接恢复。
