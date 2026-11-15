# X2 RPC deadline: independent 20/25/30ms experiment

The user asked whether the20ms gate can be relaxed. It is a software acceptance deadline tied to the nominal50Hz period, not a physical maximum. The16:53 powered run successfully handed over but latched fault at RPC24.346ms while feedback age was about1.72ms.

Saved first600 samples:587 reply-ready; RPC above20/25/30ms was13/2/1 respectively, maximum40.029274ms. This is only a retrospective timing screen: post-fault samples cannot be reclassified as successful physical control.

Independent source generator validates original native controller SHA ca3f80b4b8a77613b243e1c903ba856475e1fb0b4618aeb0d3cbb7a46a18ee47, changes only two RPC comparisons and adds explicit budget range20..30ms/status metadata. Existing source, R6 binary/profile, model/gains/physics,50Hz requests/500Hz writer,80ms target and120ms feedback limits are preserved. Generates new20/25/30ms profiles and a new binary under an independent data root. No automatic powered adoption or MC changes.

Preparation: two local generator tests passed; shell syntax passed. Board build and read-only observation pending. Bounds: build180s/two compile workers, at least4GiB available/<80C; observation at most20s per candidate, no--control, no new training or BLE reconnection. Inspect existing writer absence before observation. Resources and actual results must be recorded separately. A30ms single-RPC deadline can still fault on40ms; this does not certify stable50Hz or propose indefinite holding of stale commands.

Next: build fresh candidate, evaluate live30ms in observe mode with the same current suit stream. Keep the20ms reference as default, do not replace R6 automatically. Recovery has already been independently confirmed atMC STAND_DEFAULT and original damping writer exited.


## 板端执行结果：候选构建完成，实效比较待真人数据恢复

2026-11-15，同一SoC1 boot `3db706a5-dbb5-4916-bfca-a7fab5089abf`。源码提交 `55cdc2119086cb43ad842d6881f381188d3281ae` 已显式push并独立40位远端核同。新根 `data/analysis/X2_RPC_BUDGET_R7_20261115`，构建外层180秒/两线程，真实exit0；CMake writer gate测试1/1通过。原R6二进制SHA `f131011527c1aa453af6fc7360a0b55a637f4ae64b340e6940c764246c3fcec0`、原控制源码/adapter/gate/传感器profile共五件前后SHA同。

独立候选真实加载20ms和30ms配置，各observe 20秒；包含启动与收尾的墙钟20.647/20.650秒。两组均正常退出、SDK发布0、MC修改0，结束无Morph native/control进程；MC只读实核Running(42508)/STAND_DEFAULT。最低可用11082338304B，最高66.968°C，未触发保护。

| 配置 | 请求 | 有效人体目标 | 接受 | 拒绝原因 |
|---|---:|---:|---:|---|
| 20ms | 976 | 0 | 0 | human_reference_unavailable |
| 30ms | 993 | 0 | 0 | human_reference_unavailable |

**这次不能评估门槛收益。** API实读上下衣均fresh=false/hz=0，reconstruction=stale_hold；复核时最后有效包年龄约546秒。设备状态字符串仍显示receiving，所以应看fresh与age，不能只看连接标签。原标定静止门已通过。服务无有效HumanFrame时只回拒绝，约2.4ms的空回复不属于Morph策略推理速度。没有重连衣服、重启FGP/策略或自动接管。

原16:53前600保存样本中587个reply-ready，超过20/25/30ms分别13/2/1，最大40.029ms；这是历史阈值筛查，不能当30ms实控结果。之前24.346ms会落在新30ms时限内，但40ms慢调用和人体断流仍会失败。

## 如何使用这项放宽

20ms是单次RPC接纳期限，来源于名义50Hz周期，不是硬件不可逾越值。候选将单次接纳期限提高到30ms；请求调度仍名义50Hz、writer仍500Hz、目标最大年龄80ms/反馈120ms和故障锁存不变。超过20ms的调用可能错过一次策略更新，放宽没有加速推理，也不能认证持续50Hz。实时衣服30Hz与策略50Hz是两个时钟；不因衣服30Hz直接更改训练策略步长。

候选二进制：`data/analysis/X2_RPC_BUDGET_R7_20261115/build/x2_morph_native`；三个配置为同根 `profile_20ms.json` / `profile_25ms.json` / `profile_30ms.json`。原终端三仍使用R6默认；本阶段没有自动采用新配置。

Done：独立30ms实现、板端编译/保护测试、两个有界只读运行与资源/原件终检完成；原机器人已恢复MC站立。

To do：衣服恢复有效输出后先检查/api/state的fresh/age与SMPL推进，再对30ms候选做有有效目标的observe，检查连续就绪、所有拒绝原因和>30ms长尾；通过后才由现场操作者安排有支撑短控制。不得用本次零ready数据放行实控。后续若仍有长尾，优先优化策略调度/CPU资源隔离，再评估有界容错，不能无限增加目标年龄或自动清除故障。

原始日志/每组summary/resources/比较数值保存在同根；构建stdout `logs/x2_rpc_budget_r7_build_20261115.log`。九原件及清单9667B压缩包已Mac大小SHA同，逐件清单验证通过；本地保存在主根data/analysis同名目录。包SHA `c356b8c906fa719c85c86062d7ab0abaf3c3232e745306ac3cdeff4c5698863d`。小记录：[执行收据](../logs/session_records/x2_rpc_budget_r7_observed_20261115.json)。新原始native逐帧日志只在机器人，不是本机或云备份。阶段文档设备镜像及Git发布以后续publication为准。
