# Q1 RTX 3090 故障定位与恢复建议

记录日期：2026-11-03，北京时间。状态：已确认持续的 GPU 复位需求，首次触发原因未定位；设备恢复阻塞。详见[结构化结论](../data/manifests/morphmimic_q1_gpu_error_diagnosis_20261103.json)。

## 当前结论

当前优先由平台运维保留数据后重启承载本卡的 GPU 宿主机，或迁移到健康 GPU。仅重开 SSH 或重启训练容器不能保证恢复宿主机设备。用户本次询问最优方案，尚未实施宿主机、容器或驱动重启。

这项建议针对已持续的设备故障状态，不代表已证明显卡永久损坏，也不代表软件代码已排除。恢复后应先核设备身份、健康状态与短验证，通过后再恢复正式训练。

## 两次错误须区分

| 记录 | 直接证据 | 支持的结论 |
| --- | --- | --- |
| 共享 E2048 V03 | 各 rank 申请158/174MiB时全局仅余92.12/94.12MiB，原栈为 CUDA OOM | 当次明确为显存容量失败；完整轮0，部分 optimizer 已更新 |
| 共享 E2048 V04 | Torch Dynamo 报 unspecified launch failure，随后 Warp 错误719；进程退出后仍 Reset | CUDA 执行失败与持续的设备复位需求；首次故障 kernel/原因未隔离 |

V04 两路各已采集413696个物理转移，完整轮0、五optimizer全0、cache释放0。零更新不能排除之前的仿真/CUDA执行问题，cache释放未发生也不能据此验证或否定交接假设。启动准入未记录 recovery_action，无法证明设备首次进入Reset恰在V04。

## 错误栈与内核证据

V04 首个可见异常在 Torch Dynamo 恢复 CUDA RNG 状态时报告，随后 Warp 的设备内存清理也报719。CUDA 异步报告使这些位置不能直接当作最初出错位置。NVIDIA 的[运行时错误说明](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html)将719定义为kernel执行异常；仅凭该码不能区分程序非法访问、编译/驱动问题或硬件原因。

09:52:19的只读快照仍为原3090、driver/GSP595.112.07、recovery_action=Reset、used1MiB、45C、利用率N/A；repair字段仍要求复位。SSH可用与GPU恢复是两件事。

内核反复出现 `rpcSendMessage failed with status 0x00000062` 及 `rpc.c:2090` 断言。[NVIDIA状态枚举](https://github.com/NVIDIA/nvidia-persistenced/blob/main/nvstatuscodes.h)中0x62对应 `NV_ERR_RESET_REQUIRED`。这里只借官方枚举解释状态，未声称核实了本机595.112.07的对应源码或首次故障。

现有dmesg最早时间571733.320973秒，已晚于V04启动559702.35秒，环形日志不再覆盖首次故障。正确的journalctl -k查询返回无journal文件/无记录，kern.log、syslog、messages不存在，未取得原始Xid。日志中GPU0至GPU7的NVLink状态查询告警不能证明八卡故障或NVLink是起因。

## 一次授权复位已用完

用户确认独占并授权一次GPU复位。09:39:32针对原UUID仅执行一次nvidia-smi GPU reset，返回255/Not Supported，耗时0.118637秒；后查仍Reset，boot不变，设备未恢复。驱动同时列出七个Unlisted GPU不代表执行了其他复位命令。

[原始收据](../logs/session_records/morphmimic_q1_gpu_reset_once_20261103.json)/[结果](../data/manifests/morphmimic_q1_gpu_reset_once_results_20261103.json)保留完整输出，剩余复位次数0。后续诊断没有新CUDA context、积分、策略推理、optimizer或复位。旧文“尚未获授权/未尝试复位”为历史状态。

## 恢复与后续

给平台运维的信息：RTX3090，UUID `GPU-26028785-9fb9-45af-e614-4cc53af15d08`，PCI `0000:38:00.0`，driver595.112.07。设备持续报告Reset，nvidia-smi定向复位不受支持；请保留 `/opt/data/private/ZHY` 后恢复GPU宿主机，或迁至健康设备，并提取2026-11-03 02:35至03:05北京时间原始Xid/NVRM日志。

平台若提供的“重启实例”实际只重启容器，应让运维确认会恢复GPU设备。[NVIDIA管理说明](https://docs.nvidia.com/deploy/nvidia-smi/index.html)也说明复位有平台限制，复位不可用或恢复失败时需要更高级别的系统恢复。恢复结果以实际健康核查为准。

恢复前不启动新GPU作业。恢复后先核boot/GPU/唯一根及数据保留、recovery action与正常利用率，再做有界GPU检查；正式训练仍待共享高容量完整更新验证，不续失败profile权重。单Q1 E4096八全更新及独立CPU保存审计保持，不将其当正式训练或步态效果。

## 保存与验证

故障设备唯一根为 `/opt/data/private/ZHY`；原V04执行冻结 `dbfd8199694f59d96035bb4bc3d027ef2a2d59d1`，未修改训练源码、失败根或终态。原1820输入独立SHA审计沿用已保存证据，本轮只读系统信息/旧日志，不重复加载模型或数据。

原始[首次只读记录](../logs/session_records/morphmimic_q1_gpu_error_readonly_v01_20261103.json)保留最初journalctl选项错误，[补查记录](../logs/session_records/morphmimic_q1_gpu_error_readonly_followup_20261103.json)保留正确-k查询、日志时段和两rank首异常上下文。小文件按README/索引/显式暂存/payload audit/提交推送Q1及独立40位远端核验规范发布，实际镜像与差异见 `data/manifests/morphmimic_q1_gpu_error_diagnosis_publication_20261103.json`；不提前标记未核发布完成。

权重、大NPZ/BVH仍SERVER_LOCAL_ONLY；采纳null，不创建或恢复goal/agent，不改变其他自动任务。
