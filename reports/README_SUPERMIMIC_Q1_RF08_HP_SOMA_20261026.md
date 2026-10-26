# RF08 Q1 / SOMA / HP 一小时训练

2026-10-26 23:27，北京时间。状态TRAINING_RUNNING，23:23:07正式启动、23:23:22进入实际采集/训练时钟。用户再次明确env=4096并要求定目标观察；MG03工具active、新每10分钟监督q1-rf08-4096实际创建ACTIVE。本轮最多3600秒，延长2/4/8小时仅后续考虑，不在本次自动执行。

## 目标与配置

验证整套师兄SuperMimicHP方法适配Q1后是否形成稳定迈步、支撑切换和全身/上肢跟踪。独立RF08，从零/seed0；保留RF06/RF07，不把同时改变多项设置解释为单变量因果。

4096真实GPU环境，WM1024×10/actor1024-H32/critic5×4/UTD.4；历史10×80、未来10×59、context26。500组WM-only、64步采集预热、原网络/BF16/compile。100条SOMA直接Q1训练/15演员分离验证，类别分配与完整来源在[AP01](README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md)。heading XY/相对朝向进观测历史，未来包含相对根位移及完整朝向。Q1模型推导隐式PD/完整effort/动作尺度按HP公式；不是复制X2数值。

师兄实际1024环境而本轮4096；其逐条100/15源清单未取得，本轮独立选片；Q1机器人、cu128软件与一小时预算等差异保留，不称完全等条件复现。

## 版本与实际命令

上游冻结310244b73e45775744a75b789eb25a6c9c5a81c9。本项目冻结d8b1dcf76b7819ddd5be39a5177e14fe1737a3ad已提交/push Q1并独立ls-remote核验；40文件387207字节在Mac/25252逐大小SHA一致。代码目录`script/supermimic_q1_hp`，独立runtime在`data/references/supermimichp/AP01_runtime_20261026`；plan冻结146项适配/上游源码、配置/资产身份。后续仅报告提交不改变冻结代码。

服务器自己的25252，唯一根`/opt/data/private/ZHY`；新Python`data/environments/supermimic_q1_hp_20261026/bin/python`。正式输出计划为`data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026`，日期为计划日，跨日启动据实记录。

```sh
/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python -u \
  /opt/data/private/ZHY/script/supermimic_q1_hp/run_trial.py \
  --mode train \
  --runtime /opt/data/private/ZHY/data/references/supermimichp/AP01_runtime_20261026 \
  --library /opt/data/private/ZHY/data/supermimic_for_q1/AP01_SOMA_Q1_20261026/q1_candidates/pilot_manifest.json \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026 \
  --expected-oom-kill 0 --code-commit d8b1dcf76b7819ddd5be39a5177e14fe1737a3ad --detach
```

## 预算与验收

真实采集/训练软3500秒，在完整更新后正常结束；外层3540秒+45秒退出宽限保最多3600秒，不自动续训、重试、并行第二训练。初始化/资源profile/等待/评测/绘图不作训练效果预算。保存initial/final/last和实际WM/actor更新、转移/训练秒数，不能用墙钟或模型loss宣称完成/改善。

守护保留100GiB余盘、12GiB单实验、30GiB共享cgroup、启动现场OOM基线0、GPU86度、900秒无进展。连接失败UNKNOWN，一次有界核查，不凭PID或断线重启。核boot/PID/start_ticks、controller/status、train_guard和run/status，不热改冻结输入，不操作他人任务。

训练正常结束后顺序执行初末全115片无噪声真实GPU物理评测，保存逐片完成/跌倒、根净移/沿目标/路径、腿/上肢/腕误差、腕高度与时序及足几何近地比例。近地比例不是接触力证据；共同前缀比较用于不同终止长度，摔倒位移不称行走。六宫格细原值/粗500-iteration平均，初末MP4保存Mac项目data对应目录并核大小SHA，不建网站、模型/大轨迹不拉Mac。训练失败保真实终态；终后交付故障可独立补齐评测，不能重跑训练掩盖。

## 目前结果与下一步

4096完整资源profile、Q1 FK/梯度、GPU观测/未来/时序及真实回合记录验证通过，详AP01。RF08正式运行controller21522/start_ticks3632019311、worker21525/start_ticks3632019335，boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，GPU UUID GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4。23:26:34观察现场三status均RUNNING、命令/进程身份匹配、actual_config num_envs4096/WM1024/actor1024-H32，191.48实际秒/271学习轮/3047424转移/2710次WM/0actor，500轮WM-only预热正常。GPU9766MiB/53度，共享15.53GB、OOM0，余盘1.644TB。

当前没有策略训练效果结论。预计10月27日00:22左右结束训练后接评测；恢复先读现场，禁止重复启动或修改任何冻结脚本/参考。启动快照`data/manifests/supermimic_q1_RF08_launch_observation_20261026.json`，部署收据同前缀code_deployment。日志`logs/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026_train.log`及controller日志；status/guard/run均在RF08根。MG03 A/B完成、C进行中、D/E待，不将“已启动”标目标完成。

## 终后共同前缀比较入口

23:39新增独立`script/project_tools/analyze_q1_hp_trial.py`，不在本轮冻结入口内，未改运行训练/评测源码。以已保存初末轨迹做同初态、连续前缀和参考时轴检查，按训练/验证及walking、gesture等类别比较共同存活帧的根/腿/上肢/腕误差、相对根腕高度时序与几何支撑切换；另保留各自整段完成/跌倒和位移，完整参考抬手峰尚未到达则明确标记。只重算FK/几何，0动力学积分、0策略推理、0优化器更新。

本机语法检查与短/长终止前缀、错误时轴拒绝、RMSE算术小测试通过；真实RF08共同前缀结果仍待初末评测产物，不把工具准备好当效果已验证。恢复先确认输出不存在，终后执行：

```sh
/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python \
  /opt/data/private/ZHY/script/supermimic_q1_hp/activate.py \
  /opt/data/private/ZHY/script/project_tools/analyze_q1_hp_trial.py \
  --runtime /opt/data/private/ZHY/data/references/supermimichp/AP01_runtime_20261026 \
  --library /opt/data/private/ZHY/data/supermimic_for_q1/AP01_SOMA_Q1_20261026/q1_candidates/pilot_manifest.json \
  --trial /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026 \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026/common_prefix_analysis.json
```

23:35:39实际controller/worker身份仍匹配，873学习轮/9211904转移、WM8730次/actor373，三status RUNNING；GPU14538MiB/52度、共享16125796352字节、OOM0、余盘1642826940416字节。已跨过500轮预热，记录实际更新但不据此宣称学会行走。以上为时间点观测，后续依现场，不新增训练。

23:57:18再次现场核验：controller21522/worker21525的boot/start_ticks及命令均匹配，三status RUNNING；2091学习轮、21684224转移、WM20910次/actor1591次，实际训练2036.057519秒。GPU14538MiB/52度、共享16445321216字节/OOM0、余盘1629030232064字节，实验分配554426368字节，保护正常。采集可在本轮优化记录之后继续推进，运行时转移数不强套最终边界算式。小收据`data/manifests/supermimic_q1_RF08_observation_20261026_235718.json`；这不是固定协议策略评测，不据此宣称改善。关于用户RF07图的单调性/过拟合疑问，解释已写入RF07阶段README，RF08冻结配置与一小时上限继续保持。
