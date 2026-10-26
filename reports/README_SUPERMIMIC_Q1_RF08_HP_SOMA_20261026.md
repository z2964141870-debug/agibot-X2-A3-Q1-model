# RF08 Q1 / SOMA / HP 一小时训练

2026-10-26 23:05，北京时间。状态PLANNED_NOT_STARTED，准备验证已完成，待本阶段源码发布/部署核验后启动。用户再次明确env=4096。本轮最多3600秒，延长2/4/8小时仅后续考虑，不在本次自动执行。

## 目标与配置

验证整套师兄SuperMimicHP方法适配Q1后是否形成稳定迈步、支撑切换和全身/上肢跟踪。独立RF08，从零/seed0；保留RF06/RF07，不把同时改变多项设置解释为单变量因果。

4096真实GPU环境，WM1024×10/actor1024-H32/critic5×4/UTD.4；历史10×80、未来10×59、context26。500组WM-only、64步采集预热、原网络/BF16/compile。100条SOMA直接Q1训练/15演员分离验证，类别分配与完整来源在[AP01](README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md)。heading XY/相对朝向进观测历史，未来包含相对根位移及完整朝向。Q1模型推导隐式PD/完整effort/动作尺度按HP公式；不是复制X2数值。

师兄实际1024环境而本轮4096；其逐条100/15源清单未取得，本轮独立选片；Q1机器人、cu128软件与一小时预算等差异保留，不称完全等条件复现。

## 版本与实际命令

上游冻结310244b73e45775744a75b789eb25a6c9c5a81c9。本项目发布身份待Git历史及启动plan记录，不预写已同步。代码目录`script/supermimic_q1_hp`，独立runtime在`data/references/supermimichp/AP01_runtime_20261026`；运行时冻结所有适配/上游源码、配置/资产、参考身份。

服务器自己的25252，唯一根`/opt/data/private/ZHY`；新Python`data/environments/supermimic_q1_hp_20261026/bin/python`。正式输出计划为`data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026`，日期为计划日，跨日启动据实记录。

```sh
/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python -u \
  /opt/data/private/ZHY/script/supermimic_q1_hp/run_trial.py \
  --mode train \
  --runtime /opt/data/private/ZHY/data/references/supermimichp/AP01_runtime_20261026 \
  --library /opt/data/private/ZHY/data/supermimic_for_q1/AP01_SOMA_Q1_20261026/q1_candidates/pilot_manifest.json \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026 \
  --expected-oom-kill 0 --code-commit PUBLISHED_40_HEX_SHA --detach
```

## 预算与验收

真实采集/训练软3500秒，在完整更新后正常结束；外层3540秒+45秒退出宽限保最多3600秒，不自动续训、重试、并行第二训练。初始化/资源profile/等待/评测/绘图不作训练效果预算。保存initial/final/last和实际WM/actor更新、转移/训练秒数，不能用墙钟或模型loss宣称完成/改善。

守护保留100GiB余盘、12GiB单实验、30GiB共享cgroup、启动现场OOM基线0、GPU86度、900秒无进展。连接失败UNKNOWN，一次有界核查，不凭PID或断线重启。核boot/PID/start_ticks、controller/status、train_guard和run/status，不热改冻结输入，不操作他人任务。

训练正常结束后顺序执行初末全115片无噪声真实GPU物理评测，保存逐片完成/跌倒、根净移/沿目标/路径、腿/上肢/腕误差、腕高度与时序及足几何近地比例。近地比例不是接触力证据；共同前缀比较用于不同终止长度，摔倒位移不称行走。六宫格细原值/粗500-iteration平均，初末MP4保存Mac项目data对应目录并核大小SHA，不建网站、模型/大轨迹不拉Mac。训练失败保真实终态；终后交付故障可独立补齐评测，不能重跑训练掩盖。

## 目前结果与下一步

4096完整资源profile、Q1 FK/梯度、GPU观测/未来/时序及真实回合记录验证通过，详AP01和小清单。RF08尚无正式训练或效果结果。下一步仅一次发布/部署核验后启动并监视，重要结果立即更新本README、索引、CURRENT_STATE及Git小收据。
