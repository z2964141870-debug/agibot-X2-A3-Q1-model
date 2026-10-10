# A3 微调退化定位

日期：2026-10-10。实验 E01；状态：评测进行中。服务器根目录 `/media/yu/FAFF-E9771/YUANQI_A3`，main。

## 目标与范围

比较同一 R05 训练轨迹中的 0、2、10、20、40、80、120 次更新模型，复用官方和 200 次更新的原评测，定位退化最早已观测阶段。只评测，不训练、不修改旧账本，不操作真机。留出动作从此也用于诊断，不能再作为全新最终验收集。

## 方案与依据

所有模型先独立核验大小、SHA、CPU 重载、网络/优化器有限性、16 环境与优化器更新计数；0 更新模型还核对两网络逐张量等于官方。相同资产、selected20、stride4/30Hz参考、50Hz策略、frame0初始化和0延迟完整回放。

记录各阶段逐动作首次跌倒、完整与跌倒前误差；额外按每个动作所有模型首次跌倒之前的共同窗口比较关节、根位置、手腕与腿部误差，避免不同截断长度造成误判。首次已观测退化不等于精确起点或因果归因；重启仅登记并按完整动作恢复。

独立服务 `a3-regression-e01.service`，CPUQuota100%，复用串行评测锁；CPU90/GPU85连续2次、5秒采样保护，重启后300秒冷却。普通失败/热停不自动解除；连续2次启动没有保存动作进度停止。没有累计启动次数/到期上限。

## 版本、命令与产物

vendor `fe6868ba37034f89b912f0fb851bce19f120266d`，新代码 `script/a3/regression_diagnostic.py`；版本以阶段 Git 提交为准。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
data/environments/a3-sonic/bin/python -m script.a3.regression_diagnostic verify
systemctl --user start a3-regression-e01.service
```

产物 `data/experiments/a3_regression_20261010_E01/`；日志 `logs/a3_regression_20261010_E01/`；小型最终清单 `data/manifests/a3_regression_20261010_E01.json`。全部大型工件 LOCAL_ONLY。

## 当前核验与边界

2026-10-10 12:41北京时间SSH超时，暂记连接UNKNOWN；最后已核验step0/2/10/20/40完整，step80有17个动作持久登记。尚未取得新boot，不认定断电或重启。后续连接恢复以boot/service/manifest重新核验，原已安装控制器保留300秒冷却。

12:42连接恢复，boot实际变为 `b13f532d-64fc-42eb-8a09-187639a79b3f`，uptime96秒、服务activating/exit75，GPU无计算进程。旧账本completed包含80且其完整20动作摘要可读：2/20跌倒、关节RMSE0.197315rad；120只有已初始化manifest、0完成。冷却后自动核验恢复，不人工解除blocked、不查重启根因。恢复及统计已有5项单元检查通过；完整动作真实SHA恢复仍以服务后续结果为准。

中间完整阶段：step10为1/20跌倒、关节RMSE0.151929rad；step20为0/20、0.135700rad；step40为0/20、0.142816rad。首次已观测跌倒位于2→10区间，但20/40恢复无跌倒，跌倒数不单调。所有这些阶段总体跟踪误差均高于官方；不是已经找到好的微调方向。80/120仍评测中。

启动前服务器同 boot `9a9f35c1-c5b3-48d8-ab97-412730c8c937`、GPU空闲，Git main干净。0/2/10/20/40/80/120/200独立大小/SHA/CPU重载/网络和优化器有限性、计数与16环境均通过；step0网络逐张量等于官方。两个优化器参数组保存的LR所有阶段均为2e-5。评测尚未完成；已有官方0/20、step200为5/20跌倒是原实验结果。

step10所属R05_002_s2/config.yaml为0字节，原文件不改。已读固定vendor的A3Policy加载器，MuJoCo结构由权重形状和固定MLP构造、直接strict加载PT，不读取旁边训练YAML；故此次同条件仿真仍可执行，但空配置为历史记录缺损，不能用于独立复现该attempt训练配置。

## 恢复与下一步

CPU固定输入诊断 `script/a3/fixed_input_diagnostic.py` 已执行：读取官方慢走最初100真实策略输入，官方适配器与原模拟器动作通过atol1e-4/rtol1e-3，step0编码和动作均与官方完全相等。step2量化编码元素变化27.3594%，动作输出对官方RMSE0.0905764（原始policy单位，不是rad）。仅在推理时交换部件：当前decoder配官方token误差0.0742825；官方decoder配当前token误差0.0355094。两部分都有变化，此小样本中decoder变化影响更大；不能据此声称冻结encoder会修复策略。200次对应0.499309/0.482526/0.136428，均只限同一慢走前2秒固定输入，不是闭环或因果验收。证据 `fixed_input_diagnostic.json` 与 `fixed_input_outputs.npz`。

只读学习率审计 `script/a3/audit_finetune_updates.py` 已执行，证据 `update_audit.json`。step2/200的args.learning_rate均1e-5，optimizer两个参数组及scheduler._last_lr/base_lrs均2e-5；配置actor2e-5/critic1e-3/desiredKL0.01/adaptive下限1e-5。vendor KL handler逐minibatch写所有参数组，轮末lr_scheduler.step再次写入；故记录的自适应LR和checkpoint实际参数组LR不一致，不能把保存点2e-5认定为全程实际优化步长。该冲突值得下一轮前修正/验证，不是已经证明的退化根因。KL日志实际是旧采样策略对当前策略的高斯KL、按动作维求和；clipfrac为clipped surrogate更大的比例。

step0与step2均完整20动作/24304步、无跌倒。step0各聚合指标与已有官方一致；step2关节RMSE0.0590597→0.0915286rad，根位置0.0841691→0.1528472m，手腕anchor相对0.0119790→0.0333251m，腿部anchor相对0.0189769→0.0262490m。故最早已观测的跟踪退化在step2，位于0→2更新区间；没有step1保存点，不能断定发生在第1还是第2次。step2属于初次连续2更新冒烟，早于后续恢复attempt，不能将全部退化归因于后续重启。

审计同时读取全部可用R05 TensorBoard标量。首更新记录KL1.32985、clipfrac0.668229，第2次KL0.249387；Train/lr日志1e-5，与轮末scheduler/optimizer2e-5不一致。R05_002_s2和R05_006_s40事件缺所选标量，其他事件有尾部丢失；缺失不补造。训练标量不替代同条件评测，后续阶段评测仍进行中。

先看本 README 与独立 `state.json`，再核验当前进程/GPU/boot；已验证完成动作跳过，中断动作完整重跑。完成后停用独立服务，更新 README/索引/current state，审计后 scoped 提交推送 main 并核验。根据退化时点和优化器真实参数提出单变量实验，尚未授权或启动新训练。
