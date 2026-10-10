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

启动前服务器同 boot `9a9f35c1-c5b3-48d8-ab97-412730c8c937`、GPU空闲，Git main干净。0/2/10/20/40/80/120/200独立大小/SHA/CPU重载/网络和优化器有限性、计数与16环境均通过；step0网络逐张量等于官方。两个优化器参数组保存的LR所有阶段均为2e-5。评测尚未完成；已有官方0/20、step200为5/20跌倒是原实验结果。

step10所属R05_002_s2/config.yaml为0字节，原文件不改。已读固定vendor的A3Policy加载器，MuJoCo结构由权重形状和固定MLP构造、直接strict加载PT，不读取旁边训练YAML；故此次同条件仿真仍可执行，但空配置为历史记录缺损，不能用于独立复现该attempt训练配置。

## 恢复与下一步

先看本 README 与独立 `state.json`，再核验当前进程/GPU/boot；已验证完成动作跳过，中断动作完整重跑。完成后停用独立服务，更新 README/索引/current state，审计后 scoped 提交推送 main 并核验。根据退化时点和优化器真实参数提出单变量实验，尚未授权或启动新训练。
