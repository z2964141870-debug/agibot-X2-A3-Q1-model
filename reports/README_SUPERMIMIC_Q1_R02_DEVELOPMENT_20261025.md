# PHUMA R02第5000轮开发动力学对照

记录日期：2026-10-25。阶段：R02-DEV-A。状态：实际完成，跟随门未通过。

## 目标与范围

用现有PHUMA资源，判断内存保护退出前保存的Q1策略是否值得直接续训。原R02 FAILED终态不改写。只评测原六段development，不评test、不新增训练；初始与5000权重均来自原随机初始化R02。方法仍世界模型/想象BPTT/critic。

## 版本、命令与实际验证

入口发布`a30cfc64fef0f48ae3065461af771d16d5207124`，已推送并独立核远端Q1。前提为[5000保存16门审计](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md)。实际输出`huanyu:/opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R02_DEV_A_step5000_20261025`，四个阶段均原生退出0：初始自由根MuJoCo、初始轨迹独立重算、5000 MuJoCo、5000轨迹独立重算。参考首帧初始化、完整片段、无噪声/DR，原接触/PD/力矩/奖励/0.15rad与0.10m完整跟随门保持。

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/evaluate_phuma_R02_checkpoint.py \
  --root /opt/data/private/YuanQi_Q1 \
  --checkpoint-audit /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/RESOURCE_RECOVERY_A_20261025/checkpoint5000_audit.json \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R02_DEV_A_step5000_20261025
```

原样小证据：[plan](../data/manifests/supermimic_q1_R02_DEV_A_plan_20261025.json)、[status](../data/manifests/supermimic_q1_R02_DEV_A_status_20261025.json)、[对照](../data/manifests/supermimic_q1_R02_DEV_A_comparison_20261025.json)、[初始评测](../data/manifests/supermimic_q1_R02_DEV_A_initial_20261025.json)/[独立审计](../data/manifests/supermimic_q1_R02_DEV_A_audit_initial_20261025.json)、[5000评测](../data/manifests/supermimic_q1_R02_DEV_A_step5000_20261025.json)/[独立审计](../data/manifests/supermimic_q1_R02_DEV_A_audit_step5000_20261025.json)。完整数组留服务器LOCAL_ONLY。结束内存13444337664字节、OOM14，0新optimizer更新。

## 结果与边界

初始5/6完整、5000为6/6完整，但两者均0/6过完整跟随门。六段中五段关节RMSE降低，一段0.22649→0.23049rad微退。非移动几段根部误差多在1–3cm范围，不能替代全身关节门。

唯一开发移动片`humanml/009636`（CMU/137/137_07别名）参考最大水平位移0.521165m。5000策略最大位移仅0.078251m（约15.0%），根位置RMSE0.256907m、终点误差0.592081m、关节RMSE0.452543rad、腿部0.519185rad。初始在126/199步失败，0.431946m位移含跌倒，不能把它当可用移动或与完整片误差直接排名；训练后199/199步完整，回报0.29072→0.56896主要支持稳定性改善，不支持学会移动。

## 决定、保存与下一步

不把本次较大库/320000转移、模型保存有效或完整率上升当作复现成功，也不把扩量当唯一充分解。先继续现有PHUMA独立扩库和移动训练参考诊断，检查世界模型对移动的建模与控制信号，再选是否用新实验从5000追加7000轮；原失败目录不得重启。独立未见移动测试仍未形成充分证据，P07已完成HumanML别名审计但全源隔离尚未证明。

本阶段README、索引及原样小收据本地保存，审核后显式推送Q1并独立核远端；大型工件仅LOCAL_ONLY。BONES仍由师兄处理，代理不恢复获取。
