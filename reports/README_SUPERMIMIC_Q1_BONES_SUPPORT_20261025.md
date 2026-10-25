# BONES真实移动参考的足底与碰撞比较

记录日期：2026-10-25，北京时间。阶段：SM-Q1-BONES-D01。状态：只读诊断入口已实现，真实执行待；没有新增训练参考。

## 目标与范围

[A02真实接入与8片结果](README_SUPERMIMIC_Q1_BONES_RECEIPT_20261025.md)已完成。8片train全部支撑/最大间隙失败、7片自穿透失败，0准入。仅对同8个冻结train来源比较：原生120Hz G1、原速50Hz G1、映射后Q1脚刚体目标、导出前Q1、整段共同移高后的Q1。区分来源几何、映射目标和IK输出的缺口，不查看held-out或改原失败工件。

## 实现与决定

新增`script/supermimic_q1_sources/diagnose_bones_support.py`，复用独立CSV读取与实际MuJoCo碰撞几何，按左右踝刚体重建sphere/box/mesh足底。每一帧以实际Q1脚姿态核验重建足底与fresh FK一致，另逐片对照A02独立验收原指标及全局位移关系；任一超容差拒绝成功。对源G1与Q1报告实际最深自碰撞身体对。

不优化IK或动力学、不修改参考/原准入门槛、不执行optimizer。输出仅新诊断数组与小收据，统一根内100GiB盘余/12GiB实验/250GiB来源区、28.75GiB内存/OOM14/900秒进展线及进程组清理。保持原包LOCAL_ONLY、不复制原包。

## 版本、命令与证据

```bash
ROOT=/opt/data/private/ZHY
PY=$ROOT/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u "$ROOT/script/supermimic_q1_sources/diagnose_bones_support.py" \
  --root "$ROOT" --batch "$ROOT/data/supermimic_for_q1/BONES_A02_received_20261025" \
  --output "$ROOT/data/supermimic_for_q1/BONES_D01_support_20261025" --expected-oom-kill 14
```

此为计划命令，真实结果执行后补充。预期summary.json、guard.json及8份小型数组保留服务器；日志`logs/BONES_D01_support_20261025.log`。输入和代码前后逐大小/SHA核验，模型声明仍为宇树官方G1运动学等价候选，不证明BONES精确生成XML身份。

## 结果与边界

真实脚底比较尚未执行，不能先归因源数据、映射或IK。该检查只给运动学几何证据，不证明动力学可行、策略跟随或已形成合格库。8片当前仍0准入，A02终态保持。

## 保存与下一步

本入口与阶段README在Q1分支同次审核提交，远端核验后部署；实际运行结束即补充原样小收据与结论。以比较结果决定后续接触/映射修正，避免直接全量转换或仅追加训练预算。
