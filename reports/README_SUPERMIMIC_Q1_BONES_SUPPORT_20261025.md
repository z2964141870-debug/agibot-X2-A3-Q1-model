# BONES真实移动参考的足底与碰撞比较

记录日期：2026-10-25，北京时间。阶段：SM-Q1-BONES-D01。状态：只读比较已结束，用户随后取消本工程额外物理准入门；本结果仅历史诊断，不阻止新训练。

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

## 实际结束与用户覆盖决定

12:12:32至12:13:29真实退出0、guard完成回收，8个冻结train全部完成原G1/映射目标/Q1比较；刚体足底逐帧闭合、原8片指标及共同Z位移关系均通过。新增分配827392字节、余盘约2.095TB/OOM14；原数组与来源保持。原样[summary](../data/manifests/supermimic_q1_BONES_D01_support_20261025.json)和[guard](../data/manifests/supermimic_q1_BONES_D01_support_guard_20261025.json)归档。

用户明确要求取消代理自定训练数据硬门槛，优先复刻SuperMimic。该决定覆盖本文旧“以几何修正通过再训练”的安排；不再扩展D01或以支撑、间隙、自碰撞及2rad/s挡住新数据。后续以[新主线](README_SUPERMIMIC_Q1_PAPER_FIRST_20261025.md)为准，历史结果原样保留。
