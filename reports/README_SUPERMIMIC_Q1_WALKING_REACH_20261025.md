# PHUMA P13步行目标可达性

记录日期：2026-10-25，北京时间。阶段P13。状态：只读诊断与控制已实现，正式服务器运行待。

## 目标与范围

继续现有PHUMA/AMASS/SMPL-X及SuperMimic世界模型路线；BONES交师兄处理。[P12C](README_SUPERMIMIC_Q1_WALKING_CONTACT_20261025.md)加足地不等式后仍浮脚，本阶段仅用已完成、独立审计通过的四个train/24组保存数组，量化根与踝部目标的几何兼容。不读取P10补充test来选参数，不修改当前R02D的340/6/7输入、7000新轮或516冻结，不启动第二训练。

## 实现与决定

新入口`script/supermimic_q1_sources/diagnose_walking_reach.py`读取P12C原数组，核receipt/guard/独立审计和全部绑定输入前后SHA。沿Q1的hip_pitch到ankle_roll刚体链累加平移长度，作为忽略关节范围/足姿态的保守外球半径。必须先核各关节均为原点处hinge、hip直接接pelvis；实际FK脚踝距离不可超此半径。目标距离超过此上界，就至少产生超出量那么大的踝部位置误差；球内不代表实际可达。

本地三项解析控制分别原生退出0，两个新文件语法通过；Mac无MuJoCo，实际Q1控制仅安排在服务器执行，不把本地三项称四项全过。部署前复核NPZ字段，采用原生产者实际`stance_mask`；没有运行错误字段的真实诊断。

同时对原映射目标与接触修正目标分别计算。保持root XY/方向时，给出左右脚外球允许的根Z修正区间及交集，区分水平跨度无解、双脚竖直区间无交集和必要降低量。区间非空只是必要条件，不能直接生成参考或称动力学可行；此处不做逐帧根高投影、IK或积分。新控制覆盖解析距离/竖直区间、原点平移不变性、非法输入拒绝及真实Q1 256组关节限位内FK/固定hip前提。

07:10实读R02D仍3825新增轮/244800转移、原三个实际进程及三状态RUNNING，GPU39摄氏度/OOM14、盘余393.39GB；未宣称移动有效。初步只读探查提示修正目标可能超腿长，正式绑定结果待；不以聊天探针代替本阶段收据。

## 版本、命令与产物

权威Git`/tmp/yuanqi-supermimic-q1-20261023`，Q1分支；服务器`huanyu:/opt/data/private/YuanQi_Q1`。先控制/审查/提交推送/独立核远端，逐SHA部署并运行服务器全部控制后，执行：

```bash
ROOT=/opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u script/supermimic_q1_sources/diagnose_walking_reach.py \
  --root "$ROOT" \
  --output "$ROOT/data/supermimic_for_q1/PHUMA_P13_walking_reach_20261025" \
  --expected-oom-kill 14
```

单CPU/nice10/无CUDA；共享内存28.75GiB/OOM14、盘余100GiB、输出32MiB、无进展900秒停止，只回收自身worker。输出`diagnosis/{receipt.json,*.npz}`和`guard.json`，stdout同名logs/；0新IK/积分/optimizer/训练参考。正式结果、独立核验及诊断后的方案选择待，不提前称任何门通过。

## 保存与下一步

大型原件保留服务器LOCAL_ONLY，Mac仅收小证据；代码、README、索引、当前状态及小收据审计后按Q1规范发布并核远端，再同步恢复镜像。当前目标未达、训练最终配对待。优先完成正式可达性证据，再决定参考映射或根高度的独立有限试验；不重复单纯增加脚任务权重。

## 服务器控制通过

准备发布`024a68f9cd96788dd24be7b569369b641bffe3eb`已独立核Q1远端。首次Git推送因本机6234代理连接关闭而原生失败128；原会话结束后，用当前SSH配置/10秒连接超时重新推送退出0，独立ls-remote完整号一致，未强推或改远端。两新文件部署前均不存在，部署后SHA：producer`a6dd647a36fb10130e0f8d1fb6d5419d04b6e2678accd76f99009db8ad41d223`、test`1e9ad7a7e9d370105beef8c45c42b0ebce5550a4f23fa02928d452bc9106dca5`，与发布一致。

服务器四控制原生退出0、0跳过；真实Q1 256组关节限位内姿态的双侧外球上界及hip中心固定前提通过。原样[控制日志](../logs/session_records/supermimic_q1_P13_reach_controls_20261025.log)SHA`28cfbe5ff7d15fd67984332a1432e8aa0a5f310f5f2ee0c0817c9c0dff2e87c8`，服务器`logs/PHUMA_P13_reach_controls_20261025.log`。这没有读取正式24组数组或改变参考，正式P13待执行。

## 首次启动预检失败

正式入口首次原生退出1：新main在输出目录创建前调用完整`budget.check()`，而现有StorageBudget的allocated_tree要求目录已存在。错误发生在mkdir/worker/正式数组读取前，没有创建预定实验目录，也无guard或服务器日志。[终端回执节录](../logs/session_records/supermimic_q1_P13_bootstrap_failed_20261025.txt)保留，不能称真实P13已运行或四控制覆盖了启动顺序。

修正对齐既有P12包装器：建目录前仅检查内存，建目录后由未改guard完整核100GiB盘余/32MiB输出及共享内存再启动worker；worker继续独立完整检查。未放宽任何停止线，当前R02D未变。修正发布/逐SHA部署后才运行正式P13。
