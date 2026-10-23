# Q1独立根高支撑参考候选

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D05。状态：独立入口与验证代码已实现，服务器控制及真实候选生成待执行；未接入训练。

## 目标与范围

针对[D04](README_SUPERMIMIC_Q1_REFERENCE_DYNAMICS_20261024.md)确认的参考足底悬空，准备单一变量的独立候选。R03仍按原固定预算训练，不改变其输入、冻结的两个训练源码目录、控制设置、预算或保护，不启动额外GPU作业。

## 完成内容与关键决定

新增`script/supermimic_q1_analysis/build_support_projection.py`、`audit_support_projection.py`及控制测试。候选只调整每帧根部Z及相应竖直速度；保持原50Hz时间轴、水平路径、根姿态、全部关节位置/速度、来源和三片覆盖。重新计算身体FK/速度，不直接修改原参考。

使用SciPy稀疏有界最小二乘，将较低足底限制在地面下0.05至0.5mm，目标0.25mm；最小化根高修正相对目标的误差及二阶差分，二阶差分权重10、求解单位1mm，绝对根高修正上限3cm。这是固定的几何候选，不是优化接触力或已验证支撑平衡；微小穿透只为避免精确零高度受数值误差变成无接触，不证明所产生支撑力正确。

每片独立记录绝对原目标的身体位置误差、全片自碰撞和足底接触，复用D04的三点/五点差分对新候选做逆动力学检查。原D04不重跑，候选不修改关节，因此预期自碰撞不会消失。出拳仍为复用开发数据，不提升为独立测试。所有候选明确`qualified_for_training_pilot=false`，须先完成独立导出审计及后续匹配动力学评测再决定是否接入。

## 版本、命令与证据

工作副本`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`；服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。服务器已确认SciPy1.15.3/NumPy1.26.0；未安装或修改共享依赖。代码身份由本次Git提交和部署大小/SHA记录。

本地AST三文件通过。新增工件控制后，本地7项中4项通过、3项因缺SciPy明确跳过；真实导出阳性及错误关节、速度、身体字段阴性控制通过。服务器要求所有7项通过且0跳过后才生成真实候选。启动要求30GiB保护线下至少896MiB余量，运行每128帧保留128MiB并核对OOM；本分析失败不会向R03发送信号。

```bash
$PY -m unittest discover -s script/supermimic_q1_analysis -p test_build_support_projection.py -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 $PY -u script/supermimic_q1_analysis/build_support_projection.py \
  --root "$PWD" \
  --admission data/supermimic_for_q1/R03_B128_extend12000_20261023/run/admission.json \
  --diagnostic data/supermimic_for_q1/D04_reference_dynamics_20261024/receipt.json \
  --audit data/supermimic_for_q1/D04_reference_dynamics_20261024/summary_audit.json \
  --output data/supermimic_for_q1/D05_support_projection_20261024
```

上述候选生成命令尚未执行，输出不能重复覆盖。生成后按新receipt执行独立导出审计，并以`dynamics.json`调用已有`audit_reference_dynamics.py`重算新候选统计与差分；不重跑原D04。

## 结果与边界

本阶段尚无真实候选或改善结果。零新增训练、策略推理、积分或GPU渲染。R03最近实核为10月24日01:26:51，原controller762428/start3606081278与worker762431/start3606081380同boot非僵尸，3900/12000追加轮、62400新交互、累计6900轮，冻结条件通过，GPU40摄氏度/空闲16671MiB、容器31056261120字节/OOM14；恢复仍重新核对，不能把本快照当实时证明。

## 保存与下一步

源码、报告与后续小JSON/日志按`Q1`保存，提交前审查暂存区和payload，显式推送并独立比对远端提交。大NPZ及模型只留服务器LOCAL_ONLY，不取回Mac。

先发布/部署并通过服务器7项控制，再在独立目录生成一次候选和完成审计；如几何改善同时动力学残差变坏，保留失败结论，不因足底接触增加自动接入。主线等待原R03结束后的完整配对评测、独立重算及V02渲染，以移动/腿部/全身误差和实际画面判断可用性，不以候选生成代替策略成功。
