# PHUMA奖励与共同轨迹世界模型诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-D03。状态：独立入口实现、本地四项控制通过；服务器真实模型诊断尚未执行。R01/V01/D02C已完成，当前无新训练。

## 目标与范围

用户追问PHUMA是否已有改善、能否帮助训练Q1专用运控。R01已显示小样本姿态收益，但0/18通过跟随门，开发移动根RMSE未改善。为选择下一轮训练，读取PHUMA自己的保存证据，区分奖励增量来源以及世界模型H1/H8/H32的根部与关节预测误差。旧A3来源模型的诊断不直接迁移为本轮结论。

新增`script/supermimic_q1_analysis/diagnose_phuma_models.py`，位于121冻结文件之外。只使用注册的11训练/4开发动作进行数值诊断，原评测和审计元数据完整核对，test不参与方向筛选。不新增物理积分、策略推理或optimizer更新，不修改参考、保护线、质量门和R01终态。

## 方法与关键决定

奖励复用固定核心的逐步奖励函数，逐步核对保存reward。分量按实际执行步数之和除以完整参考片长与最大步回报，分量之和必须等于原normalized_return；提前失败仍使用完整分母。回报变化包含完整率变化，是算术拆分而非消融或相同前缀的因果比较。

初始和最终世界模型都预测同一批最终保存状态与实际动作，以同一真实后续状态为目标，避免各自在不同访问状态上打分。H1/H8/H32分别为20/160/640ms，每8控制步一个窗口，终止后无目标；同时记录相同状态的常速度基线、完整预测数组及大小/SHA。窗口重叠，不作为独立样本计数。名义context为脚摩擦参数1、偏置/COM偏移0；控制动作沿原模型诊断接口限制在[-20,20]。

全程CPU、两线程，至少2GiB初始内存余量，距原30GiB线不足256MiB或新增OOM主动拒绝。先核原七阶段终态、四份评测/审计和121冻结文件，完成后再次核验。仅加载本项目自有已审计checkpoint，不调用策略或构造optimizer。

## 已执行验证

本地`python3 -m unittest discover -s script/supermimic_q1_analysis -p test_diagnose_phuma_models.py -v`四项通过：只返回注册train/development；拒绝初末参考/完整片长/划分变化及跨划分重复；手算2/8执行时奖励分量为0.05/0.125/-0.025、合计0.15，拒绝误用执行前缀分母得到的0.6；38步H32只选起点0，40步可选0/8，短于时域及非法窗口拒绝。该控制不加载真实权重/仿真，不能代替真实诊断成功。

## 版本、命令与产物

训练发布保持`ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`，新代码由本阶段Q1提交历史提供。审核/推送/独立核验后部署，重核原进程组无活跃成员、内存/OOM、源文件SHA和输出不存在，再执行：

```bash
cd /opt/data/private/YuanQi_Q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$PY" -u \
  script/supermimic_q1_analysis/diagnose_phuma_models.py \
  --root /opt/data/private/YuanQi_Q1 \
  --experiment /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024 \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_D03_common_models_20261024 \
  --code-commit "$DIAGNOSTIC_COMMIT" --expected-oom-kill 14
```

stdout/stderr预定`logs/PHUMA_D03_common_models_20261024.log`，须`set -euo pipefail`保留tee前的真实退出码。输出排他创建，失败保留，不覆盖重跑。NPZ/权重留服务器LOCAL_ONLY，Mac仅取小JSON和日志；没有新大文件下载或网盘备份。

## 结果与下一步

当前只有入口与四项控制，没有真实模型结果，也未启动下一轮训练。若共同轨迹上H32根预测明显差于常速度且H8更可靠，下一先验证PHUMA模型的动作响应，再决定缩短想象时域的匹配试验；否则按实际奖励/参考动力学证据选择一个主要变量。预测误差不等于动作梯度错误，不预先认定唯一失败原因。

本阶段源码/准备记录审计后显式推送Q1、独立核验远端。执行成功或失败后立即补本README、报告索引与当前状态，并同步用户工作区与服务器；原已完成/停止实验不恢复。
