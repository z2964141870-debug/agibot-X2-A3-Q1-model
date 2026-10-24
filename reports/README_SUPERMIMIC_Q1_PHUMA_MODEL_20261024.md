# PHUMA奖励与共同轨迹世界模型诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-D03。状态：D03真实执行因评测划分标签接口不匹配原生退出1，尚未加载模型张量或创建输出；本地修正和四项控制通过。用户转向论文数据用法复核，D03B延后，当前无新训练。

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
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_D03B_common_models_20261024 \
  --code-commit "$DIAGNOSTIC_COMMIT" --expected-oom-kill 14
```

上述为修正后尚未执行的D03B恢复命令，stdout/stderr须改用新`logs/PHUMA_D03B_common_models_20261024.log`，并用`set -euo pipefail`保留tee前的真实退出码。输出排他创建，失败保留，不覆盖重跑。NPZ/权重留服务器LOCAL_ONLY，Mac仅取小JSON和日志；没有新大文件下载或网盘备份。

## D03真实失败与修正

发布`0206e6bc108bdff92b455ed2d79dafa6247620ff`已推送，独立远端查询首次因SSH代理短暂断线失败，重试后完整ref一致。服务器源码SHA`a09ed657f4c397d0590875fbf9e269b5564c9a1b4385bafe3128aa37c570e7c5`，两端四项控制通过。15:20:03资源预检通过，原七阶段完成/进程组无活跃成员，内存27,366,604,800字节、OOM14，输出和日志均不存在。

真实D03原生退出1：`diagnostic_rows()`误要求原评测逐片split直接为train/development/test，但核心评测实际使用training/heldout；PHUMA三划分由冻结spec/源清单另行维护。错误发生在重核原证据之后、导入torch及加载模型张量之前。原`PHUMA_D03_common_models_20261024`输出未创建；日志742字节/SHA`22b172deb48334223e5ef53d8e218d07029f543665922fcc9ec41bc47f6709ef`，取回一致。

修正只改新诊断入口：先核对真实核心标签training/heldout，再按冻结clip名单恢复train/development/test，并保留`evaluation_split`原字段；不修改原评测、审计或训练文件。控制改用真实核心标签，四项重新通过。尚未部署/真实运行修正后D03B，不能报告世界模型数值结论。[失败收据](../data/manifests/supermimic_q1_phuma_D03_failed_20261024.json)和[原生日志](../logs/session_records/supermimic_q1_phuma_D03_failed_20261024.log)保留。

## 结果与下一步

当前没有真实模型结果，也未启动下一轮训练。用户要求重读论文数据用法后，当前优先级转为[数据规模与协议差距](README_SUPERMIMIC_Q1_PHUMA_DATA_USE_20261024.md)；D03B保持待执行。保留H32基线，后续再按匹配模型/动作响应证据决定时域试验。预测误差不等于动作梯度错误，不预先认定唯一失败原因。

本阶段源码/准备记录审计后显式推送Q1、独立核验远端。执行成功或失败后立即补本README、报告索引与当前状态，并同步用户工作区与服务器；原已完成/停止实验不恢复。
