# Q1 R03最终奖励与世界模型诊断

记录日期：2026-10-24，北京时间。阶段：SM-Q1-D07。状态：独立入口与本地3项元数据控制完成；服务器执行尚未开始，没有新诊断结果。

## 目标与固定范围

针对R03“上肢误差下降、走路腿部与位移停滞”，区分当前奖励分项与世界模型预测的证据。沿用SuperMimic的解析奖励、k8拟合/H32策略路线；不改变奖励/学习参数，不重跑已完成训练、动力学评测、D04/D05/D06或张量保存审计。

预先固定：初始/final两组、走路/浅蹲/复用开发出拳全部纳入；奖励按8项拆分、与原保存reward及回报核对。两个对应世界模型各在自己已保存真实轨迹上，用实际动作/初态评估H=1/8/32、stride8；同时计算零残差恒速积分。报告根位置/朝向、线/角速度、关节位置/速度和归一化状态误差，不按结果挑片或时域。

这是访问分布上的描述性诊断：两个模型使用各自策略轨迹，不能称相同状态的模型因果对照；预测准不等于动作梯度正确；奖励分项增量不是消融。出拳始终复用开发数据。若某项失败则保存失败与原日志，不伪造完成。

## 实现与控制

新增`script/supermimic_q1_analysis/diagnose_completed_run.py`，只在独立analysis目录添加。先verify_plan，再检查原流水线终态、匹配评测/审计覆盖及checkpoint/reference/trace身份，核对原admission与新接触资产。复用冻结`analyze_learning.describe_evaluation`和`measure_world_model.main`，后者只在本进程绑定ContactCorrectedAssets，finally恢复原构造器及argv。两个冻结源码目录不编辑。

拒绝覆盖输出；运行前/各阶段/收尾查容器30GiB保护线至少768MiB余量、GPU空闲≥2GiB、温度<86摄氏度、OOM不增加；失败不向其他进程发信号。结尾再次verify_plan。GPU只做已有模型的预测，不进行策略推理、物理积分或optimizer更新。

本地`python3 -m unittest discover -s script/supermimic_q1_analysis -p test_diagnose_completed_run.py -v`原生退出0、3项通过：匹配元数据接受，未verified/错误步数拒绝，被改真实输入身份拒绝。fixture不加载真实模型，不构成科学效果验证。服务器同控制及真实执行仍待运行。

## 版本与执行入口

副本`/tmp/yuanqi-supermimic-q1-20261023`、分支Q1；服务器`huanyu:/opt/data/private/YuanQi_Q1`，Python`/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python`。输入科学冻结发布`d5f9476125f2c3c7ed2d7c1e6ac898e1f69d6429`，本次新增源码由提交历史/运行收据SHA标识。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 $PY -u script/supermimic_q1_analysis/diagnose_completed_run.py --root "$PWD" --experiment data/supermimic_for_q1/R03_B128_extend12000_20261023 --output data/supermimic_for_q1/D07_R03_reward_model_20261024 --device cuda:0
```

计划输出`D07_R03_reward_model_20261024/{world_model_initial,world_model_final,receipt}.json`，日志`logs/supermimic_q1_D07_R03_reward_model_20261024.log`。0新物理/策略/optimizer；实际世界模型预测次数以执行结果为准，不把“无策略推理”写成“无模型前向”。不覆盖失败目录。

## 保存与下一步

先审查diff/payload、发布Q1并独立核对远端，再部署源码并核对SHA、跑服务器控制及本次独立分析，保存结果/失败及解释。模型/轨迹不下载Mac，原件LOCAL_ONLY。V02已发布核对`c990bae577e5789263c9a9b7c1ad49990546610e`，入口18865供看R03；R03可用性目标仍未达，不追加原预算，A3保持停止。
