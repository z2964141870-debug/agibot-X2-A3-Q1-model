# PHUMA配对评测图表

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-V01。状态：绘图入口实现、语法及手算数值控制、服务器原生加载通过；真实18片绘图未执行。原PHUMA R01仍训练，冻结源码/输入/3000轮预算不改。

## 目标与范围

将原流水线的18片初始/最终完整动力学评测转成可检查的关节/根部误差和真实移动轨迹。绘图直接读取已保存数组，不加载checkpoint、策略或仿真模型，不做新物理积分、策略调用或optimizer更新。继续世界模型方法和PHUMA-only输入；不引入旧A3动作或权重。

新增入口`script/supermimic_q1_analysis/plot_phuma_evaluation.py`位于121冻结文件之外。它要求原controller终态为完整流水线完成、comparison与spec身份相符、18片与train/development/test名单完全一致；重核原评测/审计四份小证据和逐片参考/轨迹大小/SHA。根部和关节误差另外从保存状态用向量范数重算，与比较结果核对后绘图。它是图表及局部数值检查，不替代原保存审计或完整动力学评测。

## 图表与边界

`paired_errors.png`覆盖全部18片的初始/最终全关节、根部、回报以及腿/腰/臂误差，保留真实划分和短标签到完整clip的清单。标题明示提前失败数及误差仅覆盖已执行前缀；回报继续使用完整参考分母。关节0.15rad/根部0.10m虚线沿用本轮预登记门，不按结果修改。

`locomotion_paths.png`纳入全部合格移动片，当前为train2/development1/test0；逐片比较参考、初始策略和最终策略XY轨迹及随参考时间的根部误差。提前失败轨迹仍只显示实际执行长度，时间轴保留完整参考跨度。没有移动test，不据此声称独立移动测试泛化。图表不证明真机能力，训练是否有效仍依据完整结果。

## 已执行控制

源码AST语法检查通过。Mac默认和内置Python均未发现matplotlib，本次没有安装依赖，也没有声称本地绘图已通过。提取纯数值函数并用现有NumPy执行手算控制：两帧根误差0.3/0.4m得到范数RMSE`sqrt(0.125)`；22关节中单项0.22/0.44rad得到全关节RMSE`sqrt(0.0055)`。两执行帧/三参考帧仅计算实际前缀的误差，完整分母独立保留；空状态、超长状态、错误状态维度、错误参考维度及NaN五种输入均拒绝。控制源码和原生结果留在logs，不把这些控制当作真实18片绘图或视觉验收。

14:10:15只读观察原controller799647/worker799650均为原身份且存活，2300轮/36800控制转移，GPU39摄氏度、cgroup30,854,037,504字节、OOM14。观察读取/proc和原状态文件，没有启动新训练或并行模型诊断。

## 版本与复现

训练版本仍`ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`，本图表入口先单独审核/发布Q1后再部署。完整提交身份由Git历史及独立远端核验提供。服务器已有环境沿用，不增加依赖安装；须等原训练/审计/评测/比较流水线终态后再执行：

实际图表代码发布`801d901c9fc227b3340a107d82fae7838316bcf9`已推送并独立核验`refs/heads/Q1`。服务器部署SHA`4968dc042dacac244754b214ad5ed423393d9717d962f623411c20ff949c5771`与本地发布一致；服务器原环境运行`--help`原生退出0，证明NumPy/matplotlib及CLI可加载，0新物理/策略/optimizer。该检查没有加载checkpoint、仿真模型或真实评测数组，也未创建图表输出。[原生加载记录](../logs/session_records/supermimic_q1_phuma_V01_native_help_20261024.log)留档。

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
  script/supermimic_q1_analysis/plot_phuma_evaluation.py \
  --experiment /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024 \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_V01_paired_plots_20261024
```

输出目录排他创建，若失败保留部分输出与日志，不能覆盖重跑。预定stdout/stderr为`logs/PHUMA_V01_paired_plots_20261024.log`；真实完成后收据保存输入身份、36组原始数组重算、全部移动名单和PNG大小/SHA。图像/原始数组留服务器data，Mac仅取小图表/JSON；大型原始数据与权重仍LOCAL_ONLY，不冒称网盘备份。

## 下一步

等待原3000轮及18片初始/最终完整配对流水线，再核验controller原生事件与结果。执行本入口后目视两图、核对PNG及receipt身份，保存真实结果后更新本文/两入口并审计推送Q1。D02C仍等原流水线结束及新资源核验后执行；不在训练期间重复D02B。目标仍未完成，尚无新跟随提升结论。
