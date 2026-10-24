# PHUMA配对评测图表

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-V01。状态：真实18片配对绘图原生退出0；36组根部/关节误差从保存数组重算一致，两图已目视核验，取回文件大小/SHA与服务器一致。原PHUMA R01全部完成，冻结源码/输入/3000轮预算不改。

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

训练版本仍`ed4217384e7abfc8a9cea0b23b95592d4a3e19b0`，本图表入口单独审核/发布Q1后部署。完整提交身份由Git历史及独立远端核验提供。服务器已有环境沿用，没有增加依赖安装；原训练/审计/评测/比较流水线完成后已执行以下命令：

实际图表代码发布`801d901c9fc227b3340a107d82fae7838316bcf9`已推送并独立核验`refs/heads/Q1`。服务器部署SHA`4968dc042dacac244754b214ad5ed423393d9717d962f623411c20ff949c5771`与本地发布一致；服务器原环境运行`--help`原生退出0，证明NumPy/matplotlib及CLI可加载，0新物理/策略/optimizer。该检查没有加载checkpoint、仿真模型或真实评测数组，也未创建图表输出。[原生加载记录](../logs/session_records/supermimic_q1_phuma_V01_native_help_20261024.log)留档。

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PY" \
  script/supermimic_q1_analysis/plot_phuma_evaluation.py \
  --experiment /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R01_B128_3000_20261024 \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_V01_paired_plots_20261024
```

输出目录排他创建，不能覆盖重跑。stdout/stderr为`logs/PHUMA_V01_paired_plots_20261024.log`；收据保存输入身份、36组原始数组重算、全部移动名单和PNG大小/SHA。图像/原始数组留服务器data，Mac仅取小图表/JSON；大型原始数据与权重仍LOCAL_ONLY，不冒称网盘备份。

## 真实结果与核验

原生状态为`PHUMA_MATCHED_EVALUATION_PLOTS_COMPLETED`、退出0；读取全部18片初始/最终保存状态，36组关节及根部RMSE与原comparison一致。输出目录为服务器`data/supermimic_for_q1/PHUMA_V01_paired_plots_20261024/`，没有新增物理积分、策略调用或optimizer更新。

| 产物 | 字节 | SHA-256 |
| --- | ---: | --- |
| receipt.json | 30339 | 20f65237e74a578f2cda446b8579c8586267ad07220a07f32e65fded3b741421 |
| paired_errors.png | 192938 | a7dbf905a0935fc18f970ab0582574280f7be566075574c7b37853b61cbc3712 |
| locomotion_paths.png | 312488 | 32ac4073c8e54443b96a64f841399166cba2fedf94bd9bafdd57c325a807cf5c |
| stdout/stderr日志 | 504 | f6b8d9c14bfad539ee0271ff0caca98c89ba5e11a41f275ff548150583ece745 |

[小收据](../data/manifests/supermimic_q1_phuma_V01_plots_20261024.json)与[日志](../logs/session_records/supermimic_q1_phuma_V01_plots_20261024.log)已取回，大小/SHA与服务器独立检查一致。Mac小图在`/Users/yu/Documents/ChatGPT/元启/data/visualizations/supermimic_q1_phuma_R01_20261024/`。已实际查看2560x1440的误差图和1920x1776的移动图：两图非空，18片标签、图例、三段轨迹和提前失败标记清楚，没有明显遮挡。

图表支持的结论仍为：最终16/18完整、0/18通过跟随门；一段训练移动38/184帧提前失败，另一段训练移动实际/参考最大水平位移0.100590/0.288144m；开发移动为0.066904/0.373315m，轨迹方向也与参考不一致。开发移动根RMSE初始0.262304、最终0.263104m，未改善。test三片均非移动且均完整，关节RMSE均值0.272425降至0.240971rad；这是本小样本的姿态改善，不能证明移动泛化。提前失败片只计算执行前缀，不把跨前缀误差均值变化认作公平整片改善；图像本身不定位坐标、奖励或模型误差的因果来源。

本次恢复时补查仍为原boot，worker799650不存在、controller799647为Z/RSS0，GPU38摄氏度、使用6348MiB/利用率0%；没有恢复原训练。该快照只说明检查时状态。

## 下一步

本阶段完成，记录与小证据按Q1审计提交/显式推送并独立核验远端。下一重新核验资源，执行独立D02C八训练片足底诊断，再以train/development证据选择下一项独立试验。原完成流水线和停止实验不重跑；当前尚无新训练，目标仍未完成。
