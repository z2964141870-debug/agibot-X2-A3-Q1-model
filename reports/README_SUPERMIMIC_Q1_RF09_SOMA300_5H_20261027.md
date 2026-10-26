# RF09 300条SOMA与五小时训练

记录日期：2026-10-27，北京时间。状态：准备完成，待发布/部署并启动；不要据此重复运行旧命令。[目标MG04](README_SUPERMIMIC_Q1_MG04_EXPAND_RETARGET_20261027.md)。

## 目标、方法与差异

用户明确要求完成扩库、上肢重定向优化和至少30秒多类型对比视频后，以4096环境尝试一次5小时训练。RF09独立从零/seed0，保留RF08；此次18000秒预算覆盖旧一小时上限，不自动再延长、重试或并行训练。

沿用RF08/固定HP实现：4096 MuJoCo Warp环境、WM1024×10、actor1024/H32、critic5×4、UTD.4、BF16/compile、64步采集预热和500轮WM-only；heading XY/相对朝向10帧历史、完整未来根/关节参考。Q1 22关节、模型armature推导PD15.791367/1.005310、完整URDF effort，动作尺度和奖励/终止原样。师兄实际1024环境；不把此设置说成论文所有条件一致。

RF09主要变化为300训练/30演员分离验证（原100/15）、Q1默认pelvis静止原点归一化，以及更长预算。三项共同变化，不是单变量消融；上肢参考诊断有小幅改善，训练效果未知。详细来源、同源对照与视频清单见[AP02](README_SUPERMIMIC_Q1_AP02_SOMA_RETARGET_20261027.md)。

## 预算与保护

软目标18000秒实际采集/训练，在完整更新后结束；外层18040秒要求退出，45秒宽限，最大18100秒仅容纳更新边界/退出。初始化、准备、profile、等待、绘图、评测不计训练。保留320000最大iteration，但只跑本次五小时时间预算，不以迭代数代替真实时长。

100GiB余盘、12GiB单实验、30GiB共享cgroup、OOM基线0、86度、900秒无进度保护；不动其他项目或任务。SSH失败记UNKNOWN，核boot/PID/start_ticks和命令再判断；断线不重启。所有运行源码、资产、配置、输入和展示清单冻结，文档可更新，禁止热改源码。

## 位置与启动命令

只在自己的25252容器，唯一根`/opt/data/private/ZHY`。Python `data/environments/supermimic_q1_hp_20261026/bin/python`；CUDA/EGL经`script/supermimic_q1_hp/activate.py`。权威Git `/tmp/yuanqi-supermimic-q1-20261023`/Q1，远端按AGENTS。固定上游`310244b73e45775744a75b789eb25a6c9c5a81c9`，本次冻结提交待发布后登记，后续报告提交不改变冻结来源。

实验根`data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027`。库`data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json`，SHA `ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337`。runtime仍`data/references/supermimichp/AP01_runtime_20261026`，不用profile权重。

```sh
/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python -u \
  /opt/data/private/ZHY/script/supermimic_q1_hp/run_trial.py \
  --mode train \
  --runtime /opt/data/private/ZHY/data/references/supermimichp/AP01_runtime_20261026 \
  --library /opt/data/private/ZHY/data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027 \
  --training-seconds 18000 --maximum-training-seconds 18100 \
  --external-training-stop-seconds 18040 --expected-oom-kill 0 \
  --video-suite /opt/data/private/ZHY/data/supermimic_for_q1/AP02_SOMA_Q1_20261027/demonstration_suite_V02.json \
  --benchmark-library /opt/data/private/ZHY/data/supermimic_for_q1/AP01_SOMA_Q1_20261026/q1_candidates/pilot_manifest.json \
  --baseline-evaluation /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026/evaluation \
  --trial-note 'MG04: expanded SOMA300, pelvis-relative rest calibration, user-authorized five-hour trial' \
  --code-commit <已发布并独立核远端的40位SHA> --detach
```

命令只执行一次。目录已存在先核状态与真实进程，不重复启动；失败保原终态。

## 自动终后与交付

保存initial/final/last、真实训练秒数、转移、WM/actor更新及完整回合统计。训练正常结束后自动初末330片无噪声真实动力学，先保存两组定量JSON/轨迹再生成视频；原RF08固定115参考另做两策略评测（不重复生成基准视频），再按共同存活前缀比较RF08 final与RF09 final。同初态/参考时轴/名称必须一致，不能只对比不同参考库均值。

`evaluation/initial.mp4`与`final.mp4`各覆盖预定8类示例片约54秒，原速，独立动作切换和终止后保持明确标记；并非连续54秒一次rollout。六宫格PNG/PDF/CSV保持细原值/粗500学习iteration均值，并用实际RF09/预算标注。共同前缀、全程位移/路径/速度、几何支撑、肩肘/腕高度与相位分开解释；近地比例不作接触力测量，摔倒位移不作行走成功。

图表、视频、JSON等小产物复制Mac同名data目录并逐大小SHA，权重/大轨迹留服务器LOCAL_ONLY，不建网站。若媒体交付失败，可独立补授权后处理，不重训或把FAILED改为成功。训练和性能改善是不同验收项。

## 预检与恢复

330片转换、12片同源重定向对照、完整4096 P02采集/WM/actor/critic有限性通过，profile worker已reap/返回0。新基准比较入口对同一RF08 final做身份对照，0新积分/推理/优化，初末统计一致；动态六宫格在既有真实RF08记录上生成通过。12片真实动力学接口smoke（训练/验证各6，旧RF08初末权重、0optimizer）已完成，证明新参考库/评测入口可运行，不算RF09性能证据。旧保存轨迹多类长视频仍在渲染，不将其算本次训练。

恢复读现场`status.json`、`train_guard.json`、`run/status.json`、`run/training_clock.json`及当前身份；重要阶段写本README/MG04/索引/CURRENT_STATE，审核提交push Q1、独立核远端，记录和小收据镜像Mac/服务器。
