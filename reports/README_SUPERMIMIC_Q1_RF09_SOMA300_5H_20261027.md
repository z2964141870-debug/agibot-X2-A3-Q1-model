# RF09 300条SOMA与五小时训练

记录日期：2026-10-27，北京时间。状态：02:19:56已启动唯一RF09，训练进行中；不要重复启动。[目标MG04](README_SUPERMIMIC_Q1_MG04_EXPAND_RETARGET_20261027.md)。

## 目标、方法与差异

用户明确要求完成扩库、上肢重定向优化和至少30秒多类型对比视频后，以4096环境尝试一次5小时训练。RF09独立从零/seed0，保留RF08；此次18000秒预算覆盖旧一小时上限，不自动再延长、重试或并行训练。

沿用RF08/固定HP实现：4096 MuJoCo Warp环境、WM1024×10、actor1024/H32、critic5×4、UTD.4、BF16/compile、64步采集预热和500轮WM-only；heading XY/相对朝向10帧历史、完整未来根/关节参考。Q1 22关节、模型armature推导PD15.791367/1.005310、完整URDF effort，动作尺度和奖励/终止原样。师兄实际1024环境；不把此设置说成论文所有条件一致。

RF09主要变化为300训练/30演员分离验证（原100/15）、Q1默认pelvis静止原点归一化，以及更长预算。三项共同变化，不是单变量消融；上肢参考诊断有小幅改善，训练效果未知。详细来源、同源对照与视频清单见[AP02](README_SUPERMIMIC_Q1_AP02_SOMA_RETARGET_20261027.md)。

## 预算与保护

软目标18000秒实际采集/训练，在完整更新后结束；外层18040秒要求退出，45秒宽限，最大18100秒仅容纳更新边界/退出。初始化、准备、profile、等待、绘图、评测不计训练。保留320000最大iteration，但只跑本次五小时时间预算，不以迭代数代替真实时长。

100GiB余盘、12GiB单实验、30GiB共享cgroup、OOM基线0、86度、900秒无进度保护；不动其他项目或任务。SSH失败记UNKNOWN，核boot/PID/start_ticks和命令再判断；断线不重启。所有运行源码、资产、配置、输入和展示清单冻结，文档可更新，禁止热改源码。

## 位置与启动命令

只在自己的25252容器，唯一根`/opt/data/private/ZHY`。Python `data/environments/supermimic_q1_hp_20261026/bin/python`；CUDA/EGL经`script/supermimic_q1_hp/activate.py`。权威Git `/tmp/yuanqi-supermimic-q1-20261023`/Q1，远端按AGENTS。固定上游`310244b73e45775744a75b789eb25a6c9c5a81c9`，本次冻结提交`a553b6a5a3c44167042355ff2f2fbd42ff558342`已push并独立核远端；24发布文件463177字节在权威Git/Mac/25252逐大小SHA一致，[部署收据](../data/manifests/supermimic_q1_RF09_deployment_20261027.json)。后续报告提交不改变冻结来源。

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
  --code-commit a553b6a5a3c44167042355ff2f2fbd42ff558342 --detach
```

命令只执行一次。目录已存在先核状态与真实进程，不重复启动；失败保原终态。

## 已启动与监督

启动2026-10-27 02:19:56，实际训练时钟02:20:12.650051开始；controller40210/start_ticks3633080196、worker40213/start_ticks3633080219，boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`。02:20:47三层status/guard均RUNNING，真实进程身份/命令匹配；plan冻结153个来源，预算18000/18040/18100秒正确。实际GPU物理契约确认4096env、obs80/goal590/state57/context26；实际args确认WM1024×10、actor1024/H32、BF16/compile、seed0。[启动/配置收据](../data/manifests/supermimic_q1_RF09_launch_20261027.json)。

02:22:44已206学习轮、2379776转移、WM2060次、actor0，仍处500轮WM-only正常预热。02:20资源采样GPU9894MiB/43度、cgroup10529509376字节/OOM0、盘余1605449195520字节；这些是时点采样，不当作峰值。预计07:20左右结束训练，之后自动评测/媒体，不把启动或预热算效果改善。

沿用`q1-rf08-4096`原任务ID，已通过工具更新为“Q1 RF09 4096环境五小时训练监督”并ACTIVE，每10分钟；本地结构化回读核MG04/RF09/18000秒/冻结提交/目标聊天均正确。正常进度静默，异常、需要处理与重要最终结果才通知；supermimichp-x2/yuanqi不改。

02:35:46增量核验：同boot/真实两进程start_ticks和命令保持，1050学习轮/11022336转移、WM10500/actor550，实训933.134414秒，已越过预热并真实更新策略。GPU14616MiB/54度、cgroup11189334016字节/OOM0、余盘1604336644096字节、实验约0.416GB；未见保护/有限性异常。[现场收据](../data/manifests/supermimic_q1_RF09_observation_20261027.json)。只证明执行正常，未进行中途效果评测。

## 自动终后与交付

保存initial/final/last、真实训练秒数、转移、WM/actor更新及完整回合统计。训练正常结束后自动初末330片无噪声真实动力学，先保存两组定量JSON/轨迹再生成视频；原RF08固定115参考另做两策略评测（不重复生成基准视频），再按共同存活前缀比较RF08 final与RF09 final。同初态/参考时轴/名称必须一致，不能只对比不同参考库均值。

`evaluation/initial.mp4`与`final.mp4`各覆盖预定8段、6类示例片约54秒，原速，独立动作切换和终止后保持明确标记；并非连续54秒一次rollout。六宫格PNG/PDF/CSV保持细原值/粗500学习iteration均值，并用实际RF09/预算标注。共同前缀、全程位移/路径/速度、几何支撑、肩肘/腕高度与相位分开解释；近地比例不作接触力测量，摔倒位移不作行走成功。

图表、视频、JSON等小产物复制Mac同名data目录并逐大小SHA，权重/大轨迹留服务器LOCAL_ONLY，不建网站。若媒体交付失败，可独立补授权后处理，不重训或把FAILED改为成功。训练和性能改善是不同验收项。

## 预检与恢复

330片转换、12片同源重定向对照、完整4096 P02采集/WM/actor/critic有限性通过，profile worker已reap/返回0。新基准比较入口对同一RF08 final做身份对照，0新积分/推理/优化，初末统计一致；动态六宫格在既有真实RF08记录上生成通过。12片真实动力学接口smoke（训练/验证各6，旧RF08初末权重、0optimizer）已完成，证明新参考库/评测入口可运行，不算RF09性能证据。

RF08保存轨迹的55.24秒多类长视频已交付本地，1381帧/25fps/1280×480/H264，全帧解码退出0、5个分散帧目视非空且策略/参考可见。3文件13795681字节Mac/服务器大小SHA一致，0新物理积分/推理/优化，[交付收据](../data/manifests/supermimic_q1_RF08_multitype_local_delivery_20261027.json)。MP4路径`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026/evaluation/multitype_20261027/final_multitype.mp4`。此旧示例用较早清单；RF09 V02另含更明确的双手抬起动作。RF09五小时训练、真实效果与终后交付仍待完成。

恢复读现场`status.json`、`train_guard.json`、`run/status.json`、`run/training_clock.json`及当前身份；重要阶段写本README/MG04/索引/CURRENT_STATE，审核提交push Q1、独立核远端，记录和小收据镜像Mac/服务器。
