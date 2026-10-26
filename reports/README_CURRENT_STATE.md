# YUANQI 当前状态与下一步

2026-10-27最新：[MG04](README_SUPERMIMIC_Q1_MG04_EXPAND_RETARGET_20261027.md)active，[RF09](README_SUPERMIMIC_Q1_RF09_SOMA300_5H_20261027.md)02:19:56已实际启动，真实训练时钟02:20:12.650051。本次用户明确授权4096环境五小时18000秒，覆盖本次旧一小时限制；从零、不启用子agent/续训或自动重试。controller40210/start_ticks3633080196、worker40213/start_ticks3633080219，boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`。现场三status/身份/4096契约/预算已核，02:22:44已206学习轮/2379776转移/WM2060/actor0，500轮WM-only正常，无新效果结论。

[AP02](README_SUPERMIMIC_Q1_AP02_SOMA_RETARGET_20261027.md)已330片转换、300训练/30验证、82/11演员无交集，保留RF08原115源。确认Q1默认pelvis0.419m在静止骨架偏移中被重复叠加，归一化后同源D02腕误差3.52→3.12cm、峰高4.11→3.04cm；参考改善不代表策略改善，不断言X2也有同一问题。4096完整P02已通过且回收，不重复profile/转换/解包/hash45GB原包。RF09 V02预选8段6类约53.83秒，含普通前向走与明确抬手；数据、重定向和预算共同变化，不冒称单变量消融。

## 运行与监督

原10分钟任务`q1-rf08-4096`同ID已改RF09五小时并通过工具/结构化回读确认ACTIVE，正常进度静默。预计07:20左右训练结束，之后自动330片初末动力学、六宫格、原RF08固定115参考基准与共同前缀比较、初末长MP4；终后评测/交付均未完成。恢复读取现场`status.json`、`train_guard.json`、`run/status.json`、`run/training_clock.json`和boot/PID/start_ticks/命令/mtime，不能只凭旧RUNNING或PID，SSH断线不重启。

02:35:46[增量现场](../data/manifests/supermimic_q1_RF09_observation_20261027.json)：1050学习轮/11022336转移、WM10500/actor550、实训933.134414秒，已越过预热；两进程身份与三状态正常，GPU14616MiB/54度、cgroup11189334016字节/OOM0、余盘1604336644096字节、实验约0.416GB。执行正常不代表效果已改善。

软训练18000秒、外层18040秒+45秒退出宽限、最大18100秒只容纳更新边界/退出。保留100GiB余盘/12GiB实验/30GiB cgroup/OOM基线0/86度/900秒无进展。02:20:47时点采样GPU9894MiB/43度、cgroup10529509376字节、OOM0、余盘1605449195520字节，非峰值。共享盘变化不等于本项目新增量，不动他人文件/任务。正常训练receipt需TRAINING_COMPLETED且真实秒数满足、guard正常reap；全pipeline COMPLETED、资源释放和本地交付再验MG04。

## 版本与连接

权威Git `/tmp/yuanqi-supermimic-q1-20261023`，分支Q1，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。RF09冻结提交`a553b6a5a3c44167042355ff2f2fbd42ff558342`已push/独立核远端，24文件463177字节三端大小SHA通过，[部署](../data/manifests/supermimic_q1_RF09_deployment_20261027.json)/[启动](../data/manifests/supermimic_q1_RF09_launch_20261027.json)。plan锁153来源，包括所有`script/supermimic_q1_hp/*.py`、`script/project_tools/analyze_q1_hp_trial.py`、runtime/配置/资产/参考/视频清单；全pipeline结束前禁止热改。后续报告提交不改变冻结身份。

正式计算仅用户25252，唯一根`/opt/data/private/ZHY`、data/logs/script/reports。SSH：`ssh -S /tmp/yuanqi-q1-25252.sock -T -p 18869 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25252' root@127.0.0.1`。保留Windows桥，不读写认证、不操作师兄容器；失败UNKNOWN，一次有界排查。

Python `data/environments/supermimic_q1_hp_20261026/bin/python`，CUDA/EGL经`script/supermimic_q1_hp/activate.py`。runtime `data/references/supermimichp/AP01_runtime_20261026`、固定上游`310244b73e45775744a75b789eb25a6c9c5a81c9`。RF09库`data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json`、SHA `ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337`；实验根`data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027`。

4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4、BF16/compile、seed0，heading XY/朝向历史及完整未来根/关节参考。Q1 full effort、armature=.004推导PD15.7914/1.00531及原动作尺度/奖励/终止。师兄实际1024环境，Q1数值依自身模型，不是厂家逐关节特调；不称全部条件等同。

## RF08基线与媒体

[RF08最终结果](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)、[MG03完成审计](../data/manifests/supermimic_q1_MG03_completion_audit_20261027.json)、[运行记录](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)。RF08全pipeline00:27:46 COMPLETED、进程退出、MG03已完成；旧RF05/06/07均封存，不恢复旧命令。

RF08实际3500.923046秒/3510学习轮/36204544转移、WM35100/actor3010。100训练片提前终止100→10、15验证片15→0；验证walking3/3完成，沿目标均移1.1234m/参考2.3943m。共同约1秒前缀上肢RMSE训练0.560→0.868rad、验证0.517→0.861rad，原地gesture净漂移1.162m/参考0.023m。存活/移动改善不等于全身跟踪合格。原AP01固定115参考保留供RF09同协议基准，不能直接把两套库均值当因果证据。

原六宫格/短MP4的25项本地大小SHA交付已完成，[清单](../data/manifests/supermimic_q1_RF08_local_delivery_20261027.json)。Mac RF08根`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026`；图`dashboard/training_dashboard.png`及PDF/CSV。新增长对比`evaluation/multitype_20261027/final_multitype.mp4`55.24秒/1381帧/25fps/1280×480，8段6类独立原速片段，左策略动力学右运动学参考。3文件13795681字节大小SHA、全帧解码/5帧目视通过，[收据](../data/manifests/supermimic_q1_RF08_multitype_local_delivery_20261027.json)。这是旧保存轨迹，0新积分/推理/优化，不当RF09效果或连续55秒rollout。

## 交付与恢复要求

RF09终后先报告量化效果，再交付已有小图/JSON/CSV/初末长MP4至Mac对应目录并逐大小SHA；权重/大轨迹只留服务器LOCAL_ONLY。六宫格为原值与500学习iteration均值；MP4各至少30秒、原速且明确切片/失败，不建网站、不为图像加工拖延结论。近地几何不当接触力，跌倒位移不当行走，参考与策略误差分开，不加数据/性能硬门。

重要阶段更新README/索引/本摘要、小收据，显式暂存/audit/push Q1/独立核远端并镜像Mac/25252。只有MG04 A至E实证和交付齐备才complete；失败保原终态、不自动重训。RF09交付完成后暂停本训练监督，独立supermimichp-x2/yuanqi保留。无关`script/supermimic_q1_sources/download_bones.py`不执行/修改/暂存/删除。
