# RF07 根位姿反馈短对照

记录日期：2026-10-26，北京时间。状态：20:45:21全pipeline COMPLETED，真实训练3058.032232秒，初末动力学评测、保存/身份审计、跨轮比较及本地MP4/曲线交付完成。根反馈产生沿参考方向的移动，但113/365步、2.26秒摔倒，尚未学会完整跟踪。用户随后要求尽量全面对齐师兄方法，见`README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md`；新训练仍最多一小时，不恢复MG02。

## 目标与单一主要变量

比较RF06的actor缺少当前根误差，和RF07加入当前参考相对pelvis的位置XYZ/朝向6维反馈。启用既有anchor_obs想象分支，同函数用于动力学采集与确定性评测，540维goal变549维，本体历史仍72×10。参考、重定向、奖励、终止/时间约定、控制、seed和从零初始化不变。反馈经原motion encoder的32维latent进入actor，与X2把heading XY及相对朝向放在观测历史的具体路由不同；本轮验证机制，不称完整对齐X2。

## 固定配置与预算

| 项目 | 设置 |
| --- | --- |
| 正式容器/唯一根 | 用户25252，/opt/data/private/ZHY |
| 独立输出 | data/supermimic_for_q1/BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026 |
| 实际启动 | 2026-10-26 19:48:55，北京时间 |
| 环境/模型槽/线程 | 4096 / 8 / 8，CPU MuJoCo |
| 世界模型/actor | WM2048×10，actor4096/H32，UTD0.4 |
| 控制 | 12腿50%/其余25%型号限矩，原25%动作尺度，PD50/2 |
| 参考 | train_3419f09ba93c91d75fc1，365帧/50Hz |
| 参考SHA256 | f9bce4c3452720cda88a5adc4111fe2354440dd04e8c8ecb50cbf94d7181b8c3 |
| seed/初始化 | 2026102513 / 随机从零 |
| 训练预算 | 软目标3000秒，真实训练≤3600秒，阶段3500秒退出+55秒回收，完整25轮边界 |
| 资源保护 | 100GiB余盘、12GiB单轮、30GiB共享内存、OOM0、86度、900秒无进度 |

初始化/验证/等待/绘图/评测均不计训练时长。最大12000采集轮仅为上限，不自动扩大预算。RF06实际3132.859650秒/600轮/139次actor；本轮同软目标，仍按实际时长/更新数比较，不能事先宣称等更新量对照或效果改善。

## 版本、入口和验证

新增代码`script/supermimic_q1_reproduction/root_feedback/{feedback,entry_root_feedback,run_single_motion,test_feedback,verify}.py`，复用既有训练/保存/评测/上肢腿部分析/渲染管线。原核心、RF06文件及已完成工件未编辑。新入口用唯一名称，避免旧leg_control/entry.py同名导入。

冻结代码`8ba7f820cffbe0e2f0ece7a3b34ce79d0683f7a6`已push origin Q1并独立核远端。71项源码/小记录服务器SHA一致，15文件164817字节在Mac/服务器逐大小SHA通过。Mac tar附带的15项AppleDouble传输元数据已按本次精确文件名清理，启动前新源码目录无`._*`；未触碰别的文件。后续报告提交不改变冻结代码。

19:50:10现场核验：controller4805/start_ticks3630734136、worker4808/start_ticks3630734152，boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`，GPU UUID`GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4`。status/guard/run均RUNNING，step0，实际manifest的goal_dim549、anchor_obs=true、4096/WM2048/actor4096、腿部控制及3000/3600秒预算/代码身份匹配。worker命令和start_ticks现场一致；GPU2706MiB/34度，OOM0，guard共享约7.87GB/盘余约1.783TB。step0属于初始化/早期采集，不据此推断学习效果。原q1-2048已更新为RF07唯一监督，ACTIVE/每10分钟；旧RF06不重启，每小时仓库跟踪继续。

结构化启动/现场配置/71项SHA与15文件镜像记录见`data/manifests/supermimic_q1_rf07_launch_20261026.json`，状态均是采样时快照，后续监督重新读现场。权威Git报告同步完成后，小记录镜像Mac与25252；运行源保持冻结。

随后现场已推进100采集轮/409600转移，WM0/actor0为正常128轮采集预热尚未结束；记录耗时391.774609秒，GPU37度、共享内存7883939840字节/OOM0。该训练指标不能当作最终动力学效果。

V01因测试入口歧义FAILED，0optimizer，失败收据保留；V02六项测试全通过，实际新训练dispatch配置和native/slotted物理等价、H32梯度与实仿目标、存取维度均通过；随后真实评测与算术身份审计通过，0optimizer/无CUDA、33.582713秒。小收据见`data/manifests/supermimic_q1_root_feedback_V0{1,2}_20261026.json`。随机窄网络夹具不充当RF07模型或效果结果。

已按发布与部署身份核验执行一次，以下仅作复现记录，运行期间禁止重复执行：

```sh
/opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python \
  /opt/data/private/ZHY/script/supermimic_q1_reproduction/root_feedback/run_single_motion.py \
  --root /opt/data/private/ZHY \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026 \
  --code-commit 8ba7f820cffbe0e2f0ece7a3b34ce79d0683f7a6 \
  --expected-oom-kill 0 --detach
```

先核输出不存在，RF06 pipeline完成且worker/controller已退出、GPU释放。失败/保护停止保留终态，不自动重试；运行中冻结代码与输入。实际boot/PID/start_ticks/命令和现场status/guard/run进度合并判断，不凭PID或SSH中断重启。

## 结果交付与下一步

训练后核真实时长与保存审计、初末MuJoCo/算术身份、根位移/方向/路径速度、双支撑与换脚、腿/上肢关节误差、肩肘/腕轨迹/抬手高度时序。使用既有对比工具与RF06共同前缀比较；原2×3图PNG/PDF/CSV和初末MP4直接核大小/SHA复制到Mac对应data目录，权重留服务器LOCAL_ONLY，不建网站、不增加不必要图像加工。

## 终后实际结果

20:40:08训练worker正常退出并reap，575轮/2355200转移、WM179组/actor129次；真实训练3058.032232秒，满足3000至3600秒及完整25轮边界。20:45:21全部10阶段退出0，13项保存检查和两份算术/身份审计通过。20:48:07 controller/worker均不存在，GPU14MiB/26度、共享内存5132034048字节、OOM0、盘余1765355159552字节，未延长或重启。

| 指标 | RF06最终 | RF07最终 |
| --- | --- | --- |
| 真实训练秒数 / actor更新 | 3132.859650 / 139 | 3058.032232 / 129 |
| 完成控制步 / 跌倒 | 365/365 / 否 | 113/365 / 是 |
| 根净移 | 0.014933m | 0.508650m |
| 沿整段参考方向位移 | -0.014915m | +0.508302m |
| 路径长度 | 0.158446m | 0.605795m |
| 双支撑边界比例 | 98.904% | 94.690% |
| 全段归一化回报 | 0.437431 | 0.148666 |
| 共同前113步腿RMSE | 0.269550rad | 0.254372rad |
| 共同前113步上肢RMSE | 0.715981rad | 0.687486rad |
| 共同前113步根XY平均误差 | 0.109869m | 0.031088m |

RF07执行前缀的参考位移0.475148m，最终根高0.153956m，0.508650m净移包含失稳和摔倒，不能称稳定行走。左单支撑0帧、右单支撑3帧、无支撑3帧、接触状态变化5次；腿部边界限矩饱和0.07375%，没有证据将本轮失败单独归于力矩截断。上肢误差共同前缀略低，但左右完整抬手主峰在233/236帧，本轮提前结束未到达，腕相对躯干误差0.119483/0.116551m，尚不能称完整手势改善。

本轮从初始365步不倒/净移2.28cm转为有方向的移动后倒地，支持“新增根反馈会改变移动响应”，不证明唯一根因，也不证明根反馈方案已足够。RF06/RF07时长与更新数略有不同，不冒称等更新量消融。跨轮比较调用`script/project_tools/compare_q1_physical_trials.py --root /opt/data/private/ZHY --trial RF06 <RF06根> --trial RF07 <RF07根> --output /opt/data/private/ZHY/data/supermimic_for_q1/RF06_RF07_COMPARISON_20261026.json`，同参考/共同初态/协议核验通过，0新积分/推理/优化器。

原PNG/PDF/CSV、初末MP4和六张代表帧共11文件1539139字节，已复制Mac对应实验目录并逐大小/SHA一致，训练图已目视检查；收据`data/manifests/supermimic_q1_RF07_local_delivery_20261026.json`。最终视频`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026/render_final/comparison.mp4`，训练图在同根`run_dashboard/training_dashboard.png`。模型和大轨迹留25252，LOCAL_ONLY，不建网站。

用户最新指示要求后续尽量对齐师兄的观测历史根反馈、完整未来参考、逐关节PD/full effort、100条SOMA和GPU后端。RF07已封存；这是多项方法适配的下一独立阶段，不将多项同时改变解释成单一变量因果。21:37用户提供canonical骨架23062字节/SHA匹配并已接收25252统一inbox，元数据已在，SOMA人体主包由用户上传；师兄实际100/15清单用于精确选片，非代理追加的数据门。下一阶段见`README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md`，总体双向核查见`README_SUPERMIMICHP_Q1_REVIEW_20261026.md`。21:43尝试更新旧训练监控q1-2048，工具返回Automation does not exist，未成功修改或新建任务；不将历史ACTIVE描述当当前事实。
