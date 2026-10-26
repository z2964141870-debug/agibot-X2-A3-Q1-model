# RF07 根位姿反馈短对照

记录日期：2026-10-26，北京时间。状态：适配和现场V02验证完成，等待代码发布/部署核验后启动。用户明确要求核查并积极采用可用方法，沿用每轮训练最多一小时，不新增目标、不恢复MG02。

## 目标与单一主要变量

比较RF06的actor缺少当前根误差，和RF07加入当前参考相对pelvis的位置XYZ/朝向6维反馈。启用既有anchor_obs想象分支，同函数用于动力学采集与确定性评测，540维goal变549维，本体历史仍72×10。参考、重定向、奖励、终止/时间约定、控制、seed和从零初始化不变。反馈经原motion encoder的32维latent进入actor，与X2把heading XY及相对朝向放在观测历史的具体路由不同；本轮验证机制，不称完整对齐X2。

## 固定配置与预算

| 项目 | 设置 |
| --- | --- |
| 正式容器/唯一根 | 用户25252，/opt/data/private/ZHY |
| 独立输出 | data/supermimic_for_q1/BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026 |
| 实际启动 | 待现场记录，尚未启动 |
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

V01因测试入口歧义FAILED，0optimizer，失败收据保留；V02六项测试全通过，实际新训练dispatch配置和native/slotted物理等价、H32梯度与实仿目标、存取维度均通过；随后真实评测与算术身份审计通过，0optimizer/无CUDA、33.582713秒。小收据见`data/manifests/supermimic_q1_root_feedback_V0{1,2}_20261026.json`。随机窄网络夹具不充当RF07模型或效果结果。

发布并核验40位提交、部署逐文件大小/SHA和空闲资源后执行一次：

```sh
/opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python \
  /opt/data/private/ZHY/script/supermimic_q1_reproduction/root_feedback/run_single_motion.py \
  --root /opt/data/private/ZHY \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026 \
  --code-commit <published-and-independently-verified-40-character-HEAD> \
  --expected-oom-kill 0 --detach
```

先核输出不存在，RF06 pipeline完成且worker/controller已退出、GPU释放。失败/保护停止保留终态，不自动重试；运行中冻结代码与输入。实际boot/PID/start_ticks/命令和现场status/guard/run进度合并判断，不凭PID或SSH中断重启。

## 结果交付与下一步

训练后核真实时长与保存审计、初末MuJoCo/算术身份、根位移/方向/路径速度、双支撑与换脚、腿/上肢关节误差、肩肘/腕轨迹/抬手高度时序。使用既有对比工具与RF06共同前缀比较；原2×3图PNG/PDF/CSV和初末MP4直接核大小/SHA复制到Mac对应data目录，权重留服务器LOCAL_ONLY，不建网站、不增加不必要图像加工。

当前没有RF07训练效果可报告。训练结束与性能改善分开判断。后续欠缺包括未来目标的相对根轨迹/完整朝向信息、片尾H32起点约定、20ms参考时间差异、GPU后端/实际优化更新吞吐、扩展动作覆盖；不因发现这些差异同时改进本轮。总体双向核查见`README_SUPERMIMICHP_Q1_REVIEW_20261026.md`。
