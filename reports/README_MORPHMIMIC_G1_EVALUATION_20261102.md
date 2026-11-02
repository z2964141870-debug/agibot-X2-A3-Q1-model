# MorphMimic G1足量训练检查与评测

2026-11-02用户要求检查此前复现是否完成并测试。本阶段只审计保存内容、导出和动力学评估，不启动、恢复或延长训练。G1单本体基线不等于G1/X2/T2共享跨本体复现，也不等于Q1控制已通过。

## 训练结束证据

北京时间2026-11-02 10:17:07经原生hp_han Windows系统SSH至同一25386，核唯一根/opt/data/private/ZHY、原boot4d859008及GPU-26028785相同。原status COMPLETED、guard WORKER_COMPLETED/worker_reaped=true/return0；原controller362416和worker362421均不存在。GPU15MiB/0%、无计算进程，容器OOM0。原登录提示跨上下文等待后超时，先登记本独立故障一次恢复，Windows会话未断；未重启宿主机、容器或训练。

原worker于北京时间2026-11-02 06:21:57.737完成，controller 06:22:03.117结束；训练开始为11-01 01:55:48.838，worker总墙钟约28小时26分。跑满160000轮、非有限0；WM1600000、motion158000、critic3160000、KP158000、KPcritic3160000次更新。原训练冻结71859ca8abe238477aa18245a31d81204da9197f，183上游blob由原训练前后核验保持。12个编号checkpoint及latest存在，最终274170107B；这一步只有状态/文件大小核查，权重内容审计另记。

[完整现场收据](../data/manifests/morphmimic_faithful_g1_completion_20261102.json)保存原状态、guard、进度、原进程退出和当前身份；不把文件存在或训练曲线当作物理效果。

## 固定评测范围

沿用固定aa40fe46原版deploy.export_onnx、deploy.sim2sim.export_assets和deploy.run；未改上游183个源码。该版本没有独立evaluate脚本，部署CPU MuJoCo自带原训练同定义的高度/倾斜失败、关节误差、身体误差和根漂移。资产导出核编译模型一致和IMU FK；ONNX导出自带随机输入的PyTorch动作一致校验。

审计全部12个保存检查点的迭代/时钟、五Adam步数、网络/归一化和Adam状态有限性；核latest与160000 SHA相同。固定初始0、40000及160000轮：初始测motion，两个已训练快照各测motion和keypoint，共15片独立回放。最终两个接口各三片原速视频，共6份；失败保留真实终止时刻，不换动作或填充成功画面。

| 固定动作 | 帧数/参考时长 | 在真实1h训练子集中 |
| --- | --- | --- |
| walk_forward_loop_001__A022 | 1032 / 20.64秒 | 是 |
| walk_forward_start_002__A023 | 152 / 3.04秒 | 是 |
| walk_forward_stop_005__A026 | 253 / 5.06秒 | 否 |

第三片沿用此前登记的固定停走，不按结果换片。资产导出读取原all.yaml中该片，训练仍是原1h559条。分别报告训练内和训练外，不将三片当完整泛化评测。原summary的joint_err_rms_rad实际先逐帧算关节RMSE再对帧求均值，遵循原字段并注明口径；失败片按已执行帧计算误差，不能与完整片无条件比较。

新入口script/morphmimic_faithful/evaluate_completed.py与g1_evaluate_20261102.json使用独立评测根，原共享GPU锁、24GiB准入/30GiB运行内存、OOM/身份/温度/存储守卫不变。零optimizer，原训练目录不写；权重和NPZ仅服务器。每条命令有停止时限、整阶段4小时上限和10秒回收，不自动重试。旧短训练的300秒物理门/1200秒训练预算不适用。

训练图工具仅新增显式标题/预热/平滑窗口参数以标注1024环境、WM512/imag-KP1024和2000轮预热，原默认烟测图口径保持。独立CPU完成契约反例与语法检查通过后，按显式stage/audit/commit/push/独立40位远端及三端发布快照大小SHA核验再执行。

计划输出：data/supermimic_for_q1/MORPHMIMIC_FAITHFUL01_G1_EVALUATION_20261102。当前只有完成检查与入口准备，尚未执行权重审计/物理回放，尚无步态改善结论；实际启动和结果见后续收据。未调用goal结算或变更其他自动任务，无agent；衣服Q1演示仍搁置、采纳null。

## 入口已发布与内容审计

afb34a030b8164654451a76a6dad82d181722119已独立核远端40位SHA，9记录417205B Git/Mac主根/25386及发布快照逐大小SHA一致；本地和原服务器各5项完成契约测试通过。[入口发布旁收据](../data/manifests/morphmimic_faithful_g1_evaluation_entry_publication_20261102.json)。并发研究新增两提交已正常rebase并保留，未强推或覆盖研究记录。controller542501/start50075060、worker542506已启动独立评测，启动前183upstream匹配。

北京时间10:38:53已完成全部12编号checkpoint内容审计，五Adam计数/网络归一化/Adam状态有限性均通过，latest与160000 SHA相同。10:41:08中间快照40000的motion/keypoint各三片均完整；未训练0轮三个motion片分别在46/43/44帧失败。最终160000物理和视频仍在执行，上述中间结果不代表最终结论。媒体工具仅补完整评测状态/跳过无视频对照/接口标注；不改变原仿真和终止定义。
