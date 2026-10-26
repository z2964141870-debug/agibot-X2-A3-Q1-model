# AP02 SOMA扩库与Q1重定向修正

记录日期：2026-10-27，北京时间。状态：数据、短对照和4096实测完成。服务[MG04](README_SUPERMIMIC_Q1_MG04_EXPAND_RETARGET_20261027.md)，尚无RF09训练效果结论。

## 数据与来源

原包仍为已核45,188,088,297字节的`data/references/bones_seed/inbox/soma_uniform.tar.gz`，canonical和元数据沿用AP01。没有重复全量hash/解包或引入人体模型下载。沿用固定HP提交`310244b73e45775744a75b789eb25a6c9c5a81c9`的无道具平地元数据过滤、演员hash划分、每演员最多4片；没有增加几何、IK残差或性能硬门。

保留原181个BVH候选，再从原归档流式提取149片，共330片全部转换可读、有限的Q1参考，没有处理错误或类别缺额。原RF08的100训练/15验证源身份全部保留。

|类别|新训练|新验证|
|---|---:|---:|
|walking|90|6|
|jogging|50|4|
|turning|40|4|
|gesture|60|8|
|dancing|45|6|
|jumping|15|2|
|合计|300|30|

训练82演员、验证11演员，交集为空。原速参考时长训练1555.2秒（25.92分钟）、验证159.333秒；源120Hz按既有加载器取30Hz，MotionLib再重采样50Hz，未减速或裁速。根/关节布局仍29列，策略/世界模型维度未变。数据增加同时伴随重定向/训练时长改变，不能由RF09单轮把改善全归因于数量。

服务器根`data/supermimic_for_q1/AP02_SOMA_Q1_20261027`；`pilot/selection.json`、`pilot/quotas.json`、`q1_candidates/retarget_report.json`、`q1_candidates/pilot_manifest.json`。Git小清单见`data/manifests/supermimic_q1_AP02_soma_library_20261027.json`。原BVH/CSV留服务器LOCAL_ONLY。

## 确认的问题与修正

Q1场景`qpos0`的pelvis为`[0,0,0.419]`。HP Retargeter在MjData默认姿态收集世界坐标`self.rest`，构建hip/torso/arm目标时又把这些量直接加在动画pelvis上；这是默认模型根不在零点时不成立的隐含前提。在Q1上出现0.419m重复偏移，后续足底投影又统一平移目标，导致pelvis目标与肢体几何冲突。本项目先前Q1适配没有处理该条件，应由本项目修正；不能仅凭这一点认定师兄实际X2资产也受影响。

`retarget_soma.py --root-relative-rest`仅在Mink求解器初始化后把静止位置减去默认pelvis原点。保留Mink求解、姿态/关节余量项、接触锁定和足底投影、迭代数及速度等既有方法，不换奖励或控制公式。默认不加该flag仍保留原转换行为；新库明确启用且元数据记录移除的原点。当前采用已验证的适配修正；没有声称集成/验证了新的学习式重定向模型，GMR外网入口本轮未成功读取。

## 同源短对照

12片覆盖6类，各2片，分别原转换和pelvis-relative转换，0物理积分、0策略训练。D01保存两组Q1轨迹，但首版腕部指标各自使用不同的pelvis目标，口径不适合比较；保留原件，D02复用其轨迹并以同一个修正后人体相对目标重算，不重复IK。后续只引用D02的上肢对照。

|指标|继承转换|修正后|
|---|---:|---:|
|逐片pelvis目标残差p95均值|0.401263m|0.067096m|
|共同人体相对目标下腕位置RMSE均值|0.035195m|0.031160m|
|腕相对根峰值高度绝对误差均值|0.041148m|0.030390m|

pelvis残差各自对其求解目标，腕指标使用相同的身体相对目标。不是“手臂误差降低40厘米”，也不代表策略跟踪误差已经改善；部分困难舞蹈/转身残差仍大。新增参考是否更易控制，须由RF09真实动力学评测判断。D02完整收据`data/manifests/supermimic_q1_AP02_retarget_D02_20261027.json`。

## 视频与验证

新`video_suite.py`在训练前只按参考选择展示：2条普通前向走（不同演员）、明确双手抬起、原地gesture、转身、慢跑、舞蹈、跳跃，共8片约53.833秒。展示规则仅选视频，不过滤训练数据；不查看学后效果。清单为`demonstration_suite_V02.json`及Git同名前缀小收据，V01保留为早期候选。所有片段原速、明确跨片切换/训练或验证身份，左物理策略右运动学参考；终止后冻结最后姿态并显式标记，不伪造连续30秒rollout。

旧RF08保存轨迹正在独立生成多类长视频，用于验证新输出入口，0新积分/推理/优化；不改原5.88秒MP4。六宫格改按实际阶段号与预算取标题，避免RF09误标RF08/max1h。

P01实测因全局日志basename和旧AP01实测重名，在worker创建前被单次运行保护拒绝，0训练/更新，失败原件保留。新独立`AP02_E4096_PROFILE_P02_20261027`测试330片/4096env、WM1024×10、actor1024/H32/critic、BF16/compile。当前尚未完成，不能用低GPU利用率或已采样当通过。过程中只对已核SOMA原包和AP02源BVH建议释放干净文件缓存，不删除数据、不全局drop_caches、不改变冻结内容；建议不保证回收字节数。

01:54:57 P02实际PROFILE_COMPLETED，guard WORKER_COMPLETED、worker38868/start_ticks3632915869返回0并reap，controller38865/start_ticks3632915848退出。64步262144转移、10次WM、1次actor及critic更新均有限；Torch峰值分配6310380032/保留6834618368字节，不含Warp。运行整卡曾观测12332MiB，不称采样值为真实峰值；结束GPU14MiB/OOM0。仅一次完整实测有效，不复用其权重。缓存建议331文件无错误，cgroup采样22355963904→9080844288字节，属干净缓存回收而非删数据，随工作负载会变化。

`render_probe.json`实际确认llvmpipe软件EGL；关闭MSAA/阴影/反射后10帧1.325秒，完整640×480 Q1图像已目视检查且像素非空。渲染只读保存姿态做FK，不改物理/策略/参考；不下载GPU图形驱动或新建网站。初末定量结果先于长视频保存，固定115参考基准不重复渲染。

## 命令与后续

Python为`data/environments/supermimic_q1_hp_20261026/bin/python`。扩库`script/supermimic_q1_hp/expand_soma.py --runtime <AP01_runtime> --archive <原包> --metadata <原元数据> --previous <AP01/pilot> --output <AP02/pilot>`；转换使用`retarget_soma.py --runtime <AP01_runtime> --pilot <AP02/pilot> --canonical <原canonical> --gmr data/references/GMR_SOMA_AP01_20261026 --output <AP02/q1_candidates> --workers 4 --iterations 4 --root-relative-rest --quotas <AP02/pilot/quotas.json>`。

每次输出单次使用；恢复不重复转换、解包或哈原包。P02真实终态/回收已核，随后发布冻结代码并部署SHA核验，再启动[RF09五小时](README_SUPERMIMIC_Q1_RF09_SOMA300_5H_20261027.md)。手工语法、共同前缀算术self-test、真实RF08同final身份比较及动态六宫格入口通过；新库12片初末真实GPU评测接口也已完成，0optimizer、使用RF08权重，不当作RF09效果。完整新终后链路仍待真实RF09结束，不冒称已完成五小时实验。
