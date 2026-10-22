# A3解剖肩点独立候选

记录日期：2026-10-22（北京时间）；2026-10-23补记退出结果。实验：T13。状态：FAILED_PRE_IK，实际退出1。

## 目标与固定范围

根据T12已核验的肩点错位，只修正肩位置/朝向任务的对应关系：位置任务使用`shoulder_roll_Link`原点，朝向任务继续使用`shoulder_yaw_Link`。两任务分别只承担位置、朝向代价，避免给1或2自由度身体强加完整肩朝向。旧代码、历史T10轨迹不改写；腿部映射、真实足几何、接触、限位、求解预算和5cm门保持。

新驱动`script/teleop_research/a3_anatomical_shoulder_probe.py`在构造实例时临时替换BODY_MAP并立即恢复，原文件没有改动。继承原Mink/实际足几何求解及完整质量检查；保存qpos后用独立MuJoCo FK验证，并重算位置误差。任务点变化会改变目标语义，T10两臂同时保留对原yaw目标合同的误差，不能将新旧定义直接包装为同一指标改善。

## 输入、命令与产物

固定8个离线运动学转换：原T10 walk标签/ONNX两臂各30帧，再用T11三段相同源帧`[59,300)`的标签/tanh配对共6臂，各241帧。不重跑人体估计器；完整数据、已存输出和继承源码在启动时重新核对SHA。根和接触均由标签辅助，不代表完整在线遥操。

```bash
cd /opt/data/private/yuanqi_ops
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /opt/data/private/hechengzhu/ZHY/data/env/bin/python -u \
  script/teleop_research/a3_anatomical_shoulder_probe.py \
  --root /opt/data/private/hechengzhu/ZHY \
  --t10 data/experiments/teleop_20261022_T10_paired_a3_kinematic \
  --t11 data/experiments/teleop_20261022_T11_clotho_activation/receipt.json \
  --t12 data/experiments/teleop_20261022_T12_geometry_audit.json \
  --data data/fgp/train_released.zip \
  --output data/experiments/teleop_20261022_T13_anatomical_shoulder
```

终端日志`logs/teleop_20261022_T13_anatomical_shoulder.log`。输出目录只准首次创建；`admission.json`用于恢复识别，只有最终receipt才表示完成。新NPZ/CSV留服务器，`LOCAL_ONLY`；小清单、代码和README提交指定GitHub main。

## 验收与恢复

每臂完整记录IK P95/最大值、关节限位/速度比、地板、接触覆盖/滑动、足底高度、根修正和根跟踪。未过任何原门不称合格，不因肩修正就启动策略训练。无策略加载、动力学步进、硬件/Unity连接或优化器。

实际运行在构造候选任务时退出，错误为`ValueError: Requires coincident rest roll/yaw orientation for this correction`（脚本118行）。A3静止位的roll/yaw朝向并不重合，候选实现的前提错误；8个IK转换均未开始，只有`admission.json`，没有最终receipt。不能把实现保存或admission称为候选通过，也不能据此推断肩修正的效果。

执行脚本SHA-256：`b2abe19e179094fbdf66b41e8567dd4dc19c80ff5335d269e1c16f2a6cc87b11`；失败日志SHA-256：`29906380e9fc894f615878b1e8f3a55950db23dcb28cca3113ef58c90b6c881b`，恢复时在服务器重新核对。执行会话已返回退出1；未遗留该试验的计算进程。0新增IK结果、动力学或训练，035仍为固定基线。

## 关键决定与下一步

保留失败目录和代码，不覆盖重跑、不移除保护条件冒充修复。后续若继续本候选，必须显式处理rest-frame旋转关系，再建立独立试验。用户已另行明确新目标`supermimic_for_a3`，优先阅读论文、审计独立复现并推进A3训练；T13不作为新目标的通过证明或前置任务。

## 保存与同步

README、索引/current与候选源码在指定仓库main保存；实际推送及远端核验以Git历史和发布回执为准。服务器原始admission/日志保持`LOCAL_ONLY`，未声称网盘备份。
