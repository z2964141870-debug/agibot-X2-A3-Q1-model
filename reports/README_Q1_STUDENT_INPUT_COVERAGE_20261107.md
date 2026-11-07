# Q1 因果学生输入与来源覆盖审计

记录日期：2026-11-07。实验：INPUT_AUDIT01。当前为实现和本地合同检查完成，真实服务器数据审计待执行。

## 目标与依据

低延迟目标仍为减少 H10/200ms 真实人体未来帧依赖，优先复用 32k 教师、冻结 decoder 和 WM。此前聚合、下一状态、教师 latent 和 FP32 物理蒸馏均未选出优于原 R2 的学生。不能继续以相同数据增加更新量代替失败定位。

原采集源码每个来源组只取一片：train 12 个来源组/12 片，validation 6 组/6 片，每片最多 256 步。完整来源划分实际有 train 22 组/167 片、validation 6 组/98 片。覆盖有限是待检验假设，尚不能认定是闭环失败原因。原教师已见全部 393 片，来源划分只约束本轮学生拟合，不能当新教师泛化证明。

## 本次范围

新增独立 `student_input_audit.py`、固定计划和五项检查，不修改原 184 执行文件、200 冻结来源或原 773 保护输入。不加载 checkpoint、不调用 Torch、不读取原始人体或 test NPZ，不重新计算既有动作残差。

只读 R1/AGGREGATE01 已保存的实际物理状态、教师动作、f0/history 和 trace，核对逐行状态/动作身份、完整来源划分、终止边界及历史 valid/exact/age。按原包事件和 120→50Hz 支持时钟重建历史可见性；这核实保存标志和合成时钟，不重新认证原始几何或真人时钟。

使用原 R2 train-only 统计，报告按来源/几何块的 ±10 截断率和超原训练范围比例，明确保留启动前 200ms 与首秒样本数。复用既有逐来源动作残差，仅读不重新拟合。

有限近邻诊断只用实际规范化后的 f0/history 输入块，各按 RMS 等权；训练查询排除整来源组，validation 查询只查 train 库。最多每训练片/原点 32 个参考、每查询片/原点 8 个查询。距离和标签差仅作有限样本覆盖代理，不能证明因果或不可约误差，也不是学生策略得分。

## 资源与版本

固定输出：`/opt/data/private/ZHY/data/supermimic_for_q1/Q1_LOW_LATENCY_CKPT32000_INPUT_AUDIT01_20261107`。外层 300 秒、进程地址空间 2GiB、观测 RSS 上限 1GiB、输出 64MiB，输入含 trace 不足 100MB。原 22/24GiB、GPU18GiB/80度和磁盘100GiB边界保留；只读任务不占用 GPU、也不新增 GPU 成本。

先核 root/boot/GPU、原训练暂停和五 PID 已退，再核计划、执行源码、数据及 trace 大小 SHA；退出后再次核资源和所有来源 SHA。旧训练不恢复、采纳 null。既有累计 GPU 成本 1606/14400 秒、R3 485/7200 秒保持。

本地五检查全过，0.053 秒：包时钟重建对原可见性接口、无效行不造截断、状态/年龄/重复/分组错误拒绝、实际 encoder 两输入块、近邻排除整来源。真实数据审计尚未执行，不以夹具当真实通过。

## 复现

发布后通过既有原生交互 SSH 会话，使用固定 Primebot Python、单 BLAS 线程和 `timeout -k 5s 300s`，调用：

```text
student_input_audit.py --plan <published plan> --output <fresh audit root>
  --manifest <published code manifest> --manifest-sha256 <verified SHA>
  --evaluation-commit <independently verified 40-character commit>
```

输入固定计划：[student_input_audit_plan_20261107.json](../script/supermimic_teleop/student_input_audit_plan_20261107.json)。源码清单、实测结果和发布收据随后补入；服务器部署与真实执行前不称已同步。

## 下一步

依据真实审计决定是否扩展同一划分内的来源组和轨迹阶段。若支持覆盖不足，下一实验优先扩大真实教师状态覆盖并保留配对控制，冻结原 decoder；需独立预算、来源/资源门和发布后才能采样或训练。仍须完成选定学生闭环、事件压力、六宫图和原速媒体。原同策略 0/7 历史失败不重跑。
