# 严格原始依赖截止下的短预览补全

2026-11-08，北京时间。本阶段把既有原始帧可见性规则接到阻尼补全，尚未得到新的策略或步态质量结果。原共享训练继续暂停，低延迟自然遥操完整目标继续。

done：

1. 核清旧84例阻尼回放和后续R1学生的输入口径差异：前者只登记合成capture/arrival，原始插值依赖为未知；后者明确要求120→50Hz右侧原始支持在控制截止前到齐。不能直接用二者的质量差异判断短预览训练优劣。
2. 新增独立`DependencyPreviewReceiver`，复用R1的`OfflineStudentStream.student`与原有`forecast_window`。先筛原始依赖/到达时间，再对实际连续前缀做masked、hold或damped_cv；没有改旧接收器、worker、预测器、训练输入或动力学。
3. 固定393片元数据的121659个有效锚点已完成时序统计；未读取人体姿态、模型、物理轨迹或新留出事件结果。10项新CPU检查通过/.551秒，实际元数据分析.002299125秒，RSS33079296B。
4. 获批SSH90/60/5仍生效，45014在12:44:17返回Windows日期。最近一次第二跳12:38返回Connection refused，未认证，现有会话在PowerShell；本阶段没有再次循环第二跳，也没有发送Linux命令。

todo：

1. Linux恢复后优先检查已提交的人体条件CPU审计与七份原JSON，补双训练六宫图；不重派已完成6000更新。再按已发布来源和真实资源门执行原多步WM诊断。
2. 为本接收器准备独立GPU包装/保存状态动作门及来源、资源、预算准入，核同一组目标的原生pack/encoder/decoder动作2e-5等价，覆盖原始支持五相位、事件、回退和完整worker计时。当前这些尚未实现或执行，不能把CPU合同当动作等价。
3. 动作门通过后才做新的匹配物理小样本：严格依赖感知的H10 masked、H4 masked、H4阻尼共用相同接收规则和回退。先固定前走筛查，不重跑旧84；完整多片/事件、原速视频及新留出验证仍需后续独立有界方案，不能用小样本结算目标。

## 为什么重新对齐

已有[来源计划](../data/manifests/q1_teleop_forecast_split_20261106.json)已登记120→50Hz重采样的右侧支持。本次没有重新做96组离线预测，而是审查两条已用实验入口是否都执行同一规则：

- 旧`event_inputs.EventReceiver`将包的`dependency_end_ns`置为None；其结果保留`offline_unverified`，不能认证严格原始数据截止。
- `student_data.OfflineStudentStream.packet_frame`计算`max(合成capture, source_support(source_index))`，到达也不能早于该依赖；`student_inputs.short_preview`在构坐标和送策略前筛除未到齐支持。
- 新[接收器](../script/supermimic_teleop/dependency_preview.py)复用后一条路径，并把实际连续前缀送入既有阻尼预测；新父侧校验从不可变事件包重建依赖/到达/前缀和逐槽来源。

这说明旧84不能单凭H4标签充当严格80ms的基线；不据元数据推断旧实验读到了真实人体的未来意图，也不改写其真实物理指标或原FAILED根。

## 时序结果

[元数据分析入口](../script/supermimic_teleop/dependency_preview_audit.py)只读固定SHA `f40c1d5a856a08d89f1a8c23dfe6d990fc7439232b9da9a658d9860456353145`，每片锚点取frame0至frames-11，十个目标均在片内。167 train/98 validation/128 test只用于来源长度统计，不打开新事件或质量标签。

| 请求窗口 | 名义等待 | 在截止前真实就绪的连续前缀 | 若坚持等完整请求窗口，额外原始支持等待最大值 |
| --- | --- | --- | --- |
| H1 | 20ms | 0帧97489锚点；1帧24170锚点 | 6.666667ms |
| H2 | 40ms | 1帧97408；2帧24251 | 6.666667ms |
| H4 | 80ms | 3帧97257；4帧24402 | 6.666667ms |
| H5 | 100ms | 4帧97169；5帧24490 | 6.666667ms |
| H10 | 200ms | 9帧97169；10帧24490 | 6.666667ms |

H4有79.9423%的锚点只能使用三帧真实前缀，再生成七个槽；其余使用四帧，再生成六个槽。H10同样受原始支持规则约束，不能只对短窗口严格筛选、却给H10提前可见的原始帧。

上述是固定插值时间算术。姿态估计、传输、推理、执行器和身体响应延迟不在本统计中，硬件端到端字段保持null。`offline_unverified`不改成已验证传感器因果。

## 接口与验证

阻尼时间常数保留原.12秒，不引入新的调参或训练；position与SO(3)仍用原预测器。生成帧只标generated，完整送十槽不意味着观测到十帧真人未来。只有一个真实前缀时退为hold；第一目标缺失时仅允许最近200ms内、原始支持已就绪且capture不晚于第一目标的显式过去姿态保持；无此支持则拒绝出目标，由未来包装器处理停止或动作保持。该回退未经真机安全验证。

旧R1无前缀时保持上一动作，旧事件回放保持过去姿态，二者回退不同；不能复用旧R1物理当新三方法的完整匹配对照。新小样本须共用同一回退，并如实记录发生位置。

[10项新检查](../script/supermimic_teleop/test_dependency_preview.py)覆盖：五相位实际前缀；把截止后未就绪插值及其后姿态全部污染仍不改变任何目标；同可见前缀复用原预测；原始支持全齐时与旧目标一致；停止/反向/丢包/突发晚到；H1和单帧回退；陈旧输入拒绝；伪造依赖、mask、因果声明拒绝；同前缀三方法；周期聚合与独立逐锚点计算一致。全部为合成合同，零GPU动作推理、物理或Adam。

实际统计文件在Mac：

```text
/Users/yu/Documents/ChatGPT/元启/data/analysis/Q1_DEPENDENCY_PREVIEW_TIMING_20261108/summary.json
10939B SHA256 a74a69233241a2aecec5f44f3181347a19ef3ef53a1baf6b4415630058c7c6aa
/Users/yu/Documents/ChatGPT/元启/data/analysis/Q1_DEPENDENCY_PREVIEW_TIMING_20261108/phases.csv
934B SHA256 c587669235f70c9ac69d13d955df4a38abd7b86b5de081f4923c15a6ebff698a
```

复现仅需既有本地环境；输出必须为未存在的新路径：

```sh
cd /Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028
/Users/yu/Documents/ChatGPT/元启/data/environments/q1-teleop-mps-20261106/bin/python \
  script/supermimic_teleop/dependency_preview_audit.py \
  --split-plan data/manifests/q1_teleop_forecast_split_20261106.json \
  --output /absolute/fresh/output
```

本轮新增服务器模型调用/物理/训练/预算预留均0；累计仍5036=1614+2051+1371秒，R1余1086/R2余2449/R3余5829，总余9364秒，不重置。未来GPU门和小样本也必须核台账及原22/24GiB、GPU18GiB80度、100GiB盘余、500GB项目与非阻塞锁，不争用现有任务。

原多步WM五执行文件和12128B包SHA保持，新接收器未部署。人体条件原CPU审计/曲线、WM真实诊断、完整新匹配物理、实测人体/硬件时钟和最终质量结论仍待；原final完成门与采纳null保持。服务器本阶段镜像PENDING。[阶段收据](../logs/session_records/q1_dependency_preview_prepared_20261108.json)。
