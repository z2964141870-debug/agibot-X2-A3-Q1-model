# MorphMimic：与Q1复现的关系与借鉴边界

记录日期：2026-10-27，北京时间。阶段MM01，状态：只读初评完成，未运行复现。

## 目标与范围

用户要求服务器停服期间简单了解`lzyang2000/morphmimic`。本次读取论文、README及少量源码，不连接旧服务器、不安装依赖、不下载模型/数据、不运行上游脚本、不修改训练方案。MG05仍暂停且未完成，本次评阅不恢复或新建训练目标，不开子agent。

## 完成内容与判断

**值得作为独立实现对照，当前先提取可验证的组件，不整体切换Q1主线。** README声明其为依据SuperMimic论文开展的独立实现，并非原作者正式发布的代码。

| 项目 | 核查结果 | 对Q1的意义 |
| --- | --- | --- |
| 方法 | 可微世界模型与策略可跨G1、X2 Ultra、T2共享，以形态描述条件化，并对齐对应动作的潜变量 | 延续世界模型路线；新增重点是多形态共享，不是单独的重定向工具 |
| 输入 | 全动作或pelvis、双腕、双踝五点的位置/朝向 | 五点接口对未来实时跟随有价值；不能据此断言现有上肢偏差会消失 |
| 位置反馈 | 默认目标使用参考自身heading，只有critic取得位置偏差；`goal_anchor=robot`另提供机器人相对参考的位置纠偏 | 默认模式仍可学习移动，但不能期待其主动消除同一种世界轨迹偏差；需先明确评测目标 |
| 关节契约 | `Translator.from_names`要求G1的29个规范关节各有唯一的本体关节匹配，核名称、符号及零位偏移 | 现Q1的22控制关节无法直接套入该契约；接入需要扩展映射/缺失关节处理并重训验证 |

源码核查：`model/rewards_torch.py:268`的`goal_features`实现两种目标坐标，默认位置项不读取机器人位置，但朝向始终相对机器人anchor；`:322`的`critic_anchor_pos`另提供位置输入。`env/observations.py:51`将真实目标接入同一函数，`:93`按机器人IMU所在body计算惯性量；`robots/spec.py:79`的`Translator.from_names`拒绝缺少或重复的关节匹配。位置反馈结论只针对已读实现，不能等同“默认模式不会走路”。

## 对现有问题的启发

优先对照关节符号/零位、IMU安装坐标、未来目标时序与坐标。论文把IMU约定列为跨机器人迁移的重要因素；这些应做契约检查，不能直接抄G1/X2数值。其世界模型加入重力与身体高度，以及归一化、回报尺度和合法想象起点的处理，也适合作为独立比较项。

这不推翻既有[MG05离线结论](README_SUPERMIMIC_Q1_MG05_SIGNAL_ANALYSIS_20261027.md)：当前主要上肢偏差首先体现在策略目标，真实原生接触和H32闭环方向尚未完成验证。MorphMimic没有给出Q1接触、重定向或策略误差已经解决的证据。后续只选择与已测问题对应的措施，保留原基线。

## 结果与边界

- 论文大规模评测使用三种训练动作库交集的128786片，不能当作未见动作测试或新Q1零样本迁移证据。真机展示为G1，X2/T2真机仍列后续工作。作者报告与本项目独立复现明确分开。
- README报告4096环境即使约一小时动作子集也需约28GB显存，全量数据约51至58GB/GPU。多机器人配置按每机器人一张GPU；不能根据同为4096环境就认定本项目既有24GB卡可原样运行。以上为上游报告，当前未实测。
- 加入新机器人仍需关节映射、物理描述与该机器人重定向动作；现仓库不含Q1。五点接口、跨形态对齐并不免除Q1资产/控制/参考验证。

## 版本、命令与来源

权威工作副本：`/tmp/yuanqi-supermimic-q1-20261023`，分支`Q1`。本次只新增评阅及入口记录，不改变训练源码/配置。

已执行的只读Git查询：

```sh
git ls-remote --symref https://github.com/lzyang2000/morphmimic.git HEAD
```

结果：默认`main`，HEAD `aa40fe46a7fa995ba7a7354ad065a2c8ad64a302`。源码读取固定此提交，README与论文为本日网页快照；后续更新不自动代表本次核查范围。

```text
repository: https://github.com/lzyang2000/morphmimic
README: https://raw.githubusercontent.com/lzyang2000/morphmimic/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/README.md
paper: https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf
observations: https://raw.githubusercontent.com/lzyang2000/morphmimic/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/src/morphmimic/env/observations.py
joint contract: https://raw.githubusercontent.com/lzyang2000/morphmimic/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/src/morphmimic/robots/spec.py
analytic goals: https://raw.githubusercontent.com/lzyang2000/morphmimic/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/src/morphmimic/model/rewards_torch.py
```

本次为论文/源码初评，0仿真、0网络前向、0optimizer。未审核全部训练路径，不能声明上游无bug或其性能已在本机复现。

## 保存与恢复

阶段记录、索引与当前摘要保存到Q1；发布按payload audit、显式push及独立远端SHA核验执行，提交身份以Git历史为准。小记录镜像Mac，服务器同步等待新入口；无新增大型工件。

服务器恢复后仍按MG05先取回原生诊断、验证现有候选与受控对照。MorphMimic本次不触发安装、训练或常驻仓库监控。
