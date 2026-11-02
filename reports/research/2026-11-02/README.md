# YUANQI 研究简报：2026-11-02

今天只报一项实质更新：**FRDV 作者的运动学重定向代码已公开，动力学验证仍未发布。** 这是10月22日已读论文的代码发布更新，不是新论文。当前继续按最新Q1工程记录推进MorphMimic足量复现；衣服SMPL到Q1官方policy演示仍按用户决定搁置。

## FRDV：先把人的动作变成机器人参考轨迹

**解决什么问题。** 人和机器人身高、腿长、关节范围不同，直接复制人体关节角度，机器人可能弯膝过深、身体侧倾或脚穿地。FRDV先标定人体与机器人之间的比例和坐标，再求一串机器人姿态：关键部位尽量跟上人体，同时考虑关节限位、地面和自碰撞约束，并使相邻帧平滑。可以把它理解为“制作机器人动作参考”的工具；能生成参考，还不等于控制器能在重力和接触下稳定跟随。[作者页](https://daroslab.github.io/frdv/)、[固定源码](https://github.com/DARoSLab/FRDV/blob/3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9/src/mocap_retargeting/kinematic_retargeting/optimization/retargeting_problem.py)。

**这次实际公开了什么。** 核验到MIT代码、G1任务配置，以及LAFAN1 BVH和原始AMASS SMPL+H NPZ入口；树中列有G1模型与示例BVH，但本次未下载这些资产。作者README明确限定为运动学流水线，论文的动力学验证不在此次发布范围。完整树中未见Q1/A3/X2任务配置或常见策略权重文件；这也没有提供可直接用于Q1的训练策略。虽然带有平滑、力矩/地面反力估计等辅助脚本，不能据文件名把它当作论文CI-iLQR验证已开源。[固定README](https://github.com/DARoSLab/FRDV/blob/3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9/README.md)、[辅助脚本](https://github.com/DARoSLab/FRDV/blob/3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9/tools/dynamics_processing/dynfilter.py)。

**对我们最有用的两点。** 第一，标定要随人体比例变化，不能默认别人的缩放参数适合自己的衣服数据。第二，把一条动作拆给多个进程可能在接缝处产生跳变；作者默认每条动作一个worker，单worker跳过细化，跨多条完整动作并行则是另一种设置。以后如果参考轨迹有突变，可以先检查坐标、人体比例与接缝，再讨论策略或世界模型误差，避免把不同来源的问题混在一起。[标定与并行说明](https://github.com/DARoSLab/FRDV/blob/3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9/README.md#5-workers-and-refinement)。

**主要边界。** AMASS加载器要求`poses[T,156]`、`trans[T,3]`、16维`betas`、性别和帧率，并另外提供16-beta SMPL+H人体模型。现有衣服24关节SMPL或SOMA BVH不能仅改扩展名接入。MP4导出源码逐帧写入`qpos`再调用`mj_forward`，没有通过`mj_step`推进受控动力学，因此视频好看不能证明机器人能站稳或走稳。[加载器](https://github.com/DARoSLab/FRDV/blob/3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9/src/mocap_retargeting/data_loading/amass/loader.py)、[渲染源码](https://github.com/DARoSLab/FRDV/blob/3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9/tools/video_export/render_mujoco.py)。

**建议：基线完成后再试。** 现在只把标定和轨迹连续性列入阅读候选，不替换冻结训练、不新增数据硬门槛，也不重启已搁置的衣服演示。后续若评估FRDV，需要单独落实Q1本体映射，并分别报告参考轨迹质量、策略跟踪和物理效果。本轮没有提出新原创假设。

## 日期与证据

| 事件 | 原始日期与时区 | 本次核验 |
| --- | --- | --- |
| 论文v1 | 2026-10-20 16:22:00 UTC，即北京时间10-21 00:22:00 | 原摘要/历史；相关正文已在10-22阅读，本轮未重做全文审阅 |
| 仓库创建 | 2026-11-01 06:19:03 UTC，即北京时间14:19:03 | GitHub仓库元数据；不据此断言精确转为公开的时刻 |
| 固定提交 | 2026-11-01 06:30:10 UTC，即北京时间14:30:10 | `3600f965a8e02e430fed11e42b8ed4ed1f0e5fc9`，提交信息写v1.0，GitHub Releases为空，pyproject版本0.1.0 |
| 检查 | 2026-11-02，北京时间 | 完整树209项、12文本116014字节，逐大小/Git blob/SHA核验；仅静态阅读，未运行或复现 |

## 覆盖与保存

本轮106个缓存请求105个成功；一次非固定版本摘要请求超时，随后固定v1成功，失败记录保留。18个既有仓库头与已有记录对照未发现实质变化；FRDV是新增核查仓库。RO/LG/CV六份完整公告列表仍显示10月30日，条目顺序与上次一致，RSS为空；不能由此推断之后没有新论文，历史公告及资产核验缺口保留。

服务器原生Windows登录成功，但本轮容器自动认证未通过，未执行容器命令；服务器Git与文档镜像仍未核验/完成，不据旧训练记录声称作业实时正常。Mac记录与GitHub实际发布状态见[检查](../../README_RESEARCH_CHECK_20261102.md)和[收据](../../../data/manifests/research_watch_sync_20261102.json)。
