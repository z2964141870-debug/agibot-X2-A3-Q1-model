# RF16接触相位短训练结果与本地交付

2026-10-29 06:55，北京时间。训练和非训练交付均完成；没有确认稳健自然步态改善，数值推荐仅为待审候选，用户采纳仍为null。

## 结论

把足端相位特征和奖励分段改为原重定向保存的接触意图后，600秒候选的停走承重滑速三次下降16%-24%，但起走一次增加23%，另一次停走倾斜增加11.7%；1201秒末段没有保持该部分收益。各阶段整组改善/继续门均0/3。不能将存活、回报或单片平均改善称为正常步态，也不能将此结果归因于一个已隔离的原因。

上肢误差约.296rad、抬手峰高误差约.07m的主要收益，已存在于未训练的静态alpha.5参考混合，不归因新腿腰head学习。独立ACTIVITY动态混合是另一份零优化对照，不与RF16叠加冒充已测方案。

## 真实预算与终态

- 训练06:09:09正常短预算结束：1201.1200940818526实训秒、282轮、27721728转移，critic5640/actor5440、0新WM更新。
- 全pipeline于06:32:05 COMPLETED；06:33:01现场核run TRAINING_COMPLETED、train/delivery两guard WORKER_COMPLETED/reaped0。
- controller161302/start13696801、训练worker161313/start13696954及delivery164926/start13899834均已不存在；GPU15MiB/25度、无计算进程、OOM0。
- 正式训练来源仍为2c83b944059d4a007f0ca74fa54c0076fda3a357；一次终后保存审计12/12通过，1532冻结项未变、最终模型/优化器与最后阶段一致。报告和独立只读工具提交不改变训练来源。
- outcome为contact_phase_short_budget_completed，不改记physical_regression；未恢复、重试、扩时或补预算。权重/大NPZ只留服务器。

## 三重复物理数据

原300训练/30验证、固定两普通走及抬手不换片。每列三个数是同一checkpoint的三次评测，不是三次独立训练；验证已参与搜索，不能作为独立最终测试。

| 候选 | 实训秒 | 训练失败/300 | 验证失败/30 | 固定两走/抬手 | 停走滑速/静态 | 起走滑速/静态 |
| --- | ---: | --- | --- | --- | --- | --- |
| 未训练静态alpha.5 | 0 | 11/11/8 | 0/0/0 | 各3/3完整 | 1/1/1 | 1/1/1 |
| stage01 | 300.830 | 12/10/12 | 0/0/0 | 各3/3完整 | 1.314/1.013/.882 | 1.248/.989/1.053 |
| stage02 | 600.962 | 10/14/14 | 0/0/0 | 各3/3完整 | .808/.839/.764 | .909/.815/1.232 |
| final / stage03 | 1201.120 | 13/13/12 | 0/1/0 | 各3/3完整 | 1.171/.930/1.338 | 1.045/.941/1.229 |

最终验证完成99.44%-100%，上肢RMSE.2947-.2963rad，相对腕.0829-.0833m；抬手漂移.0776-.0779m、峰高误差.0713-.0714m。固定停走前进2.1195-2.2745m、起走2.1551-2.3338m，但真实前进及完整执行仍不足以确认自然步态。根XY仅诊断。

原值入口：[全部4950逐片/45固定/共同前缀表](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/review/final/README.md)。阶段JSON/NPZ及失败证据保留。

## 实际脚底几何

新增独立CPU工具只对已保存姿态执行mj_kinematics，没有积分、策略推理或optimizer。72足行/6参考足行/19224时序行，使用同一实际碰撞mesh/sphere和地面；不是踝高估计。

| 固定片右脚 | 未训练静态脚底高度p95，mm | stage02脚底高度p95，mm | 最终脚底高度p95，mm | 参考，mm |
| --- | --- | --- | --- | ---: |
| 普通停走 | 4.60/4.31/4.78 | 4.49/3.93/4.68 | 19.56/19.07/29.13 | 22.41 |
| 普通起走 | 5.27/3.09/3.60 | 4.57/2.23/2.66 | 2.76/3.79/3.57 | 25.49 |

600秒的低滑速没有伴随右脚可靠离地；最终停走右脚抬高增加，也没有保住滑移收益。这支持把负重转移、摆脚离地和足姿态放在后续问题定义中，尚不证明具体原因或解决方法。最低脚底的短暂负值可来自软接触穿入，不能直接当成绘图错误或另立数据准入门。

[六格脚底曲线PNG](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/review/sole_plot_v01/collision_sole_clearance.png)；同目录PDF与来源收据。曲线为20ms原值，线为repeat0、阴影为三次范围；不是500iteration均值。几何离地段数不是正确步数，姿态差分材料点速度也不等于原生solver承重滑速。

工具第一次因沿用旧JSON的trace字段而KeyError，未产生部分输出；修正为读取阶段决策的历史SHA后完成。初始轨迹只有完成收据路径/JSON维度时轴及当前读取前后SHA，没有历史初始SHA清单；该限制保留，未伪造历史校验。修订工具不在1532项训练来源内。

## 固定本地视频

四视频均54秒、1350帧、25fps、8段6类型原速合集，不是连续54秒rollout。舞蹈提前终止继续保留失败hold，普通走/抬手没有事后换片。

- [纯冻结HP教师base](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/evaluation/base.mp4)：标题中的gait compensation是通用模板，实际不含新head或手臂参考混合。
- [未训练静态alpha.5 initial](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/evaluation/initial.mp4)。
- [600秒候选recommended](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/evaluation/recommended.mp4)：原数值rank推荐stage02，仅供审阅。
- [1201秒final](/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF16_CONTACT_PHASE_SOMA300_E4096_MAX20M_20261029/evaluation/final.mp4)。

Mac一次全帧解码/声明检查24/24通过，12联系图、六格/PPO/奖励四图和脚底图目视完成。上肢抬起可见，走路仍有低离地及短滑步；解码与目视验收不等于自然步态认证。六格PNG/PDF/CSV含原值及500训练iteration均值，PPO/奖励分项另附，WM/想象两项N/A。

## 同步与下一步

102个JSON/CSV/PNG/PDF/MP4/README工件共68096950字节，Mac/25386逐大小SHA一致；包含最终COMPLETED审阅快照及脚底分析，不含权重或大NPZ。[本地交付与完整清单](../data/manifests/supermimic_q1_RF16_local_delivery_20261029.json)。Q1发布与三端记录核验以随后publication收据的实际40位SHA为准，不以报告变更改写训练冻结来源。

本夜不再开启物理或优化作业，09:00前提供综合数据与视频。原监督保持ACTIVE每10分钟静默等待用户审阅，不自动采纳stage02或继续训练，不创建旧goal或启用agent。ACTIVITY动态上肢的单独结果见[动态混合报告](README_Q1_ACTIVITY_BLEND_20261029.md)。
