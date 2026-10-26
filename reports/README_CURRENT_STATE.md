# YUANQI 当前状态与下一步

2026-10-26 23:17恢复入口：[MG03观察目标](README_SUPERMIMIC_Q1_MG03_OBSERVATION_20261026.md)、[RF08一小时训练](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)与[AP01准备](README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md)。MG03工具active，用户再次确认4096环境并要求定目标观察。AP01数据/真实GPU/完整更新验证已完成；RF08尚未启动，正在发布与部署最终源码，核验后只启动一次。每轮最多3600秒，更长2/4/8小时需效果后另行决定；不开子agent、不增加数据硬门。

## 已完成材料与环境

SOMA主包已从private顶层同盘移到`/opt/data/private/ZHY/data/references/bones_seed/inbox/soma_uniform.tar.gz`，45188088297字节，完整SHA256为`0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c`，通过，不重复全包hash。canonical骨架23062字节/SHA d844abc9…已双端核验，metadata已有。师兄实际100/15源清单未取得，可按发布方法重建并如实标注，不是阻碍。

新环境`data/environments/supermimic_q1_hp_20261026`隔离于旧环境：torch2.9.1+cu128、unilab1.5.0、unisim-core2.1.0、MuJoCo/mujoco-warp3.11.0、warp1.16.0。Q1 FK/梯度、16环境GPU观测/未来/f+1时序与真实回合记录检查通过。4096完整profile已PROFILE_COMPLETED/WORKER_COMPLETED并回收：262144转移、10WM1024及1actor1024-H32/critic更新有限，Torch分配/保留峰值5.83/6.32GiB（不含Warp等非Torch显存），运行nvidia-smi曾12252MiB；OOM0、结束GPU14MiB。不重复profile或复用profile权重。继承Isaac/torchvision包有版本冲突，未称全环境pip check通过。

上游固定`310244b73e45775744a75b789eb25a6c9c5a81c9`，原learner不改，独立Q1任务在`data/references/supermimichp/AP01_runtime_20261026`。SOMA181候选转换完成，独立100训练/15演员分离验证，原30Hz到50Hz不改速度。heading XY/相对朝向进10×80历史，未来完整根目标10×59。Q1每轴armature=.004，按HP公式PD约15.7914/1.00531、完整effort，不是厂家逐关节特调。4096env按用户要求，WM1024×10/actor1024-H32按师兄；师兄实际1024env和其确切100/15源清单差异保留。没有正式RF08效果结论。

## 最近训练终态

[RF07根反馈](README_SUPERMIMIC_Q1_RF07_ROOT_FEEDBACK_20261026.md)已全流程COMPLETED，实际3058.032232秒、575轮/2355200转移，WM179/actor129。最终113/365步、2.26秒倒地，沿目标0.50830m含摔倒，双支撑94.690%，未完成全片抬手。与RF06共同113步根XY误差0.109869→0.031088m，但不构成稳定行走；两轮更新数不同。13项保存核验、初末物理/算术/身份审计和11个本地媒体/图表大小SHA通过。controller4805/worker4808已退出，禁止重启或重做已完成媒体加工。

[RF06](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)3132.859650秒、600轮，365步不倒但仅净移1.49cm/参考1.60m、双支撑98.904%，未形成行走。MG02已完成，不恢复旧目标或旧RF05/RF06命令。RF07本地最终MP4在`data/supermimic_for_q1/BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026/render_final/comparison.mp4`；模型/大轨迹留服务器LOCAL_ONLY。

## 恢复与监督

权威Git`/tmp/yuanqi-supermimic-q1-20261023`的Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。正式计算只在用户25252容器，根`/opt/data/private/ZHY`，四目录不变。SSH：`ssh -S /tmp/yuanqi-q1-25252.sock -T -p 18869 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25252' root@127.0.0.1`。保留Windows桥，不读写认证，不操作师兄容器。

旧`q1-2048`更新返回“Automation does not exist”；不能声称它仍ACTIVE或新监控已生效。`supermimichp-x2`和`yuanqi`本轮未改。新训练启动后需核实实际监督任务，避免重复。普通无变化静默，仅实质发现、异常、完成或需用户处理才通知。

下一步：最终代码/小收据发布Q1并独立核远端、部署大小SHA核验，复核OOM0/资源与无重复输出后启动一次RF08。软3500秒、外层3540+45秒退出保护，最多3600秒；100GiB余盘/12GiB单实验/30GiB共享内存/86度/900秒无进展不变。初末115片动力学/移动/上肢、六宫格与本地MP4自动接续；结束状态与源码路径分离，CSV转移数使用实际sim_steps。新监督须实际创建并核验，当前不预称已生效。
