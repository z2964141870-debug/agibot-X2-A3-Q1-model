# YUANQI 当前状态与下一步

2026-10-26 23:59恢复入口：[MG03观察目标](README_SUPERMIMIC_Q1_MG03_OBSERVATION_20261026.md)、[RF08一小时训练](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)与[AP01准备](README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md)。MG03工具active；RF08已于23:23:07启动，训练时钟23:23:22，唯一4096环境GPU正式训练进行中。每轮最多3600秒，更长2/4/8小时需效果后另行决定；不开子agent、不增加数据硬门。23:57:18现场iteration2091/21684224转移、WM20910次/actor1591，实训2036.057519秒，身份/资源正常，暂无固定协议策略效果结论。

## 已完成材料与环境

SOMA主包已从private顶层同盘移到`/opt/data/private/ZHY/data/references/bones_seed/inbox/soma_uniform.tar.gz`，45188088297字节，完整SHA256为`0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c`，通过，不重复全包hash。canonical骨架23062字节/SHA d844abc9…已双端核验，metadata已有。师兄实际100/15源清单未取得，可按发布方法重建并如实标注，不是阻碍。

新环境`data/environments/supermimic_q1_hp_20261026`隔离于旧环境：torch2.9.1+cu128、unilab1.5.0、unisim-core2.1.0、MuJoCo/mujoco-warp3.11.0、warp1.16.0。Q1 FK/梯度、16环境GPU观测/未来/f+1时序与真实回合记录检查通过。4096完整profile已PROFILE_COMPLETED/WORKER_COMPLETED并回收：262144转移、10WM1024及1actor1024-H32/critic更新有限，Torch分配/保留峰值5.83/6.32GiB（不含Warp等非Torch显存），运行nvidia-smi曾12252MiB；OOM0、结束GPU14MiB。不重复profile或复用profile权重。继承Isaac/torchvision包有版本冲突，未称全环境pip check通过。

上游固定`310244b73e45775744a75b789eb25a6c9c5a81c9`，原learner不改，独立Q1任务在`data/references/supermimichp/AP01_runtime_20261026`。SOMA181候选转换完成，独立100训练/15演员分离验证，原30Hz到50Hz不改速度。heading XY/相对朝向进10×80历史，未来完整根目标10×59。Q1每轴armature=.004，按HP公式PD约15.7914/1.00531、完整effort，不是厂家逐关节特调。4096env按用户要求，WM1024×10/actor1024-H32按师兄；师兄实际1024env和其确切100/15源清单差异保留。没有正式RF08效果结论。

## 最近训练终态

[RF07根反馈](README_SUPERMIMIC_Q1_RF07_ROOT_FEEDBACK_20261026.md)已全流程COMPLETED，实际3058.032232秒、575轮/2355200转移，WM179/actor129。最终113/365步、2.26秒倒地，沿目标0.50830m含摔倒，双支撑94.690%，未完成全片抬手。与RF06共同113步根XY误差0.109869→0.031088m，但不构成稳定行走；两轮更新数不同。13项保存核验、初末物理/算术/身份审计和11个本地媒体/图表大小SHA通过。controller4805/worker4808已退出，禁止重启或重做已完成媒体加工。

[RF06](README_SUPERMIMIC_Q1_RF06_LEGS50_20261026.md)3132.859650秒、600轮，365步不倒但仅净移1.49cm/参考1.60m、双支撑98.904%，未形成行走。MG02已完成，不恢复旧目标或旧RF05/RF06命令。RF07本地最终MP4在`data/supermimic_for_q1/BONES_RF07_UP01_E4096_ROOTFEEDBACK_MAX01H_20261026/render_final/comparison.mp4`；模型/大轨迹留服务器LOCAL_ONLY。

## 恢复与监督

权威Git`/tmp/yuanqi-supermimic-q1-20261023`的Q1分支，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。正式计算只在用户25252容器，根`/opt/data/private/ZHY`，四目录不变。SSH：`ssh -S /tmp/yuanqi-q1-25252.sock -T -p 18869 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25252' root@127.0.0.1`。保留Windows桥，不读写认证，不操作师兄容器。

RF08冻结代码`d8b1dcf76b7819ddd5be39a5177e14fe1737a3ad`已push Q1并独立核远端；40文件387207字节在Mac/25252大小SHA一致，plan冻结146源码/配置/资产身份。controller21522/start_ticks3632019311、worker21525/start_ticks3632019335，boot d994e71f-bae0-46ba-a5f7-6c2163771c6c，GPU UUID GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4。现场三status均RUNNING，训练实耗191.48秒，GPU9766MiB/53度、共享15526113280字节/OOM0、余盘1643911651328字节。原始状态/配置/进程命令见`data/manifests/supermimic_q1_RF08_launch_observation_20261026.json`，恢复重新核现场，不凭旧RUNNING。

最新23:57:18三status与进程身份匹配，GPU14538MiB/52度、共享16445321216字节/OOM0、余盘1629030232064字节，实验554426368字节，见`data/manifests/supermimic_q1_RF08_observation_20261026_235718.json`。用户对曲线非单调的疑问已记录于RF07阶段README：不以曲线单调性证明配置正确，不把模型训练loss下降当真实控制改善；RF07实测仍差，过拟合需训练/验证差距证据。

旧`q1-2048`不存在，不恢复。新`q1-rf08-4096`已通过工具实际创建ACTIVE，每10分钟观察，正常无变化静默，仅异常、重要结果或需处理通知。`supermimichp-x2`和`yuanqi`本轮未改。RF08终后交付与MG03结束后暂停本新监督，不改其他任务。

下一步：监督RF08继续，禁止修改冻结适配器/上游/配置/参考，不并行、不自动延长或重试。软3500秒、外层3540+45秒退出保护，最多3600秒，预计10月27日00:22左右训练结束；100GiB余盘/12GiB单实验/30GiB共享内存/OOM0/86度/900秒无进展不变。初末115片动力学/移动/上肢、六宫格与本地MP4自动接续，终后程序完整链路待实际执行。23:39已备独立`script/project_tools/analyze_q1_hp_trial.py`（前缀/时轴/算术小测试通过），真实轨迹到齐后按RF08 README命令生成common_prefix_analysis.json；0新积分/推理/更新，不改冻结入口。足近地为几何诊断不等于接触力。MG03 A/B完成、C进行中、D/E待；本轮无性能改善或目标完成结论。
