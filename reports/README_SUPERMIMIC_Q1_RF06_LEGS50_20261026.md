# Q1 RF06腿部限矩对照与一小时预算

记录日期：2026-10-26，北京时间。RF06于17:22:03启动，18:14:29正常完成训练，18:21:52.280946全pipeline COMPLETED；全部10阶段WORKER_COMPLETED。初末动力学、保存/算术身份审计、全身/腿部分析、六宫格和本地MP4交付完成。MG02-A至E结果记录已发布并完成三端大小SHA核验，q1-2048训练监督已暂停。

## 训练与最终效果

训练收据实耗3132.8596501871943秒（52分12.86秒），600轮/2457600真实转移、WM189组/actor139次，完整25轮边界，低于用户3600秒上限；初始化/绘图/评测未计入。guard为WORKER_COMPLETED、worker3304已退出、GPU14MiB/OOM0。保存13项检查全通过，三Adam计数1890/139/2780，运行源码/输入未变。

18:15:22前初末真实MuJoCo、算术身份、全身和腿部分析均WORKER_COMPLETED。初始/最终均365步不倒，但最终根净移仅0.014933m、目标1.601504m；整段回报0.475341→0.437431，腿RMSE0.250740→0.257165rad，上肢0.663430→0.692123rad。双支撑98.904%、左单支撑0/右单支撑0.274%（仅一帧），支撑状态变化2次，未形成连续换脚；腿部控制边界饱和0.0228%。本轮加大腿部控制权限没有带来有效行走，训练按预算完成与性能改善分别记录；初始也不倒，不能把最终站住全部记为训练收益。

原样训练、保存、初末评测/审计、全身、腿部、渲染和pipeline终态小收据已接收到`data/manifests/supermimic_q1_RF06_leg50_*_20261026.json`。终后controller3301和worker3304均已退出，GPU14MiB/26度，共享内存4941115392字节/OOM0、盘余1788185260032字节；RF06实际分配277934080字节。没有恢复或重复训练，没有重跑已通过的评测、比较或绘图。

## RF05与RF06完整段比较

已有`script/project_tools/compare_q1_physical_trials.py`已执行一次，验证同参考、同初态、确定性评测协议和控制资产身份，使用保存轨迹重建运动/运动学，0新积分、0推理、0优化器更新。原样[RF05/RF06比较](../data/manifests/supermimic_q1_RF05_RF06_comparison_20261026.json)状态为Q1_COMPLETED_PHYSICAL_TRIAL_COMPARISON_COMPLETED；共同前缀365步，即整段7.3秒。

| 指标 | RF05原25%限矩 | RF06仅腿部50% |
| --- | ---: | ---: |
| 实际训练秒数 | 7237.627906 | 3132.859650 |
| 真实转移数 | 4608000 | 2457600 |
| WM组/actor更新 | 399/349 | 189/139 |
| 完成步数/倒地 | 365/365，不倒 | 365/365，不倒 |
| 整片归一化回报 | 0.500930 | 0.437431 |
| 根净位移m | 0.0187372 | 0.0149333 |
| 沿目标方向位移m | -0.0168110 | -0.0149155 |
| 路径长度m | 0.105109 | 0.158446 |
| 沿目标方向平均速度m/s | -0.00230287 | -0.00204322 |
| 根XY平均位置误差m | 1.003879 | 0.997689 |
| 腿关节RMSE rad | 0.257180 | 0.257165 |
| 上肢关节RMSE rad | 0.680416 | 0.692123 |
| 双支撑边界比例 | 99.178% | 98.904% |
| 左/右单支撑边界比例 | 0%/0% | 0%/0.274% |
| 支撑状态变化次数 | 1 | 2 |
| 腿部边界限矩饱和 | 0.2740% | 0.02283% |

两轮都没有有效向前行走；RF06路径变长不等于向目标移动，仍近乎全程双支撑。提高限矩降低了观测到的边界饱和，没有带来步态改善。边界采样不能代表物理子步的所有力矩峰值；RF05与RF06时长/转移/更新量不同，不能称等时单变量训练消融，也不能据一次负结果排除控制、重定向、奖励或世界模型梯度的其他影响。

上肢不只检查全局位移。左右手腕相对躯干的轨迹RMSE，RF05为0.133857/0.107987m，RF06为0.128610/0.108673m；RF06左右抬手高度跨度仅为参考的47.62%/42.71%（RF05为50.88%/44.27%），参考高于肩部时同步抬高手的比例两轮均0。RF06左右峰值比参考晚0.88/0.90秒，较RF05的-4.48/-4.54秒更接近，但手腕峰值仍在肩下0.07829/0.07888m，不能称抬手成功。RF06左右肘RMSE为1.290764/1.012285rad，肩部各轴误差及每侧腕轨迹/幅度/时序保留在比较JSON中。

## 本地媒体与数值交付

Mac根`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/`已接收11项PNG/PDF/CSV、初末MP4和六张代表帧，共1706855字节，逐文件大小/SHA与服务器相同；[交付收据](../data/manifests/supermimic_q1_RF06_leg50_local_delivery_20261026.json)保留全部身份。RF05原11项1723401字节同步再次核验通过。没有复制模型或大轨迹，没有网站。

- [最终动力学MP4](../data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/render_final/comparison.mp4)
- [初始动力学MP4](../data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/render_initial/comparison.mp4)
- [2×3训练图PNG](../data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026/run_dashboard/training_dashboard.png)、同目录PDF/CSV及来源收据

六项面板齐备，细线原值/粗线500训练iteration平均，独立核129条CSV记录，原值最大差0、均值最大差1.11e-16，见[数值检查](../data/manifests/supermimic_q1_RF06_leg50_dashboard_check_20261026.json)。训练图与最终代表帧已目视检查，继续直接使用现有产物，无额外图片加工。

## 实际启动

RF05在17:10:45全流程COMPLETED，worker和controller退出，11项图/初末MP4本地大小SHA通过。启动前现场6个RF06源码大小/SHA与已发布`0b1da033148fff0b705fd85cdac5a773b5a7f332`一致，GPU14MiB/27度、共享内存4.66GB/OOM0、盘余1.789TB，目标目录不存在。

按下方准确命令只启动一次，实际2026-10-26北京时间17:22:03，controller3301/start_ticks3629852873、worker3304/start_ticks3629852891、boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`。17:23:36两者身份/worker命令匹配，pipeline/guard RUNNING、step0初始化，GPU2704MiB/37度、共享内存7.62GB、OOM0、余盘1.789TB；不把step0当真实更新或效果。现场admission确认为12腿关节50%、其余10关节25%，动作尺度按25%、PD50/2，预算软3000秒/最大3600秒、阶段3500秒及55秒清理，禁止续训。

冻结[计划](../data/manifests/supermimic_q1_RF06_leg50_plan_20261026.json)及[启动观测](../data/manifests/supermimic_q1_RF06_leg50_launch_observation_20261026.json)保存真实身份；日志`logs/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026_controller.log`和同前缀`_train.log`。正式运行中不热改训练来源、配置或输入。RF05最新双支撑99.178%、腿部边界饱和0.274%，提示限矩不一定是本轮不迈步的直接原因；RF06检验训练期控制权限，改善尚待真实终后评测。

## 目标与范围

RF05验证扩大采样和更新规模后的原配置表现，不是腿部修正试验。相对RF04，环境2048→4096、世界模型批量1024→2048（每组仍10次）、actor批量2048→4096（H32）；PD、限矩、动作尺度、参考、奖励、网络和seed保持。该比较同时改变环境和更新批量，固定两小时也不保证相同转移或更新数，不能单独归因于环境数。更多环境没有自动扩大动作噪声幅度，也没有增加参考动作种类；当前仍只训练同一段365帧参考。是否改善只能由训练后的真实MuJoCo跟踪结果回答。

RF06验证更具体的问题：仅提高腿部允许力矩，能否缓解失稳并帮助策略学习支撑切换和移动。依据是[D06B闭环对照](README_SUPERMIMIC_Q1_LEG_CONTROL_20261026.md)：同一固定RF04策略，原25%限矩122步倒地，仅腿部50%完整365步不倒，但净移0.040410m、双支撑99.18%，仍未行走。这支持控制权限是失稳因素之一，不能证明它是唯一原因或已经解决行走。

## 控制与预算

- 独立目录`script/supermimic_q1_reproduction/leg_control/`；不改RF05冻结入口。`control.py`只将12个髋/膝/踝关节模型effort比例25%→50%；其他10关节25%，PD50/2，动作尺度仍按原25%计算。
- `entry.py`使训练、评测、算术审计、全身分析和渲染一致使用该控制设置，逐关节记录比例和版本`Q1_LEGS_CAP50_OTHER25_SCALE25_V1`。这是仿真干预，未经厂家真机验证。
- `run_single_motion.py`复用原SuperMimic训练与流程，固定4096环境/8线程/8槽、WM2048×10、actor4096/H32、UTD0.4、seed2026102513、同一参考/原速时轴、从零初始化，最大12000轮。
- 用户最新要求以后每轮最多一小时。RF06真实训练最多3600秒，软目标3000秒（50分钟），完整25轮边界结束；整个训练阶段3500秒时限及55秒退出宽限，预留一小时内退出的余量。超时保留保护停止，不续训、不自动重试。初始化、绘图和评测不计真实训练时长。
- RF05仍为已启动的原两小时基线。两轮同时存在限矩与时长差别，不称等时训练消融；报告必须列实际时长、转移及更新数。

## 实际验证与失败记录

`test_control.py`的8项测试全部通过、0跳过，耗时4.577秒：恰好12个关节限矩改变，尺度/PD/奖励/接触保持；原生与slotted模型实际force range及4步状态完全一致；保存manifest和训练绑定正确；各控制相关流程路由一致；未完成或不足预算的RF05阻止启动；训练时限生效，超过一小时的收据不被接受。

最终同8项CPU检查再次通过，4.441秒/退出0，原始stdout和准确命令保存在[最终检查日志](../logs/session_records/supermimic_q1_RF06_leg_control_final_checks_20261026.log)。该检查没有优化器训练，不计RF06预算，不再重复执行。

CPU集成验证`RF06_LEG_CONTRACT_C02_20261026`完成真实评测、算术审计、全身分析、腿部分析和两帧渲染，全部退出0：365步状态/动作与独立D06B最大差均0，支撑分析相同；两帧图像像素标准差最低73.65395，编码实际成功；43.716秒、0优化器更新、未使用CUDA。固定RF04策略fixture仅检查入口一致性，不是RF06训练结果，不作为正式初始化。原样[成功收据](../data/manifests/supermimic_q1_RF06_leg_contract_C02_20261026.json)保留来源SHA和每阶段日志。

首个C01在评测/审计/分析通过后因新容器缺系统字体/ffmpeg入口而渲染失败，[原失败收据](../data/manifests/supermimic_q1_RF06_leg_contract_C01_failure_20261026.json)与服务器目录保留；另一次初始调用的模块搜索路径和测试backend路径问题已修正。复用项目已有DejaVu字体和ffmpeg7.0.2二进制，建立标准系统入口，无下载或新安装，大小/SHA和入口见[依赖收据](../data/manifests/supermimic_q1_RF06_media_dependencies_20261026.json)。16:20:42检查RF05的78项冻结来源全部未变；该依赖修复也使RF05后续自动编码可用。

`analyze.py`从已保存评测状态重建每个控制边界的双脚支撑和PD需求/限矩比，0新积分/0策略推理/0优化器更新；边界采样不等于子步力矩峰值，不能独立作为行走成功判据。RF06自动执行；RF05完成后用相同分析器另存独立产物。

## 命令与产物

正式计算仅在自己的25252容器、根`/opt/data/private/ZHY`。启动前须RF05全流程COMPLETED、实际训练至少7200秒、worker正常回收、controller退出、GPU空闲，且RF06输出目录不存在；入口再次强制检查。保留100GiB余盘、12GiB单实验、30GiB共享内存、OOM基线0、86摄氏度和900秒无进度保护，不并行第二训练。

代码版本`0b1da033148fff0b705fd85cdac5a773b5a7f332`已经push并独立核验`refs/heads/Q1`，13个代码/报告/小收据在Mac和25252逐大小/SHA一致，目标AGENTS的一小时预算已核；见[部署收据](../data/manifests/supermimic_q1_RF06_C_deployment_20261026.json)。下列命令已于17:22:03执行一次，恢复时禁止重复启动：

```sh
/opt/data/private/ZHY/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python \
  /opt/data/private/ZHY/script/supermimic_q1_reproduction/leg_control/run_single_motion.py \
  --root /opt/data/private/ZHY \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/BONES_RF06_UP01_E4096_LEGS50_MAX01H_20261026 \
  --code-commit 0b1da033148fff0b705fd85cdac5a773b5a7f332 --expected-oom-kill 0 --detach
```

集成验证的复现入口为`leg_control/verify.py --root /opt/data/private/ZHY --output <新的独立验证目录>`。C02已完成，不重复运行。权重/fixture/大轨迹留服务器LOCAL_ONLY，不拉Mac、不冒充网盘备份。

## 下一步与恢复

RF05和RF06均已交付，目标不保证性能改善；本轮负结果是提高腿部限矩尚未解决不迈步/抬手不足。下一优先候选为[师兄X2源码对照](README_SUPERMIMICHP_WATCH_20261026.md)发现的actor根位置/朝向反馈接口：当前Q1根位置误差只进入critic，既有anchor_relative_obs尚未接入actor。源码差异是可验证假设，不是已证实唯一根因；未实现或启动RF07，不自动增加数据门或训练预算。

结果小记录已发布Q1并独立核远端373088b2d75bd50827f0c857b005d9ba6b826631，16文件232478字节Mac/25252大小SHA一致；[最终收尾审计](../data/manifests/supermimic_q1_MG02_completion_audit_20261026.json)17项要求全部通过，该收据及最新完成文本的提交身份见Git历史。全量权重/大轨迹仍为服务器LOCAL_ONLY。q1-2048训练心跳已暂停，独立supermimichp-x2每小时仓库跟踪继续，yuanqi不变。完整证据表见[MG02](README_SUPERMIMIC_Q1_MG02_LEG_GOAL_20261026.md)，恢复不重启任何已完成或旧STOPPED/FAILED实验。
