# Q1 对齐 SuperMimicHP 与 SOMA 接入

2026-10-26 23:05，北京时间。AP01准备阶段已完成：SOMA接收、直接Q1转换、运动学/GPU时序核验和完整4096资源实测通过。下一独立RF08尚未启动，发布/部署核验后执行。用户再次确认env=4096；每轮最多3600秒，2/4/8小时待本轮效果后决定，不自动延长。

## 目标与来源

RF07已完成但113/365步、2.26秒摔倒，未形成稳定行走。用户要求转为尽量全面对齐师兄SuperMimicHP的成功手法，按Q1自身模型适配。固定来源`humanplus-ai/SuperMimicHP`提交`310244b73e45775744a75b789eb25a6c9c5a81c9`；纯源码72文件及LICENSE/NOTICE保存在被忽略的`data/references/supermimichp/AP01_runtime_20261026`。未执行上游安装脚本，保留共享learner原实现；适配代码位于`script/supermimic_q1_hp/`。这一阶段同时改变观测、控制、数据和后端，不能当作单变量消融。

只在用户25252容器的`/opt/data/private/ZHY`工作，沿用四目录，不开子agent、不操作师兄任务。25677两次登录未成功，内容UNKNOWN，未认定其有或没有相关文件。

## 输入与数据处理

| 输入 | 实际核验 |
| --- | --- |
| `data/references/bones_seed/inbox/soma_uniform.tar.gz` | 45188088297字节，完整SHA256 `0c1fd91628d2644d0d7c0b735ae64faa5fa79834280bc1ce7860a871e55f912c`匹配；从private顶层同盘移动，无新增副本 |
| `soma_base_skel_minimal.bvh` | 23062字节，Mac和服务器SHA `d844abc909b12a1cd8bbeda93fd72473f5278fb8408946f8167d1519f72957a2`一致 |
| `seed_metadata_v004.parquet` | 4499876字节，SHA `7cbe5624f91ae19af3feee69cff74b2cf553fb26367c8edc4b674c3d8ea97e7d`；已在原metadata目录 |
| GMR `lafan_vendor/utils.py` | 实际下载10594字节，SHA `1181093a4641a7981e3e83589d32755b8b17301862d043f40c9fcbcb3926ac5e`匹配上游记录；仅必要小工具，不安装全库或SMPL模型 |

已按师兄发布的metadata条件、演员hash划分、归档顺序和类别候选配额流式提取181条SOMA；没有全量解包。复用其SOMA加载器：translation channel替换OFFSET、声明的Euler顺序、canonical校准、cm到m、120Hz按4取样为30Hz。复用Mink求解器、支撑点/接触锁定、按目标机器人骨长缩放，明确映射Q1固定腕和无头关节。Q1脚含球形碰撞，用Trimesh细分3球面选支撑点，最终足底投影仍用解析球体高度。

181候选转换完成，按类别及SHA(name)选出100训练/15验证，配额无缺项；不使用X2 QC或代理额外几何硬门。仅保留正确读取、29列Q1状态、时间/数值/关节有效条件。训练类别walking28/jogging22/turning16/gesture16/dancing14/jumping4；验证3/3/2/3/3/1，演员集合不重叠。师兄实际pilot/selection清单未取得，因此不是其逐条相同的100/15名单；差异不阻碍本轮。训练MotionLib按原时轴重采样到50Hz。

数据根`data/supermimic_for_q1/AP01_SOMA_Q1_20261026`：`pilot/selection.json`、`q1_candidates/retarget_report.json`和`pilot_manifest.json`保留完整处理证据。Git小清单为`data/manifests/supermimic_q1_AP01_soma_library_20261026.json`；原数据不入Git、不拉Mac。原包只做过一次完整hash，不重复。

## 已实现的方法对齐

| 项目 | RF08独立实现与边界 |
| --- | --- |
| actor观测 | heading水平XY误差+相对根朝向进入10帧观测历史；Q1单帧80维 |
| 未来参考 | 第1至10帧相对根位移、完整朝向、速度和22关节信息；59×10=590维 |
| Q1状态/context | 57/26维，22关节自然顺序、14跟踪刚体，包括固定腕 |
| 控制 | 模型惯量按师兄10Hz/阻尼比2公式计算隐式PD，完整URDF effort上限，动作尺度`.25*effort/kp` |
| 控制边界 | Q1提供模型各轴armature均.004，故kp15.791367/kd1.005310数值相同；不是厂家逐关节特调，不复制X2惯量数值 |
| 物理 | MuJoCo Warp，sim .005/ctrl .02，实际Q1碰撞模型；保留原后端对部分圆柱碰撞最多单接触点的警告 |
| 规模 | 用户明确4096环境；WM1024×10、actor1024/H32、critic5×4、UTD.4；师兄实际环境数1024，区别明确保留 |
| 预热/网络 | 64步采集预热、500组WM-only、BF16/compile及原网络/优化器/奖励/终止，seed0，从零 |
| 预算与出口 | 正式训练软3500秒，外层3540秒触发退出保护+45秒宽限，最多3600秒；无自动续训/重试 |

新Python为`data/environments/supermimic_q1_hp_20261026`，torch2.9.1+cu128、unilab1.5.0、unisim-core2.1.0、MuJoCo/mujoco-warp3.11.0、Warp1.16.0、Mink1.3.0。`activate.py`在进程启动前提供继承的CUDA库路径。旧环境未升级；上游当前锁文件CUDA13不适配570驱动，保留本轮cu128差异。继承Isaac/torchvision等有版本冲突，未称全环境pip check通过；本轮实际链路以下列计算验证。

## 实际验证

Q1运动学24随机状态：26含固定link刚体/14跟踪体，最大位置/朝向/线速度/角速度误差分别4.39e-7/8.58e-7/1.15e-6/2.32e-6，关节梯度有限非零。首版测试把COM速度与link原点速度相比较而失败，测试改为原点Jacobian后通过；没有据错误比较改FK公式。收据`supermimic_q1_AP01_fk_contract_20261026.json`。

GPU V01遇到包装器未init_state而失败，未作优化；修复后V02真实16环境核观测80/goal590/context26/state57、根XY、未来第1帧、动作后f+1奖励和有限性通过。随后REPORTING_V01真实1152转移/0optimizer，最终回合数/回报和长度求和记录通过，不用瞬时终止率冒充回合率。

4096完整P01于22:51:19至22:56:28执行：64步262144真实转移、10次WM1024更新、1次actor1024-H32及critic更新，全部有限；BF16/compile、replay1000均实际启用。Torch CUDA分配峰值6259699200字节（5.83GiB）、保留6782189568字节（6.32GiB），不包含Warp等非Torch分配；运行nvidia-smi曾观测12252MiB，不能把Torch峰值称整卡占用。worker12048正常reap/返回0，controller12044退出，GPU回落14MiB，OOM0，末次共享内存12173934592字节、盘余1666156916736字节。实测不是正式训练或性能结果，权重不复用于RF08。收据`supermimic_q1_AP01_profile_20261026.json`，guard和回合记录小收据已单独归档。

发布前修复controller结束状态路径与上游源文件循环变量重名的问题，避免终后写入覆盖冻结源码；训练算法未变。六宫格CSV使用实际采集sim_steps，区别于UTD=.4下的学习iteration×环境数。最终评测/绘图入口编译及接口检查完成，完整终后链路待本轮结束，不冒称已端到端验证。

## 可复现入口与下一步

使用新Python运行`script/supermimic_q1_hp/activate.py`后运行`build_runtime.py configure`、`retarget_soma.py`、`verify_runtime.py`与`worker.py --mode smoke`；精确输入、版本和命令见脚本CLI、各现场收据和固定来源JSON。已有输出均单次使用，不为恢复重复执行配置/转换/profile。

下一正式入口见[RF08一小时训练](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)。发布到Q1、独立核远端40位SHA、逐大小/SHA部署后启动；保留100GiB余盘、12GiB单实验、30GiB共享cgroup、OOM0、86度和900秒无进度保护。终后自动初末全115片动力学/移动和上肢分析、六宫格与MP4；几何近地比例如实区别于实测接触力。完成训练和性能改善分开报告，不保证行走成功。

RF07旧终态、比较与11项本地交付已写阶段README，不重做旧仿真/图像加工。旧q1-2048更新曾返回不存在，不伪报ACTIVE；新RF08监督需以实际创建结果为准。supermimichp-x2及yuanqi本轮不改。阶段代码/结论/小清单提交指定Q1，原数据/模型保持服务器LOCAL_ONLY。
