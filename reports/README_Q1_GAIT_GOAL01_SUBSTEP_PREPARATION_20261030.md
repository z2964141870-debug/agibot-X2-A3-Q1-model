# Q1逐子步执行层观测准备

记录日期：2026-10-30，北京时间。阶段：GAIT_GOAL01_SUBSTEP_PREPARATION。状态：代码、CPU替身测试、真实模块导入及只读清单准备完成；新物理观测尚未授权、尚未启动，普通自然步态目标未达。

## 目标与已有证据

仍在SuperMimic内分开推进底座与PPO。原底座V02实际301.1074845249823秒、PPO600.0709074319457秒，合计901.178391956928秒，均已按physical_regression结束并交付；唯一V02启动授权已用，不能据本阶段恢复训练、补预算或自行采纳候选。

[离线PD核查](README_Q1_GAIT_GOAL01_SAVED_PD_DEMAND_20261030.md)发现局部踝pitch的控制起点请求会超模型22Nm范围，但多数参考抬脚窗口的控制起点并未超限。原NPZ缺少逐5ms实际力矩，不能据边界重建把贴地归因于整段饱和。[保存PPO链路](README_Q1_GAIT_GOAL01_SAVED_PPO_CHAIN_20261030.md)已确认head和critic实际更新，仍未隔离失败单因。

本阶段补齐可审阅、可单独启动的执行层观测工具，回答实际力矩、卸载和离地的先后关系。实际操作只有本机权威Git和用户25386内的小脚本、CPU替身测试、模块导入与文件身份读取，**0环境构造、0策略前向、0积分、0optimizer**。没有生成新步态效果。

## 实现与时轴

`script/supermimic_q1_gait_goal_analysis/substep_observer.py`只临时包装后端内部`_execute_device_steps`，每个原4子步控制调用中依次观察4次`original(1)`；外层`step_tensor(ctrl,4)`仍只调用一次，维持原ctrl上传、外力跨4子步生效及随后清除、completion/time cache更新次序。钩子单次使用，超界或失败后不恢复；退出恢复原实例属性或类方法，其他钩子的所有权变化会报错。

每个5ms子步分别保存：积分前后qpos/qvel/time/warmstart/qacc/ctrl/施加外力；原生求解的actuator_length/velocity/force、广义执行器力、约束力、bias/passive；同次求解的脚姿态、碰撞脚底高度、足地世界Z力绝对值及力加权接触点水平滑速。原生执行器顺序通过明确ID映射回Q1关节名。

当前Warp的step先forward/solve再implicit积分。因此力矩、接触和脚姿态按该步**积分前**时刻解释，post_qpos/qvel/time为积分后；工具不调用额外forward去刷新几何，也不把post-state误配给前一求解力矩。足力是求解样本，非20ms冲量。接触提取前后逐值保护已列的动态/求解数组；该保护不覆盖全部求解器内部状态。

`probe_substeps.py`提供prepare、check-imports、supervise、worker入口。prepare只读取文件和冻结身份；执行入口要求新决定对应精确plan SHA与独立路线，并使用预登记唯一输出根，不能换名自动重试。外层复用原GPU锁、启动<=24GiB准入、30GiB/OOM/存储/温度保护及进程组回收，每秒检查、1800秒停止阈值/10秒回收，训练optimizer没有构造路径。

## 拟议测量与固定上限

下面是准备好的新测量，**尚未获新运行决定**：

| 路线 | 两个条件 | 每路线范围 |
| --- | --- | --- |
| 底座 | 原initial与原301秒stage_01 actor | 原固定停走320步、起走210步，各条件3重复 |
| PPO | 同原冻结教师、同600秒stage_02 head，置零与开启 | 同上，保持alpha.5、PD/力矩限制/参考/控制不变 |

每路线12个单片单环境case，最多3180控制转移/12720物理子步/6360策略批调用；两线总计最多6360转移/25440子步，0optimizer。策略批调用上限包含教师与head，单环境网络内部仍按原4096块填充执行。图捕获/reset forward属于单列初始化，不能当作新控制转移或训练收益。

每片单独创建环境，避免短片结束后陪长片继续积分而多算物理步。三个重复使用环境/算法配置和全局RNG均在创建前设置的32000/32001/32002，并在rollout前重设；复用现有prepare_seeded，避开旧worker.prepare内部强制seed0。两条件按片/重复配对，这些不是三个训练seed。

首个策略输入、语义初态、可用原生初态/warmstart、实际gain/bias/限幅以及Python/NumPy/Torch CPU/CUDA RNG分别保存、比较；发现差异保留逐字段证据，不重跑。即使这些字段相等，也不称完整求解状态相同或历史330环境rollout重放。新单环境结果应与历史指标区分。

未来保存原始initial_inputs/native_substeps/control_trajectory NPZ只在服务器；小JSON/关节CSV分别报告实际力矩近限样本、足力/脚底时序与原gait/tracking，不能以这些诊断自动推荐checkpoint。近限容差1e-4Nm，仅模型执行器样本，采样时长和不等于连续饱和时长，更非硬件电流/电机饱和证明。

唯一新根分别为data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_BASE_20261030及Q1_GAIT_GOAL01_SUBSTEP_PPO_20261030。尚无这两根的真实观测作业；不重复原全330库、媒体或旧物理预检。

## 已执行验证与失败修正

13项CPU测试最终全过：从当前安装backend.py抽出真实step_tensor和内部循环，在替身上分别检查graph/eager的4次launch、外力生命周期、一次time cache/completion、恢复/钩子所有权、次数上限、部分launch错误的未知边界与禁止恢复；检查pre/solve/post时轴、控制变化/接触读取意外修改拒绝；检查两线配对种子/固定预算/新决定与只读清单构造。

测试未加载或创建实际物理后端，graph/eager也是替身调用，**未验证真实观测前后轨迹数值等价**。真实组件导入检查另返回OBSERVATION_IMPORTS_PASSED，环境/前向/积分/更新计数均0。

保留三项准备错误：首12测试有一项把float32高度与float64字面量逐位比较，差1.49e-9，已改同类型期望值；首次只读清单把runtime目录当文件hash，已改为冻结目录内源码；新合成清单回归测试的supervisor相对路径错，已改真实script布局。均发生于CPU/文件准备，不是仿真或训练重启，也未放宽原物理容差。

冻结清单共281个实际使用来源，包含两线历史检查点/实际用到的原冻结文件及5个当前后端Python身份。原试验未冻结这些后端Python，故当前SHA不能证明历史未变。没有覆盖或改写原训练/奖励/后端文件。

## 命令、工件与发布

权威Git：/Users/yu/Documents/ChatGPT/元启/data/repositories/q1-rf12-20261028，Q1分支；准备基础提交f340821be87bf11ca067bf0cf0eefacf83850eca。服务器唯一根仍/opt/data/private/ZHY，沿原25386/Windows桥，无认证或环境改动。

实际CPU测试命令：

```sh
substep_root=/opt/data/private/ZHY
env CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$substep_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$substep_root/script/supermimic_q1_hp/activate.py" -m unittest discover \
  -s "$substep_root/script/supermimic_q1_gait_goal_analysis" -p 'test_*substep*.py' -v
```

只读prepare已执行一次成功，输出data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_PREPARATION_20261030/observation_plan.json，85901字节，SHA-256 b5f4e7fb211a0f5445549d26b3b0cb943c9168941099164821849b3576db35ac。Git可审阅的[精确plan副本](../data/manifests/supermimic_q1_gait_goal_substep_observation_plan_20261030.json)与该文件同内容；无需重做prepare。

[本轮清单](../data/manifests/supermimic_q1_gait_goal_substep_preparation_20261030.json)及[执行记录](../logs/session_records/q1_gait_goal_substep_preparation_20261030.log)包含实际检查和错误。发布提交、独立40位远端及Mac主根/25386逐大小SHA结果以[publication旁收据](../data/manifests/supermimic_q1_gait_goal_substep_preparation_publication_20261030.json)为准；本报告写入不预称发布完成。

新执行命令的固定形状如下，仅在用户明确新决定并由代理登记对应plan SHA后使用，不能据示例自行启动：

```sh
"$substep_root/data/environments/supermimic_q1_hp_20261026/bin/python" \
  "$substep_root/script/supermimic_q1_hp/activate.py" \
  "$substep_root/script/supermimic_q1_gait_goal_analysis/probe_substeps.py" supervise \
  --plan "$substep_root/data/manifests/supermimic_q1_gait_goal_substep_observation_plan_20261030.json" \
  --line base \
  --output "$substep_root/data/supermimic_for_q1/Q1_GAIT_GOAL01_SUBSTEP_BASE_20261030" \
  --authorization-record USER_DECISION_RECORD --code-commit PUBLISHED_40_HEX
```

另一条PPO线须使用line ppo和其已登记独立根，串行运行；该准备不增加原训练预算。先审阅本方案并决定是否允许两条各一次新零优化观测；目前0新物理测量。goal保持ACTIVE、采纳null，原监督、缓存维护与其他任务保持，未启动/激活agent。
