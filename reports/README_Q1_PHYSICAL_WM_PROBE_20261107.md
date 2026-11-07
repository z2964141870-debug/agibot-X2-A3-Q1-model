# Q1冻结世界模型的保存转移校准

2026-11-07。当前状态：WM_PROBE02于18:47实际完成，来源和回收门通过，一步预测校准通过、物理梯度差分筛查未通过，整体科学筛查FAILED。原共享训练保持用户暂停，采纳仍为null。

## 用户版done与todo

done：Agent自行按用户固定入口进入Windows及原Linux，沿33391完成实际GPU诊断和原值传回，不再要求用户先测连通性。256条保存转移/368次WM.step/4次物理VJP已完成；验证一步关节误差为保持当前状态的9.58%和11.72%，身体位置误差为4.80%和6.93%。

done：预测校准通过，但88个存在关节梯度检查仅4个通过，整体科学筛查FAILED。实际BF16配置及全部差分原值保留；六宫诊断PNG/PDF、原值CSV和可追溯包已生成，没有把诊断冒称新训练或自然遥操验收。

todo：后续[同权重BF16/FP32独立精度对照](README_Q1_WM_PRECISION_20261107.md#真实完成结果)已于19:18完成；原阈值下FP32为88/88、原生为4/88，原生FAILED未改写。下一单项物理特征蒸馏设计已登记，精确控制审计与实现/发布/准入尚待，实际新训练为0；精度筛查不等于自然遥操验收。

## 为什么先做这个检查

原latent监督3000更新改善了latent MSE，却使validation动作风险退步17.168%，选中update0。[原结果与六宫图](README_Q1_TEACHER_LATENT_STUDENT_20261107.md)保留，不追加同配方拟合，也不重复精确相同的七条失败物理。

[十五份冻结源码审计](README_Q1_FROZEN_WM_INTERFACE_20261107.md)确认有效接口为state269/context62。WM下一状态末29维直接等于投影动作；只对总269维求梯度会在物理网络完全不响应动作时产生假阳性。本入口必须排除命令缓存，分别检查实际物理预测和物理动作导数。

这仍是共同Primebot CPU物理保存数据上的迁移校准，不是原MorphMimic训练动力学内的WM拟合认证。失败不能直接解释成原训练世界模型整体无效；通过也不代表长滚动稳定或自然遥操合格。

## 精确样本和预登记门

来源限定原R1和AGGREGATE01已有train/validation，不读取test或新采样。每个origin/split最多64行，共至多256行；先按source配额，再在合法行中均匀选取，原始clip/frame/dataset-row全部记录。文件数组会为来源核验、选样和状态对应读取，不声称只读了256行磁盘数据。终止行、跨片及状态/teacher/实际混合动作错配均拒绝。

动作必须为保存的真实executed canonical29，不能改用同状态teacher查询，也不重复乘0.5执行器比例。当前和下一状态通过原native_state/FK转换为269维；下一命令缓存必须与记录执行动作核同。实际32k WM按原metadata和plan严格加载，保持cfg frame、残差尺度、morph154、presence、height、normalizer和全部buffers；参数冻结但动作输入保留autograd。

延迟模型的开关与本段实际延迟分开。原Primebot可以开启延迟建模，但保存采集的`randomize=False`沿固定`nominal_context`把delay设为零；不能仅因开关True就拒绝这个已知名义条件，也不能将True误读成实际存在20ms延迟。未知开关类型拒绝，实际零延迟结论仅用于本次固定采集上下文。

预测报告逐例和逐source的存在关节位置/速度、世界身体位置、符号无关四元数角、线/角速度，以及单列命令缓存。identity为原当前物理状态。每个validation origin按source等权；预登记研究筛查要求关节位置和身体位置均值不超过identity的0.95，且任一source不超过1.25。阈值在新结果之前公开，不是硬件或机器人质量验收标准。command-memory不参与筛查。

每组首个选中状态做一次固定当前坐标系的物理特征VJP，归一化沿原268维std，排除缺失feature和命令缓存。22存在关节各用0.01与0.001两个中心差分步长；至少一个步长须满足绝对0.002加相对5%容差。每个关节至少在四状态之一有大于1e-7的物理导数；七个缺失关节导数和扰动物理响应须严格为零。这是四个标量VJP和逐关节差分，不称完整Jacobian。四组预测和梯度按当前实现共368次WM.step前向，硬上限512；CUDA原语kernel数量不据此虚构。

## 工程边界

独立四执行文件：`wm_probe.py`、`wm_probe_worker.py`、`wm_probe_data.py`、`wm_probe_math.py`，另有固定JSON计划和八项聚焦检查。原旧执行清单/200冻结输入/训练/保存频率/动力学不改。测试覆盖命令缓存假梯度、被detach的物理梯度、正确有限差分、缺失关节、调用预算、真实行为与终止mask、XML摩擦、四元数符号和坏source不能被均值掩盖；本机CPU夹具0.093秒全过，不等于真实权重测试。

总300秒/输出64MiB、原22GiB准入24GiB退出、GPU18GiB/80度、100GiB磁盘余量及private500GB保持，使用原非阻塞GPU锁、自有子树守卫和回收。复用原R2缓存边界，不再次advice/drop或删证据。GPU持锁时间计入原r3/总四小时持久账本；开始前累计1371秒、r3已250秒，无重置。

先核output/log/dispatch/receipt/实际进程防重复，再原身份和暂停/资源、源码/计划、原运行时与独立准入发布，最后单次派发。worker诊断完成可返回筛查FAILED，控制器仍如实记录诊断已完成/source PASSED；实现错误与模型筛查失败分开。训练、物理、媒体均为0，本阶段没有新训练六宫图或新视频可宣称。

发布、部署、运行时检查、准入、实际结果和三端镜像分别以随后收据为准。原final门及采纳null保持；不新建goal或agent，不让用户先做连通性确认。

## 真实部署与准入

独立实现`f417c8475b77295904404c6b5746d032dfb5ebe8`、名义零延迟修订`1566834e3634eb9d73b0f4c68a42c77365237298`及清单`e28dd7febf5fe96bb13f141f967d1eb564d7591a`已显式push并独立40位远端核同；修订前代码23215B归档保留，修订三成员15237B包SHA `be0bafa3c156eb94e45d1f7463b2aa46808bfc07ae5f5c7eb2097fff9f68d1b8`部署时旧751份保护来源前后同。原共享/冻结200输入没有修改。

原运行时八项检查通过。额外实际权重CPU加载首次调用遗漏项目原有Git环境，原失败原样保留；随后沿`preview_review.environment(runtime)`的原Git可执行路径与CUDA禁用环境，于18:19:40完成严格加载，20.561966秒、Q1 present22/state269/context62/CUDA未初始化。WM全张量SHA为`4eccbe169cf22ce7729930a8d45c763fe0646c0a957b3b9e39d22beff0ebf330a`。没有更改全局Git或SSH设置。

首条CPU准入命令误用了缺numpy的系统python，在模块导入处即退出；改用固定Primebot Python后通过。零延迟能力修订的八项检查又在原Morph运行时通过；这是代码变更后的必要检查，未重复旧物理或训练。

18:26:00修订准入：175执行/200冻结/731保护输入和原锁通过，cgroup8107569152B/限34292629504B、GPU1MiB25度/计算空/OOM0；原五训练PID退出及STOPPED_SIGNAL由准入重新确认。累计成本1371秒/r3已250、全SETTLED。原输出与派发仍未创建，不据准入称诊断已完成；发布本准入后唯一派发新WM_PROBE01。

七份原始回执共17254B已沿现有SSH传回Mac并逐件大小SHA核同，4784B八成员归档SHA `140aa3b3418271551e10561362230ca6a2750285ff2938d9c927508e33f0c1d2`。[初部署](../logs/session_records/q1_physical_wm_probe_deployed_20261107.json)、[原检查含首次加载失败](../logs/session_records/q1_physical_wm_probe_native_checks_20261107.json)、[真实严格CPU加载](../logs/session_records/q1_physical_wm_probe_strict_cpu_load_20261107.json)、[初准入](../logs/session_records/q1_physical_wm_probe_admitted_20261107.json)、[名义上下文修订部署](../logs/session_records/q1_physical_wm_probe_r2_deployed_20261107.json)、[修订八检查](../logs/session_records/q1_physical_wm_probe_r2_native_checks_20261107.json)、[修订实际准入与完整参数](../logs/session_records/q1_physical_wm_probe_r2_admitted_20261107.json)。新说明服务器镜像仍待回执；认证材料未记录。

## 首派与共同物理脚底映射修订

准入发布`c1e899af9762452f9915819b7dda5f0241bf3329`已显式push、独立40位远端与Mac十二文件765098B核同。33391会话再次返回原host/root/boot后，唯一派发357419/start36045340；真实守卫于18:32:39启动357492，18:33:03退出1并完成回收。[派发原收据](../logs/session_records/q1_physical_wm_probe_dispatch_20261107.json)、[原result](../logs/session_records/q1_physical_wm_probe01_result_20261107.json)、[原guard](../logs/session_records/q1_physical_wm_probe01_guard_20261107.json)及[原错误](../logs/session_records/q1_physical_wm_probe01_worker_failure_20261107.log)保留。

错误来自本诊断的资产名称假设：Morph资产构建器把脚底重命名为`left/right_foot[1-5]_collision`，共同Primebot XML保留无名mesh和四角sphere。名称正则匹配为空，`nominal_context`在样本与WM.step之前拒绝；这不是模型预测筛查失败。该诊断代码改为按左右`ankle_roll_link`上的collision-enabled geom选取，排除visual和其他身体上即使匹配旧名字的geom，读取全部实际XML摩擦并要求两脚完整/同值/有限。原固定采集`randomize=False`没有重写XML摩擦；仍不能把随机化向量的默认friction=1替代实际接触参数。

九项本地检查0.037秒通过，新增无名脚底/不同身体假同名、视觉geometry过滤、缺一脚和不同摩擦拒绝。只修独立`wm_probe_data.py`与检查，原训练资产、动力学、采集和原失败输出不改。必须先发布新清单，在原服务器跑修订检查，并用原compiled XML验证context和十个实际碰撞geom，再在新`Q1_LOW_LATENCY_CKPT32000_STUDENT_R3_WM_PROBE02_20261107`根运行；复用已成严格权重加载证据。

18:36:25[退出后独立实核](../logs/session_records/q1_physical_wm_probe_first_failure_20261107.json)：175执行/200冻结/731保护输入SHA全同，两个进程已退、原GPU锁可独占后释放、计算空、原共享暂停/OOM0。总8018432000B/限34292629504B、GPU1MiB25度；25秒持锁已SETTLED，累计1396/14400秒/r3已275/7200秒。实际WM.step/新物理/优化均0，CUDA可用性检查不冒称执行了模型预测。六原文件4530B七成员归档SHA `eaeb8dc80bcb42e75f2df01a7060161e24cfa201bedc1aa0b95c222fec825f20`已Mac逐件大小SHA核同，原大authorization继续保存在失败根。新文档服务器镜像待，后续诊断结果另记。

## 共同物理修订的真实准入

修订`15d6acb4f3cf4e6a22b54c14384508d1434862b7`与175项新清单提交`5cb124c8c0e45100f9e25ef364ad91fa5e5f5a5b`已显式push并独立40位远端核同，修订十三份782632B Mac镜像逐件核同。三成员15314B代码包SHA `9045e7a7cd9834900089b5732d3334f0c41b5fc120fb6cae2ccdb2c6104bda53`沿33391部署，除两份已审定诊断文件外746份旧来源前后SHA同；旧失败根和日志保持。

原Morph运行时九项检查全过/进程2.430354秒（测试本体0.071秒）。另只编译实际原XML、没有创建模拟器或加载WM：左右踝分别找到geom15..19和31..35，共十个无名碰撞geom，摩擦均1，重建62维context为`[1,0,...,0]`且CUDA未初始化。这个1是本次真实XML的测量值，不是套用随机化默认值。

18:43:45独立准入重新通过175执行/200冻结/731保护输入、原暂停与GPU锁；总8107974656B/限34292629504B、GPU1MiB25度/OOM0、原累计1396秒/r3已275，全SETTLED。新根/派发尚未创建，待发布本准入后唯一WM_PROBE02；256样本/512前向/300秒/64MiB和原资源边界不变，严格WM CPU加载不再重复。

[部署](../logs/session_records/q1_physical_wm_probe_r3_deployed_20261107.json)、[原运行时九检查](../logs/session_records/q1_physical_wm_probe_r3_native_checks_20261107.json)、[实际编译XML context](../logs/session_records/q1_physical_wm_probe_r3_compiled_context_20261107.json)、[完整准入参数](../logs/session_records/q1_physical_wm_probe_r3_admitted_20261107.json)四原始回执8623B已Mac逐字节大小SHA核同，3000B五成员包SHA `c30e1111b70af253a5d2fcab8e1065bd7120a55e79068a63782547c6b4ee2703`。汇总见[来源与准入发布记录](../logs/session_records/q1_physical_wm_probe_r3_runtime_admission_20261107.json)，服务器新文档镜像仍待。

## 真实诊断结果

准入发布`e4b637774e594004decfe8c0612145687578f162`已显式push、独立40位远端及Mac九文件766045B核同。复用33391，确认原host/root/boot后唯一派发357888/start36128560，worker357960于北京时间18:46:32至18:47:04完成。控制器状态`WM_PROBE_COMPLETED`、守卫`WORKER_COMPLETED`且return0/reaped、来源门`PASSED`；科学筛查`FAILED`，二者不能混为执行故障。

实际使用256条保存train/validation转移、368次CUDA `WM.step`和四个标量物理VJP。末29维命令缓存未纳入物理预测门或梯度。每个validation origin包含六个source，各source均值等权：

| 验证状态来源 | 关节WM / identity (rad) | 关节比值 | 身体位置WM / identity (m) | 身体比值 |
| --- | --- | --- | --- | --- |
| 原教师状态 | 0.002325322 / 0.024267059 | 0.095822 | 0.000448462 / 0.009339778 | 0.048016 |
| 教师学生混合状态 | 0.002896251 / 0.024707958 | 0.117219 | 0.000545368 / 0.007870760 | 0.069290 |

一步预测校准门通过，包括逐source上界；原始六项物理误差与全部source保留。梯度门未通过：四组各22个存在关节，共88项，仅4项至少在一个预登记epsilon通过，四组VJP均未全过。22个存在关节都曾有非零物理响应，七个缺失关节的导数和扰动物理响应严格为零。该结果排除了只靠命令缓存造成的假阳性，但没有给出可直接用于训练的可信物理梯度。

实际原配置`bf16=True`。例如original/train的joint0：autograd=-9，epsilon0.01中心差分约-9.789705，epsilon0.001差分为0。精度量化是待验证假设，尚不能认定根因，也不能由差分失败推论原WM整体无效。

33秒持锁成本已SETTLED，累计1429/14400秒、r3已308/7200秒；本阶段新增物理/渲染/优化均0。退出后两个进程已退、原GPU锁可独占后释放、计算空，175执行/200冻结/731保护输入SHA全同。峰cgroup10521755648B/GPU428MiB，退出总8019955712B/限34292629504B、GPU1MiB26度/OOM0。原共享训练保持人工暂停，采纳null。

六个原始文件62818B包SHA `0164c6dc5bec9e09f36e13c1aa2ddf8c586d6b6aa40067e0e1272d6faf183618`已Mac逐件核同。[派发](../logs/session_records/q1_physical_wm_probe02_dispatch_20261107.json)、[控制器结果](../logs/session_records/q1_physical_wm_probe02_result_20261107.json)、[守卫](../logs/session_records/q1_physical_wm_probe02_guard_20261107.json)、[全部诊断原值](../logs/session_records/q1_physical_wm_probe02_probe_result_20261107.json)、[退出实核](../logs/session_records/q1_physical_wm_probe02_completed_20261107.json)和[累计账本](../logs/session_records/q1_low_latency_gpu_cost_20261107.json)可追溯。

独立`deliver_wm_probe.py`只读上述来源，已生成预测六宫PNG/PDF、四组CSV、完整梯度CSV及绘图审计，PNG已目视。它是物理预测诊断图，不是训练曲线；没有新增视频。证据包存于`data/deliveries/Q1_PHYSICAL_WM_PROBE02_EVIDENCE_20261107.zip`，包SHA/成员及链接检查见随后交付收据。服务器新文档镜像仍待实际回执。

下一项[精度对照计划](../script/supermimic_teleop/wm_precision_probe_plan_20261107.json)已登记：严格相同权重，保留原生BF16并创建独立FP32影子；复用同256行、四锚点和原生已存差分，只新增四原生VJP及FP32预测/VJP/差分。保持原差分阈值、TF32关闭和512前向/8反向/300秒/64MiB上限，要求FP32仍保留预测校准。实际FP32调用仍0，须独立实现、发布、原运行时/来源/资源准入；原生FAILED不改写。此处是探索性train/validation诊断，不是未见holdout、闭环效果或硬件延迟认证。
