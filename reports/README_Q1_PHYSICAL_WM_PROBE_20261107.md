# Q1冻结世界模型的保存转移校准

2026-11-07。当前状态：独立实现和八项本地检查已成，实际32k权重预测/梯度诊断尚未执行；原共享训练保持用户暂停，采纳仍为null。

## 用户版done与todo

done：Agent自行按`source /Users/yu/.zshrc`再`ssh -tt hp_han`进入Windows，再用原生第二跳交互认证到原Linux。18:01北京时间取得原boot/GPU/root/旧训练PID退出、GPU计算空、OOM0与约8.0GB容器总内存的真实回执。该会话后来明确Timeout；一次同入口自主恢复33391又进入原Linux。没有要求用户先测网络，也没有将单条会话结束泛化为服务器停机。

done：找到并实核了保存物理数据对应的原模拟器与randomization源码。保存NPZ没有逐帧context，不能将任意零向量称为实际context。新入口在原`randomize=False`采样源码和固定XML下重建名义上下文，摩擦读实际脚底geom，明确标注为源码/XML重建而非记录张量；遇到未知延迟或不一致摩擦即拒绝。

todo：发布并部署独立入口，通过原运行时/来源/资源准入后，只做最多256个保存转移和512次CUDA WM.step前向的校准与梯度检查。诊断通过后才考虑单项误差差分训练；本阶段不增加物理、渲染或优化更新。

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
