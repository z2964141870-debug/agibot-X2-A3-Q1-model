# Q1低延迟探索：冻结世界模型接口与下一项验证

2026-11-07。原共享训练继续保持用户暂停；本阶段没有新训练、物理回放或CUDA调用，没有新建goal或agent，采纳仍为null。

## 给用户的done与todo

done：教师latent监督已完成3000更新，latent验证误差下降9.255%，动作验证风险却比初始高17.168%，仍选择update0。训练六宫图、原值和两段明确标注为历史失败的原速视频已经归档；[证据ZIP](../data/deliveries/Q1_TEACHER_LATENT_R2_EVIDENCE_20261107.zip)。本轮结果已发布到Q1，完整提交为`ee0639a553551d3c617e9e280b2dd32d34192616`，远端40位和Mac十一份782301B均已独立核同。

done：找到了与保存的服务器授权清单逐字节一致的十五份本地源码，共296205B，完成静态接口核查。真实Q1世界模型是state269/context62，不能直接套用上游state240/context33接口；末尾29维命令缓存不能冒充物理动作梯度。

todo：恢复后先实核现场，再用实际32k世界模型和保存的训练/验证转移检查物理预测及动作梯度。通过后再登记同曝光的独立误差差分训练对照；当前尚未执行该诊断或新训练，不再延长已失败的同数据latent拟合。

## 原始证据与适用范围

新增[只读审计入口](../script/supermimic_teleop/wm_interface_audit.py)只使用标准库、JSON和AST，不import MorphMimic、加载checkpoint或初始化CUDA。依据Mac已保存的PAIR02 `authorization.json`，对九份v0.1核心源码与六份本项目适配源码核验原服务器路径、大小和SHA；前后复核文件与授权原值一致。[实际十五文件清单与函数位置](../data/manifests/q1_frozen_wm_interface_20261107.json)状态为`LOCAL_FROZEN_WM_INTERFACE_AUDIT_PASSED`。

这证实本地副本对应历史冻结输入，不是新的服务器现场核验，也不是实际32k权重的梯度或预测精度认证。三个聚焦检查实际全过/0.001秒：同大小篡改拒绝、嵌套来源身份冲突拒绝，以及正确解析嵌套NativeWorldModel覆盖方法/梯度装饰器。没有重跑旧45项检查或已成回放。

核心文件为上游`model/world_model.py`、`state.py`、`rewards_torch.py`、`policy.py`、`nets.py`、`train/imagination.py`、`normalizers.py`、`trainer.py`及constants；适配文件为`bootstrap.py`、`model_adapter.py`、`canonical.py`、`assets.py`、严格推理加载器和`q1_native_review.py`。

## 实际有效接口

| 项目 | 冻结代码所定义的语义 | 下一入口必须保留 |
| --- | --- | --- |
| 原生状态 | 前240维为29关节位置/速度和14身体世界位置/四元数/线速度/角速度；后29维为上一命令 | state269；不能把79维物理trace直接送入WM |
| 物理trace桥接 | 原`native_state`用同一Q1 FK和稀疏关节映射转换79维状态，57:79作为原22关节命令缓存 | 逐行核当前状态、源clip/frame、实际执行动作和下一状态对应关系 |
| 上下文 | 摩擦1、编码偏差29、torso CoM偏移3、固定延迟秒29，共62维 | 使用保存的实际上下文；禁止默认为零或重估；原缩放含延迟×50 |
| 动作 | 29维原始canonical策略输出，7个缺失关节投影为0 | 不能再乘一次执行器0.5比例，或用同状态教师查询替换实际执行动作 |
| 归一化 | 268维canonical物理特征按保存mean/std处理；缺失关节feature mask、重力/身体高度、形态输入由原代码处理 | cfg的frame、残差尺度、形态154和所有buffers严格从原checkpoint加载 |
| 一步结果 | NativeWorldModel.step返回`physical240 + projected action29`；下一命令缓存确定性更新 | 预测误差和可导性必须单独报告物理部分，不能只检查269维总梯度 |
| 误差坐标 | 原相对身体目标按robot anchor xy/yaw对齐；anchor全局误差另计，四元数wxyz | 两个时刻使用同一对固定比较坐标；不能直接差分两个各自移动坐标系的特征 |
| 梯度 | 源码step无no_grad，原imagination只冻结模型参数，仍允许梯度传回动作 | 不调用包在inference_mode里的NativePolicy作为训练路径；实际权重物理梯度尚待测 |

即使动态网络预测完全不依赖动作，命令缓存的恒等更新也会让“269维输出对动作有非零梯度”成立。因此下一次实际检查必须只看关节/身体物理预测，对存在关节检查有限非零响应，对缺失关节检查零响应，并用有限差分核对。源码声明可导不表示实际训练权重的物理Jacobian已经合格。

## 下一项有界只读诊断

先查输出、日志、派发收据与实际进程，防止重复提交；再核原root/host/boot/GPU、原共享暂停与旧五PID退出、GPU原锁/计算进程、cgroup实际限额/总量/OOM及磁盘。沿原22GiB准入/24GiB退出、GPU18GiB/80度、100GiB磁盘余量、private≤500GB、非阻塞原锁和自有子树回收。复用原R2缓存边界，不重新advice、drop或删证据。

拟议诊断只读最多256条原train/validation保存转移，测试集不读；控制器总300秒、CUDA调用至多512、输出64MiB，0新物理、0优化更新。必须先登记精确选样及实际来源，发布诊断实现/清单并通过原运行时与资源门，不能据本README直接派发。

诊断分开记录：原WM对实际执行动作的一步物理预测与identity基线；物理关节/身体位置与姿态的误差分布；存在/缺失关节的物理动作梯度；固定方向有限差分与autograd一致性；原WM/教师/decoder/normalizer前后SHA。命令缓存误差另列，不参与物理精度或梯度通过判断。训练/验证分开、终止和clip边界逐行mask，不用整体均值掩盖失败来源。

如实际WM预测或梯度不可靠，先记录失败，不直接叠加模型内优化。若通过，再独立登记`L_action + lambda * L_error_delta`，保持初始化、来源、batch曝光、decoder、WM与预算相同。误差差分是本项目候选；未来参考只能作离线监督，不进入f0/history部署输入。不能把该项改称动作差分，也不能仅以新loss下降宣称自然遥操改善。正式选点规则须在新训练前确定，保留初始化候选；闭环、留出事件、接触/侧倾/上肢时序和完整软件周期仍必须验收。

## GigaBrain原始v5纠正

本次直接下载并用PDF技能解析17页原始v5文件：22350232B，SHA`dc9a66864a77beea3fced6a8fa730fe65d262fe4ce0fc0b7f5e584628158f916`，PDF元数据arXivID为2608.18234v5，首页日期2026-11-05。[本地原始PDF](../data/research/2026-11-07/sources/gigabrain_v5_exact.pdf)。文中仍为GigaBrain-WBC-0.5；第7页和附录A第15页明确十帧未来/50Hz/固定0.2秒缓冲。附录B第15页说明独立privileged critic。此前把摘要入口标题差异写成方法名InterTrack，以及未核全文的3.5/3.6节号，均撤回；[研究对照](research/2026-11-07/LOW_LATENCY_SUPERMIMIC_REBASE.md)已纠正。

固定公开源码提交`fa44c2418e7be86c97655e5cb0c809213b2c7dca`工作树干净。`tracking_body_linvel_error`与`tracking_body_angvel_error`直接比较reference/robot世界系速度；tracking.yaml引用这两个项。原始v5也未给出本项目拟议“跟踪姿态误差差分”目标，不能再把该目标当作已经移植的论文性质。附录A参考接口没有单独列参考速度，但包含帧间根平移；不能据此宣称运动信息或未来要求被去掉。下一状态/下一行为联合预测与命令可行性过滤仍是可研究机制，但本轮小型辅助头的负结果不等于复现或否定整套BWM。

## 连接与成本恢复点

Agent本轮自主原样`source /Users/yu/.zshrc`后`ssh -tt hp_han`，19098实际进Windows，并经第二跳原生交互认证收到原Linux `root@interactive67462:~#`。之后提交身份/cd/stty命令没有返回新核验回执；发送只读WM库存检查时，客户端已明确Timeout255退出。没有获得新的boot/资源/源码库存结果，不以登录提示代替完整现场核验。

Agent随即自主同入口有界重试90374，在Mac→Windows TCP阶段明确Operation timed out，未Windows shell，未尝试Linux。两句柄已结束，不再探旧句柄；这不证明Linux或后台停止。下一周期仍由Agent自行尝试固定入口，不要求用户先测网络、不读取或保存文件认证，不换IP/第一跳参数、代理、转发或socket。本阶段新服务器文档镜像仍待实际回执。

最新已结算账本仍1371/14400秒、r3已250/7200秒，剩13029秒；此次本地静态源码核验和PDF阅读未发生服务器GPU持锁，不增加训练账本。原LATENT02和控制器已完成并回收的收据保持，不因观察会话退出重发。完整目标尚未完成，原final门和采纳null保持。
