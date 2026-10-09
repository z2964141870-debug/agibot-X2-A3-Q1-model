# A3 全链路测试与短微调

记录日期：2026-10-10，北京时间。状态：当前独立可执行项已完成，剩余任务全部依赖已登记的搁置项。用户已授权执行完整计划，卡点登记后继续独立工作；短微调最多本实验累计200次更新。旧R01/R02/R03继续暂停，不操作真机。官方PT未取得，本轮官方基线和PPO更新次数均为0，不将准备代码或诊断结果记为训练完成。

## 目标、推进与验收

官方PT校验、四段SMPL审查、A3重定向、官方与动捕仿真、短微调对照、180ms参考缓冲、ONNX及可用RKNN工具链依次推进。任务台账 `data/experiments/a3_fullchain_20261010/tasks.json` 使用pending/running/passed/failed/deferred，保存阻塞证据和用户待办。普通卡点最多定位10分钟；重启只核验boot/进程/工件后恢复，温度90°C/85°C连续两次和无保存进度保护保留。

衣服数据质量未验收时，使用官方selected20完成微调链路：固定种子42整动作划分16训练/4留出，16环境、官方优化配置，2更新冒烟后完整恢复至200，每10次保存；不改变衣服TEST_ONLY标识。结果不等同于真实衣服效果或硬件通过。

## 版本与恢复入口

- 主机 `/media/yu/FAFF-E9771/YUANQI`，分支main；开始HEAD `a96cb4ad8070edccc40e95675a012b83b5171658`。
- vendor `fe6868ba37034f89b912f0fb851bce19f120266d`；官方发布固定revision与SHA来自 `a3_hf_manifest.json`。
- 初始boot `40497006-bd02-449e-931a-dac12218a7eb`；旧service inactive/MainPID0/disabled，GPU30°C、92MiB。
- 四段SMPL来自Q1目录，Q1并行暂存修改保持。依赖、输入和各实验日志 `logs/a3_fullchain_20261010/`，数据与模型均在服务器data下，LOCAL_ONLY。
- 启动台账：`data/environments/a3-sonic/bin/python -m script.a3.fullchain_support init`；后续模块与命令按实际运行证据补充。

## 当前结果与待办

01:57已完成四段输入的数值、旋转、父树、SHA与30Hz规范化，原帧1766/1767/1969/1923，规范化1766/1767/2251/2253。旧录制平均26.23/25.60Hz、最大间隔157.9/178.0ms，不能称实时固定30Hz。新两段trace以frame_index对齐，身体首帧匹配trace下标30，不按数组下标直接配对。

同一时段四段完整A3运动学参考已生成，29主动关节沿固定URDF匹配，Mink/quadprog与URDF速度限位；使用拟合被动脚开链MJCF生成参考，闭链动态评测另用配套loop资产。合成中立姿势检查通过，真实参考为首帧相对校准诊断，保留TEST_ONLY、training_allowed=false；不证明高保真/足接触/平衡。

官方HF连接本轮全部超时且无下载字节，固定官方PT和配套4个小文件均deferred。清单 `data/experiments/a3_fullchain_20261010/official_download.json`，目标根 `data/models/a3_official_035/checkpoints/035_step200000/`：

| 文件 | 字节数 |
| --- | ---: |
| model_step_200000.pt | 402050931 |
| config.yaml | 37231 |
| meta.yaml | 40 |
| model_config.yaml | 9529 |
| LICENSE | 11358 |

源仓库 `sonic-for-a3/sonic`，固定revision `db1f40787d580af6d1e53df0df72242904a1dae5`，远端目录 `035_step200000/`；清单有完整源地址/目标路径/SHA，等待用户迁移或连接恢复后独立验证。服务器既有候选路径搜索未发现step200000模型。Isaac的 `smpl_sim.smpllib.smpl_eval` 不可导入；快速检索未发现本地模块，先搁置。RKNN工具链缺失。MuJoCo/Mink/quadprog/smplx/ONNX/ONNXRuntime存在。

官方模型基线与200更新微调暂被官方PT阻塞；继续参考格式、延迟逻辑、包装器及导出实现测试。所有新工件LOCAL_ONLY；每个重要阶段补充记录，审计后scoped发布main。只有完成或所有剩余项被已登记卡点阻塞时停止。

## 参考、视觉与代码链路结果（02:18）

开链默认qpos0不是官方站姿，初轮数据另存 `mocap_qpos0_diagnostic/` 与 `retarget_qpos0_diagnostic.json`。正式诊断参考已改用固定资产官方keyframe，补上单左膝0.1rad扰动检查，29关节恢复最大误差6.10e-9rad，中立检查通过；速度、限位保持。四份真实A3网格与源骨架抽帧图已生成，单腿图实际视觉检查确认抬腿/朝向变化及穿地，不能据小数值通过称高保真。

| 参考 | 帧数 | IK位置P95（cm） | 脚网格最低z（cm） | 穿地超过1mm的帧数 |
| --- | ---: | ---: | ---: | ---: |
| FGP走路 | 1766 | 14.47 | -14.68 | 503 |
| FGP单腿 | 1767 | 31.54 | -12.45 | 697 |
| 9/23录制 | 2251 | 10.56 | -4.87 | 1938 |
| 9/24录制 | 2253 | 26.28 | -1.83 | 1747 |

这些是A3诊断首帧映射、脚网格及自身IK目标残差，不是FGP相对人体真值误差，不能混为原38–39cm指标。全部限位违例0，但参考质量不合格，衣服微调搁置，官方样例微调准备保留。足滑真实接触验收仍缺可靠接触/地面，未用踝速度冒充足滑结论。

26份CSV已通过官方读取器：4真实参考、2合成、20官方样例；位置/旋转/命名关节与qpos往返及时长核验通过，官方120Hz stride4→30Hz，新参考30Hz stride1。官方训练16/4两部分PKL转换均通过，16个训练输入preflight通过（仅数据入口，不宣称权重加载或训练通过）。

4段因果参考回放共13351个窗口，无未来样本读取；与延迟对齐的离线位置插值最大差4.1e-14m。30Hz数据/50Hz目标格点使实际参考年龄180–220ms；这是模拟到达、参考输入检查，策略性能及现场端到端延迟仍待验证，不能把重采样提前读取原始未来称实时通过。

MuJoCo包装器新增显式checkpoint/motion/采样/长度/输入捕获，保留旧500/2000模式；Isaac新增dataset/config并保留validate。用已有独立验证2950模型只做包装器100步/100真实输入诊断，2秒内已触发跌倒，不能称官方基线。输入按官方tokenizer特征顺序保存，schema为command580→ori6d60→proprioception930，共1570维。首轮捕获顺序错误已纠正并另存新实验，没有覆盖原诊断。

独立微调R04配置已保存：官方200000仅权重初始化，初始优化器/计数清空；固定200更新学习率计划，以回调在2更新冒烟停止，然后完整恢复至200，每10更新保存，无次数/到期限制。当前state为deferred/official_pt_unavailable，没有启动PPO。14项测试已通过，涵盖未来帧边界/缺帧、数据划分、CSV往返、合成单关节、计数/优化器清空和历史训练隔离。

ONNX适配器与捕获的100组PT动作先行核验一致；opset13不支持aten::atanh，首次导出失败已留journal。当前只做一次log恒等式导出兼容修复与数值复验，不修改策略权重。最终结果补在下方。

## 最终验证（02:31）

ONNX第二次导出通过：输入 `[1,1570]`、输出 `[1,29]`，100组真实MuJoCo输入，PT/ONNX最大动作差 `8.344650268554688e-7`，满足atol=1e-4、rtol=1e-3。PT适配器与模拟器动作差 `9.5367431640625e-7`。文件57673235字节，SHA-256 `18eceeda8e42c4a77c9864364a79b8dc0473f3e77a695bdef5485b9a347c3004`，位置 `data/experiments/a3_fullchain_20261010/onnx_diagnostic_local2950_v2/a3_fast.onnx`。这是旧2950模型的MuJoCo A3Policy适配器验证，未验证官方PT或原Isaac导出器，模型LOCAL_ONLY。

评测恢复再次调用同一协议后跳过已验证动作，仍仅attempt_001，没有重复仿真。新增摘要独立核对timeseries、参考SHA及请求步数，保存首次跌倒前和全程指标。旧2950前向慢走诊断首次跌倒1.88秒：跌倒前94步关节RMSE 0.22268rad，全程100步0.22833rad；原动作需1652步，本次仅100步冒烟，动作未完整回放、未稳定完成。不能将计算完成状态passed理解为策略有效。

四份动捕A3 CSV进一步转换为独立命名的29DOF/30Hz MotionLib PKL，逐文件重载、有限性、四元数及帧数/时长通过；原始TEST_ONLY与training_allowed=false保持。官方16训练/4留出PKL有限性与格式复核通过。`smpl_joints`是官方转换器的占位字段，不能当成人体真值；未用于衣服训练。

新微调作业添加独占进程锁、R04路径隔离和重启/监督器中断300秒冷却检查。保持异常/温度停止、连续两次无保存进度停止，无累计启动/到期限制。R04没有安装自动启动服务，当前账本deferred且attempts为空；实际官方权重加载、2更新冒烟和2→200完整恢复都尚未运行。固定200更新学习率计划与短冒烟停止逻辑仅经代码检查和模拟接口测试，不能代替真实Isaac训练验收。

56项项目测试通过（本轮17项，加既有保存/恢复39项），官方RKNN转换器6项测试通过。实际诊断ONNX通过该转换器schema检查；只证明输入契约，不是RKNN二进制包结构或板端推理验证。工具链缺失后搁置，没有生成RKNN模型。

最终主机核验：同boot `40497006-bd02-449e-931a-dac12218a7eb`，CPU43°C、GPU30°C、RAM可用31018774528字节，无GPU计算进程；旧longtrain service inactive/dead/MainPID0/disabled。本轮网络下载和首轮ONNX失败unit保留为历史记录，没有正在执行的评测或训练。Q1原有两份PredActor暂存修改仍在，未修改。最终证据 `data/experiments/a3_fullchain_20261010/final_host.json`。

## 测试矩阵与卡点

结构化台账共22项：11项通过（均有各自验收范围）、1项质量失败、10项搁置，待做/运行中为0。Git清单 [a3_fullchain_20261010.json](../data/manifests/a3_fullchain_20261010.json) 保存任务、用户待办、官方迁移信息及148项代码/产物/日志的大小与SHA；大文件仍在服务器LOCAL_ONLY。

| 任务 | 状态 | 结果或受影响任务 |
| --- | --- | --- |
| 官方PT及匹配配置 | 搁置 | HF连接超时、0字节；阻塞正式基线、微调、对照与官方模型导出 |
| 四段SMPL结构与时序 | 通过 | 数值/旋转/父树、frame_index、SHA及30Hz规范化；仅结构 |
| A3重定向与抽帧渲染 | 通过 | 官方站姿/单关节合成检查、29关节和限位；仅运动学与视觉诊断 |
| 动捕参考质量 | 失败 | IK残差和脚网格穿地，根轨迹/接触/真实姿态精度未验收 |
| 衣服数据微调 | 搁置 | 依赖动捕质量修复，原数据仍TEST_ONLY |
| CSV/MotionLib | 通过 | 26份CSV、4真实PKL及20官方PKL格式/有限性/时长检查 |
| 16/4划分与微调输入preflight | 通过 | seed42、完整动作隔离；只检查输入，未训练 |
| MuJoCo包装器 | 通过 | 旧2950模型100步、真实输入捕获、恢复跳过、跌倒前/全程统计 |
| 官方MuJoCo基线与动捕策略评测 | 搁置 | 官方PT缺失，10步官方冒烟/selected20均未执行 |
| Isaac指标依赖及基线 | 搁置 | 缺smpl_sim.smpllib.smpl_eval，同时缺官方PT；validate未通过 |
| R04 2→200短微调 | 搁置 | 独立配置/保护与权重计数重置实现完成，PT缺失，实际更新0 |
| 微调前后留出对照 | 搁置 | 无正式基线/微调产物，无改善结论 |
| 因果180ms参考输入回放 | 通过 | 13351窗口、无未来读取；实际年龄180–220ms，仅模拟到达 |
| 策略层缓冲与离线未来窗口对照 | 搁置 | 依赖官方PT；实际端到端延迟仍未验证 |
| ONNX诊断 | 通过 | 旧2950模型100真实输入满足容差，无官方导出或硬件结论 |
| 官方/微调ONNX与RKNN转换 | 搁置 | 缺正式PT/训练产物及RKNN工具链，板端未运行 |
| RKNN输入契约 | 通过 | 实际诊断ONNX schema及6项转换器测试，未生成RKNN包 |

## 用户后续处理与恢复

优先把5个官方文件迁移到服务器同一目录。文件名/大小见上表，源固定为Hugging Face仓库 `sonic-for-a3/sonic` 的revision `db1f40787d580af6d1e53df0df72242904a1dae5`、`035_step200000/`。目标绝对路径 `/media/yu/FAFF-E9771/YUANQI/data/models/a3_official_035/checkpoints/035_step200000/`。每个源地址、目标文件路径和SHA均在Git清单的user_transfer中；PT为402050931字节，SHA `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`。不用迁移到Mac；可按用户原偏好通过第三方软件迁移，代理不代发消息。到位后先大小/SHA、CPU重载/网络有限性验证，再恢复正式测试。

其他待办独立保留：提供兼容smpl_sim指标依赖；提供可用RKNN转换环境；处理个人标定、姿态质量和根轨迹/地面接触证据。后两项不妨碍官方样例基线和短微调，不能因为搁置而修改原数据用途。服务器重启只按boot/进程/GPU/产物恢复，不要求先定位重启原因。

恢复先读本README及清单，重新核验主机/Git，然后验证官方文件。代码入口如下，均在服务器项目根执行：

```bash
# 模型到位后会先验证已有文件，只有缺失文件才尝试有限下载。
data/environments/a3-sonic/bin/python -m script.a3.fullchain_support download
# 已完成的数据检查入口。
data/environments/a3-sonic/bin/python -m script.a3.reference_contract validate
data/environments/a3-sonic/bin/python -m script.a3.reference_contract motionlib
data/environments/a3-sonic/bin/python -m unittest script.a3.test_fullchain script.a3.test_checkpoint_store script.a3.test_autoresume
```

官方模型通过校验后先使用 `script.a3.evaluate_mujoco` 显式checkpoint/motion、reference-fps=30、frame-stride=4、max-policy-steps=10做官方冒烟，再以新的输出目录取消步数上限测完整selected20。新30Hz动捕CSV使用stride1。官方模型基线完成后，再用 `script.a3.finetune_job` 启动R04；同作业从最新已校验R04模型恢复，旧R01/R02/R03不能作为来源。真实重载/优化器有限性由checkpoint_store复核。细节命令通过各模块 `--help` 或代码入口确认；R04模块直接运行，不接收CLI参数。官方策略缓冲对照仍需接入并实际验证，参考输入回放通过不代表这部分已经完成。

达到200后使用相同资产/动作/初始状态/延迟做16训练与4留出评测，保留跌倒前及全程指标，再捕获100组实际输入运行 `script.a3.export_check`。只有全部待办有证据后才能改写其状态，本记录没有把剩余任务记成完成。

阶段首个提交 `5cbf3d4d847289f42222e60641aeab42570fb98e` 已同步；本次收尾版本按main最终提交和清单代码SHA恢复。发布前检查diff及暂存区，执行Git载荷审计，只提交本阶段代码/README/清单，不提交模型/数组/PKL。
