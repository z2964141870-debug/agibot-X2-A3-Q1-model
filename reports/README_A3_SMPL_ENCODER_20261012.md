# A3 官方编码器核验与自制 SMPL 编码器路线

记录日期：2026-10-12。状态：自制人体编码器/冻结控制桥已实现；11项CPU契约及E13真实128窗口/16官方动作/独立核验通过。人体编码器尚未训练，策略收益未通过。

## 目标与范围

用户纠正此前“微调怎样改善目标”的表达：从官方权重出发取得可验证提升，一直是项目目标，不是本轮新发现。当前没有候选通过完整改善门槛；修复训练精度、能执行更新、支持新输入都不能代替效果提升。需要细化的是目标工况、错误类型和比较指标，而不是重新确认是否希望提升。

本轮回答三个问题：发布版 A3 是否有可用的 SMPL 编码器；是否必须输入“A3 点骨架”；能否自己制作人体输入编码器。只读核验原机源码和新机 CPU checkpoint；无模型前向、CUDA、仿真、optimizer 更新、安装或真机操作。原 hp3090 继续停载。

## 已核验的四层证据

| 层次 | 实际证据 | 结论 |
| --- | --- | --- |
| 发布说明 | A3 README 的发行边界明确说明没有 SMPL/teleop encoder，人体输入需上游重定向 | 缺的是本次发布的可用编码器，不能推断 SONIC 架构不能容纳 SMPL |
| 配置与构造 | 基础配置保留 `smpl`/`teleop` 定义；035 `active_encoders=[g1,a3_fast]`，SMPL/teleop 采样概率为 0；构造器会跳过非 active 分支 | 有配置名字不等于实际实例化或训练过 |
| 实际发布 PT | 新机 CPU 读取 `policy_state_dict` 45 个张量；`g1` 和 `a3_fast` 各 10 个 encoder 张量，没有任何 `smpl`/`teleop` policy key | 手里的官方035确实不含这两条分支权重，打开开关不能获得已训练分支 |
| 部署资产 | 官方发行清单仅列 `g1`/`a3_fast` ONNX/RKNN 对，ONNX 全策略为 `[1,1570] -> [1,29]` | 本轮核对的是官方清单，没有重载部署二进制，也没有验证新增 SMPL 部署链 |

实际 PT 大小 402050931 字节，SHA-256 `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`；2026-10-12 UTC 10:51:08 后重新核对 SHA 一致。两 encoder 的第一层 weight 都是 `[2048,640]`，最后一层都是 `[64,512]`，输出 flattened latent 64 维；这是本 A3 035 的接口，不据此认定其他 SONIC 权重兼容。

原版 NVIDIA SONIC 官方仓库说明提供已训练的机器人、SMPL 和稀疏 VR 编码器，共用动作表示和 decoder。A3 035 的裁剪发行与原版的多输入发布范围不同；不能直接把原版 G1 SMPL 编码器接到 A3 decoder 就认定兼容，同 shape 不保证动作语义和潜空间一致。

## 当前输入并非人体点骨架

固定源码 `TrackingCommand.command_multi_future` 和 `command_multi_future_a3_fast` 实际拼接机器人参考关节位置与速度。035 两条 active 输入分别是对应的 future joint reference 和 anchor-relative root orientation；29 个关节角 + 29 个关节速度 + 6D 朝向，每帧 64 维、10 帧合计 640。`observations.command_multi_future` 的旧 docstring 写 body positions，与实际 property 不一致，本次以 property 和实际运行时 shape 为准，未改只读 vendor。

因此更准确的当前链路是：

```text
人体动作/SMPL -> 上游重定向 -> A3 目标关节角、速度、朝向
                                -> 官方 encoder -> FSQ token
机器人自身反馈 ------------------------------------> 官方 decoder -> 控制动作
```

人体骨架点可作为重定向输入，不是当前官方 encoder 能原样接收的数据。`g1` 是历史 checkpoint/ONNX 名字，本发行仍面向 A3，不代表当前策略控制 G1 硬件。

基础配置中预留的 SMPL 输入也不是未经处理的 SMPL pose 参数：E09 实际元数据保留 SMPL 局部关节位置 `[10,72]`、相对 root 朝向 `[10,6]`、六个机器人腕部关节参考 `[10,6]`。源码腕关节索引是 `[23,24,25,26,27,28]`。这些字段定义不能证明人体输入链已经具备真实、可用的数据。

## 自制编码器的候选路线

可以自制，但首个假设是补齐人体输入并保持官方控制能力，不能先承诺提升。

1. 固定官方035及其 quantizer/decoder，选定 `a3_fast` 为冻结 teacher。使用同一段可靠人体 SMPL 动作及其重定向 A3 参考，统一时间、坐标、人体尺度、根朝向与机器人 anchor。新 SMPL encoder 学习 teacher 的 64 维量化前 latent。运行时学生的可得机器人反馈/anchor 必须与教师标签定义一致。
2. 学生输出经过同一 FSQ，再与机器人 proprioception 一起进入冻结动力学 decoder。先验收 latent、token 和同状态动作偏差，再做完整闭环。量化前 MSE 小不保证 token 一致或机器人稳定；当前部署版 `round` 的零输入梯度由既有 E19 证据支持，训练量化路径和部署路径不能混同。
3. 用独立人体会话、动作和速度，对照当前“SMPL 重定向 + 官方035”完整链路。报告跌倒、joint/root/身体跟踪、动作平滑和实际延迟，保留站立/行走/转向等基础能力。新增 SMPL 接口算能力扩展；只有同任务同输入条件下指标超过原链路且保留门槛通过，才算相对官方系统提升。

这不是从零重训整个机器人控制器，也不是把现有机器人参考适配器改名成 SMPL encoder。已有参考适配试验的 prequant teacher 目标可复用作方法依据，但新机 E02 的完整闭环验收没有通过，不能视为这条新路线已验证有效。

### 未来输入与数据前提

035 A3-fast 输入含当前到未来180ms共10帧，间隔20ms。如果训练时学生也读真实未来，最多先证明离线接口；在线人体输入需单独确定历史因果学生/预测/计入延迟的缓冲方案。不要把录制中的真实未来当成实时已到达信息，也不要承诺更换 encoder 自动消除延迟。

当前公开 selected20 只有 A3 flat CSV，不包含同步人体 SMPL 标签。现有衣服/动捕仍 `TEST_ONLY` 且根轨迹/接触/姿态质量未验收，本次不改为可训练数据。下一步先核查已有资料是否有成对人体来源、时间戳与许可；缺可靠配对时，先准备小型数据合同和输入/输出验收，不盲目开始训练。A3 CSV 不能唯一反推出原始人体 SMPL。

“追随官方 teacher”主要是保持已知行为并建立新输入接口。若要在同样干净机器人参考任务上超过 teacher，还需有目标缺陷、可靠监督/奖励和独立评测；学生蒸馏损失下降本身不能证明提升。

## 版本、命令与证据

- 本轮代码基线：项目 `27d775089f27857b18202af174d19572e14b1661`，vendor `fe6868ba37034f89b912f0fb851bce19f120266d`；vendor 未更新。
- 读取地点：原机 `/media/yu/FAFF-E9771/YUANQI_A3` 的源码；新机 `/opt/data/private/hechengzhu/ZHY` 的官方 PT 和 E09 元数据。
- 小型核验清单：[a3_smpl_encoder_20261012.json](../data/manifests/a3_smpl_encoder_20261012.json)。未创建训练实验编号或新策略工件。
- 既有精度/收益证据：[E09精度修正](README_A3_ZHY_CONSISTENCY_20261012.md)、[R12结果](README_A3_ZHY_FP32_20261012.md)、[E12模块隔离](README_A3_ZHY_MODULES_20261012.md)。R12四诊断动作平均关节误差比官方高4.63%，不升级；这些动作已参与诊断，不是新鲜最终测试。
- 既有适配与量化依据：[E19/E20](README_A3_CLOSED_LOOP_20261010.md)、[冻结控制器试验合同](README_A3_CONTROLLER_ADAPTER_20261010.md)、[新机E02结果](README_A3_ZHY_SEARCH_20261012.md)。

在新机根目录复现权重清单检查，仅 CPU 读取本项目已核验 PT：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 data/environments/a3-sonic/bin/python - <<'PY'
import torch
torch.set_num_threads(1)
path = 'data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt'
state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)['policy_state_dict']
print('policy tensors', len(state))
for name, tensor in state.items():
    if '.encoders.' in name:
        print(name, tuple(tensor.shape))
print('smpl/teleop', [key for key in state if 'smpl' in key or 'teleop' in key])
PY
```

原始来源，本轮读取日期均为2026-10-12：

- [A3发行README，固定提交](https://github.com/AgibotTech/sonic_for_a3/blob/fe6868ba37034f89b912f0fb851bce19f120266d/README.md)：实际发行边界。
- [035训练合同](https://github.com/AgibotTech/sonic_for_a3/blob/fe6868ba37034f89b912f0fb851bce19f120266d/docs/a3_035_training_contract.md)：active编码器、未来窗口、公开语料范围。
- [A3训练/部署指南](https://github.com/AgibotTech/sonic_for_a3/blob/fe6868ba37034f89b912f0fb851bce19f120266d/docs/a3_training2sim2deploy.md)：A3资产、历史命名和机器人参考链路。
- [原版NVIDIA SONIC仓库](https://github.com/NVlabs/GR00T-WholeBodyControl)：已训练SMPL/机器人/VR分支的公开说明；本轮未对其权重或整条SMPL链路复现。

## 保存与下一步

重要核验、方案与边界已保存到本README和小型清单，并更新索引/current state。同步前只记本地已保存；发布提交及远端状态由Git历史和本轮推送核验确定，不在文档中预写成功。既有大型权重/数据仍在服务器 `LOCAL_ONLY`，没有新增Mac模型或原始数据。

恢复先读本README及[E12](README_A3_ZHY_MODULES_20261012.md)。下一步候选为配对SMPL/A3数据与实时输入合同盘点；现有保留官方行为的冻结/约束微调分支仍是独立候选，不把它与自制人体编码器的效果混在一起。

## E13 自制人体编码器与接口验收（已完成）

用户进一步要求不需要补充资源时自主继续。本阶段实现 `script/a3/smpl_encoder.py`：10帧人体输入，每帧24个root-local关节点72维、相对机器人anchor的root朝向6维和两个parent-local手腕朝向12维，共900维；独立MLP输出64维量化前latent，接冻结官方035 a3_fast teacher / FSQ / decoder。这个90维/帧分支是自制输入合同，区别于vendor中未激活的84维/帧定义。

位置以米为单位、采用既有SMPL到机器人基变换；手腕旋转单独保留，因为同一组关节点不能唯一表达手腕扭转。官方29关节位置/速度打包沿用全部10帧q、全部10帧v再reshape的ABI，不使用逐帧q/v交错。归一化只允许明确合格的训练会话，恒定维度尺度设为1；真实TEST_ONLY输入拒绝optimizer。11项合成CPU测试在ZHY实际通过，`Ran 11 tests in 1.235s; OK`。合成数据的单步优化只验证梯度/冻结合同，不是人体训练成果。

10帧输入分开记录 `offline_future`（0到180ms）与 `causal_history`（-180到0ms）。插值只读取窗口所需的已到达上下界；未来未到达的姿态值不参与检查/插值。200ms缺帧预算是既有诊断预算，不是实时设备精度承诺。E13使用录制中下一个上界样本模拟到达时间，anchor采用同刻运动学参考；尚未验证设备时间同步、实际机器人anchor或在线延迟。

四段完整24关节 `body_only.npz` 及 `inspection.json` 已从原Q1目录直接复制到新机 `data/test_inputs/a3_smpl_encoder_20261012/{cloth_walk,cloth_oneleg,recording_0923,recording_0924}/`。原数据保留，Mac没有增加原始数据。原TEST_ONLY、training_allowed=false及既有重定向质量记录不变；这些输入只能做接口诊断。

新独立 `script/a3/zhy_smpl_probe.py` 预登记E13：每段每模式固定16窗口（总128），CPU单线程，官方旧保存输入固定16帧动作比对，原ABI误差0、动作max误差预算1e-5、身体参考重构误差预算1e-6。0真实optimizer、0新仿真、无CUDA初始化、无自动恢复/重复；复用90/85°C、4GiB保留、5秒采样和首故障停止保护。工件位于 `data/experiments/a3_zhy_smpl_20261012_E13/`，日志 `logs/a3_zhy_smpl_20261012_E13/`，原始数据/未训练接口checkpoint仅服务器LOCAL_ONLY。

现有CLOTHO包的元数据盘点发现206序列/17个会话目录，包含24关节pose/joint标签。包内未找到来源/许可说明，也未建立合格A3配对语料；目录叫train_released不等于已验证可直接训练。这是新的原始标签候选，不把旧FGP TEST_ONLY输出改名为训练数据；后续先建立来源、时间、重定向与会话隔离证据。

代码基线 `283cc2790e26b5047c9f0e90e0e71a03a0312584`，vendor固定 `fe6868ba37034f89b912f0fb851bce19f120266d` 不变。只能在ZHY运行，需完整四段输入及此前E12核验通过：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.test_smpl_encoder
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_smpl_probe register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_smpl_probe run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.zhy_smpl_probe verify
```

E13只运行一次；已存在实验目录时不得照抄命令重置。新编码器目前未训练，不能部署、不能用有限输出宣称具有SMPL跟随能力。

### E13 实测结果与边界

新机2026-10-12北京时间19:35:49–19:35:59完成worker，19:37:43完成独立核验，状态 `VERIFIED_INTERFACE_ONLY`。仅CPU单线程，没有初始化CUDA，没有新仿真或真实数据optimizer更新。

| 检查 | 实测 |
| --- | --- |
| 真实人体输入覆盖 | 四段各32窗口，共128；未来与因果历史各64，形状128×10×90 |
| 原14点参考重构 | 四段最大位置误差均0；raw帧1766/1767/1969/1923，归一化帧1766/1767/2251/2253 |
| 官方teacher ABI | 四段全部q/v/朝向打包误差0，按joint name映射URDF顺序 |
| teacher与学生输出 | 两者均128×64、有限；学生严格重载输出逐位相同，未训练 |
| 冻结控制桥 | 16固定官方保存帧，与旧捕获动作/单帧官方act最大差均2.86102294921875e-6，低于登记1e-5预算 |
| 训练边界 | 四段TEST_ONLY均被训练门拒绝；合格训练会话0，真实数据optimizer0，新回放0 |
| 权重不变 | 官方/学生前向前后张量SHA不变；独立CPU重载/源数据重构及teacher前向复算通过 |

官方部署policy张量摘要SHA `05aee7b994b2c788c43f51e0eb7b929215ea52012c05550dfb295d53d3dbe1eb`；学生初始状态摘要SHA `7a12100d03b1fec63a2abfb9a80cb02961a31b94e6f00c410b81163a827ee971`。这些是本probe实例的state_dict摘要，不等同于完整官方checkpoint文件SHA。

5个健康样本同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`、CPU最高40°C/GPU26°C、cgroup可用最低9,401,647,104字节、OOM0，无新关键内核错误或遥测错误。worker期间按5秒采样；最大103.68秒间隔发生在worker结束到独立核验的阶段间空档，不能宣称整个注册/核验时段有连续5秒覆盖。GPU未进行计算。

产物与证据：

- `data/experiments/a3_zhy_smpl_20261012_E13/registration.json / state.json / result.json / independent_verification.json`。
- `contract_inputs_outputs.npz` 736,263字节，SHA `1f4a923469ec56e3eab7d182fbe7022044b9fa6c96f9c570d41df0fb4144686a`。
- `student_untrained_contract.pt` 2,547,061字节，SHA `d95c6676a492721279979d1e5b0e598bf9269d4a8613eafc0f7879b9e4560f15`，标记 `UNTRAINED_INTERFACE_ONLY / deploy_allowed=false`，不作为候选策略。
- `logs/a3_zhy_smpl_20261012_E13/health.jsonl / worker/console.log`，小型清单 [E13核验回执](../data/manifests/a3_zhy_smpl_20261012_E13.json)。原始/衍生数据与PT留服务器LOCAL_ONLY，Mac仅代码/README/小清单。
- [11项CPU契约运行记录](../logs/session_records/a3_smpl_cpu_tests_20261012.json)，源码身份见E13清单。

本次已完成可自主执行的编码器实现与接口验收，不需要用户再提供服务器、模型或现有录制。下一步的数据候选是CLOTHO原始人体标签；需先核对来源/使用范围，重新生成A3配对参考并通过质量验收、按会话划分，再开始学生蒸馏。未要求用户重传已有大包，也未把接口通过解释为人体跟随能力或对官方策略的提升。

### 原始标签候选清单

`script/a3/smpl_corpus_inventory.py`只使用标准库读取既有ZIP中央目录，已在原机实际运行；没有导入PyTorch、加载模型、解压或启动负载。206个完整pose/joint序列对、17个capture文件夹、412个标签文件，未发现缺一半的序列。标签原大小458,867,914字节、压缩409,787,554字节；没有README/LICENSE/notice条目。17个文件夹不能直接当作17个独立受试者，所有标签的shape、坐标与时间仍须新机独立核验。

清单 [a3_smpl_corpus_20261012.json](../data/manifests/a3_smpl_corpus_20261012.json)保存每个序列名、标签大小/CRC和中央目录摘要；沿用此前inspection的整包SHA、此次只对比整包大小，不把它记成新整包SHA核验。新清单明确 `training_ready=false`，尚无新A3配对参考、合格质量和固定会话划分；没有在Git加入原始标签或第三方资产。

此前公开网页/GitHub检索没有找到能确认此本地版本来源和训练范围的原始发布记录；本地FGP目录也不是Git仓库，README为本机运行说明。包名和目录结构只能证明本地存在标签，不能用来捏造来源或使用条件。后续若确认这些原始标签可用于本项目，独立制作新语料；旧测试录制与衍生记录仍保留原标记。

此阶段10份小型代码/记录按项目规范审阅、审计并提交推送；具体完成状态以Git历史及远端核验为准，不预写同步成功。原始模型/数据备份仍为LOCAL_ONLY。
