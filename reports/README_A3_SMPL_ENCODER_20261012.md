# A3 官方编码器核验与自制 SMPL 编码器路线

记录日期：2026-10-12。状态：官方文档、固定源码、实际发布权重及既有运行时配置已核验；自制编码器仅为候选设计，未实现或训练。

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
