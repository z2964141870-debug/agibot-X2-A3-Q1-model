# PHUMA R02审计保存的独立续训

记录日期：2026-10-25。阶段：R02C。状态：已独立启动，初始12门与496冻结通过，03:16核实新世界模型/actor/critic更新。

## 目标与选择

用户要求继续现有资源，BONES交师兄。保持SuperMimic可微世界模型、想象BPTT与critic复现，沿用原340片PHUMA训练参考及dev6/test7，完成原定12000轮中从已审计5000保存开始的剩余7000轮/448000转移。原R02第5300轮内存保护FAILED保留；失败后未保存的300轮不能算进父权重。

[开发对照](README_SUPERMIMIC_Q1_R02_DEVELOPMENT_20261025.md)显示完整率提高、五段关节误差降低，但0/6过门、移动仅0.078/0.521m。选择完成原固定预算是有限学习检验，不是已证明“只差训练时长”；不据test调方向，不自动追加完成后的预算。[资源恢复](README_SUPERMIMIC_Q1_RESOURCE_RECOVERY_20261025.md)已释放约21.20GB旧失败评测占用，为此新实验提供真实余量。

## 实现与约束

新`script/supermimic_q1_partial/parent.py`只接受真实R02内存保护失败原件、5000保存16门审计、原469冻结及已完成的六段development对照。旧`supermimic_q1_extension`只接受完整matched pair，保留其严格入口，未伪造pair/成功final。新loader复用其合同比较，将三网络、三Adam和两个return sampler数组复制；父Adam必须19490/1899/37980、学习率offset5000。

`train_partial.py`复用原续训训练循环，显式绑定已验证4槽/64环境后端。原参考/PD/力矩/奖励/H32/B128/WM B64每组10批保持；新seed20261025、重新收集replay并执行原128轮采集/50组模型预热。因此是新实验warm start，不能称精确进程恢复；父网络/Adam/采样器保留。每500轮保存，约17份权重不足4GB，实验12GiB/磁盘至少100GiB保护、30GiB内存/86摄氏度/新OOM停止及900秒训练无进度保护保持，无自动重启。

`pipeline_partial.py`先独立注册，再复用已验证expanded controller执行7阶段：7000训练、继承/保存独立审计、全部353片初始动力学/轨迹审计、全部353片最终动力学/轨迹审计、原三划分比较。初始是父5000策略，不是原随机策略；最终累计12000采集轮，跟随门仍完整片/0.15rad/0.10m。`audit_initial.py`用于独立CPU检查step0八类父状态完全相同、局部计数0、offset5000、新种子和全部冻结输入。最后`audit_partial.py`复用独立审计的计数和张量比较，替换父资格检查为真实partial合同，不把失败父实验认定完成。

新代码均在原469冻结集外；register额外冻结新入口、原续训依赖、资源配置、父原件及开发证据。新增PHUMA B015/B016/B017B不改本次输入，独立后续合库。

## 验证、命令与产物

本地语法通过；Mac缺PyTorch，两项小型Adam控制明确skip，不算通过。服务器两项原生控制、真实parent资格/合同及初始保存继承检查待。先发布核验再登记/启动，不将实现完成写成实际训练。

服务器两项原生Adam控制已2.94秒通过：完整网络/动量/计数/采样器继承，以及父Adam被清零时明确拒绝。真实`validated_parent`和五个入口模块导入预检退出0，确认父为原205348199字节/SHA d6e0f870…/5000轮、1949组WM/1899次actor，没有把5300失败进度当保存。七份新文件两端SHA全同。未启动训练、未假造父终态；真实step0独立继承检查将在新启动后立即执行。

预定目录`huanyu:/opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R02C_partial5000_extend7000_20261025`，日志同名`logs/`，新配置`script/supermimic_q1_partial/partial_R02_20261025.json`。登记命令模板如下，COMMIT须替换为实际发布全号：

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
"$PY" script/supermimic_q1_partial/pipeline_partial.py register \
  --root /opt/data/private/YuanQi_Q1 \
  --experiment /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_R02C_partial5000_extend7000_20261025 \
  --code-commit COMMIT --seed 20261025
```

实际启动需外层runtime_guard包裹新run命令并关闭stdin、独立session、保存启动身份/命令/boot/start_ticks。原FAILED目录不会重新启动，任何非零阶段停止后续评测。

## 保存、边界与下一步

先完成服务器控制与真实只读parent验证，结果立即补此README/索引/当前状态，audit提交并推送Q1、独立核远端后启动。模型/轨迹仍LOCAL_ONLY，无真机结果；训练完成不等于移动成功，最终必须报告各划分完整率、关节/根误差和实际/参考位移。目标仍未达。

## 03:12实际登记与启动

发布`d21a8804d64c338a5caf3851fdcaba6bbe9aa355`已显式推送Q1并独立核远端；七份部署SHA一致。登记实际退出0，496个冻结文件，7000新轮/448000转移，父5000轮/原FAILED；spec154640字节/SHA`949d9e52289a4f251a3c2e36298495fb63d669a86bf444d9bb57f9db6c801d2b`。[原样spec](../data/manifests/supermimic_q1_R02C_spec_20261025.json)与[启动收据](../data/manifests/supermimic_q1_R02C_launch_20261025.json)已保存。

03:12:33通过Popen独立session启动runtime_guard，stdin关闭，无自动重启。guard835011/start_ticks3616115871、controller835012/start_ticks3616115885、trainer835015；boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`。完整命令在launch.json，guard输出`logs/PHUMA_R02C_partial5000_extend7000_20261025/guard.log`，原训练stdout同目录train.log，状态为新目录storage_guard.json/controller_status.json/run/status.json。恢复必须核实际/proc身份，不能凭这些旧PID认定活跃。

03:12:55实际50/7000新轮、3200新转移、累计5050轮；WM/actor均0，仍在新replay预热，不冒充已更新。温度33摄氏度、GPU余23473MiB、内存16347291648字节/OOM14、盘余1639530557440字节，存储guard新实验分配约205.63MB。原5000权重及R02三个FAILED原件不变。下一立即独立step0 CPU八类状态继承审计，随后核首批真实世界模型/BPTT/critic更新；若审计失败，停止本新实验并保存终态，不放宽或重写父证据。

## 初始继承审计真实通过

独立CPU/mmap入口退出0、12项全过；父wm/policy/critic、wm_opt/policy_opt/critic_opt和bin_return/bin_memory八树逐张量/容器完全相同。局部计数均0、累计offset5000、固定7000新预算、新seed及code_commit/manifest相符；全部496冻结无变化、张量有限、未初始化CUDA。初始205323367字节/SHA`497eb3da1abba210242d03b2d092252a14be2966f23d3999e009645f7cbc5a11`，与原父文件字节不同是新manifest/局部计数不同，不是网络或Adam重置。[原样小收据](../data/manifests/supermimic_q1_R02C_initial_retention_20261025.json)。

03:14:38实际250/7000新轮、16000新转移/累计5250轮，49组世界模型更新，actor0仍按预定50组模型预热；WM梯度和loss有限。GPU38摄氏度/余23373MiB、内存16583479296字节/OOM14、盘余1630483546112字节。此为本轮真实新世界模型更新，尚非actor更新、保存后效果或完成证明。

## 03:16独立运行核验通过

[启动核验原件](../data/manifests/supermimic_q1_R02C_startup_20261025.json)于03:16:43保存，496冻结逐SHA未变、三个进程实际/proc活跃，guard/controller start_ticks及boot匹配；trainer实际start_ticks3616116127。实际最新300新轮/19200新转移、累计5300轮，WM69组/actor19次，actor梯度0.13746、critic loss0.00441685与世界模型指标有限，OOM14。此累计5300是5000父保存加本次300新轮，不包含旧失败未保存的300轮。

最新训练健康GPU39摄氏度/余22909MiB、内存18642915328字节，核验时guard盘余1619915890688字节、R02C分配205643776字节。PHUMA B018另一单数据worker已开始独立扩库，数据worker在28.75GiB先退，R02C原30GiB/新OOM/86摄氏度保护保持。下一继续固定7000新轮、每500轮保存及7阶段评测；新增参考不会改本轮固定340训练片，完成后不追加预算。所有RUNNING均是本时刻快照，恢复时仍须重核进程/状态/资源。
