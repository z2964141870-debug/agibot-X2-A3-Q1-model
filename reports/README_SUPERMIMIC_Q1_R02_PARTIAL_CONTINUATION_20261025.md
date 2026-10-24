# PHUMA R02审计保存的独立续训

记录日期：2026-10-25。阶段：R02C。状态：入口实现/本地语法检查完成，服务器验证、登记与训练待执行。

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
