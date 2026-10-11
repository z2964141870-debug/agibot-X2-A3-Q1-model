# A3 冻结控制器的参考残差短试验

记录日期：2026-10-10。独立实验E20，状态：执行中。前序E16-E19见[闭环诊断](README_A3_CLOSED_LOOP_20261010.md)。R10未分配，旧R01-R09/E10不修改。

## 目标与决定

E12干净未来位置/速度比CV准确，但机器人闭环更差；E19相同状态的编码/动作偏差也更大。官方部署FSQ的round使动作损失对参考输入梯度为0，量化前编码目标有有限梯度。本轮比较角度MSE与冻结官方teacher量化前latent MSE，检验控制器感知目标能否改善完整因果参考闭环。这是监督参考适配，官方policy/critic不更新，不是PPO长训或绕过E10概率问题。

两组同一零初始化线性残差145→290，加到两帧CV的未来20-200ms关节预测；六帧相对历史、30Hz到达、50Hz策略。teacher与student朝向均相同因果SO(3)预测，仅未来干净关节标签不同。训练用已固定selected20整动作16/4划分中的16个，noise0/0.01rad、seed0，训练归一化仅来自训练样本；E19诊断包training_allowed=false，不用于训练。动捕仍TEST_ONLY、质量未通过，不进入本轮训练。

训练encoder使用到达目标时刻的标准参考pelvis作anchor，**不是闭环机器人状态**，存在分布差异。推理仍使用实际robot anchor，必须由闭环验证迁移；不把teacher loss降低解释成机器人提升。

## 固定预算与验收

AdamW，LR1e-4、weight_decay0、batch256、gradclip1；两组各0→2冒烟，恢复同一组完整adapter/optimizer/随机采样状态至本实验累计200次，保存0、2和每10次。官方200000只作冻结基础，不计作适配更新；两组初值/采样/预算完全相同，唯一主变量为训练损失。禁止根据这四动作再反复调参。

实际MuJoCo先两组零adapter各10步，再200步adapter×noise0/0.01×4诊断动作，共16完整回放。相对E18同条件CV：无新增跌倒，每动作joint RMSE不增加，两档各整体joint至少改善1%，每动作root位置/朝向、躯干/腕/腿世界与anchor-local误差不增加超过5%，action delta RMS不增加超过5%；全部条件通过才保留为候选。所有失败指标保留，报告跌倒前和完整统计；门槛预先固定，不随结果修改。

四动作已经反复诊断和用于方向选择，**不是新鲜最终测试**。官方离线reset初速度保留以匹配E18，初始化因果性未通过；仅运行时参考token因果，不宣称实时衣服效果或真机提升。

## 实现、版本与复现

服务器`ssh hp3090`，根`/media/yu/FAFF-E9771/YUANQI_A3`，起始main`e9e5a877e1a328b8b696ba89c6c51480d8e0ae65`已推送并核验；vendor只读`fe6868ba37034f89b912f0fb851bce19f120266d`。官方PT SHA`9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`。执行登记job固定代码/数据/模型SHA；E16/E18历史版本不改。

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m unittest script.a3.test_controller_adapter script.a3.test_joint_forecast_replay script.a3.test_causal_orientation -v
data/environments/a3-sonic/bin/python -m script.a3.run_controller_adapter register
data/environments/a3-sonic/bin/python -m script.a3.run_controller_adapter run
```

`controller_adapter_trial.py`准备train-only标签、冻结encoder训练adapter，sidecar保存大小/SHA/CPU重载/有限性；`verify_controller_adapter.py`用另一时序/旋转公式复算采样窗口，CPU重载teacher标签和全部adapter/optimizer、PT/NPZ100输入数值对照。`run_controller_adapter.py`GPU串行锁、完整动作恢复及独立账本；无累计启动上限/到期，重启300秒冷却，连续2次无保存进展停止，异常/热保护不自动解锁；CPUQuota100%、CPU90/GPU85连续2次5秒采样。无新Isaac初始化、安装或真机。

产物`data/experiments/a3_controller_adapter_20261010_E20/`；日志`logs/a3_controller_adapter_20261010_E20/`；小型结果清单`data/manifests/a3_controller_adapter_20261010_E20.json`。大型数据/checkpoint/输入留服务器，均LOCAL_ONLY。

## 结果与恢复

9项本轮输入契约通过（含新增2项，累计独立26项）。第一启动完成21,878训练窗口（60,767,878bytes，SHA`831ced8193c19079c968d28226d42f1448958e67de8392049e56fd57ca65e820`）及角度目标step2；编码目标仅保存step0时主机重启，未把SSH超时认定为断电。boot从`f6e86267-d04c-4085-a5fc-e35dcc1fb2b5`变为`5e9f4fdc-f5d4-41fc-a41f-e9d910dff62f`。三份已登记checkpoint独立CPU重载/大小/SHA/网络及优化器有限性通过，回执`post_boot_resume_verification.json`。原采样最后CPU48/GPU34°C，不支持热原因归因，也不据旧采样排除其他问题；不开展重启根因排查。

按300秒冷却和原账本恢复；当前未声称200更新或闭环通过。运行中先查state/job、boot/GPU及日志；普通卡点最多10分钟，记录后继续独立项。目标完成后停止本实验服务，不延长预算或追加旧训练。完成项按真实产物及SHA跳过，任何不一致保留失败证据。

## 长训准备与下一步

长训基础模型/数据/保存恢复已准备；直接PPO仍受E10概率合同阻塞。E20若有正向闭环才支持继续参考adapter路线，仍需新独立动作/会话与多seed验收、真实噪声/延迟标定，再固定长期优化变量及总预算。若失败，保留官方CV基线，登记分布/多指标缺口，不盲目增加训练时长。Isaac指标、RKNN、衣服质量/现场时钟继续搁置。

重要记录保存后scoped审计提交推送main、比对远端；提交号由Git历史核验。Q1只读，Mac不新增大型工件。报告索引/current state同步更新。
