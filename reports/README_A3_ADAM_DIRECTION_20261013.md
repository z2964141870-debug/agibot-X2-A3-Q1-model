# A3同输入Adam方向与有限步诊断草案 E47

记录日期：2026-10-13。状态：静态审查/下一单变量设计完成；E47编号核验未占用，尚未实现、登记或计算。入口为[E46结果](README_A3_RETENTION_DIRECTION_20261013.md)。官方035仍baseline，无新效果提升。

## 目标与依据

E46在E45 step4和同一保存首batch上，负PPO及负总Actor梯度均局部降低384teacher MSE；aux对dyn梯度0。不能继续把“损失反向竞争”当已证实原因，也不能直接由原始梯度推出Adam有限步方向。

下一问题固定为：**同一保存状态/衣服batch/teacher下，完整Actor+Critic梯度按原裁剪、原KL调LR和保存Adam历史矩转换后，是否把局部方向转成增加teacher误差？对应有限步是否超原teacher门？** 这是反事实更新，不是重演实际第17–19batch，不是继续训练或效果评测。只复核保存的历史第16步增误差已是已知事实，不单独为此再开重复实验。

## 本次完成的静态核验

- 运行环境真实库位于`data/env/lib/python3.11/site-packages`，不由启动链接`data/environments/a3-sonic/lib`猜测。固定源码SHA见[草案小清单](../data/manifests/a3_adam_direction_plan_20261013.json)。未导入torch/加载保存张量。
- Transformers `Trainer.create_optimizer`原式按`get_decay_parameter_names`将**整个PolicyAndValueWrapper**的requires_grad参数分成decay/no-decay两组，按named_parameters顺序序列化。PyTorch optimizer的state_dict整数ID不是参数名，必须按同代码重建完整组映射，不能只按Actor同shape猜配。
- wrapper按policy、value_model顺序注册；ClothConstrainedTrainer在创建optimizer前冻结Actor非g1_dyn，Critic沿原可训练状态。下一可实例化Critic结构以匹配分组，但不能启动环境/rollout。
- 原裁剪作用于全Actor+Critic。E46没有Critic梯度，因此不能凭Actor范数猜原裁剪比例。新草案明确增加一次固定输入Critic前向和完整梯度，解决该缺口，不改旧E46含义。
- 保存日志第16optimizer实际两组LR为4.5e-6，step4保存时384teacher RMSE为.0460553877055645；下一旧首batch的KL在E46为.34723514。反事实LR必须重新按固定原公式求出并记录，不能把任一个历史LR直接假定为新候选LR。
- PyTorch AdamW源码给出bias correction、epsilon、decoupled weight decay语义；**保存的实际optim枚举、flags、moments/step与ID映射尚未加载核验**。不能先宣布已证实动量问题。

## 独立E47实现前固定范围与预算

1. 固定输入E45 step4、同一initial_token_repair批次/实际tokens、原384teacher、运行metadata、配置和第16步LR记录；引用E46结果作方向/合同对照，不再跑step0。所有源文件先SHA固定，号码与资源恢复重查。
2. 只一个保存Actor+Critic状态；最多12次Actor/decoder forward、1次Critic forward、3次autograd.grad，0optimizer.step/仿真/rollout/可部署checkpoint，900秒。构造完整参数组后严格比对ID数量/顺序、参数shape、moments dtype、每参数step及flags；不支持的optimizer/flags立即停止，不静默近似。
3. 用实际vendor PPO/aux和原96teacher计算完整梯度；Critic前向使用**独立临时副本**。它的running normalization按原训练语义在临时buffer上更新，并单独记录变化；不能使用eval模式冒充原训练，不能把E46的constant value替代解释成完整梯度。原checkpoint、保存optimizer、源Actor参数/buffer全部前后SHA不变。临时Critic参数不更新，不保存临时buffer。
4. 对同一完整梯度计算原0.1全模型裁剪比例、原KL自适应LR，再按保存Adam历史矩和原参数组计算一个**数学候选位移**。不构造/调用optimizer.step，不改历史moments。Actor分量和E46同输入总梯度应吻合（atol1e-7/rtol1e-5）；发现额外依赖先停，不扩大范围。
5. 对比全teacher沿负原始总Actor梯度与沿Adam候选位移的一阶变化；保持符号定义清晰，实际位移直接用`g_teacher·delta`。零范数余弦为null。最多再4次teacher forward在临时参数替换上测候选有限步RMSE/maxabs，原teacher门.05/.5不放宽；超门仅记拒绝候选，绝不保存或部署。不要为追求通过尝试第二个LR/权重/方向。
6. 浮点公式需标准库解析正对照与有限性/预算测试。手算Adam公式不等于原fused GPU逐位重演，需明确这个边界。记录完整梯度范数/裁剪系数、候选LR、矩状态step、位移范数、teacher一阶/有限变化，防止只看最终通过与否。

原资源入口保持：登记/执行前各3次间隔5秒可用估计>=8GiB、无其他RSS>1GiB或计算、GPU空闲、可靠遥测；CPU90/GPU85连续2次5秒、保留4GiB、900秒，不丢缓存或杀他人进程。缺资源不创建worker，保持ACTIVE；任何计算异常STOPPED不复跑。实现若不能覆盖上述预算，在**尚未登记**时说明并固定修订，不能运行中加预算。

## 结果判据与下一步边界

- 若原始总梯度对teacher向内而Adam候选一阶向外，支持**此状态/此batch下优化器变换改变方向**；不等于证明第19步原因或Adam算法错误。下一独立考虑对实际候选位移的teacher保留约束，不盲扫teacher系数。
- 若一阶向内但有限步teacher上升，记录局部近似/非线性差异；若一阶与有限步均向内，则不能指控Adam，下一问题转实际新rollout/batch差异。禁止因不符合假设换batch选结果。
- 即使候选通过teacher门也不能证明闭环旧能力或衣服效果；不能拿这个候选作为checkpoint进行效果选点。后续训练须新编号及原匹配效果门、独立seed/未见会话。

## 版本、存储与恢复

本次只标准库读源码、JSON与SHA，0新模型/张量计算。E46实现d04e750、结果71babfcc6ff34552d842c095672be11b17948cd5已推送核验并同步ZHY；该静态草案随后scoped审计/提交，实际号见Git。模型/原始日志ZHY LOCAL_ONLY，网盘未核验。

下一实现独立E47，不改或复跑E46/E44/E45。需要先读本README、草案清单与最新current state，确认未有更新的实现/真实state，避免重复。当前没有E47注册目录或worker。
