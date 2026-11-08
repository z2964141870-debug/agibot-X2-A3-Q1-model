# Q1因果人体条件对照结果

2026-11-08，北京时间。本轮尚未得到可采用的新f0策略，独立权重审计及六宫图交付待补；低成本低延迟自然遥操目标继续。

done：

1. 两组真实GPU训练各完成3000次更新，只对比新增纠偏分支使用真实或置零因果人体latent。初始化动作等价与实际CUDA梯度门已过，八个Adam计数每组均3000，source PASSED。
2. 两组最终均选本轮新增update0，验证风险仍0.2210775182271997，没有被选中的新候选。每500步的门值和退步幅度仍待原曲线，选0本身不证明全部非零点均未过旧能力门。
3. 训练guard清洁回收return0，216秒SETTLED；累计5036/14400秒，无预留，原共享保持暂停。11:03:40原资源/OOM与暂停门核验通过，后续实时状态尚待新观察。
4. 已提交独立CPU权重审计，随后Linux会话被远端关闭，未收到审计完成回执。Windows仍可用，原生第二跳重登返回Connection refused；不能据此推断原训练失败或整桥接不可达。

todo：

1. 恢复后先查已提交审计的原输出、日志与精确进程，保留成功或失败原件，再取回dispatch/result/曲线；不重派已完成的6000更新。
2. 若selected/last审计确认选中策略与原起点全输入等价，复用旧0/7物理证据。仅改变且选中的策略需要新独立闭环，当前不新增相同视频。
3. 运行已备好的只读交付入口，生成两组六宫PNG/PDF/CSV与原值ZIP并目视。当前仅CLI通过，没有虚构图表或完成包。
4. 结合已成H4真实物理与f0多轮负结果选择不同的时序或接触/相位机制；不继续单纯延长同MSE，不降低旧能力门，保留新留出事件、真实人体与完整端到端验证。

## 证据

- [完整方案及实际训练观察](README_Q1_HUMAN_CONDITIONING_20261108.md#真实训练观察与待补证据)
- [已收到终端输出与恢复点](../logs/session_records/q1_human_conditioning_training_observed_20261108.json)
- [观察发布、独立远端、Mac与q1续行核验](../logs/session_records/q1_human_conditioning_observation_publication_20261108.json)

本页为阶段观察结论，服务器原件逐字节取回、CPU证明及图表待补。原160k final完成门不改，采纳null。

## 六宫图交付校验补齐

本次补的是保存结果消费入口，没有再训练或重新做闭环。原入口只对摘要标志作有限检查，未充分绑定逐点曲线、选点权重证明、原始JSON及成本记录；错误混合的记录可能被用于生成训练图或复用旧0/7结论。

[交付入口](../script/supermimic_teleop/deliver_conditioning_pair.py)现先做不导入torch、MuJoCo或matplotlib的独立校验，再允许绘图：

1. 原`fit_result.json`字节与controller记录大小/SHA相同；原`authorization.json`与CPU审计记录大小/SHA相同，授权绑定已发布conditioning计划。CPU审计中的原result/guard必须与取回文件一致。
2. 原成本run/stage/清洁回收和累计5036秒必须相符；两组各3000更新、八Adam计数、同初始化/容量和每个保存点的来源曝光逐一保留。没有重置台账或新增预留。
3. 重算每点old/pure各50%的验证风险、原三锚点不退门和`point + 20 * retention`目标。完整七保存点、逐来源覆盖、有限值和正百分比分母须先通过，不能出图后才发现不完整。
4. selected/last权重CPU证明的原路径、选点及全输入策略等价断言须相符。只有选中robot adapter精确相同且human末层全零，才允许注明复用旧0/7；改变的候选必须保留需要独立闭环的标记。
5. 输出摘要分列`eligible_nonzero_updates`与`ineligible_nonzero_updates`。选中新增0仍可能同时存在过门但风险更高的非零点，不能把“没被选中”都记为“旧能力门失败”。

新增[11项CPU合同](../script/supermimic_teleop/test_deliver_conditioning_pair.py)在原有`q1-teleop-mps-20261106`环境全过/.008秒；两AST、CLI和diff检查通过。覆盖错配审计/选点、重复或缺失分组、非有限值、错误目标或权重、逐点曝光、原始JSON/计划字节改变，并验证改变候选的交付仍被允许但不复用旧闭环；目标合成误差仍用原绝对1e-6，不借相对容差放宽。测试全部为临时合成数值，不构成本轮真实训练曲线或新科学结果；未运行matplotlib或生成替代训练图。

恢复后需要取回同一任务的七份原件：`result.json`、`fit_result.json`、`fit.guard.json`、`checkpoint_cpu_audit.json`、`authorization.json`和两组`training_trace.json`。先核已提交CPU审计是否完成；不能重派6000次更新。随后从项目主根用既有绘图环境运行，输出和ZIP均换为未存在的绝对路径：

```sh
data/environments/q1-teleop-mps-20261106/bin/python \
  data/repositories/q1-rf12-20261028/script/supermimic_teleop/deliver_conditioning_pair.py \
  --source /absolute/original/numeric/packet \
  --output /absolute/fresh/figures --archive /absolute/fresh/evidence.zip
```

原件、CPU证明、双六宫图和ZIP依旧待服务器取证；没有据测试生成它们。多步WM五执行文件/manifest及原12128B包未改，仍待服务器来源/资源准入。45014的Windows于12:26:48返回延迟日期，本轮向同会话提交的一次原生第二跳随后明确返回`Connection refused`，未到认证且已回PowerShell；没有发送Linux监督命令或循环认证。第一跳仍在，Linux资源未重新核验。完整目标、原共享暂停与采纳null保持。[本次校验收据](../logs/session_records/q1_human_conditioning_delivery_checks_20261108.json)。
