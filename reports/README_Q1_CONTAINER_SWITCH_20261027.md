# Q1 切换25245容器与后续诊断安排

2026-10-27后续状态：用户已确认25245服务器停止，等待新SSH/密码，旧入口不再连接。MG05新授权覆盖本文件开始时的训练暂停范围，但须先完成诊断/候选验证；停服期间仅离线分析，未启动新训练。现场状态、待同步和恢复顺序以[MG05分析](README_SUPERMIMIC_Q1_MG05_SIGNAL_ANALYSIS_20261027.md)为准，下文保留切换时历史证据。

记录日期：2026-10-27，北京时间。阶段CS01：连接、共享文件与必要运行环境核验完成；后续诊断待执行，训练保持停止。

## 目标与范围

用户将工作入口换为`root@59.77.15.15:25245`，仍经Windows桥接，并询问接下来做什么。此前“先别继续训练、思考卡点”仍有效；本阶段不创建训练目标、不启动/续训、不调度子agent，也不改RF09冻结来源。

## 切换结果

- 新容器hostname`interactive63856`，boot`d994e71f-bae0-46ba-a5f7-6c2163771c6c`；本机Windows转发18872和新SSH控制socket已建立并实际验证。
- `/opt/data/private`挂载为`/dev/nvme2n1p1[/default/ZONES/p3090-8/PROJECTS/robotics01/PRIVATE/zhy]`，与CM01中25252的宿主机子树一致。统一根仍为`/opt/data/private/ZHY`，不新增平级项目根。
- 原ZHY、Python环境、HP适配runtime、RF09权重、诊断结果和300训练/30验证参考路径已经可见。新容器读取RF09最终权重、BD01完整诊断、参考清单的SHA均与已保存清单一致，因此没有重新搬运大数据或创建副本。不是新做了全量历史树校验。
- 对旧25252只做一次有界只读连接，返回`Connection closed`，其当时连接状态记UNKNOWN；不据此认定文件丢失、不重启、不再排查旧容器。旧原件/桥未删除。

## 版本与环境验证

核验时权威Git为`/tmp/yuanqi-supermimic-q1-20261023`的Q1分支，HEAD`c4003ef8813c6044ea3244c898dcb66cbffebf1d`。RF09冻结训练仍为`a553b6a5a3c44167042355ff2f2fbd42ff558342`，HP固定`310244b73e45775744a75b789eb25a6c9c5a81c9`；没有替换来源。

实际执行了导入、CUDA小张量有限性、Q1 XML加载、Q1Kinematics构造、最终checkpoint的`weights_only=True`加载和330参考路径存在性检查。未创建训练环境、未做物理积分/优化，也未复测4096环境profile。

| 检查 | 实际结果 |
| --- | --- |
| Python | 3.11.15 |
| PyTorch/CUDA | 2.9.1+cu128 / 12.8，CUDA小张量有限 |
| MuJoCo / MuJoCo Warp | 3.11.0 / 3.11.0，导入成功 |
| Warp / unilab | 1.16.0 / 1.5.0 |
| Q1模型 | nq29、nv28、nu22，与22关节接口一致 |
| 参考路径 | 300训练+30验证，缺失0；未重哈希所有动作 |
| 最终checkpoint | 208865366字节，正常加载；不拉取Mac |
| 训练进程 | 新容器进程扫描未见Q1训练命令，GPU采样空闲 |

关键文件及SHA：

```text
data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027/run/final.pt
d834ac94a9ce9d8eca91fc576f4ffb92f0ad54386c40f39539cf0c352debf6d0
data/supermimic_for_q1/AP02_SOMA_Q1_20261027/q1_candidates/pilot_manifest.json
ba9ac284a018cdd40eea5200d2c157b7da7cfdae89e6c820eed2c9748cc83337
data/supermimic_for_q1/BONES_RF09_SOMA300_HP_E4096_T05H_20261027/RF09_BD01_SAVED_TRACE_20261027.json
e59a4bda110f08cc18fe4634cf72121c8d277ae4f3375971fa6d12b32a5c0dd7
```

## 资源与恢复命令

10:20前后采样：3090显存总24576MiB，使用14MiB，26摄氏度、利用率0%；GPU UUID仍为`GPU-c602a45a-0dd8-9ad2-9adc-4f863facd1d4`。private可用`1588639227904`字节，约1.589TB。新容器cgroup v1内存用量采样22921216字节、上限34292629504字节，OOM kill0。均是采样值，不保证未来资源，也不是整个宿主机进程可见性证明。

已建立的连接：

```sh
ssh -S /tmp/yuanqi-q1-25245.sock -T -p 18872 \
  -o BatchMode=yes -o ConnectTimeout=8 \
  -o 'HostKeyAlias=[59.77.15.15]:25245' root@127.0.0.1
```

Python仍为`/opt/data/private/ZHY/data/environments/supermimic_q1_hp_20261026/bin/python`，经`/opt/data/private/ZHY/script/supermimic_q1_hp/activate.py`加载已有CUDA库与缓存路径。Windows桥进程为转发18872到25245，保留旧桥。若socket失效先做有界连通检查，不自动创建训练。认证仅交互提供，未写入项目文件或Git。

## 接下来做什么

根据[BD01](README_SUPERMIMIC_Q1_RF09_BOTTLENECK_20261027.md)，优先级如下；本轮只完成切换与安排，以下新增诊断尚未执行：

1. **上肢策略与奖励信号。** 冻结RF09权重，选清楚抬手片核动作目标、逐项奖励和梯度，区分目标编码、奖励尺度、模型优化方向。验证手势中策略目标到参考约1.184rad，而关节到策略目标约0.062rad，首先需要解释为什么策略给出错误目标；不先默认换重定向器或提高PD就能修复。
2. **腿部真实接触与控制。** 固定两条已失败的普通前向走，检查实际接触力/支撑切换、足底滑移、躯干制动、关节限矩及Q1响应。参考足底近地比例不当成受力证据；统一模型PD尚未做Q1逐关节整定，也不直接复制X2增益。
3. **世界模型动作方向。** 在匹配状态与接触阶段，用有界正负动作微扰的物理结果与世界模型方向对照，检查跟踪收益/支撑变化是否同向；预测loss小不能代替方向验证。末帧目标位置停住而速度可能非零的候选也需先量化实际占比。

先给出可重复证据与明确修改方案，再由用户决定下一轮训练。保持原基线与参考便于受控对照，不自动加数据、延长训练或启动RF10；原五小时预算例外不延续。阶段目标是定位原因，而非用更多时长解释所有短板。

## 保存与同步

CS01小收据、AGENTS与报告索引/当前状态纳入Q1提交，运行payload audit后push并独立核远端40位SHA；同批小文件镜像Mac/新25245、比对大小和SHA。最终发布提交以Git历史为准，未核验前不宣称同步完成。既有模型/大轨迹原地保留，网盘仍LOCAL_ONLY。

训练监督`q1-rf08-4096`的本机配置仍为PAUSED，本轮未修改自动任务；本机未发现`supermimichp-x2`的TOML，不据此创建替代任务或断言其状态。每日研究任务保持原设置。恢复先读本文件、CURRENT_STATE及BD01，不执行历史训练命令。
