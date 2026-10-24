# PHUMA到Q1分批扩量

记录日期：2026-10-24，北京时间。阶段：SM-Q1-PHUMA-P04。状态：新选择器与有限批处理入口已实现，本地四项来源隔离控制通过；服务器执行结果待登记。尚未启动新训练。

## 目标与范围

继续用户的SuperMimic for Q1世界模型目标，纠正R01仅11片/41.30秒的参考覆盖。AMASS原库在手但缺SMPL-X男女模型，完整BONES-SEED包尚缺，见[三源盘点](README_SUPERMIMIC_Q1_SOURCE_EXPANSION_20261024.md)。在等待这些输入时，直接处理现有PHUMA原料，不把缺少BONES当成所有工作停止的理由。

现有完整P02索引已有26.44小时温和训练候选。这是包内标称时长，不是独立或Q1合格时长。本轮先固定新候选名单、生成和独立核验实际Q1参考，再按实际合格秒数累计。工程目标仍为30分钟、再1小时；未达到时不以候选量代替。

## 实现与选择依据

- 新`select_expansion.py`直接读取已核验的73076行索引，不重扫原始ZIP。沿用P02官方来源组隔离：官方test组不进入训练，开发集仍按原固定种子保留。
- 整体排除之前112候选的命名来源组，避免重复转换和重复计算。原R01的11训练片、输入及终态保留；后续嵌套小库/扩量库可按原身份引用合格历史参考。
- 新名单每个命名来源组只选一个chunk，且每个原文件SHA只出现一次。全索引相同字节跨划分时，优先留test、再development、再train，即使高优先级那一片不满足温和运动门，也不将副本放回训练。此规则不是语义或subject去重证明。
- 新选择种子2026102407，候选预算train最多2048（移动1024/下蹲256/上肢256/其他512），development/test各32。池不足时登记实际数，不伪造补足。明确楼梯、梯子、椅子、攀爬、坐姿、爬行名称暂排除，保留精确正则。
- `prepare_expansion_batch.py`按固定offset/count生成一个独立批次，复用原始字节/划分重算、V2重定向及独立19门检查。每批独立输出selection、scan_check、references、reference_check及completion；失败保留部分产物，不覆盖重跑。
- 对项目大ZIP的完整SHA仍实际计算，但按读取区间释放文件页缓存，避免反复哈希把共享容器顶到原30GiB保护。未清空系统缓存，不读取或操作其他项目认证材料。
- 首批继续保留原19门，实际测量新候选通过率和时间开销。论文式持续悬空过滤是后续独立协议候选，本次没有删掉贴地门或把几何合格当动力学/策略合格。

## 平台与执行前证据

当前主机huanyu，项目根`/opt/data/private/YuanQi_Q1`。本轮只读实核GPU RTX3090，39摄氏度、空闲17788MiB、利用率0%；cgroup内存28,712,398,848/34,359,738,368字节、OOM14。CPU cgroup额度2400000/100000，即24核。原Q1训练进程均已终止；保留两个可视化HTTP服务与其他项目已有进程，不以这些快照代替稍后的实时启动检查。

代码副本`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1，本阶段前HEAD`ce1f958c62956feb98348692b468e5652b23067c`。新增代码本地四项控制通过：旧候选和同录制组不重用；相同字节跨划分按test/dev优先且不因不合格回流；划分内内容去重；输入顺序改变名单仍确定且明确交互名称排除。语法检查通过。服务器数值执行待下节补充。

## 可复现入口

以下命令须在项目根运行，使用当前重新核验的OOM基准与新输出目录。首次真实执行完成后登记原生日志与大小/SHA。

```bash
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
$PY script/supermimic_q1_phuma/select_expansion.py \
  --base-selection data/supermimic_for_q1/PHUMA_P02_scan_20261024/selection.json \
  --output data/supermimic_for_q1/PHUMA_P04_selection_20261024.json

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  $PY -u script/supermimic_q1_phuma/prepare_expansion_batch.py \
  --root /opt/data/private/YuanQi_Q1 \
  --selection data/supermimic_for_q1/PHUMA_P04_selection_20261024.json \
  --vendor data/references/phuma_source_2faf15b5 \
  --output data/supermimic_for_q1/PHUMA_P04_B001_20261024 \
  --offset 0 --count 32 --expected-oom-kill 14
```

同一输出只执行一次；后续批次用不同编号和不重叠的固定slice。运行时日志存`logs/PHUMA_P04_Bxxx_20261024.log`。未产生completion不能登记批次完成。不会自动启动原R01、旧A3或旧STOPPED实验。

## 实际结果与边界

待首次服务器执行后填写实际数量、合格秒数、移动覆盖、拒绝原因、耗时、资源和独立重算。当前0新动力学积分、0optimizer更新；新的选择器与批处理实现本身不构成扩量完成或运控改善证据。

## 保存与下一步

代码/README/两个入口按Q1进行diff、payload审核、提交、显式推送并核对远端；服务器部署前核验新增路径不存在及旧引用源码SHA。大型参考、IK轨迹与日志原件保留服务器，LOCAL_ONLY，Mac只保存小清单。

下一步：执行来源隔离控制与新名单选择，先运行不重叠的四个32片批次，独立核实候选/合格数量及seconds之和，再决定扩大批数及论文式几何协议。需要原速、有效移动覆盖、内存预算和固定开发/独立移动测试；不把增加片数当成可用Q1策略。
