# 取消额外数据门槛，优先SuperMimic方法复现

记录日期：2026-10-25，北京时间。状态：用户纠正已落实，新入口两项服务器控制及真实旧8片新IO准入通过；128片新批处理与有限训练待。

## 用户决定与纠正

用户明确要求：如果训练数据入选要求由代理制定，则放弃，优先复刻SuperMimic。此前19/22门中90%帧足底15mm内、最大间隙30mm、自穿透3mm、统一2rad/s、IK残差等都是本工程额外要求，不是SuperMimic原论文的数据准入。这些条件显著缩小候选并消耗了诊断时间，不能再以未通过它们阻止数据训练。之前“0/8严格门通过”保留为历史诊断，不等于8片不可用于复现训练。

新主线取消上述额外硬门槛；自定至少30分钟/1小时、移动占比、各20组未见移动及0.15rad等目标亦不作为开始复现训练的前置要求。输入来源/原速/Q1适配、明确有限计算预算与存储管理继续执行。仅保留文件可读、模型维度/身体顺序正确、时轴正确及数组数值有限等运行必要条件。

## 原论文与实现边界

2026-10-25重新读取官方项目页`https://supermimic.github.io/`和arXiv2610.28480v1附录E/Table A.7。原文先移除依赖外物的动作，再移除明显架高、严重穿地和站立近静止双脚悬空；142220条过滤后129514条/264.56h。几何阈值按G1/50Hz标定，不能变成本工程Q1的90%贴地或3mm碰撞条件。站立近静止等判别的完整实现未由论文文字给出，不伪造阈值或宣称已逐行复现官方过滤代码。

方法继续真实仿真采样、可微世界模型、八步自回归拟合、H32想象BPTT与critic，1秒bin/50-40-10采样。当前是已有Q1/CPU MuJoCo/单3090适配实现，环境数、批量和数据规模较原论文缩小；不是完整官方G1配置复现。原论文主基准也含训练库覆盖评测，未见测试不再是启动前强制门槛。后续首先扩参考/交互并按原方法训练，不继续围绕自定几何门做修正试验。

## 实际代码改变

新目录`script/supermimic_q1_reproduction/`，保留旧冻结目录和历史终态：

- `reference_admission.py`只检查可读/有限数组、Q1形状/身体名及原速50Hz时间。旧转换器质量flags不改，仅为诊断；使用明确的`admitted_for_training`新字段，绝不把旧22门改写为通过。
- `run_reproduction.py prepare`仍用原Mink重定向，但取消原2rad/s速度约束/裁速、使用Q1 XML实际关节范围，完成后直接按新IO协议组库，不调用22门拦截器。配置覆盖明确写入新manifest，原CSV/旧参考不修改。先处理原冻结train序列前128片作为有限批次，不将其解释为论文的规模上限。
- `train_reproduction.py`从现有核心训练入口保留同一采样/世界模型/BPTT/critic循环，仅改新数据协议及绑定已验证4模型槽、64环境后端，复用原数值异常保护；移除旧physical flags及动作范围作为数据拒绝条件。三网络从随机初始化开始。

两个回归控制针对：旧速度/离地条件不再拒绝可读输入，以及NaN输入仍拒绝。源码语法检查和服务器两项控制均已通过，四份新源码部署逐SHA一致。真实旧8片读取检查8/8通过，共61.80秒；原几何flags保持false，没有改写为22门通过。[原样新IO收据](../data/manifests/supermimic_q1_BONES_PF00_io_admission_20261025.json)登记了新policy和scope，原数组没有复制或修改，仅新小清单与8链接。

## 命令与产物

```bash
ROOT=/opt/data/private/ZHY
PY=$ROOT/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u "$ROOT/script/supermimic_q1_reproduction/run_reproduction.py" prepare \
  --root "$ROOT" --intake "$ROOT/data/supermimic_for_q1/BONES_A02_received_20261025/intake/receipt.json" \
  --output "$ROOT/data/supermimic_for_q1/BONES_PF01_128_20261025" --count 128 --expected-oom-kill 14
```

计划有限训练500采集轮/64环境共32000转移，actor B128、WM B64/10批、H32，保存初始/最终、观测数值与回报，不用自定跟随门阻止这轮运行。这是新入口训练通路和资源验证，不冒充论文完整训练或策略成功。实际命令和发布提交在启动记录中补充。

数据与模型仍统一根四目录、Mac无大工件，100GiB余盘/单实验12GiB及内存/OOM保护保留，防止再写满磁盘。这是计算资源管理，不是新增动作质量门槛。大型原件LOCAL_ONLY。

## 最新状态

128片准备与有限训练尚未启动。旧D01只读诊断已经结束，不再继续扩展；结果归档为历史，不是新主线的先决条件。新入口和本用户决定同次提交Q1、独立核远端后执行，结束即补充实际结果。
