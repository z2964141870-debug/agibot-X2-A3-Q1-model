# 取消额外数据门槛，优先SuperMimic方法复现

记录日期：2026-10-25，北京时间。阶段：BONES-PF01 / RF01。状态：用户纠正已落实，128片原速Q1参考全部组库；RF01的64环境500轮有限训练与保存核对已完成，实际跟随效果待评测。

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

## PF01真实128片准备完成

12:34:11至12:37:19，prepare真实退出0、guard完成回收。原冻结train前128片全部转换且128/128通过运行必要IO检查，0转换/IO错误；39694帧、791.32秒=13.19分钟。没有使用旧几何flags筛选，未人为放慢、未加统一速度cap，Q1实际关节范围保持。3个原样小收据服务器/Mac大小/SHA一致：[prepare](../data/manifests/supermimic_q1_BONES_PF01_prepare_20261025.json)、[IO准入](../data/manifests/supermimic_q1_BONES_PF01_io_admission_20261025.json)、[guard](../data/manifests/supermimic_q1_BONES_PF01_prepare_guard_20261025.json)。

产物`data/supermimic_for_q1/BONES_PF01_128_20261025/`，原生转换在converted、组库在library。128链接只引用本阶段新数组，未复制原包或旧参考。分配142741504字节、盘余2094395326464字节，约2.094TB；共享内存22.48GB/OOM14。源码发布22962d3f09fa8c29345761e54f93888874358b11已推Q1并独立核验。

PF01完成后直接开展500轮新训练，没有因13.19分钟尚未达到旧自定30分钟目标而等待。实际训练命令在前述环境变量下：

```bash
"$PY" -u "$ROOT/script/supermimic_q1_reproduction/run_reproduction.py" train \
  --root "$ROOT" --library "$ROOT/data/supermimic_for_q1/BONES_PF01_128_20261025/library" \
  --backend-receipt "$ROOT/data/supermimic_for_q1/PHUMA_P05_slots_V01_20261024/receipt.json" \
  --output "$ROOT/data/supermimic_for_q1/BONES_RF01_500_20261025" --steps 500 \
  --code-commit 22962d3f09fa8c29345761e54f93888874358b11 --expected-oom-kill 14
```

## RF01真实训练与保存核对完成

12:42:02至12:53:46，guard真实退出0、状态WORKER_COMPLETED，worker返回0并已回收。训练终态TRAINING_FINISHED_PENDING_EVALUATION，500/500采集轮、32000真实转移、训练计时689.58秒约11.49分钟；WM149组/1490个Adam步，actor99次/99个Adam步，critic1980个Adam步。数据128片/13.19分钟，三网络从随机初始化开始。最早75轮仍在暖身、0更新的快照已被完整实际结果替代，不能用它作为最终状态。

复用`script/supermimic_for_q1/audit_training.py`在新CPU进程核对初始/最终保存：三网络均有真实参数变化，参数及优化器状态有限，三Adam计数符合原循环，输入与源码SHA未变，20条指标到500轮，OOM保持14。核对本身0动力学积分/0optimizer更新，检查对象是执行与保存，不追加数据准入要求。命令：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" "$ROOT/script/supermimic_for_q1/audit_training.py" \
  --run "$ROOT/data/supermimic_for_q1/BONES_RF01_500_20261025/run" \
  --output "$ROOT/data/supermimic_for_q1/BONES_RF01_500_20261025/training_audit.json"
```

原样小证据已取回且逐一与服务器SHA一致：

| 证据 | SHA-256 |
| --- | --- |
| [原生训练终态](../data/manifests/supermimic_q1_BONES_RF01_train_20261025.json) | ec5b90ce164b3db585bd7f40b628afbae3e00ba3a12cb0917f3feaa3d90ca01a |
| [guard终态](../data/manifests/supermimic_q1_BONES_RF01_train_guard_20261025.json) | 1bf3a28b30fdd803d9086ca9269dcb04dbe7d378d043ecf9cfb655b11279ca80 |
| [启动数据/配置快照](../data/manifests/supermimic_q1_BONES_RF01_train_admission_20261025.json) | 30242816cc3516660d31b2a35da255491c77e4510095622037311c2c5f2b42b8 |
| [保存核对](../data/manifests/supermimic_q1_BONES_RF01_training_audit_20261025.json) | 736d4f018e11b06f674dbfe071981c2b98c2fa30ceba81051fc30e1b59761085 |
| [训练指标原件](../data/manifests/supermimic_q1_BONES_RF01_metrics_20261025.jsonl) | ec0e9751bd02cb2757d514d086a92d9ae99cadec5dbfec94d42509ff0ed5624c |

保存路径为RF01/run：初始step_000000.pt为68832427字节/SHA477f5d7ea773704c5df6af84fc582bb0116c591078b79fe2bd5b0d83f36ae97f；最终final.pt为205621779字节/SHA6e467822f7e6c800bcb12f00b90e9029998c2d26a629d83df6e44c2325a17bee。模型均留服务器，Mac无模型副本。run/admission.json为启动快照，其ADMITTED_NOT_TRAINED状态不能代替原生最终receipt。

最后一条训练指标reward_mean约0.04286、joint_rmse_rad约0.43466；它是训练中即时样本统计，不是整片固定评测，不能据此称学会步行或较旧PHUMA改善。采样GPU最高39摄氏度，OOM14不变；结束RF01唯一分配274984960字节，盘余2094026186752字节约2.094TB。guard结束共享内存约23.79GB，worker已回收。

## 保存、边界与下一步

新代码发布22962d3f09fa8c29345761e54f93888874358b11、PF01报告e7ee33791d4b6f63cd3bcbfe99b5834cf4f7a0b0均已推Q1并独立核远端；本次RF01结果与5份小证据同次审计提交，完整提交身份由Git历史提供，不预写尚未执行的推送结果。重要文件继续同步用户镜像与统一服务器根。大型原件均LOCAL_ONLY。

目前完成的是取消额外门后的小库、单卡Q1方法训练链路，未完成原论文完整规模、两轮G1过滤实现或本轮固定动力学跟随评测。下一按SuperMimic原方法扩参考、增加真实交互并回放观察；参考数量与Q1必要适配单列，不再用自定几何门、规模或未见测试配额阻止训练。旧D01已结束且只作历史归档，旧PHUMA/A3停止终态不恢复。
