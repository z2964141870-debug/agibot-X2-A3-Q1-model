# 统一目录后的Q1动力学评测与足地修正

记录日期：2026-10-25，北京时间。阶段：SM-Q1-M02/DEV-B/P15R。状态：DEV-B真实配对与独立重算完成，跟随0/6；P15R待执行。

## 目标与范围

用户要求继续解决Q1策略移动不足。沿用SuperMimic世界模型方法和已有PHUMA/AMASS资源，BONES由师兄处理。先独立比较原5000父策略与R02D累计10282轮退出策略，再执行已准备的P15八组足地恢复对照；不重启旧STOPPED/FAILED实验，不以数值审计代替控制效果。

主机为huanyu，唯一实体根`/opt/data/private/ZHY`。11:16只读现场GPU为RTX3090、14MiB、26摄氏度，盘余2141874618368字节，随后容器内存6708097024字节、OOM14；无相关训练/转换进程。此为当时快照，不证明后续一直空闲。

## 实现与决定

新增`relocated_io.py`与`run_relocated_diagnostics.py`，显式读取M01的inventory/result和迁移清单SHA，按原最长前缀映射及实际软链接解析新位置。核验原516冻结与5关键文件的大小/SHA；旧收据只生成内存中的新路径视图，checkpoint仅映射manifest路径字符串，不改网络、优化器或原文件。新收据明确记录原身份、新身份与映射，不宣称旧实验原位复现。

开发评测复用原`run_contact_corrected.py`、`evaluate.py`和独立`audit_evaluation.py`，保持6段development、原时轴、首帧初始化、自由根、无噪声/DR、原PD/力矩/接触及0.15rad/0.10m门。对照为R02D的step000000（保留原5000父参数）与退出final（新增5282、累计10282），不是重训。原exit-save审计的小型字节副本需部署到新根`data/manifests/`，原审计当时未向服务器写文件的事实不变。

P15R仅在独立进程中将P14引擎的JSON读取绑定到迁移视图，复用未改的P15地面缓存/恢复约束及四train×两限速输入。保存后复用独立P14数组审计；参考仍诊断用途，未经完整十九门不接库。

保护保持：100GiB余盘、28.75GiB共享内存、OOM14、128MiB新输出、512MiB来源目录；worker进程组退出回收，单阶段600秒超时及900秒无进展保护。所有新数据在统一根内，大文件不复制到Mac。

## 命令与证据

代码版本由本阶段Q1 Git提交给出；必须发布并逐SHA部署后执行。以下为计划命令，尚不能当作实际通过证据。

```bash
ROOT=/opt/data/private/ZHY
PY=$ROOT/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -u \
  "$ROOT/script/supermimic_q1_sources/run_relocated_diagnostics.py" development \
  --root "$ROOT" --output "$ROOT/data/supermimic_for_q1/PHUMA_R02D_DEV_B_relocated_20261025" \
  --expected-oom-kill 14
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -u \
  "$ROOT/script/supermimic_q1_sources/run_relocated_diagnostics.py" p15 \
  --root "$ROOT" --output "$ROOT/data/supermimic_for_q1/PHUMA_P15R_floor_recovery_20261025" \
  --expected-oom-kill 14
```

新实验根内`guard.json`、`relocation_admission.json`、`comparison.json`或`probe/receipt.json`与`independent_audit.json`为恢复入口，stdout在`logs/`同名前缀文件。不得复用已创建的输出目录重跑。

## 结果、保存与下一步

本地迁移控制首轮4/5通过，一项因Mac临时目录`/var`与`/private/var`比较未规范化而失败；修正fixture，不放宽生产路径或内容检查。补检5/5全部通过；服务器预检及正式结果待。

### 已发布、部署与实际启动

实现发布`0fac687e7215e406c73b33367115469daae93f0b`，已推送并独立核远端Q1。三个新源码及原exit-save审计小副本部署前确认不存在，部署后四文件逐SHA与本地相同；服务器迁移控制5/5通过。真实迁移预检原生退出0，523个文件身份核验、521条旧路径映射通过，原文件不修改。

已实际启动上述development命令，输出`PHUMA_R02D_DEV_B_relocated_20261025`，guard及父模型真实MuJoCo评测进行中。首段移动父模型完整199/199步、回报0.568960313147，与原开发对照匹配；完整配对与独立重算尚未结束，不能提前报告最新模型效果。P15R仍未启动。

### 11:31开发配对实际完成

11:29:44至11:31:11，guard/worker原生退出0并回收，父5000/新10282两次MuJoCo与两次独立轨迹重算四阶段全部完成。两者6/6完整、0/6严格跟随；非移动五段中三段全身关节RMSE改善、两段退化，不构成所有动作改善。

| 移动开发片 | 父5000 | 新10282 |
| --- | ---: | ---: |
| 参考最大水平位移m | 0.521165 | 0.521165 |
| 策略最大水平位移m | 0.078251 | 0.072580 |
| 根位置RMSE m | 0.256907 | 0.248638 |
| 全身关节RMSE rad | 0.452543 | 0.461758 |
| 腿部关节RMSE rad | 0.519185 | 0.535873 |
| 归一化回报 | 0.568960 | 0.593646 |

回报上升与根位置误差小降没有带来有效移动，位移与关节跟随反而退化。较好非移动片的全身RMSE为0.232079至0.153251rad，仍超过原0.15门；不放宽阈值。动作域修正越过此前数值发散位置，但本结果不支持仅继续相同输入训练就能解决步行，也不能将修正与训练时长变化拆成单变量效果。

原样证据：[comparison](../data/manifests/supermimic_q1_R02D_DEV_B_comparison_20261025.json)、[guard](../data/manifests/supermimic_q1_R02D_DEV_B_guard_20261025.json)、[父轨迹重算](../data/manifests/supermimic_q1_R02D_DEV_B_audit_parent5000_20261025.json)、[新轨迹重算](../data/manifests/supermimic_q1_R02D_DEV_B_audit_bounded10282_20261025.json)。退出原件、原审计引用七输入及516冻结在评测后全部核验保持。新实验分配4087808字节，结束共享内存25628397568字节、OOM14、盘余2118354624512字节；未执行test或任何optimizer。完整数组留服务器LOCAL_ONLY。

下一执行P15R四train×两限速，不修改原训练、旧诊断或开发/测试动作。若硬约束无解，需保存失败并检查根/足目标与初态可行性，不能删除约束来通过。

本阶段实现/报告/小证据按Q1审计、提交、推送并独立核远端；实际发布号见Git。原大型工件仍LOCAL_ONLY。开发评测只支持开发方向判断，原库全源/受试者隔离尚未认证；不评test调参。下一根据真实新模型轨迹和P15R结果决定修正，不直接追加训练预算。
