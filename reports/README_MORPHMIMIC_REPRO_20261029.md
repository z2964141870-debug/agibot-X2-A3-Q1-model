# MorphMimic源码与最小复现准备

记录日期：2026-10-29，北京时间。阶段：MM02。状态：固定源码及官方在线推理已核；独立离线训练/回放尚未复现，服务器入口阻塞。

## 用户要求与判断

用户要求继续SuperMimic底座与PPO适配两线工作，随后提供MorphMimic项目页，询问是否有代码、能否复现。结论是有公开实现，适合作为同一方法路线的独立对照。它保留可微世界模型、想象rollout和actor/critic，仓库明确说明是依据SuperMimic论文自行实现，不是SuperMimic原作者代码。

本次先查原版可复现性，不把既有Q1项目整体迁移，也不将其G1结果当作Q1已获改善。原两线脚底诊断及候选保留，见[本地时序分析](README_SUPERMIMIC_Q1_DUALTRACK_DIAGNOSIS_20261029.md)。没有新增Q1物理、策略推理、optimizer或训练，没有恢复旧作业、agent或goal，没有修改自动任务。

## 实际完成与未完成

| 层次 | 本次结果 |
| --- | --- |
| 代码是否公开 | GitHub API完整树可读，固定main提交aa40fe46a7fa995ba7a7354ad065a2c8ad64a302，与10月27日初评相同 |
| 源码快照 | 8个关键文件160435字节，逐一核字节数和Git blob SHA-1，另存SHA-256；不含完整仓库、机器人mesh和子模块 |
| 静态检查 | 5个Python文件AST解析通过；实际模块导入、依赖安装及训练CLI尚未运行 |
| 官方在线推理 | 已在应用浏览器加载官方demo，G1/X2 Ultra/T2三机器人Walk画面和运行状态可见，显示约0.7-0.8倍实时，页面已保留 |
| 训练checkpoint | 本次GitHub releases返回空列表，所查README没有成套PyTorch训练checkpoint下载入口；在线demo实际下载了显示49.1MB的推理策略，不能因此说没有任何公开模型资产 |
| 独立离线复现 | 尚未完成；官方网页运行不等同本机独立训练、部署回放或论文性能复现 |
| Q1接入 | 尚未实现；当前registry只有g1/x2/t2，不能把已有共享策略直接装到Q1 |

官方demo是第三方已有策略的在线推理/仿真，不是本项目训练结果；未将其权重另行保存为Q1本地checkpoint。浏览器低于实时不等于服务器训练速度或策略质量结论。

普通Git HTTPS查询约75秒连接失败，原始README和pyproject读取各在30秒超时；改用公开GitHub API读取固定树与blob成功，不需要认证。没有完整clone、安装大型环境或下载SEED全库。

## 依赖、规模与Q1差异

固定pyproject要求Python 3.11、mjlab 1.6.0、Torch>=2.8.0、Warp>=1.14且<1.18；Linux Torch使用cu128索引。当前Mac系统Python为3.9.6，本次仅做不导入依赖的源码检查。后续使用独立服务器环境，不能在现有HP环境里直接uv sync覆盖依赖。

上游README报告4096环境、1小时动作子集约28GB显存；全库约51-58GB/GPU，多机器人配置每机器人一张GPU。这是作者报告，不是本项目测量，也不能保证缩到64环境就必然适配现卡。先测单G1小数据/小批量，完整论文多形态规模另行评估。

`robots/spec.py`要求G1规范29槽位各自唯一匹配本体关节，缺失或重复均拒绝；Q1现有22控制关节不能满足该直接映射。需要独立设计缺失自由度表达或Q1专用维度，并核IMU、关节符号/零位、PD/effort、碰撞/足底和Q1参考。不能仅补零就宣称共享策略支持新机器人，也不能把Q1 NPZ作为G1动作库直接输入。

源代码Apache-2.0，机器人资产和子模块各保留自身许可。本次第三方原始文件仅存被忽略的data，不作为本项目源码重新发布。

## 最小复现方案与一个易误判处

首先完成原版单G1的环境创建、参考/动作契约及零optimizer检查，再执行包含真实actor和critic更新的小闭环，导出后用相同动作回放。工程连通后再比较训练稳定性与普通走效果，最后做Q1独立适配和同底座PPO对照。多形态共享与五点接口不放入首个小闭环，避免同时扩大变量。

已读`TrainCfg.wm_only_iters=500`，`Trainer`只在越过这一门后进入策略更新；入口注释的60轮短示例维持此默认时，只能证明采集/世界模型阶段运行，不能证明策略学习正常。准备的小闭环为64环境、WM/想象batch各128、H32、500轮WM预热、总600轮、compile none；这是待验证工程配置，非论文等规模复现。后续必须核实际policy/critic更新计数，而非只看进程正常结束。

以下命令仅为已检查字段的准备记录，**未执行且尚不具备直接启动条件**。先取得完整固定仓库/必要子模块、真实G1 tiny动作库、通过原生检查，另接独立实训时钟/退出保护。长期单轮最多3600实训秒作为上限，故障不自动重试/扩时；600轮不代表承诺必须耗尽一小时，也不能将初始化/编译凑作训练时间。

```sh
# 在独立服务器checkout及环境中；G1_TINY必须指向已核的G1动作子集。
uv run python -m morphmimic.scripts.train \
  --robot g1 --subset-yaml "$G1_TINY" --gpu 0 \
  --num-envs 64 --wm-batch 128 --imag-batch 128 \
  --horizon 32 --wm-only-iters 500 --iterations 600 --compile none \
  --log-every 10 --ckpt-every 100 \
  --log-dir /opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_G1_SMOKE_20261029 \
  --run-name source_smoke
```

该命令尚无本项目独立时钟guard，不能从本记录自动开训。正常退出、有限loss和小样本回放均不能代替稳健步态验收。

底座稳定性可比较的具体源码项包括WM/critic LayerNorm、critic目标上界、WM状态坐标及归一化、合法想象起点；这些与本项目HP实现的准确差异及效果仍待独立对照。它们不是已经定位的Q1退化原因。PPO/残差适配继续保留，后续分别核底座变化与适配净增益，不能相加两份独立结果替代组合实测。

## 版本、产物与来源

Mac快照根：`/Users/yu/Documents/ChatGPT/元启/data/references/morphmimic_review_20261029`。

- `tree.json`：完整文件树，183个blob约140.37MB的仓库规模仅统计，未全部下载。
- `source/`：README、LICENSE、pyproject、constants、registry、Translator、训练入口和trainer共8文件。
- `source_receipt.json`：固定提交、8文件大小/Git blob/SHA-256、5个静态解析结果。
- `repro_plan.json`：实际默认参数、工程小闭环候选和执行前缺项。
- [小型核验清单](../data/manifests/morphmimic_repro_readiness_20261029.json)：版本、来源、官方在线演示、网络失败及范围边界。

一手入口及固定源码：

```text
https://lzyang2000.github.io/morphmimic/
https://github.com/lzyang2000/morphmimic
https://lzyang2000.github.io/morphmimic/demo/
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/README.md
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/pyproject.toml
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/src/morphmimic/robots/spec.py
https://github.com/lzyang2000/morphmimic/blob/aa40fe46a7fa995ba7a7354ad065a2c8ad64a302/src/morphmimic/train/trainer.py
```

## 阻塞、同步与恢复

本轮此前唯一25386入口返回Connection refused，一次有界本地核查无18873监听且控制socket缺失，远端UNKNOWN；没有因为本次问题重复连接、改旧入口、读取认证或重启桥。当前无法在服务器取得完整环境、G1数据和实际GPU验证，离线训练复现仍待。

记录在权威Git Q1分支保存，Git推送/独立40位SHA及Mac镜像以实际发布收据为准；25386镜像明确待连接恢复，不能称三端一致。原监督ACTIVE每10分钟与其他任务保持，候选采纳null，未结算整体目标。

恢复先读本记录、[当前状态](README_CURRENT_STATE.md)及[双线诊断](README_SUPERMIMIC_Q1_DUALTRACK_DIAGNOSIS_20261029.md)。下一步是恢复授权入口后的独立原版G1环境和零更新检查，不直接改旧Q1冻结runtime或恢复RF15/RF16。
