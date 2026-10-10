# A3专用工作目录与官方PT导入

记录日期：2026-10-10，北京时间。用户明确要求在hp3090磁盘下划分 `YUANQI_A3`，与已有YUANQI和YuanQi_Q1采用相同四目录结构。状态：目录/代码副本及PT导入完成；匹配配置包未完整，未启动评测或训练。

## 目标与完成内容

后续A3开发位置为 `/media/yu/FAFF-E9771/YUANQI_A3`。通用研究/教学与已有A3历史实验保留 `/media/yu/FAFF-E9771/YUANQI`；Q1继续 `/media/yu/FAFF-E9771/YuanQi_Q1`，本阶段不修改其工作树或暂存区。

```text
/media/yu/FAFF-E9771/
  YUANQI/                         通用工程、教学/研究及原A3历史
  YuanQi_Q1/                      Q1专用工作副本（Q1分支）
  YUANQI_A3/                      A3后续工作副本（main分支）
    data/                         模型、数据、训练与评测工件
    logs/                         stdout/stderr与执行记录
    script/                       代码、脚本、配置
    reports/                      README、重要结论及恢复入口
```

新目录从YUANQI已提交main `42e7e340481e5f5665924d46de3b79b676b1e1b6` 本地独立clone，保留代码与报告历史，Git对象不使用hardlink。远端仍为用户指定的 `git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`，跟踪origin/main；用户本轮指定目录，未指定新Git分支，因此继续已有main发布规则。不同工作副本的Git暂存区独立，后续每阶段先fetch并检查远端，避免并行提交互相覆盖；不强推。

新data下已准备models/environments/training/evaluation/experiments/references。旧实验的大型工件未复制或删除，避免改写历史和重复存储。复用链接如下，作为读取入口：

| 新目录入口 | 实际资源 | 使用约定 |
| --- | --- | --- |
| script/vendor/sonic_for_a3 | YUANQI/script/vendor/sonic_for_a3 | 固定commit fe6868ba37034f89b912f0fb851bce19f120266d，只读复用 |
| data/environments/a3-sonic | YUANQI/data/environments/a3-sonic | 已安装Python环境，只读复用，不在本轮安装依赖 |
| data/references/a3_fullchain_20261010_history | YUANQI/data/experiments/a3_fullchain_20261010 | 旧22项台账/数据/诊断的历史读取入口，不从此处续写 |

新生成数据、评测和训练写新根目录；现有Python模块ROOT按模块文件位置解析，复制到新副本后会定位YUANQI_A3。旧R01/R02/R03服务和账本仍指旧位置，保持暂停，不因目录创建重启。不能把旧全链路的绝对路径JSON复制后当成新实验，恢复时先读各manifest，独立初始化新实验；旧数据按源SHA复用并写新产物。

## 官方PT实测

用户手动下载文件：`/media/yu/FAFF-E9771/model_step_200000.pt`。没有下载至Mac，也没有删除这个原文件；已在服务器复制至：

```text
/media/yu/FAFF-E9771/YUANQI_A3/data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt
```

原文件和副本均为402050931字节，SHA-256均为 `9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d`，与固定官方HF revision `db1f40787d580af6d1e53df0df72242904a1dae5` 的清单一致。复制经临时文件大小/哈希核验、fsync后发布。服务器CPU重载与actor/critic有限性检查通过，checkpoint源计数为200000；这是预训练来源计数，未来微调从0开始。可用优化器张量同样检查有限性，优化器不作为微调初始化来源。

完整证据见 [PT导入清单](../data/manifests/a3_official_import_20261010.json)，原始stdout在新目录 `logs/a3_workspace_20261010/import_official.log`。本轮检查无需网络，不把PT交给未验证来源的代码；使用已固定官方vendor与现有环境。模型仍LOCAL_ONLY，百度网盘往返校验未完成，不记为已备份。

磁盘下仅找到这个PT，以下4个配套文件尚未出现在目标目录。它们应来自同一个固定HF版本、远端 `035_step200000/`，放到PT同目录；精确大小/SHA在导入清单configuration字段：

| 文件 | 预期字节数 | 状态 |
| --- | ---: | --- |
| config.yaml | 37231 | 缺失/未验证 |
| meta.yaml | 40 | 缺失/未验证 |
| model_config.yaml | 9529 | 缺失/未验证 |
| LICENSE | 11358 | 缺失/未验证 |

本阶段只证明PT来源身份、重载和有限性，不证明完整模型包、策略跟踪效果或训练能力；不能把旧实验config.yaml冒充官方匹配配置。待这4个小文件核验后，按已有全链路计划继续官方10策略步冒烟与selected20，再200更新短微调。

## 中断恢复与验证

初次clone/小文件传输期间SSH断开，重新连接实际boot为 `5723f6c9-951e-4866-b4dd-cf4663b8084d`，首次新目录未落盘。已重新创建并同步文件系统；按用户要求只登记和恢复，不定位重启根因。PT原文件经新boot后重新计算身份，导入检查在CPUQuota100%的临时service执行，退出0，不启动GPU负载或旧训练。

新Python路径解析到 `/home/yu/miniconda3/envs/x2-sonic-isaaclab/bin/python3.11`，vendor固定commit与既有工作树身份核验。目录结构、import receipt、Git忽略模型、代码入口及旧service状态按发布前日志核验；新目录没有后台训练服务。大文件只留服务器，Mac仅保存本README、代码和小型JSON清单。

可复现命令（在服务器执行）：

```bash
cd /media/yu/FAFF-E9771/YUANQI_A3
git status --short --branch
readlink -f data/environments/a3-sonic/bin/python
git -C script/vendor/sonic_for_a3 rev-parse HEAD
data/environments/a3-sonic/bin/python -m script.a3.import_official \
  --source /media/yu/FAFF-E9771/model_step_200000.pt
git check-ignore data/models/a3_official_035/checkpoints/035_step200000/model_step_200000.pt
```

README、索引/current state、AGENTS及小型清单按阶段规范审计后提交main并推送，比较本地HEAD与refs/heads/main；原YUANQI工作副本只在确认干净后快进同一提交，Q1保持原状态。版本以Git历史和清单的代码SHA为准。

## 下一步

补齐并独立验证4个官方小文件；随后在新A3目录初始化数据/实验台账并沿此前已授权全链路测试推进。Isaac smpl_sim指标依赖、RKNN工具链和动捕质量问题仍是独立待办。评测/微调模型和日志全部写YUANQI_A3，不自动追加旧长训，不操作真机。
