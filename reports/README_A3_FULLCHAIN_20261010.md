# A3 全链路测试与短微调

记录日期：2026-10-10，北京时间。状态：进行中。用户已授权执行完整计划，卡点登记后继续独立工作；短微调最多本实验累计200次更新。旧R01/R02/R03继续暂停，不操作真机。

## 目标、推进与验收

官方PT校验、四段SMPL审查、A3重定向、官方与动捕仿真、短微调对照、180ms参考缓冲、ONNX及可用RKNN工具链依次推进。任务台账 `data/experiments/a3_fullchain_20261010/tasks.json` 使用pending/running/passed/failed/deferred，保存阻塞证据和用户待办。普通卡点最多定位10分钟；重启只核验boot/进程/工件后恢复，温度90°C/85°C连续两次和无保存进度保护保留。

衣服数据质量未验收时，使用官方selected20完成微调链路：固定种子42整动作划分16训练/4留出，16环境、官方优化配置，2更新冒烟后完整恢复至200，每10次保存；不改变衣服TEST_ONLY标识。结果不等同于真实衣服效果或硬件通过。

## 版本与恢复入口

- 主机 `/media/yu/FAFF-E9771/YUANQI`，分支main；开始HEAD `a96cb4ad8070edccc40e95675a012b83b5171658`。
- vendor `fe6868ba37034f89b912f0fb851bce19f120266d`；官方发布固定revision与SHA来自 `a3_hf_manifest.json`。
- 初始boot `40497006-bd02-449e-931a-dac12218a7eb`；旧service inactive/MainPID0/disabled，GPU30°C、92MiB。
- 四段SMPL来自Q1目录，Q1并行暂存修改保持。依赖、输入和各实验日志 `logs/a3_fullchain_20261010/`，数据与模型均在服务器data下，LOCAL_ONLY。
- 启动台账：`data/environments/a3-sonic/bin/python -m script.a3.fullchain_support init`；后续模块与命令按实际运行证据补充。

## 当前结果与待办

01:57已完成四段输入的数值、旋转、父树、SHA与30Hz规范化，原帧1766/1767/1969/1923，规范化1766/1767/2251/2253。旧录制平均26.23/25.60Hz、最大间隔157.9/178.0ms，不能称实时固定30Hz。新两段trace以frame_index对齐，身体首帧匹配trace下标30，不按数组下标直接配对。

同一时段四段完整A3运动学参考已生成，29主动关节沿固定URDF匹配，Mink/quadprog与URDF速度限位；使用拟合被动脚开链MJCF生成参考，闭链动态评测另用配套loop资产。合成中立姿势检查通过，真实参考为首帧相对校准诊断，保留TEST_ONLY、training_allowed=false；不证明高保真/足接触/平衡。

官方HF连接本轮全部超时且无下载字节，固定官方PT和配套4个小文件均deferred。清单 `data/experiments/a3_fullchain_20261010/official_download.json`，目标根 `data/models/a3_official_035/checkpoints/035_step200000/`：

| 文件 | 字节数 |
| --- | ---: |
| model_step_200000.pt | 402050931 |
| config.yaml | 37231 |
| meta.yaml | 40 |
| model_config.yaml | 9529 |
| LICENSE | 11358 |

源仓库 `sonic-for-a3/sonic`，固定revision `db1f40787d580af6d1e53df0df72242904a1dae5`，远端目录 `035_step200000/`；清单有完整源地址/目标路径/SHA，等待用户迁移或连接恢复后独立验证。服务器既有候选路径搜索未发现step200000模型。Isaac的 `smpl_sim.smpllib.smpl_eval` 不可导入；快速检索未发现本地模块，先搁置。RKNN工具链缺失。MuJoCo/Mink/quadprog/smplx/ONNX/ONNXRuntime存在。

官方模型基线与200更新微调暂被官方PT阻塞；继续参考格式、延迟逻辑、包装器及导出实现测试。所有新工件LOCAL_ONLY；每个重要阶段补充记录，审计后scoped发布main。只有完成或所有剩余项被已登记卡点阻塞时停止。
