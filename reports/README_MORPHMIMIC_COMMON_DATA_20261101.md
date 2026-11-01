# MorphMimic共同动作数据

2026-11-01，MORPHMIMIC_FAITHFUL01 ACTIVE。G1/X2 Ultra/T2共同动作源已按G1真实1h子集精确提取559条，1030127964B，仅保存在25386。现有G1训练持续运行，未启动第二GPU作业；这一步是共同人体动作源准备，尚未完成X2/T2机器人重定向或共享训练。

## 现场和缺口

原G1 controller362416/start38298223与worker362421/start38298276于06:02:05核仍存活，原boot4d859008-4c83-4a6d-814d-cbb9dfe885c9、GPU-26028785-9fb9-45af-e614-4cc53af15d08、唯一根一致，23646轮/五optimizer均更新/OOM0/非有限0；06:31:51继续到26395轮。Windows新会话36078进入容器前超时，先记logs/q1_connection_recovery_state.json，再仅一次15595原生恢复；未重启服务器、容器或训练，故障计数不得被心跳重置。

实际训练1h.yaml有559条/559唯一stem，metadata v004恰有559行对应；AP03既存583 BVH仅9个stem相交、550条缺失，不能将AP03库直接充作共享动作。已存soma_uniform.tar.gz为45188088297B，其成员前缀与元数据均为soma_uniform/bvh/<date>/<stem>.bvh。必须提取G1精确559条，保持数据集及后续配对一致，不用另外挑选的易动作替代。

## 提取入口

新增独立script/morphmimic_cross_robot/prepare_sources.py，不在正在运行的训练源码目录中。读取原subset YAML和CSV元数据，拒绝重复stem/歧义元数据/非法路径；仅按完整归档成员名复制普通BVH到新根，逐文件流式SHA，不覆盖旧输出。压缩包顺序读取，已消费的8MiB整段只发FADV_DONTNEED，记录advice字节但不称实际释放；不删文件或全局drop_caches。最多一个CPU低优先级worker，不加载Torch/Newton/Warp，不建GPU上下文、0新积分/策略/optimizer。

内存达28GiB或磁盘余量低于原100GiB时只中止提取，现有训练30GiB守卫和GPU锁不变。输入前后核size/mtime/inode；归档只记录已读取压缩前缀SHA，除非完整读完不冒称整包SHA。结果保留输入身份/计划SHA/每个BVH大小SHA/实际读量，权重和BVH仅服务器。9项CPU反例在本机和服务器均通过；首版直接调用extract未在创建输出前检查资源，测试发现后已补入口检查。该反例未造成服务器提取失败，未改训练。

执行源码7107be0c3743deefbd37654443ed877e0eed18f2已独立核40位远端，7记录380615B Git/Mac/25386发布快照大小SHA一致，随后仅执行一次：

```sh
data/environments/morphmimic_original_20261029/bin/python script/morphmimic_cross_robot/prepare_sources.py \
  --archive /opt/data/private/ZHY/data/references/bones_seed/inbox/soma_uniform.tar.gz \
  --subset /opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_FAITHFUL01_G1_DATA_20261101/run/npz50/subsets/1h.yaml \
  --metadata /opt/data/private/ZHY/data/references/bones_seed/metadata/seed_metadata_v004.csv \
  --output /opt/data/private/ZHY/data/supermimic_for_q1/MORPHMIMIC_FAITHFUL01_COMMON_SOMA_1H_20261101 \
  --source-commit 7107be0c3743deefbd37654443ed877e0eed18f2 --detach
```

北京时间06:51:19.535429至07:06:18.835833正常完成，CPU worker391817/start40072443已退出。559条与plan成员精确相同，无额外BVH；09:21:56独立逐文件重新核大小/SHA全部相同，输入前后device/inode/size/mtime一致。实际读取压缩前缀44987525120B，不是完整45188088297B归档；前缀SHA d8adfccc7e5263af8174c9bc0f88909121e9d08d9602244155ddf137e4a1e8523只对应实际前缀。已消费页advice44979716096B不能当实际释放量。

提取全程0新物理积分/策略推理/optimizer/重定向，未创建GPU上下文。09:21原G1已42217轮，原两进程/start/boot相同，唯一GPU计算PID362421、GPU8779MiB/OOM0；容器18167828480B是现场快照。结果小记录包SHA82f7344b5c0f8b9aea1e09a81d56febe0f0d6fffaf5a47466e7fb0a51b64ff4c已Mac/25386一致，未传BVH。[完整结果及559文件清单](../data/manifests/morphmimic_faithful_common_sources_results_20261101.json)/[原准备清单](../data/manifests/morphmimic_faithful_common_sources_preparation_20261101.json)。

## 原版跨本体契约

aa40fe46固定实现的5份相关源码已从服务器读取并与原Git blob核同。unified必须share=all，共享WM/motion和KP策略/两critic的参数；每机器人保留环境、reference、replay、状态normalizer、限位、默认姿态、残差尺度和形态buffer。默认pool_norms=True时策略输入统计还须跨机器人池化，此源码选择与论文按机器人统计的描述差异继续保留。各次WM小批/策略/critic小批均先平均三机器人梯度，再clip及Adam；不能按机器人依次完整Adam更新冒充同一次共享更新。

latent对齐使用共同stem、各版本最短帧长、seed*1000003+iteration的同片同帧抽样；各机器人编码与stopgrad的三机器人latent均值相配，梯度只回传该机器人编码。单体G1这一项为0，单体训练本身不证明跨本体。单卡适配须核每阶段梯度/clip/Adam顺序、独立buffer、共享参数和normalizer、对齐采样和完整保存；当前尚无可执行的单卡unified验证，不降低为3个单体后称完成。

官方入口一机器人一rank，NCCL及cuda:local_rank固定，直接将三个机器人传给单rank会被明确拒绝。当前3090单卡需要独立验证等价的同步实现，原运行文件不热改。资源profile只证明G1 E1024可运行，并未证明三个机器人可同时驻留。

## 重定向后续

官方SOMA Retargeter短提交cbdb2b4已解析并核为cbdb2b494a2af82137bf9b6f15122d60167281c4；固定README和app/bvh_to_csv_converter.py已按API Git blob取得并校验，不跟main更新。当前未发现已装bvh_to_csv_converter.py；未安装/运行新环境。后续沿原MorphMimic assemble_robots构建X2 distal effectors修正和T2目标、使用原soft limits及batch10/Newton配置、原validate_csv与convert_seed/build_subsets --clips-from，保存真正交集/帧长。原始SOMA提取完成后再准备固定依赖，GPU重定向或资源测试在G1释放GPU后串行，不干扰现训。

本阶段未做新物理评估、未得步态改善或统一模型结果。goal ACTIVE/采纳null，G1按160000轮继续；不启agent、不改其他自动任务，发布按独立40位远端及Mac/25386大小SHA收据。
