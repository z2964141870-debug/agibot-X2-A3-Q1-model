# SuperMimic for Q1：AMASS与BONES-SEED接入盘点

记录日期：2026-10-24，北京时间。阶段：SM-Q1-DATA02。状态：新增来源登记、AMASS全库头部索引及独立算术核验完成；缺少SMPL-X男女身体模型及完整BONES-SEED原包，尚未进行新增重定向或训练。

## 目标与范围

用户明确AMASS与BONES-SEED都可用于当前Q1任务，缺少的资料要告知。这扩展此前PHUMA-only的后续数据范围，继续SuperMimic可微世界模型方法复现，不恢复旧A3动作或已停止实验。历史PHUMA R01输入、代码及终态不变。

本轮实际在huanyu训练平台、Mac小型索引/文档副本及既有SMB入口进行只读查找；源数据保留原位置，没有将大型数据复制到Mac。找到的数据要进一步转换与验证，不能因目录存在就登记为Q1合格参考。

## 实际发现与关键纠正

AMASS原库已在当前训练服务器找到：

`/opt/data/private/yuanqi_ops/script/amass_bridge/data/amass_raw/AMASS_smplxg`

之前10/22和10/23盘点只确认了`data/motionx`与`data/motion_pool`等位置，没有检查到此脚本目录内的原始库；此前“当前主机未定位”不再代表现状。无需用户再次迁移这份原库。实际NPZ逻辑大小180,217,118,833字节，约180.22GB（167.84GiB），不是沿用历史87GB估算。

| 本次索引分类 | 文件数 | 实际含义 |
| --- | ---: | --- |
| 当前解析器可识别的SMPL-X动作 | 17265 | 23个来源集合，显式120Hz，标称合计56.337981小时；全部16维betas |
| 静态形体/非动作文件 | 490 | 不能计入动作数量或训练时长 |
| 元数据待处理/当前解析器拒绝 | 173 | 保留清单，不当作坏动作或强行补默认字段 |
| NPZ总数 | 17928 | 上述三类之和，不等于17928段合格训练动作 |

男女动作分别11205/6060。时长按`(frames-1)/source_hz`求和；未去重、未排除交互动作，亦未做数值全量检查、Q1重定向、接触/碰撞或动力学跟随。23集合包括ACCAD、CMU、KIT、TotalCapture等；GRAB等含交互需求，不能整库不加筛选接入平地任务。

173个待处理文件中抽核一个MOYO样本为`smplx_locked_head`、60Hz、300维betas；不属于当前标准SMPL-X/16betas入口。保留它的真实字段，不截断成16维或按120Hz强行混入。其余拒绝明细在原始summary中。

旧`script/amass_bridge/smplx2smpl24.py`第267行只识别`mocap_framerate`，缺少时默认30Hz；已接纳原始库实际使用`mocap_frame_rate=120`。因此不能直接复用旧桥接PKL并宣称原速。源码SHA`a6cbbd09c900d2c5741230666ea57c176dfee331f70114d2fcbd03d09d40e69c`，本次没有修改或重跑旧桥接；此问题不改变PHUMA R01的独立30Hz源时间证据。

## 目前缺少什么

| 资料 | 本轮状态 | 接入目标位置 |
| --- | --- | --- |
| `SMPLX_MALE.npz` | 当前搜索范围未找到；现有两个同类`SMPL_male.pkl`不能替代SMPL-X模型 | `/opt/data/private/YuanQi_Q1/data/references/body_models/smplx/SMPLX_MALE.npz` |
| `SMPLX_FEMALE.npz` | 当前搜索范围未找到；需支持至少16个形体系数 | `/opt/data/private/YuanQi_Q1/data/references/body_models/smplx/SMPLX_FEMALE.npz` |
| BONES-SEED `g1.tar.gz`及配套metadata | 当前训练机未找到完整原包；官方公开大小23,499,973,647字节，匿名HEAD返回401/GatedRepo | `/opt/data/private/YuanQi_Q1/data/references/bones_seed/archives/g1.tar.gz`，metadata放同一来源根目录下 |

所需身体模型对应原始库中的男女形体，不能仅用SMPL男性近似模型绕过。BONES-SEED选择G1包优先与论文训练原料对齐，再转换Q1；如果用户已有原人体SOMA包，也可采用独立SOMA到Q1路线。`soma_uniform.tar.gz`45,188,088,297字节、`soma_proportional.tar.gz`45,468,083,856字节是替代/可选来源，不要求与G1包全部重复准备。

北京时间2026-10-24 17:41前再次检查上述两个SMPL-X模型与`bones_seed/archives/g1.tar.gz`指定接入路径，三者仍不存在。本次是目标路径复核，不冒充重新全盘搜索；AMASS原库及此前完整头部索引已经在手，不需要再次迁移180GB原数据。BONES配套文件为`metadata/seed_metadata_v004.csv`及`metadata/seed_metadata_v002_temporal_labels.jsonl`。已在当前聊天明确告知缺项，文件到位后先核大小/SHA和格式，再接Q1。

现有`data/g1_leg/run/data/motion_lib_bones_seed`只有24个PKL，共15,817,101字节，含PHUMA命名动作、walk_onset裁片、镜像及stage/filtered副本；不能当完整BONES-SEED或24个独立原始动作。本轮未反序列化这些历史PKL。

官方仓库`bones-studio/seed`固定revision`2f59b2077b9da34dd4e43618e705c7cb962c9a66`，API标为gated=auto。公开元数据能读取不等于原包可下载；g1/SOMA匿名HEAD均为401，本轮没有接受授权条款、读取token或尝试绕过。匿名API的LFS对象ID被掩码，不能登记为已核验原包SHA。文件到位后先核大小、计算完整SHA，再按固定版本读取官方说明和匹配metadata。

搜索覆盖当前项目data/script、ZHY保留data及15个归档候选；Mac指定资料目录只有旧预览图。旧hp3090的100.70.125.121 SSH连接超时，因此不能断言其历史文件已丢失。SMB只读认证及share列表成功，未递归扫描整个传感器存储。huanyu直连HF发生网络地址错误，官网元数据和401探测在Mac完成。上述是明确搜索边界，不宣称全盘/全机器不存在。

## 实现与可复现命令

新增`script/supermimic_q1_sources/inventory_amass.py`，只解析ZIP/NPY头部和小型标量，不加载markers大数组或执行人体FK。显式读取两种合法帧率字段，缺失或冲突即拒绝，绝不默认30Hz；禁止object数组反序列化，静态文件单列。源目录只读，输出排他创建，保留内存/OOM保护。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  /opt/data/private/YuanQi_Q1/script/supermimic_q1_sources/inventory_amass.py \
  --source-root /opt/data/private/yuanqi_ops/script/amass_bridge/data/amass_raw/AMASS_smplxg \
  --output-dir /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P01_inventory_20261024 \
  --expected-oom-kill 14
```

这是本次已执行命令，不覆盖重跑原输出；以后复核需新输出名与实时资源基准。stdout/stderr保存在`logs/AMASS_P01_inventory_20261024.log`。原生退出0，用时39.912秒，OOM保持14；cgroup内存从27,419,459,584到28,961,013,760字节（含文件页缓存，不能直接认定进程常驻内存增长）。0新增IK、物理积分、策略或optimizer更新。

四项服务器控制通过：120Hz/60Hz两种字段得到1秒/2秒已知时长；缺失/冲突帧率拒绝；静态形体不计作动作；object姿态数组不解pickle。之后另用独立流式程序核对三类文件数、23集合动作数、男女计数、16betas、时长之和及索引SHA，均一致；没有重复全量源SHA。

## 产物与版本

- 服务器`AMASS_P01_inventory_20261024/all_files.jsonl`：12,198,595字节，SHA`501e723765a71167e7186f90df0791b506191cb83fbe98a103810d86caf2a233`，原件LOCAL_ONLY。
- 原`summary.json`：71,718字节，SHA`4eca7619c242ea63711cbe095523343ef0904b9bf86fa915b7b5b6107f398b2b`；小型原样副本[AMASS清单](../data/manifests/supermimic_q1_amass_inventory_20261024.json)。索引SHA不代表180GB全部源文件已校验；23集合各一份样本SHA另列。
- [来源可用性与缺项清单](../data/manifests/supermimic_q1_source_readiness_20261024.json)保存精确路径、大小、官方版本、401证据、搜索边界及独立重算结果。
- 本阶段前Q1 HEAD`a6c18f132bd39efe63e688a9fd757ac058906c68`；新脚本SHA`96af20e269f5c71b91cd209aaf614c892e4661680b2982e5d7340c6de6e9dae3`。发布提交以本阶段Git历史及独立远端核验为准。

原始来源：`https://amass.is.tue.mpg.de/`、`https://huggingface.co/datasets/bones-studio/seed`及其公开API；各人体模型/数据许可分别适用，不把已下载数据自动视为可重新分发。Git只保存本项目代码、报告、小型索引摘要和路径/身份，不上传原始动作或模型。

## 下一步与保存

补齐SMPL-X男女模型后，直接从原始SMPL-X字段构造Q1参考；不使用未核时轴的旧24关节桥接结果。先在训练/开发样本验证形体、坐标、原速、支撑与自碰撞，再按合格时长扩至30分钟/1小时。BONES原包到位则独立索引、源格式/帧率验证、再接G1到Q1或SOMA到Q1路线。保留PHUMA可用原料及原R01，不依赖拿到BONES才继续所有数据准备。

跨AMASS/BONES/PHUMA先登记原动作、裁片、镜像与来源重叠，再固定分组和独立移动评测；不能直接把三库时长相加当独立小时，或让同一动作的机器人版本跨训练/测试。本阶段未生成新增Q1合格参考、未启动新训练。长期运控目标未完成。

README、当前状态、报告索引与新增授权同次更新；检查diff、运行Git payload审计、提交并显式推送`origin Q1`，核对远端ref后同步服务器与用户工作区。大工件仍LOCAL_ONLY，没有新增网盘备份或Mac大型文件。
