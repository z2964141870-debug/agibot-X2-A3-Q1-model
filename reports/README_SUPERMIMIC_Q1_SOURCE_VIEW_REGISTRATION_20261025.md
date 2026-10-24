# PHUMA来源排除视图的训练登记

记录日期：2026-10-25，北京时间。阶段P09。状态：独立入口及四项控制完成，服务器真实预检待；未登记或启动第二轮训练。

## 目标与范围

用户要求BONES交师兄，继续现有资源。[P08B-V2](README_SUPERMIMIC_Q1_PHUMA_LIBRARY_20261025.md)已有612训练片/2191.34秒、dev6/test7。原扩库登记只接受原始组库完成状态，不能把来源排除视图改名冒充原组库。本阶段显式核验新视图的合同，再复用已验证64环境/4模型槽的SuperMimic训练与七阶段配对流程。

当前R02C继续固定340训练参考/7000新轮，代码发布d21a880不变；不修改其496冻结文件，不热换参考，不恢复原FAILED、A3或STOPPED。新入口不自动追加训练，也不以36.52分钟参考意味着已取得移动效果。

## 完成内容与决定

新增`script/supermimic_q1_sources/phuma_source_view.py`及`register_phuma_view.py`。前者逐字节绑定原组库、原独立几何记录、排除producer、排除前/后独立别名审计及相同HumanML/AMASS索引。由原冲突证据重算应排除的训练名单；完整比较2885候选的manifest/check，只有声明的训练资格和派生出处可以变化，保留参考字段、原19门和held-out名单必须完全相同。重新核对全部保留NPZ字节、原trainer准入和按独立检查求和的时长；重新计算保留别名行的跨split冲突，不仅相信收据中的零计数。

`source_isolation_certified=false`和`geometry_checks_rerun=false`保持。已知HumanML原事件/受试者目录候选排除不等于全来源或生物身份隔离证明；其他命名空间、集合更名候选与未解身份仍保留限制。

后者提供独立`check/register/run`入口。`check`只写一个新小收据，0积分/optimizer；`register`才创建新实验并冻结上述证据及原训练依赖；`run`复用原expanded controller、trainer和七阶段动力学比较。方法、H32、actor128、WM64和每组10批不变，初始化为随机、无父权重。默认预算12000轮是可登记选项，本阶段未据此启动。原expanded/PHUMA/partial代码不改。

## 验证与命令

权威工作副本`/tmp/yuanqi-supermimic-q1-20261023`、Q1分支，阶段前发布`d7565cc47acedeb0734706325d84fdacefaeb9ed`已独立核远端。Mac四项合同控制全部通过、无跳过，三个新文件语法通过：精确排除且输入不变；拒绝几何/参考/held-out/时长变动；拒绝旧审计、索引变化、漏行或遗留冲突；拒绝伪造原组库状态、隔离认证、几何重跑或排除名单。这些是入口控制，尚非真实服务器预检或策略效果。

先发布并独立核远端，再逐SHA部署，检查新文件不与496冻结路径相交。服务器预检使用原Python、无新安装：

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python \
  -m unittest discover -s script/supermimic_q1_sources -p test_phuma_source_view.py -v
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  script/supermimic_q1_sources/register_phuma_view.py check \
  --root /opt/data/private/YuanQi_Q1 \
  --library /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P08B_subject_excluded_library_V2_20261025 \
  --source-aliases /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/SOURCE_P08B_aliases_20261025/receipt.json \
  --expected-oom-kill 14 \
  --output /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/PHUMA_P09_view_registration_check_20261025.json
```

OOM14是本次现场基准，恢复时重核。预检保持共享内存28.75GiB/磁盘100GiB余量及view512MiB保护，先检查再读库，完成再检查；不安装依赖或初始化CUDA。已有输出必须拒绝覆盖。预计小收据和stdout分别在上述JSON及`logs/PHUMA_P09_view_registration_check_20261025.log`，实际结果完成后补记，不能预写成功。

## 结果、保存与下一步

P09当前只完成实现与本地控制；不登记新实验、不启动新训练。先取得真实预检并保存原样小证据，再依据R02C固定预算后的配对效果决定下一独立实验。参考浮脚/滑动缺口详见[D04](README_SUPERMIMIC_Q1_PHUMA_REFERENCE_DYNAMICS_20261025.md)，需要单独处理，不能因入口接通就认为参考动力学已解决。

代码、本文、索引与当前状态按Q1审查、payload审计后提交并显式推送，独立核远端。大型NPZ和模型仍服务器LOCAL_ONLY，Mac只保留小记录。本阶段从476166d以来已发布25份文件/732699字节机械同步到用户恢复镜像、逐字节一致，未覆盖未知内容或加入BONES脚本。

准备代码已发布`ec58cf2b414da04ef0311ecdae3d12b0d8037d32`并独立核远端Q1，三份新文件未部署。04:17复核发现R02C已于04:07:44非有限数值失败，本阶段不据先前RUNNING快照继续登记新实验。失败证据先入[R02C记录](README_SUPERMIMIC_Q1_R02_PARTIAL_CONTINUATION_20261025.md)，优先解决动作接口/数值异常；P09真实只读预检仍可随后完成，四项本地控制不等同服务器结果。
