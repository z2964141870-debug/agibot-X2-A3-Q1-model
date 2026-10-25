# PHUMA P17独立目录排除视图

记录日期：2026-10-25，北京时间。阶段P17。状态：实现/本地五项控制/真实父库只读预览和2885行变换核验通过；正式新库、动力学评测和训练受共享盘0可用阻塞。

## 目标与完成内容

继续使用PHUMA、AMASS和SMPL-X复现SuperMimic世界模型/BPTT/critic。承接[P16](README_SUPERMIMIC_Q1_SOURCE_DIRECTORY_AUDIT_20261025.md)的GRAB/s1、s10目录候选，准备独立排除视图，同时保留原HumanML排除来源链、十九门及held-out名单。BONES获取由师兄处理；P15正式八组及退出权重动力学评测仍待存储恢复。

新增`filter_phuma_directory_conflicts.py`及对应测试。先调用已控P09 `validated_view`验证父P08B来源链、参考字节及trainer准入；从完整manifest/check/split重算GRAB/sN目录候选，只排除与held-out目录候选相同的合格train。通用subset目录不自动推断为受试者。父输入、旧HumanML排除记录及所有参考/十九门均保留。

另绑定原P10 selection SHA`1b23939b731dcdfedc8b27732e4f22f92a4faf07412dff0dbea8529e6d969ba9`，对整个父库重核原source_group、原字节SHA、HumanML原事件及集合更名后的目录候选均不进入预留P10测试。原P10十五片即使0准入仍继续预留。

正式构造仅新建小型JSON/软链接，逐保留NPZ重读50Hz时轴/大小/SHA，调用原trainer准入，最终重核全部父证据。另设`--preview`，只读验证父库并输出排除算术，不创建目录、不做IK/积分/optimizer，不把算术称新库准入。新视图完成后还须独立来源复查及专用登记合同，原P09登记器不能直接将新状态冒充原P08B。

## 版本、控制与资源

权威副本`/tmp/yuanqi-supermimic-q1-20261023`/Q1，阶段前提交`3f5589a085fb1c7b75005aa5e4cdf245a828b72e`。生产者SHA`601cf3a43fe3e174fa5ea96da783e887f06df37ddf724f838696103696eef987`，测试SHA`e0ad340a2496b255c2f44bbf4765038b092bc6a1766ca1039385c02c180697f6`。既有P08/P09/P10、P12C/P14/P15与训练代码字节未改。

Mac五项控制/0跳过，0.004秒、原生退出0：仅合格GRAB目录候选排除且输入不变；拒绝重复/划分/参考/几何门/来源SHA异常；新文档保持十九门、旧排除史、held-out及原拒绝项，拒绝伪造排除/认证；P10组/字节/原事件/目录候选及不完整证据拒绝；100GiB保护失败发生在父库读取和新目录创建之前。[资源与本地控制小日志](../logs/session_records/supermimic_q1_P17_preparation_20261025.log)保存原生输出。

08:47:04只读资源仍为盘余0、OOM14/内存13493166080字节，原R02D三终态保持，trainer/controller不存在、guard只余Z，P15正式目录不存在。正式模式在已有父目录先完整检查100GiB盘余，避免建目录前预算根不存在的旧错误；再使用28.75GiB/OOM14、64MiB输出、512MiB父元数据/链接预算。当前不部署文件到已满服务器或运行正式构造。

## 可复现命令与产物

服务器根仍为`huanyu:/opt/data/private/YuanQi_Q1`，Python仍为现有环境。实际OOM基线和盘余须在正式执行前重核。代码发布后可先将精确SHA代码在内存中执行`--preview`，保存stdout到Mac小收据，不向服务器写新文件。正式文件部署及以下建库调用等待100GiB盘余：

```bash
ROOT=/opt/data/private/YuanQi_Q1
DATA=$ROOT/data/supermimic_for_q1
PY=/opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -B "$ROOT/script/supermimic_q1_sources/filter_phuma_directory_conflicts.py" \
  --root "$ROOT" \
  --library "$DATA/PHUMA_P08B_subject_excluded_library_V2_20261025" \
  --source-aliases "$DATA/SOURCE_P08B_aliases_20261025/receipt.json" \
  --walking-holdout "$DATA/PHUMA_P10_walking_selection_20261025.json" \
  --expected-oom-kill 14 \
  --output "$DATA/PHUMA_P17_directory_excluded_library_20261025"
```

在已有正式代码路径时，末行替换为`--preview`即为只读预览。当前预定正式目录未创建；预览待实测，不能称593片新库已经通过。上述代码和本地控制不能代替真实来源绑定、NPZ时轴检查或动力学跟随。

## 保存、边界与下一步

`source_isolation_certified=false`保持，目录候选不是生物身份认证。P16估算排除19片/75.24秒后593 train/35.27分钟、dev6/test7，但必须以真实新构造结果为准。原R02D输入与STOPPED终态不改，不新增第二路训练。

本阶段代码、测试、README、小日志和恢复入口审查/payload audit后显式推Q1、独立核远端，再同步用户镜像；大工件仍服务器LOCAL_ONLY。下一先真实父库只读预览及变换不变量核验；正式转换/动力学评测/训练保持等待存储恢复。

## 08:59真实父库只读验证完成

准备代码发布`49bef24926a48570e0bcc1d3f52e13a5262b50ea`已推Q1、独立远端ref匹配。未在已满服务器部署文件；经已有SSH把发布的16187字节/SHA`601cf3a4...`精确代码编译到临时Python进程内存，使用现有训练Python、`-B`、单线程/无CUDA；显式确认预定生产者路径和P17正式目录均不存在。

实际调用原P09父库完整验证通过（来源链、全部保留参考字节及原trainer准入），再对625父参考/207 HumanML别名核P10的15个预留原事件/目录候选。新入口真实重算恰19片/75.24秒，候选593 train/2116.10秒、dev6/test7，和P16独立影响计算一致。

内存调用真实documents变换后，对原manifest的2885行及原check的2885行逐行核对，除了声明的19个train资格/排除字段，所有源记录、参考、十九门、原拒绝项完全保持；父文档前后相同、原HumanML排除史保留、held-out名单不变。606个保留候选仍只是文档投影，0新Q1文件、IK/积分/optimizer及正式新库，未据此称新时轴检查或新trainer准入。

A首次投影通过，但包装器的health_after_projection字段复制了预览阶段值；B只修包装器在变换后重新采样资源，生产代码/输入不变，再完成相同真实核验。A/B完整源码和各自原生stdout均保存于[执行记录](../logs/session_records/supermimic_q1_P17_real_preview_20261025.log)，A原件不修改。B的[原生stdout JSON](../data/manifests/supermimic_q1_P17_readonly_preview_20261025.json)状态`PHUMA_P17_REAL_PARENT_READONLY_PREVIEW_AND_DOCUMENT_CHECKS_PASSED`；08:59:52预览采集时间，变换后内存13837062144字节/OOM14、盘余0。

Mac独立将P17候选集合与P16原生影响收据连接，19片逐clip一致；625/207/15来源覆盖、两份2885行计数、实际变换后资源字段及未建库/未认证边界一致。B小JSON4492字节/SHA`3350fd9d92e90e1e7bcf26165c51190c4954d8ced45547f121b9d672b7e32d92`，本地复核原生退出0、输出附同一日志。

## 存储边界与恢复顺序

09:02:02只读du原生退出0，Q1工程`data/`实际分配19951194112字节（19.95GB），R02D为2464641024字节。该统计不跟随外部数据链接、不统计其他用户空间；[整理小收据](../data/manifests/supermimic_q1_P17_owned_storage_summary_20261025.json)明确不是服务器原样文件。即便删除整个Q1 data目录也不足100GiB余量，因此不能把清理几个本项目checkpoint视为已解除阻塞；没有删除资产或把工件改写到overlay/tmp。共享盘仍0，原资源协调问题保持等待，不追加重复询问。

连续goal turn的同一存储阻塞持续存在；本轮完成新入口及真实只读核验，但下一条科学执行必须等待外部空间恢复，不能通过继续重复元数据检查证明移动成功。恢复至少100GiB后先核当前资源/源码输入身份，执行新P15八组及独立审计、完成原退出权重的独立配对动力学评测/视频；随后据真实效果决定P17新构造及来源复查/专用训练登记。原R02D终态保持，复现最终目标未达。

阻塞审计已回读当前聊天最近五个turn，确认同一存储条件连续三次goal turn出现：P15/退出审计、P16来源复核、本轮P17准备与真实预览。前两次均有实质进展；本轮可在原地完成的准备和真实只读检查已完成，正式科学验证需要外部存储状态改变，且没有活跃训练可供等待。对应turn身份及判据附执行日志，不把聊天意向或旧状态文件当作活跃作业。

09:06:56最终原生复核仍为盘余0，OOM14/内存13514121216字节、原三停止终态保持、guard为原身份Z而controller/trainer不存在，P15目录仍未创建。完整只读源码及原生输出续入同一执行日志，后续没有启动科学计算。

本次真实结果、小收据及源码日志更新当前状态和索引，payload审计后提交推送Q1、独立核远端并同步用户恢复镜像；没有新服务器大工件或备份。
