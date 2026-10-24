# SMPL-X身体模型接收与AMASS数值验证

记录日期：2026-10-24，北京时间。阶段：SM-Q1-SMPLX-P01。状态：文件迁移、完整性及16维AMASS载入验证完成；未生成新Q1参考或训练策略。

## 目标与范围

用户告知D:\tmp里的SMPL下载完成，要求先处理。实际接收文件为`smplx_lockedhead_20230207.zip`，独立保留lockedhead版本身份。Windows直接scp到huanyu，Mac仅保存脚本、报告和小型校验收据，未落地模型或原包。

既有AMASS原始库和PHUMA输入保持原身份；本阶段不恢复停止实验。模型由用户提供，Git不重新分发模型；包内文档不作为操作指令。数值载入兼容与原始AMASS拟合时所用模型字节完全一致是不同结论。

## 完成内容与关键决定

- Windows来源`yu@100.65.183.91:D:\tmp\smplx_lockedhead_20230207.zip`，411448739字节；哈希计算前后大小与修改时间稳定。
- Windows和服务器完整SHA-256均为`88d35123fc97151bc258dc6f28f3967ae4c16117e4b452445f20598c43c43099`。直接传输scp显示约5秒，原Windows文件保留。训练平台ED25519指纹与已有Mac固定主机记录一致后进行会话认证；凭据未写文件。
- 服务器原包：`/opt/data/private/YuanQi_Q1/data/references/body_models/archives/smplx_lockedhead_20230207.zip`。
- 固定版本模型目录：`/opt/data/private/YuanQi_Q1/data/references/body_models/smplx_lockedhead_20230207/smplx/`。仅提取预期的三个NPZ及包内md5sums.txt；三模型MD5均符合包内清单，完整SHA另记。
- 预定入口`/opt/data/private/YuanQi_Q1/data/references/body_models/smplx/SMPLX_{MALE,FEMALE,NEUTRAL}.npz`为指向该版本的相对软链接，不额外复制模型。新实验优先显式登记版本路径与SHA。

| 模型 | 字节数 | SHA-256 |
| --- | ---: | --- |
| FEMALE | 137193916 | a32f91b24e13023b979ec0b5c8e9086f5918197132329c8a258ad1f1c6c557f6 |
| MALE | 137147214 | 44e57050d127538e20e10e27f8274ba9ca3dd09c992e954d3d2b5ae5e5bb8313 |
| NEUTRAL | 137106406 | 43d8f3a1375d7c5baae207870a5d51def0f7e6b507df709b4937598b5e7d965d |

## 实际验证

北京时间19:47:48完成，原生退出0，约10.998秒。状态`SMPLX_PACKAGE_AND_16_BETA_AMASS_FK_PASSED`。

三个模型`shapedirs.shape`均为`[10475,3,400]`，J_regressor为`[55,10475]`。400是文件中的混合形体/表情基数量，不能称400个betas。官方smplx载入器显式`num_betas=16`、`use_pca=False`、`flat_hand_mean=True`；实际模型保留16维，未截到默认10维。

从既有AMASS头部索引各选一份男女小样本：ACCAD/Female1General_c3d/A10_-_lie_to_crouch_stageii.npz与ACCAD/Male1General_c3d/General_A10_-__Lie_Down_to_Crouch_stageii.npz，各取首/中/末三帧。真实帧率120Hz、16betas、165维完整pose与root/body/jaw/eye/hand分字段一致；未使用旧桥接的30Hz默认值。中性模型另用三帧零姿态验证，不冒充新增中性AMASS动作。

男女和中性均输出有限的`[3,127,3]`关节坐标；55关节刚体链得到的前22身体关节与完整模型forward最大绝对差均为0。第16个beta增加0.5后，前22关节的最大变化分别为女1.3492mm、男0.9751mm、中性0.5711mm，证明第16维实际参与计算。

软件为numpy1.26.0、torch2.8.0+cu128、smplx0.1.28；本轮为CPU有限样本计算，0新Q1参考、0动力学步、0optimizer更新。容器OOM保持14；内存采样26,233,942,016到26,953,728,000字节，包括文件页缓存，不作为进程常驻内存增长结论。

## 版本、命令与证据

新脚本`script/supermimic_q1_sources/intake_smplx.py`，实际部署SHA`4b63907d2abc4657bc969942fa2dec9aec2fd65553e163c4c4e2b7a25b065be6`；阶段前Q1提交`e0387f05c3b094e9826bc07925be775e3f8d0220`。输出排他创建、完整包SHA核验、成员白名单/大小约束、包内MD5、100GiB磁盘余量与既有OOM检查均启用。

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /opt/data/private/yuanqi_ops/envs/supermimic_a3_20261023/bin/python -u \
  /opt/data/private/YuanQi_Q1/script/supermimic_q1_sources/intake_smplx.py \
  --archive /opt/data/private/YuanQi_Q1/data/references/body_models/archives/smplx_lockedhead_20230207.zip \
  --expected-bytes 411448739 \
  --expected-sha256 88d35123fc97151bc258dc6f28f3967ae4c16117e4b452445f20598c43c43099 \
  --output-dir /opt/data/private/YuanQi_Q1/data/references/body_models/smplx_lockedhead_20230207 \
  --amass-index /opt/data/private/YuanQi_Q1/data/supermimic_for_q1/AMASS_P01_inventory_20261024/all_files.jsonl \
  --source-root /opt/data/private/yuanqi_ops/script/amass_bridge/data/amass_raw/AMASS_smplxg \
  --expected-oom-kill 14
```

上面为实际已执行命令；不可覆盖原输出重跑。复核需新输出名、实时OOM基准与磁盘检查。

- 日志：服务器`logs/SMPLX_P01_intake_20261024.log`。
- 原始收据：版本目录`verification.json`，6898字节，SHA`2eb669daae1eb4c382ff7c5e715e2e86aabed1b93eb801fd4a19c1ddac896cd8`；[原样小型副本](../data/manifests/supermimic_q1_smplx_models_20261024.json)。
- [迁移和软链接清单](../data/manifests/supermimic_q1_smplx_transfer_20261024.json)登记Windows身份、服务器身份和所有目标路径。Mac独立核对收据大小/SHA、三模型校验及16维/FK结果通过。

## 结果与边界

男女身体模型缺项已补齐，可用该固定版本继续AMASS原始字段到Q1参考的候选转换。现有56.34标称小时仍非Q1合格训练时长；未跑完整AMASS数值检查、坐标/支撑/碰撞验证、Q1重定向或策略评测。当前证据支持API、16维形体及有限样本FK兼容，不能证明lockedhead模型与所有标准smplx动作原拟合模型完全一致；后续登记为此版本的候选参考，先做身体几何和Q1质量检查，不静默混入已固定库。

AMASS索引中另有MOYO的`smplx_locked_head`/60Hz/300betas数据；本轮没有解除其隔离或截断参数。BONES metadata已在Windows D:\tmp可见但未迁移，本轮不处理其未完成下载文件或完整动作包。

## 保存与下一步

原包、三模型和收据合计逻辑822903325字节，约0.823GB，入口软链接不额外占模型副本；检查时服务器可用1964811501568字节，约1.965TB。原件全部保留，网盘状态`LOCAL_ONLY`。

阶段报告、当前状态和索引同次更新，代码/小收据按Q1分支审计、提交、推送并独立核远端；本提交号由Git历史给出。大模型和压缩包不进入Git。下一从独立AMASS候选小样本验证原速、形体、坐标和Q1准入，再扩大合格移动参考；本阶段未启动训练。
