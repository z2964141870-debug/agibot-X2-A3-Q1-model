# A3切换humanplus工作机

记录：2026-10-17，北京时间22:33起，23:29补充。阶段HP01：代码工作区、只读盘点及固定MJCF的96网格实化完成；WAIT_INPUT_TRANSFER_AND_EXTERNAL_COMPUTE，运行验收/训练未启动。

## 目标与授权

用户明确要求改在humanplus@100.101.30.68工作。本指示覆盖本工程继续在ZHY启动的旧安排；Mac仅保存小代码/记录，原hp3090仍不训练。新独立工作根为/home/humanplus/YUANQI_A3，沿用data/logs/script/reports；没有改动既有EgoLocate、SONIC、x2_teleop_final或projects/ZHY工程。

本次不是恢复旧STOPPED实验。ZHY E51最后成功快照未登记，本次未能重新查其账本；不在新机沿用旧boot/内核基准、不伪造旧账本，也不直接运行固定到ZHY路径的finite_retention_trial。

## 已完成

SSH登录成功，建立仅保存在Mac进程中的复用连接；没有保存密码、复制私钥或新增authorized_keys。GitHub HTTPS只读clone成功，工程HEAD为48449c427cd37f247f7918a624b0eb5b99d212f3/main，725个跟踪文件落地。

官方vendor独立clone，固定fe6868ba37034f89b912f0fb851bce19f120266d，工作树干净。跳过LFS实化，模型/mesh等大型资产尚未迁入；LFS指针存在不算资产齐全。没有升级到仅下载统计变更的main。

23:29后续仅补齐固定评测MJCF所引用的网格。标准库ElementTree读取compiler.meshdir及asset/mesh，确认无include，按固定Git对象中的LFS指针得到96个路径、预期大小和SHA；合计19,912,414字节。限定这96个路径执行一次git-lfs pull，返回0，再逐文件核验实际大小/SHA，96/96通过、vendor工作树仍干净。未下载模型、XR工具链或完整文档资产。清单为data/manifests/a3_humanplus_mjcf_assets_20261017.json；这只证明MJCF网格闭包齐全，不证明MuJoCo/Isaac运行通过或训练资产全部齐全。来源与许可已读固定vendor的docs/THIRD_PARTY_NOTICES.md及LICENSE；机器人资产声明MulanPSL-2.0，既有声明保留，工件LOCAL_ONLY、不进本项目Git。

新增script/project_tools/humanplus_inventory.py：只读主机/GPU/可见进程/包METADATA/旧计划输入是否存在，不import torch/Isaac，不加载模型或计算张量，不遍历凭据。22:39实际执行完成，原始JSON保存在新机logs/a3_humanplus_20261017/inventory.json。

## 硬件、占用与环境

- 主机Legion-Y9000P，Ubuntu24.04.4、kernel7.0.0-38-generic，boot afb6796b-3883-4df4-99ea-bc381c52ea13。
- GPU RTX5060 Laptop，8151MiB，驱动595.99.02；22:39显存4599MiB/利用率6%/50°C。PID808677为zed环境Python、cwd=/home/humanplus/Project/EgoLocate，计算显存3278MiB、RSS约3.61GiB；父子小Python也在。不停止其他任务。
- 内存33042489344字节，可用19416887296字节；home所在盘可用196499017728字节。22:33 CPU封装78°C，单次值不是过温保护证明。未改驱动/功率/BIOS、未操作真机、未全局清缓存。
- 候选环境/home/humanplus/anaconda3/envs/x2-sonic-isaaclab：Python3.11.15（文件/conda元数据）、torch2.8.0+cu128、IsaacSim5.1.0.0、IsaacLab0.54.2、MuJoCo3.3.7、Accelerate1.14.0、Transformers4.57.6、NumPy1.26.0。IsaacLab editable指向x2_teleop_final/third_party/IsaacLab-v2.3.2，只读保留。包存在不代表兼容或验收通过，不修改共享环境。
- torch/optim/{adam,adamw,optimizer}.py和accelerate/optimizer.py的4份源码与ZHY原登记SHA相同；其余包、配置、ABI、实际fused CUDA、Isaac/PhysX和闭环均未验证。

## 遥测与输入缺口

普通用户dmesg被拒绝；经sudo仅只读取得内核日志，未改sysctl或sudo配置。保存kernel_initial.log，117936字节、SHA256 86925291fd8aaaecb5cf76273d736df035d72bcb1a4f86eb9bbdcfdbe52865e4；项目现有FATAL表达式匹配0条。这只是新机初始观察，不是已登记连续健康监控。后续原CPU90/GPU85连续两次5秒、4GiB保留、900秒/阶段与3次5秒启动检查仍需实现并实际通过；当前cgroup v2桌面环境不能套ZHY v1容器记账。

旧计划13个直接输入中11个缺失，关键为官方035 PT、E11 teacher rollout、E12 module_outputs、四段已验MotionLib、E45配置/prepare和E50结果。还需E23合格参考及E26/E02合同/轨迹作为来源证据。只做限定目录检查，没发现可直接复用的035资产根；不声称全机都没有模型。完整缺失条目在inventory.json；迁移清单见data/manifests/a3_humanplus_bootstrap_20261017.json。

humanplus到59.77.15.15:25115的TCP端口可达，但没有已验证SSH连接：严格主机密钥校验缺少已信任键而停止，认证尚未测试，不关闭StrictHostKeyChecking、不猜密码。hp3090离线，旧复用socket仅在旧机，不能当现有迁移入口。已向用户询问恢复hp3090或提供59直接迁移登录方式，不请求私钥。暂未传大型文件，后续仅按原输入SHA选择性迁移，不复制整套旧缓存。

用户已选择“提供59服务器直接登录方式”，实际认证信息仍待提供，已只追问缺失的认证入口及可用主机指纹，没有重复询问选择。23:24只读复验：同boot，MemAvailable 19,460,734,976字节，CPU封装68°C；EgoLocate PID808677仍占3278MiB计算显存/RSS 3,882,487,808字节，GPU总4599MiB/6%/50°C。没有启动张量控制、仿真或训练，也没有清缓存、终止既有任务；内存有余量，当前阻塞不是文件缓存不足。该快照不是三次启动验收或连续内核监控。

静态补查cloth_constrained_trial.register确认，旧13项直接输入之外还读取E12 independent_verification.json、E35 result.json、E23 independent_verification.json和六份合格reference.csv、E26 registration/independent_verification、E02 job/independent_verification，以及两组explicit_summary/metrics/timeseries。四官方CSV路径由E02 job.diagnostic_motions给出，六衣服顺序由E26 registration.selection给出。源连接建立后先读取这些小合同并核验原清单，再列精确路径/大小/SHA选择性迁移；不整目录搬迁。旧评测输出只作来源和历史证据，新机比较仍必须重新测035，旧ZHY内核审查材料不进入新机健康基准。

## 决定与下一步

先保留新机现有任务，继续可独立完成的代码/清单工作。源连接建立后取回已验证参考、teacher锚点及必要合同，逐文件大小/SHA核验；传输缓慢按既有规范提供具体源/目标/大小，不持续强传。官方vendor必需LFS资产也必须实化并核验，不能用指针运行。

无其他计算且原资源/温度/可靠遥测条件满足后，先单独登记新主机环境和运行验证。再固定新主机035六衣服/四官方匹配基线，最后另登记原有限更新单变量短训练与同条件效果出口。8更新/32提交/3072交互以及原效果门不因换机器增加或放宽。运行库或仿真条件不同不得用ZHY旧数值直接宣称提升；旧baseline/拒绝结果保留。任何实际异常按原规则停止，不自动纠错复跑。

## 保存与同步

本记录、最新AGENTS、索引/current state、inventory源码和小清单复制新工作根；20:31连接阻塞和22:21上游核验的未推送记录一并保留。原始JSON/内核日志LOCAL_ONLY，不进Git、不声称网盘备份。

工程可HTTPS读取，但新机和Mac的GitHub SSH鉴权均Permission denied(publickey)，新机未配置提交身份；提交时使用该仓库既有提交身份，仅本次命令指定，不改全局账号。代码/记录先scoped审diff、audit_git_payload.py、本地提交，再尝试用户指定SSH远端；推送实际结果由logs/a3_humanplus_20261017/publication记录。未成功核对远端前不称已同步。

实际发布：10文件/95815字节暂存审计及diff检查通过，已在新机main提交084387e1554482ce60fa77d80afd2026f9cb149a，包含HP01、前述transport/upstream未推送记录和新增盘点代码。指定SSH push失败Permission denied(publickey)，GitHub未同步；不新增密钥、不复制原机私钥。此次补充状态也在新机保存为后续本地提交，准确版本由Git历史提供，不为写入自身hash循环提交。日志/内核仍LOCAL_ONLY。

23:24恢复核验HEAD为f0713161e0d5c0d7829955fa22d462775d5f378a、main领先origin/main两提交且工作树干净。本次网格阶段的清单/README更新将另作scoped审计和本地提交；已知SSH写权限未恢复，不循环尝试同一失败推送，仍为LOCAL_COMMITTED_UNPUSHED。旧文提到的publication目录当时没有独立回执文件，不能作为既有证据；此次生成明确小回执，实际提交号以Git历史为准。

长期任务沿用zhy-a3的原id，更新目标为此工作根，不另建自动任务、不继续旧ZHY启动。heartbeat复用Mac临时SSH socket /tmp/yuanqi-humanplus-a3.sock，失效时记SSH_UNKNOWN，不保存密码。旧yuanqi-a3/hp3090巡检仍PAUSED，每日研究yuanqi不变。原暂停设置在迁移前读取为PAUSED，本次按用户继续在新机工作的指示恢复本工程；只报新结论/阶段/需处理阻塞。
