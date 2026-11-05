# 同权重human短预览闭环对照

2026-11-06 03:12北京时间：独立入口已实现，因果尾部已接入 worker/controller；完整本地 teleop 检查共36项，34项通过、2项 pinned-runtime 检查因本机缺服务器资产而跳过。尚未服务器原运行时验证、物理回放或新GPU渲染；本页是可执行准备，不是低延迟效果结果。长期goal实读active，最低执行至今日10:00，训练/必要验证可继续延长。

## 对照定义

默认 masked 路线固定同一已审计16000轮Q1共享权重、human接口、全24点空间掩码、原Primebot CPU物理/重置、三条配对冻结参考和seed0，仅变H10/H5/H4/H2/H1时间前缀。共15个名义确定性回放；不是15个独立随机试验。真实保存配置为full/h10，短预览尚未验证，不能把mixed空间点变化引入该消融。

因果尾部是独立策略参数，不改变默认 masked 结果：`hold`、`cv` 和固定 `tau=.12s` 的 `damped_cv` 保留十个20ms目标槽，只用H4/H5已见前缀生成尾部。每种方法必须用 `--baseline-output` 指向已完整完成、同权重/配置/运行时/冻结训练输入的 masked 对照，仅新增H5/H4共六次物理回放和六个原速视频，写入独立的 `tail_<method>/` 根。H10及同缓冲masked轨迹复用而不重跑。预测槽位带有 `measured_mask`、`generated_mask`、`forecast_method` 和逐槽 provenance，任何预测都不能宣称为实测未来。当前实现没有骨骼/接触约束，也没有学习型预测器。

每例输出完整/失败信息、有效共同前缀、关节/body误差、脚底离地与交替、承重滑移、侧倾、上肢幅度及原始时序CSV、根末误差、逐帧输入时序JSONL和原速视频。H*20ms只表示预览预算；端到端实测延迟始终null，离线sidecar预处理因果性标为未验证。

## 下发顺序

1. 原设备/根/五PID启动tick/guard锁和完整实时训练记录恢复核验，按原资源门准入。
2. 原Morph CPU运行时执行新策略等价/隐藏pose-heading/冻结normalizer与RNG检查。跳过测试、零测试或失败均拒绝后续。
3. 用真实16000权重，旧human/H10进程和新流式human/H10进程分别读取三片完整参考状态序列，所有canonical/native动作逐值相等后才允许物理积分，覆盖原片尾clamp。此阶段无积分，不是质量结果。
4. masked 按H10/H5/H4/H2/H1和前走/起走/停走顺序；因果尾部先核已完成masked对照身份，再按H5/H4和前走/起走/停走顺序。每例先隔离CPU策略与共同物理，完全退出后另起NVIDIA EGL媒体进程。预测尾部每种方法共6个物理、6个视频阶段，加运行时/动作等价/比较共15个受守卫阶段。
5. masked的四组H10配对比较、每种预测方法相对H10及同缓冲masked的四组比较，只使用真实共同有效前缀；完整片和失败分别计数，不能拿失败后的hold视频当完整走完。保留自然性未合格状态。

入口：script/supermimic_teleop/preview_review.py。原interim入口、训练源码、inference.py和final CLI均未改；复用原16000双型CPU保存审计，不重复原motion三片。GPU用于EGL图像，CPU仅用于已固定共同物理及隔离策略。

## 守卫与发布代码

全流程最多1800秒、输出3GiB、另留100GiB磁盘。初始总内存低于22GiB，运行达到24GiB或GPU18GiB/80度即只退出本评估。RSS仅诊断；训练原30GiB/86度/guard55672锁保持，不争锁、不停训腾资源、不回收全局cache、不自动重试失败。

新freeze_preview_code.py从权威Mac Git导出当前已发布40位commit的Python文件身份，逐文件核对Git与工作目录，不读取认证。服务器无.git，以显式manifest SHA和每个文件大小/SHA核验；每阶段前再查manifest、config/runtime/audit身份。新模块必须不属于冻结训练输入。服务器同步只能写独立新模块及清单；旧输入不一致须停止核查，不能覆盖正在执行的训练文件。

发布并独立核远端后生成清单：

```sh
python3 script/supermimic_teleop/freeze_preview_code.py \
  --repo '<权威Mac仓库绝对路径>' \
  --evaluation-commit '<独立远端核同的40位提交>' \
  --output '<独立准备目录>/published_code_manifest.json'
```

服务器入口需要--config、--runtime-config、--output、--evaluation-commit、--reuse-audit、--code-manifest和--manifest-sha256。输出必须是唯一ZHY根下的新独立目录，不能位于原训练根。复用审计路径为Q1_SHARED_INTERIM_CKPT16000_20261105/checkpoint_audit.json，必须先过原reusable_audit身份/双型330有限张量/退出守卫要求。

## 因果尾部实现

`causal_tail.py` 提供 `hold`、常速度 `cv` 和阻尼常速度 `damped_cv` 三个可审计基线。位置以末帧为锚点外推，旋转在 SO(3) 上按相对四元数外推，支持线速度/角速度上限；四元数符号翻转、片尾声明和隐藏尾部均有独立检查。`forecast_policy.py` 只把已验证的十槽因果结果交给原 human pack/act 数学，未改训练输入、inference.py 或 final CLI。

本地已验证：36项 teleop 测试中34项通过、2项真实运行时测试因 Mac 缺 pinned Torch/Q1 资产而跳过，耗时0.125秒。覆盖隐藏尾部不可影响预测、四元数符号和pi边界、速度上限、篡改来源拒绝，以及片尾恰好位于H4/H5边界时禁止继续外推；controller检查覆盖六例复用和两种比较基线。服务器阶段仍必须先通过 pinned runtime、H10 动作逐值等价、资源/输入身份和完整性门，再做物理与GPU EGL媒体。

## 已验证与缺口

原controller检查覆盖代码清单来源和改动/路径逃逸/软链接拒绝、真实预览时戳和来源约束、33阶段下发顺序、前置测试跳过/失败不下发、环境和信号处理恢复；新增因果尾部掩码、SO(3)外推、速度上限、片尾及预测 provenance 检查。编排和比较测试使用夹具，没有把它称为真实服务器执行；2项原运行时待验不能计为通过。

```sh
python3 -m unittest discover -s script/supermimic_teleop -p test_preview_review.py -v
```

预测尾部独立入口示例：

```sh
python3 script/supermimic_teleop/preview_review.py \
  --tail-method damped_cv \
  --baseline-output '<已完成masked输出根>' \
  --config '<config.json>' --runtime-config '<runtime.json>' \
  --reuse-audit '<16000审计/checkpoint_audit.json>' \
  --code-manifest '<published_code_manifest.json>' \
  --manifest-sha256 '<manifest SHA256>' \
  --evaluation-commit '<40位提交>' \
  --output '<独立输出根>'
```

最近一次原Windows桥接于01:33到达22端口密码提示，交互提交后Broken pipe退出255，未取得Windows shell或Linux；认证未写入文件。03:12本次恢复未新增认证尝试，实时训练仍UNKNOWN，最后健康快照是11月5日23:34各38600轮，不能用旧数据作新准入。因果尾部代码尚未下发或执行真实回放，原训练未停或重启。继续发布/清单和后续预测数据准备，待链路恢复后先全只读现场再执行。[controller原审计](../logs/session_records/q1_human_preview_controller_preparation_20261106.json)、[预测尾部准备审计](../logs/session_records/q1_causal_tail_controller_preparation_20261106.json)。

03:24发布：`1e394126bba877e75f6f3059ffd3c06a8256db15` 已显式推送origin Q1并独立40位远端核同，16份662375B Git/权威工作树/Mac主镜像大小SHA通过。已生成177项代码清单32927B/SHA `b512bc7655f515de13b432041f77a8b91b754bef8ce9ac2742aa412fc7841c71`；Linux部署包43141B/SHA `db8b2ebb7504627c0da28fd9fd2f4bf70af41a51ba28097c2fb6f9e222059f64`，16普通文件且无AppleDouble/路径逃逸。TCP22探测可达后作一次有界原生登录，Windows密码提示后仍Broken pipe255，未取得Windows/Linux shell；本次不重复认证，训练实时UNKNOWN，清单和包尚未上传。见[发布收据](../logs/session_records/q1_causal_tail_controller_publication_20261106.json)。
