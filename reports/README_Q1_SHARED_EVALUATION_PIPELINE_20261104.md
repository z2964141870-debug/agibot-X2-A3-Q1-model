# Q1共享评估、比较与本地交付串行入口

2026-11-04。现有五阶段goal与原q1心跳均已再次实核：active、ACTIVE每10分钟、同一会话。没有另建目标或暂停监督。共享V03的30GiB内存失败保持，追加一次同档验证的原问题仍待答复；V04和正式共享训练未启动。

## 已完成的代码

新增`script/q1_five_stage/shared_review.py`。入口先核正式训练pipeline、固定计划、双机型初末CPU审计、两rank完整训练与共享一致性、393同片数据和原Primebot/Mac交付门，再创建独立评估目录。未通过时不导入策略、不创建Runner或启动物理。保留训练pipeline、计划、CPU审计和数据清单的独立小副本，后续控制器写状态不会丢失评估时的证据。

评估严格串行，共24个回放阶段：共同Primebot物理环境下Q1的motion/human，原生mjlab下G1/Q1的motion/human，各自初始/最终、固定三走三种子/全393条seed0。固定seed0均保留原速视频，合计36份；失败终止帧和hold保留。视频不是独立随机扰动测试，不能将三种子名义重复解释为泛化。

回放后串行执行14项比较和两型训练曲线。原生初末只在同机型、同接口中比较；Q1直接对照沿用相同Primebot CPU物理环境，KP另列。完整片误差与共同有效前缀分开，缺前缀明确缺失。G1初末不能替代等训练量的G1单体受控实验，见过的Q1不称未见本体零样本。

`shared_delivery.py`核全40阶段成功回收和输入身份、4824例逐片JSON/脚底姿态CSV/上肢时序CSV、36视频及各三张联系帧、比较、曲线和综合报告，再生成本地交付清单。大NPZ、权重和场景MJB仅SERVER_LOCAL_ONLY。Mac独立逐文件大小/SHA、完整FFmpeg解码和每个确切视频SHA的目视收据全部通过才写VERIFIED；不装ffprobe。目标是否完成仍须后续审阅实际结论，不由工具自行结算。

每项回放的轨迹和媒体放在该guard实际统计的输出目录内；整轮按冻结帧数计算轨迹和原始RGB视频上界，启动前另保留100GiB磁盘。沿用原24GiB准入/30GiB运行/86度/设备身份/GPU锁/回收保护，不失败自动重试。V04仅增加未来正式完成后自动接入评估的配置，原各1024/160000训练配方不改；没有据此执行V04。

同物理Q1入口补齐正式完成和双机型审计门，并将渲染身份同步写回逐片结果。策略加载器额外核保存的154维形态描述等于实际固定资产，避免仅mask一致或两个保存副本同时错误时仍放行。CPU反例会同时修改两个形态副本；恢复原值使用clone，不用加减回算。

## 验证与限制

本地68项测试，63通过、5项依赖跳过；新增串行替身完整运行覆盖24回放→14比较→2曲线→待Mac，首回放失败覆盖无重试/无后续比较。另验正式未完成时不建目录、guard回收/输入变化/设备错误、预算包含全部输出、缺片或重复/大型交付成员、视频联系帧缺失和本地SHA变化。`git diff --check`通过。

这只是代码和CPU替身验证。正式共享检查点尚不存在，真实共享策略回放、GPU求解器接触读取、36视频渲染和Mac验收均未执行，不称五阶段已经实际跑完。既有真实CPU运动学结果不重复；新形态校验的实际固定源码CPU兼容性检查须在本次发布与三端核验后执行。

Primebot10000轮正常早停及9/9、393/393交付保持，仍低离地/承重滑移，不能称自然步态合格。其102563840转移达不到410009600，当前等采样量明确不可用；本入口保留该负面结论，不额外训练补齐。若后续等曝光证据状态改变，先审计和规划对应比较，不静默忽略。

## 入口

V04的`evaluation_runtime_config`指向既有`data/manifests/primebot_q1_monitored_config_20261103.json`。原controller只有八全更新、共享参数、CPU保存和正式训练全部过门才调用本入口，评估失败单独记`SHARED_TRAINING_COMPLETED_EVALUATION_FAILED`，不把已完成正式训练改成未完成或自动重训。

如独立调用，`FULL_COMMIT`须独立远端核验：

```sh
data/environments/morphmimic_original_20261029/bin/python script/q1_five_stage/shared_review.py --config SHARED_CONFIG --pipeline COMPLETED_SHARED_PIPELINE --runtime-config data/manifests/primebot_q1_monitored_config_20261103.json --output NEW_SERVER_EVALUATION_ROOT --evaluation-commit FULL_COMMIT
```

实际执行后工具保存`evaluation.json`、`conclusions.json`、`README_RESULTS.md`和`delivery_manifest.json`。独立Mac验收入口为`shared_delivery.py --manifest ... --local-root ... --visual-receipt ... --output ... --ffmpeg ...`，不得伪造目视收据。当前仅准备，具体发布和服务器CPU结果以后续publication收据为准。
