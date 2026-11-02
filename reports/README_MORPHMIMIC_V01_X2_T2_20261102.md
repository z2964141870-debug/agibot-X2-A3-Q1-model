# MorphMimic v0.1 X2 Ultra / T2 权重测试

用户要求优先测试另外两个机型，并明确同意直接测试 v0.1 发布权重。本阶段不训练；分别验证真实 SOMA 到 X2 Ultra/T2 的重定向、发布 ONNX 的动力学跟踪和本地原速视频。作者训练包含这两个机型，不能将本测试称为未见本体零样本迁移，也不能当作我们已完成共享训练复现。

## 固定来源

- 发布页：`https://github.com/lzyang2000/morphmimic/releases/tag/v0.1`。
- tag 对象 `70511b3da574403edbc2c89cc84d793a6182898b`，peeled commit `41c8382f127a6f5109bfe047445a9180f71f1fc0`。此前研究所记后续 main 不作为该 tag 的源码。
- `morphmimic_policy.zip` 63372873B，SHA256 `be874424beeda23cbccda08ed65b3fa146e926d2f338b8f898d28baa21d1b99f`。
- 实际 `policy.onnx` 69994291B，SHA256 `cf95194793e4732447f66432fc2381d82dd7c3d82c94da15ad074f2cfe295512`；与包内 `SHA256SUMS` 一致。`MANIFEST.txt` 中嵌入的文件哈希陈旧，与实际及 `SHA256SUMS` 不同，原文件保留，不据其文字代替校验。
- 包内自述 unified/share all、iteration60000、goalv2；实际只发布部署 policy/机器人 JSON，没有可独立审计的 WM/critic 或训练 `.pt`。不将说明文字视为这些缺失网络的内容审计。
- SOMA retargeter 固定 `cbdb2b494a2af82137bf9b6f15122d60167281c4`，使用 v0.1 原 `assemble_robots.py` 的软限位和 distal effector 修正、原 batch 配方及 converter。

全部模型、源码归档、大 NPZ 和 BVH 均留用户25386的 `data/references/morphmimic_v01_x2_t2_20261102` 或独立评测根，标记 SERVER_LOCAL_ONLY，未做网盘备份。

## 已发现并纠正的接口试跑问题

先前试跑将 G1 NPZ 关节改名/补零给 X2/T2，同时保留 G1 身体位置。这不是真实机器人重定向，身体几何和 T2 关节轴不一致。原 `MORPHMIMIC_V01_X2_T2_EVAL_20261102/synthetic_npz`、两模型导出和 X2 视频/JSON全部保留为接口烟测；X2 的1032帧及其误差不能用于评判策略质量，T2当时只完成资产导出，未进行动力学回放。

该试跑未挂项目资源守卫，且 `--bake` 改写了新隔离 v0.1 副本中的两个 JSON；这是执行偏离。原 aa40/G1 训练冻结源码未改。后续仿真须挂原共享 GPU 锁、24GiB准入/30GiB运行内存、boot/GPU/OOM/温度/磁盘保护和进程回收，运行配置另存，不继续改发布包原件。

原 SOMA 首次 assemble 因 LFS URDF 指针解析失败，取回精确 X2 URDF 后原 assemble 已完成；首次 converter 因 `newton` 缺失退出，尚未生成真实 X2/T2 CSV。错误和目录保留。retargeter要求 Python>=3.12/Newton1.3，而原训练环境Python3.11，下一步另建隔离环境，不升级原训练环境。X2必要 LFS mesh 仍须按指针 SHA/大小逐项补齐；T2资产来自 v0.1 vendored 文件。

固定首片是既有真实 SOMA `220713/walk_forward_loop_001__A022.bvh`，原 G1 转换为1032帧/50Hz；不能将此数值直接当作新重定向已完成。保留559条源 BVH，不重复提取或按结果换片。

## 原 G1 分支

G1原训练已跑满160000轮，12编号checkpoint内容审计通过，最终motion三片完整，详见 [G1记录](README_MORPHMIMIC_G1_EVALUATION_20261102.md)。为优先满足用户X2/T2要求，剩余keypoint评测经SIGTERM停止，guard为STOPPED_SIGNAL；controller因人工停止记通用FAILED，不是训练失败。6份最终视频全组尚未验收，保留未完成状态，不重启训练。

## 当前状态与下一步

原生 hp_han Windows 两段式 SSH 会话81774已再次返回唯一根；上下文前快照 GPU15MiB/无计算/OOM0。运行前须重新核实时变资源。先保存本阶段记录，再补独立retargeter环境和真实参考；通过输入/模型一致性核验后，X2与T2串行运行发布策略，保存真实失败、指标和原速视频，最后取回Mac并核大小/SHA、解码与抽帧目视。当前没有新的有效X2/T2动力学质量结论。

不创建/结算goal，不启agent，不改其他自动任务；衣服Q1演示继续搁置，用户采纳仍为null。发布状态以独立远端和三端旁收据为准。

## 隔离环境与受保护入口

复用现有uv0.13.6，新装Python3.12.16到data/environments/morphmimic_retarget_python_20261102，另建morphmimic_retarget_20261102。按固定retargeter原pyproject安装19依赖，含Newton1.3.0、Warp1.19.0、NumPy2.4.3、SciPy1.17.1；原Python3.11训练环境不变。使用源根PYTHONPATH确保运行的是固定源码及已核LFS资产，不误用安装时的指针副本。原生81774超时后已登记一次5845恢复；2026-11-02T08:11:27Z重新核原boot/GPU/唯一根、15MiB无计算/OOM0、容器16982175744B。第一次身份打印括号语法错误和首个镜像过长参数失败保留，后续独立命令才构成成功证据。

新增run_guarded.py只包裹原命令，复用Q1Budget/runtime_guard：原GPU锁、24GiB准入、30GiB运行、100GiB磁盘余量、原boot/GPU/OOM/86度保护，单阶段最多7200秒/10秒回收、不自动重试、冻结输入前后SHA。服务器CLI import及本地语法已通过，不新增训练算法或改作者部署实现。LFS下载工具仅处理固定commit所需X2资产/源零姿态，保存原指针并逐大小/SHA核验，失败原件不覆盖。

预先固定两条真实BVH：220713/walk_forward_loop_001__A022和walk_forward_start_002__A023，均来自此前559条，不按表现换片。两机器人各用原batch10/Newton/SOMA/Mujoco-facing/原软限和effector修正，再原convert_seed到50Hz、原export_assets核模型一致和原deploy.run。对应batch/subset在script/morphmimic_cross_robot/v01_*_20261102配置；导出不再--bake。只读发布包JSON与本地模型常量必须先比对，保持发布ONNX/PD/限矩/控制时步/真实失败定义。新输出根MORPHMIMIC_V01_X2_T2_REAL_EVAL_20261102与无效烟测分开。

准备阶段51598b30f056ac0a813e8579194bc3084676a76f已独立远端核实，6记录388931B的Git/Mac主根/25386逐大小SHA相同。工具入口本阶段尚待独立发布核验，实际运行状态另记，不提前称重定向或新物理已通过。

工具入口实际发布为41c9fe855902845bfcc286acbb0d7731107e2798，独立40位远端已核，7文件16172B服务器逐SHA相同。X2首个受保护controller564815传入的code_commit文本被手抄错误，原命令/状态保留；它是元数据错误，实际执行的7份文件SHA对应上述已发布提交，未热改源码或重启。后续命令从git rev-parse程序读取完整SHA。[明确更正](../data/manifests/morphmimic_v01_entry_metadata_correction_20261102.json)。

## 真实重定向完成

X2所需72网格和SOMA零姿态共73个LFS对象137180786B逐指针大小/SHA全部通过，收据data/references/morphmimic_v01_x2_t2_20261102/lfs_assets_01.json；先前已取X2 URDF单列保留。原assemble所得两机器人常量与发布包各自robot JSON逐字段完全相同，未改PD/限矩/时步。

受保护X2 controller564815/worker564822与T2 controller565341各完成两条真实BVH重定向，原batch10/两条输入、2491最长批时轴、原IK24轮和limb stabilizer，原stdout保留。两guard均WORKER_COMPLETED、回收return0/输入SHA前后不变，GPU释放回15MiB/OOM0；X2约213秒主要是首次编译，T2约26秒。产物分别为新real评测根retarget_x2/csv和retarget_t2/csv。此结果仅说明IK流程完成，不构成动力学质量或策略稳定结论。下一阶段原convert_seed、模型一致性导出及发布policy回放仍待执行。

首个X2转换根convert_x2在实际转换前因重连后缺GIT_PYTHON_GIT_EXECUTABLE而GitPython导入退出，原guard/stdout保留、未生成新轨迹或做策略回放。下一独立convert_x2_v02只补既有Git可执行文件的进程环境，原converter/输入不改，不重做重定向或训练；不覆盖失败根。

原convert_seed在convert_x2_v02及convert_t2均正常结束，两个机型各2/2通过原名称与可行性过滤，无拒绝；两guard回收return0、原CSV和metadata SHA前后相同。每型NPZ包含各自模型FK和身体几何，不沿用G1身体位置；后续模型导出和发布policy动力学才评价实际跟踪。

四条50Hz NPZ实际有限，前进1032帧/20.64秒、起走152帧/3.04秒；native关节均31，X2全身34body/T2为32body。X2原export_assets在独立新根正常结束，check_same_model与IMU FK最大误差5.16e-8通过、原baked JSON前后SHA不变；未使用--bake。下一步T2模型导出，再两型各两片原deploy.run和原速视频。

T2原模型导出亦正常完成，IMU FK最大误差6.66e-8、模型一致性/输入前后SHA/guard回收0通过。增加run_v01_suite.py串行执行固定4片，每片经原run_guarded、最多3600秒、无重试；固定motion接口和原deploy.run，不改策略或仿真。仅设置已有Mesa软件EGL/4渲染线程和12核CPU调度，完整50fps/1280x720原速录制；policy、机器人JSON、参考、模型及部署源码在每片前后校验。守卫与外层控制器分离会话后执行，可抵御SSH断线，结束必须核进程退出及本地媒体。
