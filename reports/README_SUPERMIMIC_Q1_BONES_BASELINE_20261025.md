# BONES到Q1的首版SuperMimic复现基线

记录日期：2026-10-25，北京时间。阶段：BONES-G02 / PF02 / RF02。状态：目标已登记，扩库与新训练准备中，尚未启动RF02。

## 目标与用户决定

用户要求重新定目标并继续执行，优先复刻SuperMimic，不再追加代理制定的数据质量门槛。新目标是利用现有1024个冻结train来源形成原速Q1参考库，完成64环境、12000采集轮/768000真实转移的独立训练，交付同一固定参考上的初始/最终MuJoCo动力学比较及回放视频。1024和12000是本次可执行数据/计算预算，不是质量准入阈值，也不代表原论文全库或算力规模。

用户最新限制最多一个子agent；主代理处理数据、训练、记录与部署，唯一子agent只实现评测/回放代码，现已完成，无待处理进程。恢复时实际代理清单仅主代理运行，旧多个子任务处于中断状态，不再调度。

## 数据与方法

复用已完成PF01的128段/791.32秒，不重新转换或复制其数组。已实际读取冻结名单：前1024条全部train，后256条为development128/test128；准备按offset128至896、每批128转换余896个来源，再通过软链接合库。各片只检查可读、Q1维度/身体顺序、原速50Hz时间轴与有限数值；原几何flags保留为诊断，统一2rad/s及旧19/22门不使用。

训练沿用RF01已跑通的可微世界模型、8步拟合、H32想象BPTT与critic，64环境/4模型槽、actor B128、WM B64每组10批。三网络重新随机初始化，固定12000采集轮，每1000轮保存中间模型，保留初始/最终。Q1形态、CPU MuJoCo与单3090参数是工程适配，不能称官方G1逐行复刻。

初始/最终使用同一库按clip名排序的前8段train，固定选择后才运行策略，统计实际误差、回报、完成比例、提前终止和根移动，不设置新的“必须达标才继续”门。该比较衡量训练内覆盖，不代表未见动作泛化。回放来自实际自由根动力学轨迹，不能用运动学播放冒充策略表现。

## 平台、代码与资源

唯一实体根`huanyu:/opt/data/private/ZHY`，四目录和Q1分支保持。Python为`data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python`。原boot仍d994e71f-bae0-46ba-a5f7-6c2163771c6c、OOM14；本次恢复18864跳板转发后实测盘余2093725360128字节、约2.094TB，容器内存23512776704字节。GPU在本轮首次核验27摄氏度/14MiB空闲任务显存，无现行训练进程；恢复时须重新读实时状态。

代码添加于`script/supermimic_q1_reproduction/`；保持原PF01/RF01四个源码不改，只新增多批合库、串行流程及新IO评测入口。`merge_library.py`复核原批IO协议和数组身份，仅建立新链接，保留旧false flags；合库不增加几何判定。`pipeline_baseline.py`依次准备7批、合库、12000轮训练、保存核对、初始/最终真实动力学比较与同一片回放。状态持续保存当前阶段、PID/boot/start_ticks、命令、日志和返回码；失败不自动重启，停止只通知自己的子进程，由原guard回收训练组。支持`--detach`，SSH断开不终止长任务。

合库2项服务器控制通过：旧false标记保留且只建数组链接；重复批在建输出前被识别。流程异常控制在Mac/服务器均通过：首阶段退出7时真实记录FAILED，后续阶段未启动、没有重试。4份主代理新增源码部署前后SHA逐一一致，合库源码7a3ea8e610e9de726df0be98cfa2b2402adffcc375cd654d7c4b4955fa4b8c40，流程源码7641488c3929b4813262fdbdcce8c707fc59be7d1049a758a944ffa5241079cf。

评测新入口不再依赖旧qualified flags或效果门；绑定训练一致的接触资产、clip20策略与原迁移视图。服务器3项评测控制与语法检查通过。实际执行RF01前2片初始/最终CPU评测及指标重算均退出0，结果未见整体改善：

| 固定前两片均值 | 随机初始策略 | RF01最终策略 |
|---|---:|---:|
| 归一化整片回报 | 0.27296 | 0.14400 |
| 整片完成比例 | 62.03% | 32.49% |
| 提前终止 | 1/2 | 2/2 |
| 执行段关节RMSE | 0.44609rad | 0.44551rad |
| 根平面位移 | 0.24135m | 0.47935m |

第二片最终策略虽然移动约0.480m，但108/282步提前终止，不能称有效步行。第一片最终视频实际查看末帧：参考直立行走、策略倒地，清晰标注71/266步提前终止，无空白或文字重叠。视频`data/supermimic_for_q1/BONES_RF01_E01_20261025/final_first_video/comparison.mp4`，126120字节、H264/1280×548/25fps/36帧/1.44秒，SHA`2b935c38e8d9f4b2138edf14067f6bbb7416f3ef7c67f3d4fb70af11cb81177d`，像素标准差最小73.732。上述仅覆盖500轮小试的两片训练动作，不是12k轮或1024库结果，也不据此暂停扩库/训练。

五份原样收据已从服务器取回至`data/manifests/`，本次逐一复核SHA与服务器一致：

| 原样证据 | SHA-256 |
|---|---|
| [初始评测](../data/manifests/supermimic_q1_BONES_RF01_E01_initial_20261025.json) | `e09c1f92c95d460a6074e1979a565f0a4f0a3ee31d02595b8457b8e9d28ef54d` |
| [初始算术复算](../data/manifests/supermimic_q1_BONES_RF01_E01_initial_audit_20261025.json) | `2dc2963a7e72fb561d2922cc923cfa913a93c9c84abc70703d4725640506345c` |
| [最终评测](../data/manifests/supermimic_q1_BONES_RF01_E01_final_20261025.json) | `fab013277b4a0d163b55102e0c2b1f149ae24f0eb9745a01fd03eeec8cea841d` |
| [最终算术复算](../data/manifests/supermimic_q1_BONES_RF01_E01_final_audit_20261025.json) | `cba6e022fccd3a0eb25032bab140dc01dc9b8c0ec566d595db64da22a16e1e61` |
| [最终首片回放](../data/manifests/supermimic_q1_BONES_RF01_E01_render_20261025.json) | `c0bef1a49e6fbf610336c0ff11407297c118947853c9544fd8e86efccecd336e` |

E01真实输出根`data/supermimic_for_q1/BONES_RF01_E01_20261025/`；执行日志`logs/BONES_RF01_E01_{initial,final,initial_audit,final_audit,final_first_render}_20261025.log`。评测使用`evaluate_reproduction.py --root /opt/data/private/ZHY --motions /opt/data/private/ZHY/data/supermimic_for_q1/BONES_PF01_128_20261025/library/references --checkpoint <RF01/run/step_000000.pt或final.pt> --output <新评测目录> --device cpu --count 2`，再以`audit_reproduction_evaluation.py --root /opt/data/private/ZHY --evaluation <新评测目录> --output <新评测目录>/audit.json`复算；新目录要求用于防止覆盖原证据。

计划启动命令（发布代码后实际执行，执行前不称已启动）：

```bash
ROOT=/opt/data/private/ZHY
PY=$ROOT/data/environments/yuanqi_ops/supermimic_a3_20261023/bin/python
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -u "$ROOT/script/supermimic_q1_reproduction/pipeline_baseline.py" \
  --root "$ROOT" --intake "$ROOT/data/supermimic_for_q1/BONES_A02_received_20261025/intake/receipt.json" \
  --backend-receipt "$ROOT/data/supermimic_for_q1/PHUMA_P05_slots_V01_20261024/receipt.json" \
  --code-commit PUBLISHED_Q1_COMMIT --expected-oom-kill 14 --detach
```

新增代码、测试、阶段报告与五份原样收据本次准备提交Q1；实际提交号由Git历史提供，发布后核对远端。发布提交、实际启动身份与各阶段终态在完成后补记，不预写结果。

保留100GiB余盘、单实验12GiB和温度/内存/OOM保护，控制模型保存量；不全量解压原包、不复制23.5GB归档、不把模型拉到Mac。原PHUMA/A3停止或失败实验不重启、不改终态，大型资产均LOCAL_ONLY。

## 计划产物与恢复

- 原参考：`data/supermimic_for_q1/BONES_PF01_128_20261025/library`。
- 新7批：`data/supermimic_for_q1/BONES_PF02_{offset:04d}_20261025`，offset为128至896。
- 合库：`data/supermimic_for_q1/BONES_PF02_1024_20261025/library`。
- 训练：`data/supermimic_for_q1/BONES_RF02_12000_20261025/run`。
- 流程状态：`data/supermimic_for_q1/BONES_BASELINE_G02_20261025/status.json`，计划与进程身份同目录保存。
- 日志：统一`logs/BONES_*_20261025*.log`，模型/参考/视频只写data。

后续立即运行新增代码必要控制、发布Q1并部署，转换余896个来源、合库、训练、保存核对、动力学比较和回放。每完成重要阶段更新本README/当前状态/索引并同步GitHub；未执行项目不记成完成。
