# 低延迟预测器的冻结数据划分与因果窗口

2026-11-06北京时间。已完成393条既有配对清单的独立划分、窗口读取入口和泄漏边界检查；没有读取真实human NPZ内容、训练模型、策略推理、物理积分或GPU作业。长期goal继续active，最低执行至今日10:00，必要训练/验证继续延长。

## 本阶段结果

新增 `script/supermimic_teleop/forecast_data.py`，输出冻结计划 [q1_teleop_forecast_split_20261106.json](../data/manifests/q1_teleop_forecast_split_20261106.json)。两份输入清单均在读取前后核SHA；计划240526B/SHA `f40c1d5a856a08d89f1a8c23dfe6d990fc7439232b9da9a658d9860456353145`。从原清单重新生成后逐值相同。

| 用途 | 完整片 | 来源目录组 | 帧数 | 25帧历史+10帧目标窗口 | 时长 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 167 | 22 | 54370 | 48692 | 1084.06s |
| 验证 | 98 | 6 | 31911 | 28579 | 636.26s |
| 测试 | 128 | 9 | 39308 | 34956 | 783.60s |
| 合计 | 393 | 37 | 125589 | 112227 | 2503.92s |

分组先按原始来源目录整体绑定，再将相同human或原始BVH内容SHA的目录传递合并，防止同内容别名跨集。各组由固定seed20261106的SHA规则分派；这是按组的规则，实际帧/时长比例以表中为准，不声称70/15/15样本比例。三个固定评测片所在的 `220713` 来源目录共26片整体归测试，而不是只排除三个已看过的文件。

目录和文件名后缀尚未获得被试身份认证，不能把此划分称为跨被试或跨动作族泛化。393条是已筛选的走路相关子集，对任意上肢/全身遥操的代表性不足。原V04策略已经训练过全部393条：即使新预测器在这些留出片上改善，也只能支持预测器的来源留出结果，不能支持整套策略的陌生片泛化。

## 时间依赖与输入边界

已读本地锁定源镜像的 `human_convert.py` 和 `bvh.py`。BVH FK、固定rest-frame和坐标变换按帧计算；50Hz侧车的位置线性插值和旋转Slerp从120Hz数据取左右支持。对输出第k帧，最后非零原采样支持为 `ceil(12*k/5)`，名义时刻为 `k/50` 秒。五帧相位循环的额外右支持为0、5、1.666667、6.666667、3.333334ms，纳秒使用向上取整，不能把它标成当前硬件采样的零等待输入。

源镜像身份：`human_convert.py` SHA `7d10905a472810e7bf492bb8bcd48f7cacc09ede90275c318f77be9d2d59787d`；`bvh.py` SHA `a77755b1a9cb9c3f6ea70301afcc3a54f947af5483535f7163b5fe915b3052e6`。这里核的是代码依赖，尚未重新读取服务器393份侧车或原始BVH；不能认证更上游的动捕滤波、硬件capture或钟同步，也不等于实际端到端延迟。

每个训练窗口只在同一源片内取连续25帧历史，截止帧a，目标为a+1至a+10。历史覆盖480ms的样本跨度，未来目标保持20至200ms；不跨片、不补末帧、不将声明片尾当未来意图。输入历史与目标分别复制为只读数组，显式记录截止帧的原始采样依赖。

H4/H5部署继续保留原十个20ms槽：前4/5槽为可见数据，后6/5槽调用以最后可见时刻为起点的预测时距1..6/1..5。没有压缩时间轴、跳过200ms目标或将生成槽标成实测。无缓冲的H0可用同一20..200ms预测定义，但是否适合当前策略仍需独立验证。

归一化统计只能接收 `split=train` 的数值，验证/测试拟合立即拒绝；后续选模型只能依据验证集，最终测试只作固定模型报告。两个TEST_ONLY真人小包不在计划中，也不用于参数选择。

## 实际检查

新增7项检查：乱序清单重建相同；相同BVH/human的传递别名同集；固定片整个目录留出；路径/身份错误拒绝；120→50精确有理依赖与H4/H5槽时刻保持；扰动未来目标不能改变历史输入、末端/跨集窗口拒绝；TEST_ONLY/错误姿态schema拒绝；分批归一化与独立总体统计一致且拒绝验证/测试拟合。这些是软件边界夹具，不是真实轨迹训练结果。

完整teleop共69检查，67通过、2项原服务器运行时检查跳过；`-W error` 下无警告，0.192秒。真实393条冻结计划已独立重建相同、三个固定片均test。没有为等待而重跑既有96组、60008决策或54解析对照。

## 复现与下一步

```sh
python3 script/supermimic_teleop/forecast_data.py plan \
  --admission data/manifests/primebot_q1_admission_frozen_20261103.json \
  --sources data/manifests/morphmimic_faithful_common_sources_results_20261101.json \
  --output '<全新路径>/forecast_split.json'

python3 script/supermimic_teleop/forecast_data.py audit-sidecars \
  --admission data/manifests/primebot_q1_admission_frozen_20261103.json \
  --sources data/manifests/morphmimic_faithful_common_sources_results_20261101.json \
  --plan data/manifests/q1_teleop_forecast_split_20261106.json \
  --output '<服务器全新路径>/sidecar_audit.json'
```

第二条尚未执行。默认human根就是原服务器冻结库，393份预期合计85438830B；入口逐片核大小SHA、50Hz/24槽顺序、source_frames和50Hz帧数、有限位置/单位wxyz旋转，并仅缓存一片。没有在Mac新增大型数据或安装依赖。

下一步实现只从可见历史估计锚点的SE(3)特征、姿态/速度时间一致解码及小型因果预测器的有界训练入口。侧车真实内容/原设备资源核验过后才训练，优先安全GPU；原V04持锁或实时资源不可核时不争锁、不停原训练。学习输出仍需承认骨长/接触约束未验，并与hold/CV/damped和实际human策略闭环比较。

最近原桥接仍停在Windows认证后Broken pipe，训练实时UNKNOWN；本阶段未新增认证、上传或下发。已发布3b1cd18的真实human预览campaign仍待原桥接恢复和完整身份/资源门，不能拿本地数据边界检查替代其结果。原final和完整五阶段、原训练保护线与采纳null保持。[本阶段审计](../logs/session_records/q1_teleop_forecast_data_20261106.json)。
