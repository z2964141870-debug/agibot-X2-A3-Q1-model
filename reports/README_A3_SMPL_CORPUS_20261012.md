# A3 原始人体标签与新配对语料 E14

记录日期：2026-10-12。实验：A3_ZHY_SMPL_CORPUS_20261012_E14。状态：VERIFIED_QUALITY_REJECTION；原始标签结构7/7通过，新A3配对0/7通过，未训练。

## 目标与新增来源证据

用户2026-10-12明确说明，既有CLOTHO `train_released.zip` 是自主采集、用于训练衣服模型的数据，并授权现有数据不合格时自行寻找外部来源。这解除此前来源/训练用途缺失，不要求用户重传文件或再次确认。此前包内无说明的检查仍保留为当时证据，不回写成已知来源。

从官方权重取得可验证提升仍是最终目标。本阶段先验证原始人体标签的内部一致性，再新建A3配对参考；接口、离线监督可用性和闭环改善分别验收。旧FGP预测及旧四段TEST_ONLY参考不进入训练、不修改。

## 固定方案

七段原始pose/joint标签预先选定：zj walk/warmUp、fjw walk用于train，zls walk用于validation，yy squat/walk和zzx walk用于test。按capture文件夹完全隔离；目录/姓名缩写不是经核实的独立受试者身份。zj walk此前参与FGP诊断，故放训练，不充当新鲜测试。每段完整标签做结构检查，A3 IK使用固定中间300源帧（30Hz约10秒），不按结果挑最好片段。

仅复制14个标签文件，从原机直接到ZHY；Mac只保存代码、README和小型清单。来源 ZIP 的整体SHA沿用既有证据，本轮核验目录CRC及选定标签逐文件SHA和新包SHA，不声称重新核验整个原始ZIP。

人体关节单位/局部轴角语义以固定骨架、parent-local骨向量及pose重建joint检验。时间使用已有CLOTHO工程30Hz合同；选定标签没有原始逐帧时间戳，因此不证明现场时钟/帧率或在线延迟。人体标签内部自洽也不等于独立测量的姿态真值。

复用既有A3 Mink/quadprog IK，固定vendor不变。使用从原始标签推得的canonical neutral骨架校准及平均腿链比例；与旧“首帧当中立姿态”诊断配置不同。检查IK残差、URDF限位/速度、真实foot geometry穿地和根据人体toe高度/速度推定的接触期间robot ankle滑动。推定接触不是接触传感器。预登记门槛在registration.json保存；通过仅允许有上限离线编码器蒸馏，不证明平衡或真机使用。

## 版本与执行

代码基线 `8f21d8d2e2eef71e25bdd934faa49abcdd0ec612`；新代码以登记SHA固定。vendor `fe6868ba37034f89b912f0fb851bce19f120266d` 不修改。Q1四个辅助模块只复制到新机 `data/references/a3_smpl_q1_helpers_20261012/`，登记SHA，以原实现读取BodyReference/坐标工具，未运行Q1仿真。

源码：`script/a3/smpl_corpus_pack.py` 标准库复制标签；`script/a3/smpl_corpus_qualify.py` CPU结构/IK/独立验收。原机禁止模型/仿真/训练计算，仅打包与传输；所有张量和IK计算只在 `/opt/data/private/hechengzhu/ZHY`。

原机打包（已有输出不得重置）：

```bash
/usr/bin/python3 -m script.a3.smpl_corpus_pack --archive /media/yu/FAFF-E9771/data/fgp/CLOTHO_train_data/train_released.zip --inventory data/manifests/a3_smpl_corpus_20261012.json --output data/transfers/a3_smpl_original_20261012
```

新机工作根执行（登记后只能运行一次）：

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_corpus_qualify register
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_corpus_qualify run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 data/environments/a3-sonic/bin/python -m script.a3.smpl_corpus_qualify verify
```

固定CPU单线程/无CUDA，0optimizer/0策略回放；沿用90/85°C、4GiB保留、5秒采样、900秒任务上限和首故障停止，无自动恢复。原始输出 `data/experiments/a3_zhy_smpl_corpus_20261012_E14/`，日志 `logs/a3_zhy_smpl_corpus_20261012_E14/`，小型清单 `data/manifests/a3_zhy_smpl_corpus_20261012_E14.json`。

## 当前结果与下一步

新机2026-10-12北京时间20:28:17–20:28:32完成worker，20:28:41独立核验为 `VERIFIED_QUALITY_REJECTION`。6项CPU测试通过（0.162秒），覆盖一致骨架、pose/joint不一致、单帧骨长损坏、毫米误用、接触滑动和实际脚几何穿地。

| 原始序列 | 源帧数 | A3位置误差P95 | 最低脚几何Z | 推定接触滑动P95左/右 |
| --- | ---: | ---: | ---: | ---: |
| zj walk | 1825 | 0.63164m | -0.03213m | 0.278/0.348m/s |
| zj warmUp | 1806 | 0.40290m | -0.03235m | 0.220/0.219m/s |
| fjw walk | 2061 | 0.62530m | -0.03942m | 0.284/0.283m/s |
| zls walk | 1855 | 0.64187m | -0.02057m | 0.281/0.315m/s |
| yy squat | 1895 | 0.66058m | 0.00257m | 0.437/0.405m/s |
| yy walk | 1839 | 0.66006m | -0.01431m | 0.356/0.270m/s |
| zzx walk | 2245 | 0.64690m | -0.01604m | 0.421/0.271m/s |

完整原始标签共13,526帧。pose-FK/joint最大偏差2.12e-7–5.72e-7m，固定骨长/parent-local骨向量、人体米尺度、根速度全部通过；这支持内部一致性，不是独立人体姿态精度验收。2100新IK参考帧有限、URDF关节限位/速度全部通过，但7/7位置残差门槛失败、6/7脚几何穿地、6/7接触滑动门槛失败，故三个分区均无合格pair。

独立核验从新包重新解码原始标签、重算骨架指标，核对新人体clip的原始frame identity；从保存qpos重新FK核对全部真实body位置/足几何，重构人体到机器人目标，复算位置误差和接触/速度指标，并核对CSV关节角。没有重复求IK、模型前向、策略回放或optimizer。

14个标签的新包7,812,937字节，SHA `0115e72ef863ceb54dec9f759fb92d836bb403e0a88658474cb55e015b435fda`。原始ZIP未重新计算整包SHA。6健康样本20:28:14–20:28:41，同boot `d994e71f-bae0-46ba-a5f7-6c2163771c6c`，CPU最高39°C/GPU26°C、cgroup可用最低8,886,669,312字节、OOM0，无新关键内核/遥测错误；最大采样间隔8.52秒含worker结束到独立核验。无CUDA初始化，计算已退出。原始及衍生数据服务器LOCAL_ONLY。

小型完整回执：[E14清单](../data/manifests/a3_zhy_smpl_corpus_20261012_E14.json)。质量不合格没有被改boolean放行，现有学生真实optimizer仍0。人体原始标签内部自洽而新A3目标残差大，当前优先定位校准/几何转换；不能直接归因“衣服数据不行”，也没有理由先下载替代大包。

下一步用已保存目标/实际位置按body与时间拆开残差，核对canonical neutral与机器人rest姿态、根朝向及逐链缩放。任何转换更改用独立新实验，不覆盖E14。七段都已用于转换质量诊断，后续不能把它们包装成未看过的最终泛化测试；模型最终改善还需未参与选择的新会话和闭环证据。

E13接口与旧TEST_ONLY证据见 [编码器记录](README_A3_SMPL_ENCODER_20261012.md)。原始/衍生数据继续服务器LOCAL_ONLY，未验证网盘备份；重要阶段完成后同步代码、README与小清单，并核验GitHub，不预写推送成功。
