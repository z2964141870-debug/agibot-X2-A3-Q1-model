# RF14用户审阅材料准备

记录日期：2026-10-28。状态：准备完成、最终数据/媒体待RF14终态；不是效果验收。

## 目标与范围

用户要求先看具体数据和仿真视频，再决定底座、RF15启动及后续加训。本阶段只整理已保存RF14证据，不改冻结来源、不产生优化或物理积分，不代替用户选择。

## 完成内容与验证

独立`script/project_tools/prepare_q1_gait_review.py`读取初始/阶段JSON，输出验证、固定片和共同前缀CSV、摘要JSON和阅读表格。Mac已用RF11_initial/stage_01/stage_02共9份保存评测完成集成：54项验证原值、27项固定片行、42项共同前缀比较；stage01 repeat0走后停271/320步失败和stage02验证1/0/0失败均保留，用户接受字段为null。原值范围是三重复的范围，不是置信区间；原始全可用时长与共同前缀分开标明。

新增`render_q1_gait_review.py`可在RF14全pipeline结束后，复用冻结视频套件从stage01/stage02 repeat0已保存NPZ导出固定54秒合集，包含失败hold。尚未实际渲染；`--help`、解析及RUNNING时拒绝执行的保护已通过。扩展本地`audit_q1_gait_media.py`接受stage_01/stage_02标签，待初末/数值推荐/两中间片都落盘后统一全帧解码与目视，当前仅help已通过，不把工具准备当媒体通过。

Mac已有ffmpeg/ffprobe位于`/opt/homebrew/bin`；服务器不重复安装。新工具均不在RF14的1145项冻结来源中，已检查plan名单，训练冻结仍为`4d87269d7ac815cd75a0424dcf7be56ae2c2661d`。

## 结果与边界

现有两小时走路滑移/足高有部分改善，但验证上肢/腕误差与姿态/抬手漂移有退步，不能据此认定展示达标。数值推荐仍非用户接受。22:23:43北京时间现场原controller110563/start_ticks10173450、worker110573/start_ticks10173556及boot/命令/GPU一致：累计12357.660645秒、10980轮、112967680转移，时钟ACTIVE/OOM0；继续原14300秒检查/累计硬14400秒，最终门尚未形成。

## 路径与命令

Mac中间材料：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF14_RF11_CONTINUE_SOMA300_E4096_CUM04H_20261028/review/interim_stage02/`，含`README.md`、`validation.csv`、`fixed_demos.csv`、`common_prefix.csv`、`summary.json`。其snapshot生成时pipeline为RUNNING，不能当当前进程或终态证明。权重/大NPZ留服务器SERVER_LOCAL_ONLY，镜像中间阅读材料不称云备份。

最终门后使用同一整理入口，`--labels RF11_initial stage_01 stage_02 stage_03`并指定新的`review/final`输出目录；不得覆盖中间快照。RF14已有保存审计`script/supermimic_q1_gait_comp/audit_saved.py --mode base`，执行前先看help和已存在输出，避免重复。终后再运行两次review渲染（stage_01/stage_02、repeat0不挑片），已有初末/selected沿用自动交付。实际执行及大小SHA/全帧/目视结论另记最终结果。

## 保存与下一步

准备已发布`4c5f1d9dd57cf84bacac77f670b2f015b76a1bde`并独立核远端，8文件172990字节权威Git/Mac主根/25386逐大小SHA一致；5个中间阅读文件33770字节Mac/服务器大小SHA一致。实际最终评测及5个媒体核验仍未完成，RF15未开始。[发布证据](../data/manifests/supermimic_q1_RF14_review_preparation_published_20261028.json)。

本阶段重要记录按Q1 payload审查/提交/推送、独立40位远端及Mac/25386大小SHA核验；本次提交号由Git历史提供，不预先宣称同步。小收据见[准备核验](../data/manifests/supermimic_q1_RF14_review_preparation_20261028.json)。

继续同一RF14至最终门，核TRAINING_COMPLETED、双guard正常reap0、pipeline COMPLETED、原进程退出/GPU释放与保存来源/优化器审计；复制小工件与视频到Mac并核验，再把数据和视频交给用户。等待其明确底座和RF15决定；不自动推进或扩时，不暂停现有goal/监督，不开agent。

