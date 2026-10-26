# YUANQI 当前状态与下一步

2026-10-27恢复入口：[RF08最终结果](README_SUPERMIMIC_Q1_RF08_RESULTS_20261027.md)、[MG03完成审计](../data/manifests/supermimic_q1_MG03_completion_audit_20261027.json)、[运行记录](README_SUPERMIMIC_Q1_RF08_HP_SOMA_20261026.md)。RF08全pipeline已于00:27:46 COMPLETED，全部训练/评测/绘图进程已退出；MG03 A至E证据验收齐备，目标工具随最后提交核验后结算。本轮q1-rf08-4096已实际PAUSED，没有在训作业；不恢复任何旧命令，未续训或超预算。

## 最终效果与交付

实际训练3500.923046秒、3510学习轮/36204544转移、35100次WM/3010次actor更新。100训练片提前终止100→10、15验证片15→0；验证walking3/3完成，沿目标均移1.1234m/参考2.3943m、速度0.2646m/s。双足近地仍约90.72%，只是几何诊断，不是接触力测量。

全身跟踪未达标：共同约1秒前缀的上肢RMSE训练0.560→0.868rad、验证0.517→0.861rad，训练原地gesture净漂移1.162m/参考0.023m。腕峰值高度和相位亦有明显偏差。存活和移动有进步，不等于准确复现动作；完整数据见最终结果，不用模型loss或摔倒位移称稳定行走。

初末115片共230条真实物理rollout、26项训练保存/来源检查、12项物理算术/身份检查、共同前缀及全程支撑/肩肘腕分析已完成。六宫格20560条CSV与500学习轮平滑独立重算通过，两段MP4各147帧/5.88秒/H264/1280×480全帧解码正常。25小文件8044244字节在Mac/服务器大小SHA一致，[交付清单](../data/manifests/supermimic_q1_RF08_local_delivery_20261027.json)。

本地图：`/Users/yu/Documents/ChatGPT/元启/data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026/dashboard/training_dashboard.png`，同目录PDF/CSV。最终视频：同实验根`evaluation/final.mp4`，初始为`evaluation/initial.mp4`。视频左为真实策略物理轨迹、右为运动学参考；不建网站。模型/大轨迹仅服务器LOCAL_ONLY。

## 版本、输入与连接

权威Git：`/tmp/yuanqi-supermimic-q1-20261023`，分支Q1，远端`git@github.com:z2964141870-debug/agibot-X2-A3-Q1-model.git`。冻结训练代码`d8b1dcf76b7819ddd5be39a5177e14fe1737a3ad`，固定上游`310244b73e45775744a75b789eb25a6c9c5a81c9`，146源码/资产及115输入终后SHA通过，之后只改记录/独立后处理。

正式计算仅用户25252容器，唯一根`/opt/data/private/ZHY`、四目录不变。SSH：`ssh -S /tmp/yuanqi-q1-25252.sock -T -p 18869 -o BatchMode=yes -o ConnectTimeout=8 -o 'HostKeyAlias=[59.77.15.15]:25252' root@127.0.0.1`。保留Windows桥，不读写认证，不操作师兄容器。

Python为`data/environments/supermimic_q1_hp_20261026/bin/python`，须经`script/supermimic_q1_hp/activate.py`提供CUDA/EGL库路径。runtime为`data/references/supermimichp/AP01_runtime_20261026`，库为`data/supermimic_for_q1/AP01_SOMA_Q1_20261026/q1_candidates/pilot_manifest.json`，实验为`data/supermimic_for_q1/BONES_RF08_SOMA_HP_E4096_MAX01H_20261026`。[AP01](README_SUPERMIMIC_Q1_HP_ALIGNMENT_20261026.md)转换/FK/GPU时序/完整4096资源实测已完成，不重复转换、profile或hash45GB原包。

4096env/WM1024×10/actor1024-H32/critic5×4/UTD.4，100训练/15演员分离验证SOMA。师兄实际1024env、精确100/15名单未获，本轮独立选片；Q1模型armature=.004推导PD15.7914/1.00531，不是厂家逐关节特调。软件cu128与一小时预算差异保留，不声称全部条件等同。

## 资源与后续

00:40:09 controller21522、训练21525、评测25224、绘图25780均不存在，GPU14MiB/26度、共享13318414336字节/OOM0、余盘1608945061888字节、实验约0.993GB，100GiB余盘/12GiB实验/30GiB共享/86度保护保持。[资源收据](../data/manifests/supermimic_q1_RF08_final_resources_20261027.json)。旧RF05/06/07均封存，不重启；MG02已完成。

结果提交bb8c909e44badf8346b6a8d08b5246aa75d48c67已独立核远端40位SHA，19文件1354813字节在权威Git/Mac/25252大小SHA一致；收尾记录随本次提交再核。q1-rf08-4096暂停已由工具和本地结构化回读双核，其他字段保留；supermimichp-x2每小时跟踪与yuanqi未改。之后候选是漂移/上肢动作尺度、控制响应、真实与想象奖励分项/相位诊断。本轮最多一小时，更长2/4/8小时需用户另行决定；不开子agent、不自动追加训练、不添加数据硬门。无关download_bones.py不执行/修改/暂存/删除。
