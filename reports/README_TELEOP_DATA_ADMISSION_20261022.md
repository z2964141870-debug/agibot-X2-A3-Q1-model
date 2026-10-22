# 传感衣数据准入清单：206场不等于206个完整可用配对

2026-10-22，目标轮2；LOCAL_ONLY，未推理/训练/仿真。huanyu经Windows仍在SSH banner阶段超时，exit255。本轮继续本地不依赖服务器的工作，不标记整个目标blocked。

## 本轮结果
标准库读取既有train_released.zip中央目录，不解压大文件/不加载PT：206个含acc.pt的会话目录；197个在同一目录同时具备acc.pt、rot.pt、device2bone.pt、pose.pt、joint.pt的候选。另9个taichi目录缺device2bone.pt，不默用其他人的校准；不证明文件损坏或全部不可用，只是不满足当前准入合同。

缺校准：taichi/1、2、20、3、5、6、7、8、train。需要查发布代码是否另有合法共享标定来源，未查清前不静默补齐。

197个只是同源目录候选，仍未证明张量shape、源帧对齐、坐标/单位、真实GT来源或FGP预测存在；不标为197条有效GT/FGP配对，不划训练测试。原包已知被用于FGP训练，不宣称未见泛化。

## 新工具
[候选生成器](../script/teleop_research/pairing_candidates.py)拒绝跨目录拼接、缺标定和不安全路径；[测试](../script/teleop_research/test_pairing_candidates.py)包含4个反例/正例。连同已有证据合同，20项唯一测试通过。

命令：
```bash
python3 -m unittest script.teleop_research.test_pairing_candidates script.teleop_research.test_evidence_contract -v
python3 script/teleop_research/pairing_candidates.py /Users/yu/Human+/data/train_released.zip --output data/manifests/teleop_pairing_candidates_20261022.json
```
现存输出拒绝覆盖，重建需指定独立输出名。清单为data/manifests/teleop_pairing_candidates_20261022.json，包含源entry、大小、CRC、待核实来源状态。CRC来自ZIP目录，不等于逐成员实读验证。

## 下一步
连接恢复后核服务器输入SHA/环境，先选一条完整发布包序列跑离线FGP配对（仅分布内诊断）；查GT采集来源和帧索引，再独立量化人体估计/重定向/闭环误差。demo100或SMB未见性须对照训练成员证据，未知则明确unknown。多分支探索合同见[首轮合同](README_TELEOP_EXPERIMENT_CONTRACT_20261022.md)。

## 三维评估
创新性：没有新增算法主张。实用性：避免9场缺标定输入被错误送入推理，并准备197场可追踪候选。严谨性：候选与有效配对、包内标签与独立GT明确分开；20项单测只验证工具，不代表机器人效果。
