# script：代码、脚本与配置

当前包含项目管理工具、研究任务配置快照和教学例子；已运行单步示例与官方倒立摆 PPO 完整仿真训练/保存/重载，尚未完成 X2/人形策略训练。

| 文件 | 作用 |
| --- | --- |
| `teaching/lesson02/run_lesson.py` | 最小完整训练入口：限定运行时间、保留日志、独立进程重载；只针对仿真 |
| `teaching/lesson02/cartpole_ppo.py` | 调用现有 Isaac Lab 官方倒立摆任务和 RSL-RL PPO，保存前后评测、参数差异与动作探针 |
| `teaching/lesson02/plot_results.py` | 从评测 JSON 生成训练前后对照图，不启动训练 |
| `project_tools/inventory_host.py` | 只读检查主机、GPU、磁盘、Git 与百度网盘入口；不读取认证内容 |
| `project_tools/audit_git_payload.py` | 检查暂存文件的路径、大小、模型格式及常见凭证特征 |
| `project_tools/register_artifact.py` | 为 `data/` 下的大型工件生成大小、SHA-256 和备份清单；不上传 |
| `research_watch/automation_snapshot.json` | 已创建每日任务的审计快照，不另起调度器 |
| `teaching/lesson01/policy-target-pd.html` | 目标角、实际角与 PD 力矩的交互教学片段；不连接机器人、不训练网络 |
| `teaching/lesson01/reward_vs_update.py` | 用已有 PyTorch 在 CPU 演示奖励、梯度、参数更新与评估；单步 REINFORCE，不是 PPO 或机器人仿真 |
| `teaching/lesson01/actor_critic_update.py` | 固定一条假设经历，用 PPO 裁剪策略项和价值平方损失更新两个小网络；CPU 上正、负、零优势对照通过，不是完整 PPO 训练 |

项目管理工具使用 Python 标准库；第一课使用已有 PyTorch，第二课使用既有 Isaac Lab/RSL-RL 环境，未安装新依赖。复现命令与结果见[倒立摆实验](../reports/README_LESSON02_CARTPOLE_20261002.md)；旧例见[第二小节](../reports/README_LESSON01_REWARD_UPDATE_20260930.md)与[第七小节](../reports/README_LESSON01_ACTOR_CRITIC_UPDATE_20260930.md)。人形训练配置仍待独立核验。
