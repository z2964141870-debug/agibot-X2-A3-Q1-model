# script：代码、脚本与配置

当前只有项目管理工具，还没有选定完整机器人训练框架或运行训练。

| 文件 | 作用 |
| --- | --- |
| `project_tools/inventory_host.py` | 只读检查主机、GPU、磁盘、Git 与百度网盘入口；不读取认证内容 |
| `project_tools/audit_git_payload.py` | 检查暂存文件的路径、大小、模型格式及常见凭证特征 |
| `project_tools/register_artifact.py` | 为 `data/` 下的大型工件生成大小、SHA-256 和备份清单；不上传 |

工具使用 Python 标准库。训练代码与依赖等完成 P0 后再加入独立子目录，避免把运行包误当作训练源码。
