# logs：纯执行记录

保存终端 stdout/stderr、命令执行时间、环境检查结果和小型运行记录。算法产生的数据、轨迹、权重、录制和视频放 `data/`。

`session_records/` 中检查过的小型关键日志进入 Git；其他日志默认忽略，重要大型日志需要在工件清单登记并备份。

示例（在项目根目录运行；替换为真实训练命令）：

```bash
bash -o pipefail -c 'python3 script/your_training_entry.py 2>&1 | tee logs/train_YYYYMMDD_HHMMSS.log'
```

保留失败日志。发布前检查日志没有凭证、会话 Cookie 或含认证参数的链接。
