# Windows SSH别名与当前连接断点

记录日期：2026-10-29，北京时间16:10，16:30更新。当前状态：用户明确授权本机保存Windows密码并自动输入，已实测Windows和25386均成功连接。旧“密码不保存”和UNKNOWN描述只记录16:10阶段，不再代表当前状态。

## 16:30自动密码与连接恢复

用户最新明确要求把Windows密码保存在本机，覆盖此前该快捷方式的“不存密码”偏好。OpenSSH不支持Password配置字段，现采用系统SSH的ASKPASS接口：密码仅在`/Users/yu/.ssh/hp_han.password`、权限600；助手`hp_han-askpass`为700，shell分派`hp_han-shell.zsh`为600，均在本机.ssh目录。`/Users/yu/.zshrc`加载分派，只有native ssh -G确认目标为hp_han/yu/100.65.183.91:22时才自动输入，其他SSH使用原命令。

新终端直接`ssh hp_han`；已打开的终端执行一次`source ~/.zshrc`即可生效。助手只响应密码提示，拒绝主机密钥等其他提示；不自动接受主机密钥。密码内容及摘要、本机认证文件和完整SSH配置均不进项目Git或任何服务器镜像。容器密码没有保存。

实际验证：shell语法/解析通过，助手输出在内存中与本机文件对照成功且不打印内容、非密码提示被拒绝；自动登录Windows返回`DESKTOP-3C1DPPR`/exit0。随后原Windows60607/18873及25386 master60674恢复，容器interactive40173、原boot4d859008和GPU26028785、实体根/opt/data/private/ZHY一致。现场GPU15MiB/0%无计算；当前cgroup memory.events文件不存在，未沿用历史OOM0。没有启动新推理、物理、optimizer或训练。

[授权与实测收据](../data/manifests/q1_windows_ssh_hp_han_autopassword_20261029.json)不含认证内容。之前小记录及本轮记录补镜须另核大小/SHA，不能由连接成功推断交付完成。原H32 V02失败后的停止规则仍保持。

## 初始别名配置（16:10历史）

用户要求以后用`ssh hp_han`代替Windows的完整SSH地址。已在本机`/Users/yu/.ssh/config`新增独立Host段，保留其他条目，文件权限仍为600：

```sshconfig
Host hp_han
  HostName 100.65.183.91
  User yu
  Port 22
  ConnectTimeout 8
  ServerAliveInterval 20
  ServerAliveCountMax 3
```

实际执行`ssh -G hp_han`，确认hostname100.65.183.91、useryu、port22、超时和keepalive均按上段生效。以后直接运行：

```sh
ssh hp_han
```

密码由SSH交互提示输入，没有保存在SSH配置、脚本、命令参数或Git。本地完整SSH配置和任何认证文件没有复制进项目或服务器。

## 连接断点（16:10历史）

用户再次明确要求从Windows跳转后，已先核18873无监听、Windows与25386两个控制socket均不存在。随后仅重新尝试一次原Windows本地转发；出现Windows密码提示，在非回显输入后以`ssh_dispatch_run_fatal: Connection to 100.65.183.91 port 22: Broken pipe`结束。没有建立桥，也没有执行Windows到25386的容器命令。

因此已知断点在Mac到Windows的SSH认证交互/会话阶段，还没有证据判断容器掉线、密码错误或Windows到容器的路由问题。该失败不是Permission denied。创建别名后仅做本地解析验证，没有自动重复认证。

`hp_han`指向Windows，并不自动建立18873转发或登录25386。原目标容器仍只有25386入口；没有探测旧入口、重启服务器、训练或远端作业。后续需先得到成功的Windows会话，再建立原18873到25386的转发并核服务器身份。

## 项目记录

安全映射和检查结果见[别名清单](../data/manifests/q1_windows_ssh_alias_hp_han_20261029.json)。代码/README保持原Q1分支发布流程；本机SSH配置只留在本机。

同期[H32逐步记录](README_SUPERMIMIC_Q1_H32_INIT_AUDIT_20261029.md)的代码提交1f224fd已独立核GitHub、10项308960字节Git/Mac大小SHA一致；底座/PPO原生验证仍未追加，服务器镜像待连接恢复，不把本机别名配置当作远端连接通过。

别名/后续记录已本地提交`b5ecaeda4f67c86cdaa5eb419e14857d83aa8348`，8项281680字节Git/Mac逐大小SHA一致；此次push在GitHub22端口连接超时，尚未核远端，不重复推送或认证。本条通知与[待发布收据](../data/manifests/q1_windows_ssh_alias_hp_han_publication_20261029.json)另存本地Git；已核的H32源码1f224fd不受该后续记录同步状态影响。
