# Windows SSH别名与当前连接断点

记录日期：2026-10-29，北京时间16:10。状态：按用户明确要求配置本机`hp_han`别名并验证解析；Windows认证后会话仍断开，容器连接UNKNOWN。

## 已完成

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

## 当前连接断点

用户再次明确要求从Windows跳转后，已先核18873无监听、Windows与25386两个控制socket均不存在。随后仅重新尝试一次原Windows本地转发；出现Windows密码提示，在非回显输入后以`ssh_dispatch_run_fatal: Connection to 100.65.183.91 port 22: Broken pipe`结束。没有建立桥，也没有执行Windows到25386的容器命令。

因此已知断点在Mac到Windows的SSH认证交互/会话阶段，还没有证据判断容器掉线、密码错误或Windows到容器的路由问题。该失败不是Permission denied。创建别名后仅做本地解析验证，没有自动重复认证。

`hp_han`指向Windows，并不自动建立18873转发或登录25386。原目标容器仍只有25386入口；没有探测旧入口、重启服务器、训练或远端作业。后续需先得到成功的Windows会话，再建立原18873到25386的转发并核服务器身份。

## 项目记录

安全映射和检查结果见[别名清单](../data/manifests/q1_windows_ssh_alias_hp_han_20261029.json)。代码/README保持原Q1分支发布流程；本机SSH配置只留在本机。

同期[H32逐步记录](README_SUPERMIMIC_Q1_H32_INIT_AUDIT_20261029.md)的代码提交1f224fd已独立核GitHub、10项308960字节Git/Mac大小SHA一致；底座/PPO原生验证仍未追加，服务器镜像待连接恢复，不把本机别名配置当作远端连接通过。
