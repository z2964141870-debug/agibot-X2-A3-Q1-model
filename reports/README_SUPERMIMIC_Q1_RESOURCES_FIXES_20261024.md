# Q1继续探索、数据接收与官方资源

记录日期：2026-10-24，北京时间。状态：用户已恢复探索和训练授权；Windows接收位置实际读写通过，基础库完成，后端和训练改进尚待真实验证。

19:47:48资源更新：用户指定D:\tmp已实际读到BONES metadata和smplx_lockedhead_20230207.zip；本轮优先完成SMPL-X原包直接迁移、双端SHA、三模型MD5和16betas/有限AMASS FK验证，见[身体模型阶段](README_SUPERMIMIC_Q1_SMPLX_INTAKE_20261024.md)。元数据尚未迁移、未完成下载文件不使用，其他训练工作状态沿各阶段收据恢复。

## 用户要求与当前范围

继续SuperMimic世界模型路线，优先以合格移动参考的小规模可学习性与模型诊断选择方案，再扩大覆盖和在线交互。用户将引入新数据；新增原料单独登记，不直接混入已固定评测或旧实验，不恢复旧A3/STOPPED任务。Mac不新增大型数据，服务器保留原始来源、原速、Q1转换与准入证据。

## Windows实际检查与迁移位置

桥接机100.65.183.91，现有yu账号。C:\Users\lzl\Downloads实际读取返回Access is denied；这是权限不足，不是文件不存在。本轮不改变该目录ACL或用户权限。C:\Users\yu\Downloads可读，实际0文件。

已创建C:\Users\yu\YuanQi_inbox，写入probe、读取相同内容、删除自身probe实际通过；接收目录当前0文件。用户可将SMPL-X模型、BONES-SEED包或新数据放在此目录。后续按文件名、大小、SHA和格式登记并从Windows迁往huanyu，不经Mac保存大型副本。

服务器目标：身体模型/opt/data/private/YuanQi_Q1/data/references/body_models/smplx/；BONES包/opt/data/private/YuanQi_Q1/data/references/bones_seed/archives/，metadata保留同来源目录；未知新数据先放data/references/incoming/<来源名>/，先盘点后使用。

WindowsSSH默认命令解析会展开嵌套PowerShell的变量，前两次带变量查询有语法/路径误判风险；最终使用UTF-16LE EncodedCommand实际运行并核验上述目录。密码只用于临时交互，不保存文件或Git。认证会话中断只重连，未重复转换已有数据。

## SMPL-X官方入口

官方下载页https://smpl-x.is.tue.mpg.de/download.php，未登录实际重定向到login.php；需要用户本人注册/登录、按其使用范围获取授权模型。官方代码https://github.com/vchoutas/smplx明确列出smplx/SMPLX_MALE.npz和SMPLX_FEMALE.npz，并记录2020-08-20提供完整形状/表情空间。下载SMPL-X身体模型包，选择完整形状空间版本；16维要求来自本机AMASS的betas，不是模型包文件名。到位后检查shapedirs、性别、文件身份及16维载入数值，不把默认num_betas=10当作文件只含10维，也不以SMPL/SMPL-H代替。

## Q1与其他数据调研

已找到官方开发者社区https://dev.primebot.com/和官方公开仓库https://github.com/PrimebotTech/primebot-q1-sdk；下一固定提交、核对接口和旧SDK差异，不把官网演示当作已发布可用policy。新来源优先核查LAFAN1完整连续移动动作，避免仅有PHUMA几秒chunk、且不依赖缺失人体模型；取源、许可、时间轴和Q1验证尚未完成，不能记为已接入训练。

## 实际进度、保存与下一步

19:13:16基础PHUMA库独立组装/时轴/完整源重算及原trainer准入退出0，train340/1227.10秒、移动62/221.76秒，细节见[扩库阶段](README_SUPERMIMIC_Q1_EXPANDED_RUN_20261024.md)。无新策略效果。

下一完成模型槽数值/梯度/资源验证，部署已发布的PHUMA共同轨迹诊断修正，落实磁盘保留空间与有限checkpoint预算，再登记独立改进试验。训练/评测分开报告，重要结果及时更新README/当前状态/索引。Git仍有此前连接中断导致的本地未同步记录，须重新推送并独立核对Q1远端。大工件LOCAL_ONLY。
