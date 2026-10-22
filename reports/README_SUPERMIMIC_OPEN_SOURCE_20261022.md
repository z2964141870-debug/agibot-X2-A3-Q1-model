# SuperMimic 开源状态核验

记录日期：北京时间2026-10-22 08:20。阶段：supermimic-open-source-20261022。状态：来源核验与分析完成；GitHub发布回执见本页保存状态。此为用户指定项目的单项调查，不是每日检查。

## 目标与范围

核实 https://supermimic.github.io/#compute 是否已开放可用的官方训练代码、部署实现和模型权重，并区分第三方复现。仅访问公开页面与GitHub元数据、静态阅读文档；没有安装依赖、执行第三方实现、运行仿真/训练、加载或下载模型、操作真机。

恢复时读取AGENTS、报告索引/current和阶段模板。通用研究正式目录仍为hp3090:/media/yu/FAFF-E9771/YUANQI；A3工程在humanplus的独立目录，本轮没有连接或修改A3工程。

## 结论与关键依据

**截至本次检查，没有核验到可直接使用的官方SuperMimic开源训练包或授权下载的官方权重。已公开论文和浏览器演示；演示本身明确采用专有条款。另有两个独立第三方复现仓库，不能称为作者开源。**

| 对象 | 核验结果 | 边界 |
| --- | --- | --- |
| 论文 | arXiv:2610.28480v1可访问；原提交2026-10-20 16:29:30 UTC，即北京时间10/21 00:29:30 | 本轮读摘要/提交历史，未完成论文全文审查 |
| 官方项目页 | 可访问，提供Paper、Video、Web demo入口；本次提取的链接无官方算法仓库或checkpoint发布入口 | 页面没有链接不等于证明所有平台永久不存在代码 |
| 官方训练/部署实现 | 本次官方入口与公开GitHub检索未核验到发布 | 不把页面HTML/JS或开源依赖当成官方训练实现 |
| 官方权重/动作资产 | 未核验到可独立下载与使用的官方发布；Demo参数被其专有条款覆盖 | Hugging Face检索超时，模型站的全面可见性未知 |
| 官方Demo许可 | LICENSE第1–3节将演示代码、policy/world-model参数及相关资产列为专有材料；允许个人非商业浏览器体验，限制复制、改作和用于训练/微调/蒸馏其他模型 | 仅据页面所述使用条款判断该Demo不是开源发布；不将其许可外推至论文方法或独立复现 |

许可证是比“未搜到GitHub”更强的正面证据。第三方依赖各自的MIT/Apache/BSD许可不改变SuperMimic演示自身的许可。初次读取公开Demo页面、脚本标头及许可证后，没有继续获取其参数、worker或资产，也没有在磁盘保存/镜像Demo代码与参数。

## 已发现的第三方代码

1. [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)，固定a1508d42c9d1699a52e8233ab16d08c488441ab9。README明确写Unofficial、非作者关联的独立重写，并自述快速AI编码、未充分审查。目录元数据包含训练/评测与世界模型相关Python文件；固定README及LICENSE解码后Git-blob校验通过，代码许可证为MIT。其40段LAFAN1结果为维护者报告，含关闭自碰撞、精度和观测等偏离，不能据此认定复现了原论文完整数据/真机结果。
2. [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，固定4c27c13f4d6a625376700fadbdc3d7cb2a4d81a1。README自称独立从零复现，目录元数据包含训练/世界模型/评测实现；固定README及LICENSE解码后Git-blob校验通过，代码许可证为MIT。README主动区分论文观测与额外锚点观测，标明部分结果不同口径、部分重跑中，并列出尚无ONNX导出/真机验证等缺项。不能将这些维护者自报结果记成我们已验证的能力。

以上只完成仓库身份、目录元数据和文档/许可证核验。**未审完整训练源码、未执行测试、未核验checkpoint发布、未复现其结果。** MIT针对各仓库自己的代码，不自动覆盖外部动作数据或官方Demo资产。

## Compute如何理解

项目页作者报告：1张H100的3小时覆盖采集仿真转移、拟合世界模型和更新policy的完整训练循环；训练后的checkpoint评测不计入。SONIC对照为作者引用的128张L40/7天，数据量与GPU型号不同。该结果不能直接换算为其他GPU耗时、A3/Q1适配预算或项目已经达到的收益。

项目页明确129,514段benchmark属于两方法的训练动作，因此失败数衡量训练集覆盖，不证明未见动作泛化；其“0失败”也不等于真机安全或任意衣服遥操成功。

方法线索是“用一个学习的动力学模型预测短期运动，再通过该模型的梯度改进控制策略”，而不是拿到一个现成G1权重就能直接用于A3/Q1。现阶段建议登记为候选：保留已有035匹配基线与数据/坐标/时间接口工作，后续若评估第三方实现，先审计参考输入、观测、执行器和评测口径，再决定是否另立小实验。本调查不改变训练主线或恢复已停止实验。

## 版本、命令与证据

主要原始来源：

- 项目/Compute：https://supermimic.github.io/#compute
- Demo许可：https://supermimic.github.io/demo/LICENSE.html
- 论文/提交历史：https://arxiv.org/abs/2610.28480v1
- GitHub公开候选检索：https://api.github.com/search/repositories?q=supermimic
- 上述两仓库的固定提交README、LICENSE与Git-tree API。

本地缓存为data/research/2026-10-22/supermimic/，只留项目首页、arXiv摘要和许可允许留存的第三方文档/API元数据；官方Demo材料不镜像。自编分析清单为data/manifests/supermimic_open_source_20261022.json；执行记录为logs/supermimic_20261022/。原始缓存LOCAL_ONLY，未网盘备份。

可复核的只读命令：

```bash
curl -fSL --connect-timeout 6 --max-time 25 'https://supermimic.github.io/'
curl -fSL --connect-timeout 6 --max-time 25 'https://arxiv.org/abs/2610.28480v1'
curl -fSL --connect-timeout 6 --max-time 25 'https://api.github.com/repos/zita-ch/supermimic_vibed/contents/README.md?ref=a1508d42c9d1699a52e8233ab16d08c488441ab9'
```

重新请求产生新快照，不覆盖原检查时点证据。官方Demo按其页面条款使用，不通过演示抽取参数或制作训练数据。

## 实际验证与缺口

- 官方主页/论文摘要/许可可访问；两仓库固定树元数据完整、四份文档Git-blob/大小/SHA256独立校验通过。
- GitHub默认名称检索返回2个复现库；扩展名称/描述/README检索返回18项，包含论文清单。作者qiayuanl公开profile姓名匹配Qiayuan Liao，公开仓库列表未发现SuperMimic官方实现。此检索不是全网不存在代码的证明。
- Hugging Face API连接超时；两个raw.githubusercontent.com请求被连接重置，已改用固定提交GitHub contents API读取并验hash。
- web工具调用未提供可用正文，实际来源核验以上述直接HTTP访问为准；不伪造检索引用。
- hp3090一次有界SSH Git状态请求超时，正式工作树HEAD/未提交修改未知，没有覆盖它。发布副本沿用data/research/2026-10-20/record_checkout，开始时干净；fetch和ls-remote一致为a57632e80cd0931459858ba08238ffebc8042bc7。公开仓库的较新工程记录保持原样，只添加本调查记录/入口。

## 保存与同步状态

本地报告、来源清单及索引/current新增调查入口已保存。研究内容提交a952eedcbcb65e7e0d5d5d2b074c9a768139ee18已普通推送main；独立GitHub API分支SHA一致，两份新文档的固定远端内容逐字节/Git-blob/SHA256与本地提交相符，发布副本干净。四份暂存文件120,927字节的diff/暂存区和audit_git_payload.py通过。发布核验见data/manifests/supermimic_open_source_20261022.json的publication字段及logs/supermimic_20261022/publication_verification.json。本次回执更新另行审计提交，提交号由Git历史提供。正式hp3090工作树仍未知，未改写它。没有运行中的本任务科学作业，不改每日检查时间或既有去重清单。

## 下一步

回答用户“官方未核验到训练/权重开源，Demo专有；有非官方MIT复现但未验证”。若后续官方实质发布，再单独更新状态；第三方实现是否可用需另做具体源码和资源评估，本次不启动它。
