# CPU加速已保持策略动作，软件长尾仍需处理

2026-11-08，北京时间。原共享训练保持暂停。

done：前走/起走各32保存状态，64次真实GPU前向全部通过动作等价门。最大动作差仅2.74e-6/3.34e-6，低于原定容差；906保护输入同，两个worker干净回收，零新物理、媒体或训练。本轮GPU占用59秒，累计2492/14400秒已结算。

| 去每条前两调用，ms | 前走 | 起走 |
| --- | ---: | ---: |
| 新CPU状态转换中位 | .83 | .98 |
| 完整worker中位 / p95 | 9.88 / 34.72 | 11.05 / 15.05 |
| 本机RPC中位 / p95 | 11.43 / 36.85 | 12.09 / 16.26 |

前走worker仍有4/30次超过20ms，全部调用中的约300ms初始化尖峰也保留。因此只确认CPU改进可保持动作，没有证明20ms期限、硬件端到端延迟或自然遥操已经达标。上次CPU单项约17倍加速不能外推为完整链路同倍提升。

todo：回到H4时间mask/闭环目标，处理人体未来帧依赖和真实控制质量，保留旧f0七例失败。原共享训练不恢复，采纳保持null。

[六宫图](../data/analysis/Q1_CPU_ACTION01_FIGURES_20261108/cpu_action_chain.png) / [PDF](../data/analysis/Q1_CPU_ACTION01_FIGURES_20261108/cpu_action_chain.pdf) / [CSV](../data/analysis/Q1_CPU_ACTION01_FIGURES_20261108/cpu_action_chain.csv) / [完整记录](README_Q1_CPU_ACTION_20261108.md)。[新证据包](../data/deliveries/Q1_CPU_ACTION_CHAIN_EVIDENCE_20261108.zip)327401B，21成员CRC/逐件SHA/7链接已核验；此包无新视频，旧CPU两图及步态视频证据包保留。
