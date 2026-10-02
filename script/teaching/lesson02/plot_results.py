"""Plot measured Cartpole evaluations; does not run training or a simulator."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
args = parser.parse_args()
font = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
data = [json.loads((args.run_dir / f"evaluation_{label}.json").read_text())
        for label in ["initial", "trained"]]
colors = ["#D05A4E", "#178276"]
labels = ["训练前", "训练后"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), gridspec_kw={"width_ratios": [1, 1.7]})
fig.set_facecolor("#FAFAF7")
fig.suptitle("同一个任务，训练前后发生了什么？", fontproperties=font, fontsize=18, y=.98)
values = [item["mean_survival_s"] for item in data]
axes[0].bar([0, 1], values, color=colors, width=.55)
axes[0].set_xticks([0, 1], labels, fontproperties=font, fontsize=12)
axes[0].set_ylim(0, 5.8)
axes[0].set_ylabel("平均维持时间 / 秒", fontproperties=font, fontsize=11)
axes[0].set_title("1024 个相同初始状态", fontproperties=font, fontsize=12)
for i, value in enumerate(values):
    axes[0].text(i, value + .14, f"{value:.2f} 秒", ha="center", fontproperties=font, fontsize=14)
for item, color, label in zip(data, colors, labels):
    trace = item["trace_env_0"]
    time = np.array([x["step"] for x in trace]) * item["control_dt_s"]
    angles = np.degrees([x["observation"][0] for x in trace])
    axes[1].plot(time, angles, color=color, lw=2.6, label=label)
axes[1].axhline(0, color="#858585", ls="--", lw=1)
axes[1].set(xlim=(0, 5), ylim=(-102, 25))
axes[1].set_xlabel("时间 / 秒", fontproperties=font, fontsize=11)
axes[1].set_ylabel("杆的倾斜角 / 度", fontproperties=font, fontsize=11)
axes[1].set_title("同一初始状态的一条轨迹（编号 0）", fontproperties=font, fontsize=12)
axes[1].annotate("训练前约 0.45 秒失败", xy=(.4333, -89.6), xytext=(1.15, -76),
                 fontproperties=font, fontsize=11,
                 arrowprops={"arrowstyle": "->", "color": colors[0]}, color=colors[0])
axes[1].text(3.4, 5, "0°：杆竖直", fontproperties=font, fontsize=10, color="#555555")
axes[1].legend(prop=font, loc="lower right", frameon=False)
for ax in axes:
    ax.set_facecolor("#FAFAF7")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
fig.text(.5, .02, "真实仿真记录 · 回合约 5 秒 · 一个训练种子 · 不代表人形机器人结果",
         ha="center", fontproperties=font, fontsize=10, color="#555555")
fig.tight_layout(rect=[0, .06, 1, .93])
fig.savefig(args.run_dir / "learning_comparison.png", dpi=160, facecolor=fig.get_facecolor())
plt.close(fig)
