"""Plot measured saved-checkpoint performance; no training or simulation."""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from script.a3.regression_diagnostic import OUTPUT


def main():
    result = json.loads((OUTPUT / "comparison.json").read_text())
    rows = [r for r in result["rows"] if r["label"] != "official"]
    x = list(range(len(rows)))
    labels = [r["label"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    axes[0].plot(x, [r["all"]["falls"] for r in rows], "o-", color="#b83d37")
    axes[0].set(title="Falls / 20 motions", ylabel="Count", ylim=(-.3, 20.3), yticks=[0, 5, 10, 15, 20])
    axes[1].plot(x, [r["all"]["full_rollout"]["joint_rmse_rad"] for r in rows], "o-", color="#b36a16", label="Full replay")
    axes[1].plot(x, [r["common_prefix"]["joint_rmse_rad"] for r in rows], "o-", color="#1675ad", label="Same pre-fall windows")
    axes[1].set(title="All motions: joint tracking", ylabel="RMSE (rad)")
    axes[1].legend(fontsize=8)
    axes[2].plot(x, [r["groups"]["heldout"]["full_rollout"]["joint_rmse_rad"] for r in rows], "o-", color="#16724d")
    axes[2].set(title="4 R05 held-out motions", ylabel="Full-replay RMSE (rad)")
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("Saved update stage (category axis)")
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("R05 fine-tuning regression: one trajectory, fixed MuJoCo protocol", fontsize=12)
    fig.savefig(OUTPUT / "regression_curve.png", dpi=160)
    fig.savefig(OUTPUT / "regression_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
