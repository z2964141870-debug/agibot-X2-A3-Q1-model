"""Plot saved offline forecast evidence without model loading or inference."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def identity(path):
    content = path.read_bytes()
    return dict(path=str(path), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    inputs = [
        root / "logs/session_records/q1_upper_causal_forecast_20261106.json",
        root / "logs/session_records/q1_upper_mps_forecast_completed_20261106.json",
        root / "logs/session_records/q1_upper_consistency_completed_20261106.json",
        root / "data/supermimic_for_q1/Q1_UPPER_MPS_FORECAST_20261106/training_trace.json",
    ]
    before = [identity(path) for path in inputs]
    causal, mlp, paired, trace = [json.loads(path.read_text()) for path in inputs]
    if not (mlp["actual_optimizer_updates"] == 1000
            and paired["actual_optimizer_updates_total"] == 2000
            and mlp["selected_update"] == 800
            and causal["training"]["validation_selected"] == "ridge_0.01"):
        raise ValueError("Unexpected saved forecast experiment identities")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = [output / name for name in ("offline_forecast_six_panel.png",
                                       "offline_forecast_six_panel.pdf", "plot_source_receipt.json")]
    if any(path.exists() for path in names):
        raise FileExistsError("Use new plot output paths")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10))
    blue, green, orange = "#2463a6", "#237b52", "#cf7028"
    x, width = np.arange(2), .34
    clean = causal["clean_test_mean_per_horizon_rmse_rad"]
    noisy = causal["noisy_test"]
    axes[0, 0].bar(x-width/2, [clean["damped_cv"], noisy["damped_cv_mean_rmse_rad"]],
                   width, label="Damped CV", color=blue)
    axes[0, 0].bar(x+width/2, [clean["validation_selected_ridge_0.01"],
                   noisy["selected_ridge_mean_rmse_rad"]], width, label="Selected ridge", color=orange)
    axes[0, 0].set(xticks=x, xticklabels=["Clean input", "IID noise 0.01 rad"],
                   ylabel="Mean horizon RMSE (rad)", title="A. Clean fit gains can reverse under noise")
    axes[0, 0].legend(fontsize=8)

    trained = [row for row in trace if row["last_training_loss"] is not None]
    axes[0, 1].plot([row["completed_updates"] for row in trained],
                    [row["last_training_loss"] for row in trained], "o-", color=blue)
    axes[0, 1].set(xlabel="Completed MPS updates", ylabel="Saved training MSE",
                   title="B. Small MLP: 1,000 actual updates")

    steps = [row["completed_updates"] for row in trace]
    risks = [row["validation_risk_rad"] for row in trace]
    selected = next(row for row in trace if row["completed_updates"] == mlp["selected_update"])
    axes[0, 2].plot(steps, risks, "o-", color=blue)
    axes[0, 2].scatter([selected["completed_updates"]], [selected["validation_risk_rad"]],
                       color=green, s=85, zorder=3, label="Validation-selected update 800")
    axes[0, 2].set(xlabel="Completed MPS updates", ylabel="Validation risk (rad)",
                   title="C. Selection used validation before test")
    axes[0, 2].legend(fontsize=8)

    stress = mlp["test_mlp_ten_horizon_mean_rmse_rad"]
    keys = ["clean", "iid_0.01", "iid_0.02", "ar1_0.8_0.01", "window_bias_0.01", "last5_frozen"]
    axes[1, 0].bar(np.arange(6), [stress[key] for key in keys],
                   color=[blue, green, green, blue, blue, orange])
    axes[1, 0].set(xticks=np.arange(6), xticklabels=["Clean", "IID .01", "IID .02", "AR1", "Bias", "Freeze 5"],
                   ylabel="Mean horizon RMSE (rad)", title="D. MLP stress tests: frozen inputs remain hard")
    axes[1, 0].tick_params(axis="x", labelsize=9)

    control = [paired["validation_risks_rad"][0], paired["clean_test_mean_rmse_rad"][0]]
    revision = [paired["validation_risks_rad"][1], paired["clean_test_mean_rmse_rad"][1]]
    axes[1, 1].bar(x-width/2, control, width, label="Paired control", color=blue)
    axes[1, 1].bar(x+width/2, revision, width, label="Revision loss 0.5", color=orange)
    axes[1, 1].set(xticks=x, xticklabels=["Validation risk", "Clean test RMSE"],
                   ylabel="Error (rad)", title="E. Continuity loss did not improve all errors")
    axes[1, 1].legend(fontsize=8)

    x = np.arange(3)
    axes[1, 2].bar(x-width/2, paired["noisy_dense_revision_p95_reduction_percent_by_fixed_clip"],
                   width, label="Prediction revision p95", color=green)
    axes[1, 2].bar(x+width/2, paired["noisy_dense_80ms_signal_acceleration_p95_reduction_percent"],
                   width, label="80ms reference acceleration p95", color=blue)
    axes[1, 2].set(xticks=x, xticklabels=["Forward", "Start", "Stop"], ylabel="Reduction versus paired control (%)",
                   title="F. Noisy reference continuity improved")
    axes[1, 2].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=.20)
        axis.set_axisbelow(True)
    fig.suptitle("Offline Upper-Body Forecast Studies | Saved Evidence", fontsize=17, y=.97)
    fig.text(.5, .925, "Small reference predictors, separate from the shared robot-policy training", ha="center", fontsize=11)
    fig.text(.5, .025, "Reused test set: exploratory. No new training, robot inference or physics in this plot.\n"
             "Reference prediction and continuity are not measured hardware latency or closed-loop safety.",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.065, right=.985, top=.86, bottom=.12, wspace=.30, hspace=.47)
    fig.savefig(names[0], dpi=160)
    fig.savefig(names[1])
    plt.close(fig)
    if [identity(path) for path in inputs] != before:
        raise RuntimeError("Saved input changed while plotting")
    record = dict(stage="SAVED_OFFLINE_FORECAST_EVIDENCE_PLOT", status="PASSED", inputs=before,
                  inputs_unchanged=True, figures=[identity(path) for path in names[:2]],
                  new_training_updates=0, new_robot_policy_calls=0, new_physical_steps=0)
    names[2].write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(dict(status="PASSED", figures=[str(path) for path in names[:2]],
                         inputs_unchanged=True, new_training_updates=0)))


if __name__ == "__main__":
    main()
