"""
Parse a training log and save a two‑panel plot of loss and validation pAUC.

"""

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

logger = logging.getLogger(__name__)

mpl.rcParams.update({
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


def parse_training_log(log_path):
    """Extract run seeds, folds, and per‑epoch train loss / val pAUC from the log."""
    runs = []
    current_run = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            run_match = re.search(r"Run: seed=(\d+), fold=(\d+)", line)
            if run_match:
                if current_run is not None:
                    runs.append(current_run)
                current_run = {
                    "seed": int(run_match.group(1)),
                    "fold": int(run_match.group(2)),
                    "epochs": [],
                    "train_loss": [],
                    "val_pauc": [],
                }
                continue

            epoch_match = re.search(
                r"Epoch (\d+)/\d+: train_loss=([\d.]+), val_pauc=([\d.]+)", line
            )
            if epoch_match and current_run is not None:
                current_run["epochs"].append(int(epoch_match.group(1)))
                current_run["train_loss"].append(float(epoch_match.group(2)))
                current_run["val_pauc"].append(float(epoch_match.group(3)))

    if current_run is not None:
        runs.append(current_run)

    return runs


def aggregate_runs(runs, max_epochs=6):
    """Stack runs into epoch‑aligned arrays of train loss and val pAUC."""
    train_loss_grid = np.full((len(runs), max_epochs), np.nan)
    val_pauc_grid = np.full((len(runs), max_epochs), np.nan)

    for i, run in enumerate(runs):
        for j, epoch in enumerate(run["epochs"]):
            if epoch <= max_epochs:
                train_loss_grid[i, epoch - 1] = run["train_loss"][j]
                val_pauc_grid[i, epoch - 1] = run["val_pauc"][j]

    return train_loss_grid, val_pauc_grid


def plot_training_curves(log_path, output_path):
    """Plot train loss and val pAUC curves with mean ± SD, and mark the peak."""
    runs = parse_training_log(log_path)
    logger.info(f"Parsed {len(runs)} runs from {log_path}")

    train_loss_grid, val_pauc_grid = aggregate_runs(runs)
    epochs = np.arange(1, train_loss_grid.shape[1] + 1)

    train_mean = np.nanmean(train_loss_grid, axis=0)
    train_std = np.nanstd(train_loss_grid, axis=0)
    val_mean = np.nanmean(val_pauc_grid, axis=0)
    val_std = np.nanstd(val_pauc_grid, axis=0)

    peak_epoch = int(np.nanargmax(val_mean)) + 1

    color_train = "#2c3e50"
    color_val = "#8b1a1a"
    color_runs = "#999999"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))

    for run in runs:
        ax1.plot(run["epochs"], run["train_loss"],
                 color=color_runs, linewidth=0.5, alpha=0.35)
    ax1.plot(epochs, train_mean, color=color_train, linewidth=1.8,
             label="Mean", zorder=10)
    ax1.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                     color=color_train, alpha=0.15, linewidth=0,
                     label="$\\pm$1 SD")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax1.set_xticks(epochs)
    ax1.set_xlim(0.8, 6.2)
    ax1.legend(loc="upper right", frameon=False, handlelength=1.5)
    ax1.grid(True, alpha=0.25, linewidth=0.5)
    ax1.set_axisbelow(True)

    for run in runs:
        ax2.plot(run["epochs"], run["val_pauc"],
                 color=color_runs, linewidth=0.5, alpha=0.35)
    ax2.plot(epochs, val_mean, color=color_val, linewidth=1.8,
             label="Mean", zorder=10)
    ax2.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                     color=color_val, alpha=0.15, linewidth=0,
                     label="$\\pm$1 SD")

    peak_val = val_mean[peak_epoch - 1]
    ax2.axvline(peak_epoch, color="black", linestyle=":", linewidth=0.7, alpha=0.6)
    ax2.annotate(
        f"Peak (epoch {peak_epoch})\npAUC = {peak_val:.3f}",
        xy=(peak_epoch, peak_val),
        xytext=(peak_epoch + 0.3, peak_val - 0.008),
        fontsize=8,
        color="black",
        arrowprops=dict(arrowstyle="-", color="black", linewidth=0.5, alpha=0.6),
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation pAUC")
    ax2.set_xticks(epochs)
    ax2.set_xlim(0.8, 6.2)
    ax2.legend(loc="lower right", frameon=False, handlelength=1.5)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Saved figure to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    plot_training_curves(
        log_path="phase_d_output.txt",
        output_path="results/figures/training_curves.pdf",
    )