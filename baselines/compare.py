"""
Compare naive, tabular, and image‑only baselines with forest plot and table.

"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/baselines")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.dpi"] = 300


def load_baseline_summaries() -> pd.DataFrame:
    """Load mean, CI, and std from baseline summary JSON files."""
    rows = []

    summary_paths = {
        "Naive (majority class)": RESULTS_DIR / "naive_summary.json",
        "Tabular (LightGBM)": RESULTS_DIR / "tabular_summary.json",
        "Image-only (efficientnet_b0)": RESULTS_DIR / "image_only_efficientnet_b0" / "summary.json",
    }

    for display_name, path in summary_paths.items():
        if not path.exists():
            logger.warning(f"Missing summary file: {path}")
            continue

        data = json.loads(path.read_text())
        summary = data.get("summary", data)

        rows.append({
            "baseline": display_name,
            "mean": summary["mean"],
            "ci_low": summary["ci_low"],
            "ci_high": summary["ci_high"],
            "std": summary.get("std", 0),
            "n_measurements": summary.get("n_measurements", 0),
        })

    return pd.DataFrame(rows)


def plot_forest(df: pd.DataFrame, output_path: Path) -> None:
    """Draw a forest plot of mean pAUC with 95% CI for each baseline."""
    fig, ax = plt.subplots(figsize=(7, 3))

    y_positions = range(len(df))
    means = df["mean"].values
    errors_low = means - df["ci_low"].values
    errors_high = df["ci_high"].values - means

    ax.errorbar(
        means, y_positions,
        xerr=[errors_low, errors_high],
        fmt="o", color="#4C72B0", ecolor="#4C72B0",
        capsize=5, markersize=8, linewidth=2,
    )

    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(df["baseline"].values)
    ax.set_xlabel("Partial AUC above 80% TPR")
    ax.set_title("Baseline comparison (mean with 95% CI)")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlim(left=-0.005)
    ax.grid(axis="x", alpha=0.3)

    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved forest plot to {output_path}")


def save_comparison_table(df: pd.DataFrame, output_path: Path) -> None:
    """Save a CSV table with pAUC and confidence interval strings."""
    df_display = df.copy()
    df_display["pauc_with_ci"] = df_display.apply(
        lambda row: f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]",
        axis=1,
    )
    df_display[["baseline", "pauc_with_ci", "n_measurements"]].to_csv(output_path, index=False)
    logger.info(f"Saved comparison table to {output_path}")


def compare_baselines() -> None:
    """Run the full baseline comparison pipeline (table, plot, print)."""
    df = load_baseline_summaries()

    if len(df) == 0:
        logger.error("No baseline summaries found. Run the baselines first.")
        return

    print("\nBaseline comparison:")
    print(df[["baseline", "mean", "ci_low", "ci_high"]].to_string(index=False))

    plot_forest(df, RESULTS_DIR / "comparison_forest_plot.pdf")
    save_comparison_table(df, RESULTS_DIR / "comparison_table.csv")

    print(f"\nForest plot: {RESULTS_DIR / 'comparison_forest_plot.pdf'}")
    print(f"Comparison table: {RESULTS_DIR / 'comparison_table.csv'}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    compare_baselines()