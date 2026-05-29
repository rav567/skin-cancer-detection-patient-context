"""
Load saved model evaluation results, produce a master summary table of mean partial AUC with 95% CIs, compute pairwise significance 
with paired bootstrap, and generate a dot plot.

"""

import json
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


MODELS = [
    {"label": "Naive (majority)", "path": "results/baselines/naive_summary.json", "group": "baseline"},
    {"label": "Tabular (LightGBM)", "path": "results/baselines/tabular_summary.json", "group": "baseline"},
    {"label": "Image only (EfficientNet-B0)", "path": "results/baselines/image_only_efficientnet_b0/summary.json", "group": "baseline"},
    {"label": "A: Image only", "path": "results/configurations/config_a_image_only/summary.json", "group": "configuration"},
    {"label": "B: Image + tabular", "path": "results/configurations/config_b_image_tabular/summary.json", "group": "configuration"},
    {"label": "C: Image + patient context", "path": "results/configurations/config_c_image_context/summary.json", "group": "configuration"},
    {"label": "D: Combined", "path": "results/configurations/config_d_combined/summary.json", "group": "configuration"},
]


def load_summary(path):
    """Read a summary JSON file and extract key metrics."""

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    summary_block = data.get("summary", data)

    runs = data.get("runs", [])
    pauc_values = list(summary_block.get("all_paucs", []))
    if not pauc_values:
        for run in runs:
            metrics = run.get("metrics", {})
            val = (
                metrics.get("pauc_above_80tpr")
                or run.get("pauc")
                or run.get("val_pauc")
                or run.get("pAUC")
            )
            if val is not None:
                pauc_values.append(float(val))

    mean = summary_block.get("mean", data.get("mean", np.nan))
    std = summary_block.get("std", data.get("std", 0.0))
    ci_low = summary_block.get("ci_low", summary_block.get("ci_lower", data.get("ci_lower", mean)))
    ci_high = summary_block.get("ci_high", summary_block.get("ci_upper", data.get("ci_upper", mean)))
    n = summary_block.get("n_measurements", summary_block.get("n", data.get("n", len(pauc_values))))

    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_low),
        "ci_upper": float(ci_high),
        "n": int(n),
        "runs": [float(v) for v in pauc_values],
    }


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Compute bootstrap mean and percentile confidence interval."""
    if rng is None:
        rng = np.random.default_rng(42)
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()) if len(values) else np.nan, np.nan, np.nan
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True)
    means = samples.mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap_pvalue(values_a, values_b, n_boot=10000, rng=None):
    """Paired bootstrap test: observed difference and two‑sided p‑value."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = min(len(values_a), len(values_b))
    if n < 2:
        return np.nan, np.nan
    diffs = np.asarray(values_b[:n], dtype=float) - np.asarray(values_a[:n], dtype=float)
    samples = rng.choice(diffs, size=(n_boot, n), replace=True)
    boot_means = samples.mean(axis=1)
    observed = float(diffs.mean())
    p_two_sided = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    return observed, float(p_two_sided)


def build_master_table(models):
    """Build a summary DataFrame from a list of model specifications."""
    rows = []
    for spec in models:
        if not Path(spec["path"]).exists():
            logger.warning(f"Missing: {spec['path']}")
            continue
        summary = load_summary(spec["path"])
        rows.append({
            "Model": spec["label"],
            "Group": spec["group"],
            "Mean pAUC": round(summary["mean"], 4),
            "95% CI Low": round(summary["ci_lower"], 4),
            "95% CI High": round(summary["ci_upper"], 4),
            "Std": round(summary["std"], 4),
            "N runs": summary["n"],
        })
    return pd.DataFrame(rows)


def build_pairwise_significance(models):
    """Compute all pairwise model comparisons with bootstrap p‑values."""
    data = {}
    for spec in models:
        if not Path(spec["path"]).exists():
            continue
        data[spec["label"]] = load_summary(spec["path"])

    keys = list(data.keys())
    rows = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            runs_a = data[a]["runs"]
            runs_b = data[b]["runs"]
            if len(runs_a) >= 2 and len(runs_b) >= 2 and len(runs_a) == len(runs_b):
                diff, p = paired_bootstrap_pvalue(runs_a, runs_b)
                test_type = "paired"
            else:
                diff = data[b]["mean"] - data[a]["mean"]
                p = np.nan
                test_type = "unpaired"
            rows.append({
                "Model A": a,
                "Model B": b,
                "Diff (B - A)": round(diff, 4),
                "p-value": round(p, 4) if not np.isnan(p) else "n/a",
                "Significant (p<0.05)": "yes" if (not np.isnan(p) and p < 0.05) else "no",
                "Test": test_type,
            })
    return pd.DataFrame(rows)


def plot_master_dot(models, output_path):
    """Generate a dot plot with mean, 95% CI, and jittered individual runs."""
    rows = []
    for spec in models:
        if not Path(spec["path"]).exists():
            continue
        summary = load_summary(spec["path"])
        rows.append({
            "label": spec["label"],
            "group": spec["group"],
            "mean": summary["mean"],
            "ci_lower": summary["ci_lower"],
            "ci_upper": summary["ci_upper"],
            "runs": summary["runs"],
        })

    rows = list(reversed(rows))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    color_baseline = "#8b1a1a"
    color_config = "#1f4e79"
    color_runs = "#bbbbbb"

    rng = np.random.default_rng(123)
    for i, row in enumerate(rows):
        color = color_config if row["group"] == "configuration" else color_baseline
        runs = row["runs"]
        if len(runs) > 1:
            jitter = rng.uniform(-0.12, 0.12, size=len(runs))
            ax.scatter(runs, np.full(len(runs), i) + jitter, s=14, color=color_runs, alpha=0.7, edgecolors="none", zorder=2)
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i], color=color, linewidth=1.3, zorder=3, solid_capstyle="round")
        ax.plot([row["ci_lower"], row["ci_lower"]], [i - 0.12, i + 0.12], color=color, linewidth=1.0, zorder=3)
        ax.plot([row["ci_upper"], row["ci_upper"]], [i - 0.12, i + 0.12], color=color, linewidth=1.0, zorder=3)
        ax.scatter([row["mean"]], [i], s=55, color=color, zorder=5, edgecolors="white", linewidth=0.8)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_xlabel("Partial AUC above 80% TPR")
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.axvline(0.020, color="black", linestyle=":", linewidth=0.6, alpha=0.4, zorder=1)
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_config, markersize=8, label="Configuration"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_baseline, markersize=8, label="Baseline"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_runs, markersize=6, label="Individual run"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
        handlelength=1.2,
        columnspacing=2.0,
    )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved figure to {output_path}")


def main():
    """Run the master results pipeline."""
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    table = build_master_table(MODELS)
    table.to_csv("results/master_results_table.csv", index=False)
    logger.info(f"Saved master_results_table.csv with {len(table)} rows")
    print(table.to_string(index=False))

    significance = build_pairwise_significance(MODELS)
    significance.to_csv("results/pairwise_significance.csv", index=False)
    logger.info(f"Saved pairwise_significance.csv with {len(significance)} pairs")

    plot_master_dot(MODELS, "results/figures/master_dot_plot.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()