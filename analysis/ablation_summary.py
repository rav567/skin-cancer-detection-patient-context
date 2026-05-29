"""
Build an ablation summary table (CSV) and a 
forest plot (PDF) from saved experiment results.

"""

import json
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.stats import bootstrap_confidence_interval, paired_bootstrap_test

logger = logging.getLogger(__name__)

mpl.rcParams.update({
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# The mean + residual + context condition was run twice under identical
# settings; both runs are pooled so the anchor is a single 6-run estimate.
MEAN_RESIDUAL_PATHS = [
    "results/ablations/agg_mean/summary.json",
    "results/ablations/fuse_hcr/summary.json",
]

ABLATIONS = {
    "Aggregation function": {
        "reference": "Mean",
        "conditions": [
            {"label": "Mean", "paths": MEAN_RESIDUAL_PATHS},
            {"label": "Attention", "paths": ["results/ablations/agg_attn/summary.json"]},
        ],
    },
    "Fusion structure": {
        "reference": "[h] only",
        "conditions": [
            {"label": "[h] only", "paths": ["results/ablations/fuse_h/summary.json"]},
            {"label": "[h, c]", "paths": ["results/ablations/fuse_hc/summary.json"]},
            {"label": "[h, c, h−c]", "paths": MEAN_RESIDUAL_PATHS},
        ],
    },
}

COLOR_MAP = {
    "Aggregation function": "#2c6e6b",
    "Fusion structure": "#1f4e79",
}


def load_run_arrays(paths):
    """Return one numpy array of per-run pAUC values for each existing path."""
    arrays = []
    for path in paths:
        if not Path(path).exists():
            logger.warning(f"Missing: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary_block = data.get("summary", data)
        values = list(summary_block.get("all_paucs", []))
        if not values:
            for run in data.get("runs", []):
                metrics = run.get("metrics", {})
                val = metrics.get("pauc_above_80tpr") or run.get("pauc") or run.get("val_pauc")
                if val is not None:
                    values.append(float(val))
        if values:
            arrays.append(np.asarray(values, dtype=float))
    return arrays


def summarise_condition(paths):
    """
    Pool runs across paths for the point estimate, and build a fold-matched vector (replicate-averaged) for paired testing.

    """
    arrays = load_run_arrays(paths)
    if not arrays:
        return None

    pooled = np.concatenate(arrays)
    stats = bootstrap_confidence_interval(pooled)

    if len(arrays) > 1 and len({len(a) for a in arrays}) == 1:
        paired_runs = np.vstack(arrays).mean(axis=0)  # fold-wise mean of replicates
    else:
        paired_runs = arrays[0]

    return {
        "mean": stats["mean"],
        "ci_lower": stats["ci_low"],
        "ci_upper": stats["ci_high"],
        "n": int(len(pooled)),
        "paired_runs": paired_runs,
    }


def collect_groups():
    """Load every condition and attach the paired diff vs the in-axis reference."""
    groups = []
    for axis_name, axis in ABLATIONS.items():
        loaded = {}
        for cond in axis["conditions"]:
            summary = summarise_condition(cond["paths"])
            if summary is not None:
                loaded[cond["label"]] = {"label": cond["label"], **summary}

        if not loaded:
            continue

        ref_label = axis["reference"]
        ref = loaded.get(ref_label)

        items = []
        for cond in axis["conditions"]:
            item = loaded.get(cond["label"])
            if item is None:
                continue
            item["is_reference"] = item["label"] == ref_label
            ref_runs = ref["paired_runs"] if ref is not None else None
            if (not item["is_reference"] and ref_runs is not None
                    and len(item["paired_runs"]) == len(ref_runs)
                    and len(ref_runs) >= 2):
                test = paired_bootstrap_test(item["paired_runs"], ref_runs)
                item["diff"] = test["mean_diff"]
                item["diff_ci_low"] = test["ci_low"]
                item["diff_ci_high"] = test["ci_high"]
                item["p_value"] = test["p_value"]
            else:
                item["diff"] = 0.0 if item["is_reference"] else np.nan
                item["diff_ci_low"] = np.nan
                item["diff_ci_high"] = np.nan
                item["p_value"] = np.nan
            items.append(item)

        groups.append({"name": axis_name, "reference": ref_label, "items": items})
    return groups


def build_table(groups):
    """Write the ablation table: absolute pAUC + 95% CI, and paired diff vs reference."""
    rows = []
    for group in groups:
        for item in group["items"]:
            rows.append({
                "Ablation": group["name"],
                "Condition": item["label"],
                "Reference": "yes" if item["is_reference"] else group["reference"],
                "Mean pAUC": round(item["mean"], 4),
                "95% CI Low": round(item["ci_lower"], 4),
                "95% CI High": round(item["ci_upper"], 4),
                "Diff vs ref": "ref" if item["is_reference"] else (
                    round(item["diff"], 4) if not np.isnan(item["diff"]) else "n/a"),
                "Diff 95% CI": "ref" if item["is_reference"] else (
                    f"[{item['diff_ci_low']:+.4f}, {item['diff_ci_high']:+.4f}]"
                    if not np.isnan(item["diff_ci_low"]) else "n/a"),
                "p-value": "ref" if item["is_reference"] else (
                    round(item["p_value"], 3) if not np.isnan(item["p_value"]) else "n/a"),
                "N runs": item["n"],
            })
    return pd.DataFrame(rows)


def plot_ablation_forest(groups, output_path):
    """
    Forest plot: absolute mean pAUC with 95% CI per condition, grouped by axis.
    """
    n_rows = sum(len(g["items"]) for g in groups) + max(0, len(groups) - 1)
    fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.55 * n_rows + 1.0)))

    baseline_mean = None
    for group in groups:
        for item in group["items"]:
            if item["label"] == "[h] only":
                baseline_mean = item["mean"]
    if baseline_mean is not None:
        ax.axvline(baseline_mean, color="black", linestyle=":", linewidth=0.8,
                   alpha=0.55, zorder=1, label="[h] only (no context)")

    y_pos = 0
    yticks, yticklabels, section_anchors = [], [], []

    for gi, group in enumerate(groups):
        if gi > 0:
            y_pos += 0.8
        section_anchors.append((y_pos - 0.4, group["name"]))
        color = COLOR_MAP.get(group["name"], "#444444")

        for item in group["items"]:
            yticks.append(y_pos)
            yticklabels.append(item["label"])
            xerr = [[item["mean"] - item["ci_lower"]], [item["ci_upper"] - item["mean"]]]
            ax.errorbar(
                item["mean"], y_pos, xerr=xerr, fmt="o",
                color=color, ecolor=color, capsize=3, markersize=6,
                linewidth=1.2, markeredgecolor="white", markeredgewidth=0.8,
                zorder=4,
            )
            y_pos += 1

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis()
    ax.set_xlabel("Partial AUC above 80% TPR (mean with 95% CI)")
    ax.grid(True, axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    for sec_y, sec_name in section_anchors:
        ax.text(
            -0.015, sec_y, sec_name,
            transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=9.5,
            fontweight="bold", color="#333333", clip_on=False,
        )

    if baseline_mean is not None:
        ax.legend(loc="lower right", frameon=False, fontsize=8.5, handlelength=1.6)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved figure to {output_path}")


def main():
    """Build the ablation table and the ablation forest figure"""
    Path("results/figures").mkdir(parents=True, exist_ok=True)

    groups = collect_groups()

    table = build_table(groups)
    table.to_csv("results/ablation_table.csv", index=False)
    print(table.to_string(index=False))
    logger.info(f"Saved ablation_table.csv with {len(table)} rows")

    plot_ablation_forest(groups, "results/figures/ablation_summary.pdf")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()