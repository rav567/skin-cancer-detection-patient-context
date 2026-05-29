"""
Per‑run pAUC dot plot showing that patient context (C, D) increases variance.

"""

import json
import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

CONFIGS = [
    ("A: Image only", "results/configurations/config_a_image_only/summary.json", "#1f4e79"),
    ("B: Image + tabular", "results/configurations/config_b_image_tabular/summary.json", "#1f4e79"),
    ("C: Image + context", "results/configurations/config_c_image_context/summary.json", "#c1121f"),
    ("D: Combined", "results/configurations/config_d_combined/summary.json", "#c1121f"),
]
OUTPUT_PATH = "results/figures/variance_analysis.pdf"

mpl.rcParams.update({
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_runs(path):
    """Return the list of per-run pAUC values stored in a summary.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    block = data.get("summary", data)
    runs = list(block.get("all_paucs", []))
    if not runs:
        for run in data.get("runs", []):
            val = run.get("metrics", {}).get("pauc_above_80tpr")
            if val is not None:
                runs.append(float(val))
    return [float(v) for v in runs]


def main():
    """Build the per-run variance strip plot for the four configurations."""
    os.makedirs("results/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    rng = np.random.default_rng(42)

    for i, (label, path, color) in enumerate(CONFIGS):
        if not os.path.exists(path):
            logger.warning(f"Missing: {path}")
            continue
        runs = np.asarray(load_runs(path), dtype=float)
        if len(runs) == 0:
            continue

        jitter = rng.uniform(-0.09, 0.09, size=len(runs))
        ax.scatter(np.full(len(runs), i) + jitter, runs, s=30, color=color,
                   alpha=0.7, edgecolors="white", linewidth=0.6, zorder=3)

        mean = runs.mean()
        sd = runs.std(ddof=1) if len(runs) > 1 else 0.0
        ax.hlines(mean, i - 0.22, i + 0.22, color=color, linewidth=2.0, zorder=4)
        ax.errorbar(i, mean, yerr=sd, color=color, capsize=4, linewidth=1.3, zorder=2)
        ax.text(i + 0.28, mean, f"SD = {sd:.4f}", va="center", fontsize=8, color=color)

    ax.set_xticks(range(len(CONFIGS)))
    ax.set_xticklabels([c[0] for c in CONFIGS], rotation=12, ha="right")
    ax.set_ylabel("Partial AUC above 80% TPR")
    ax.set_title("Per-run pAUC: adding patient context (C, D) widens the spread")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)
    plt.close()
    logger.info(f"Saved figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()