"""
Run the naive (majority class) baseline and save summary JSON.

"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from training.metrics import compute_all_metrics
from training.stats import bootstrap_confidence_interval

logger = logging.getLogger(__name__)

SPLITS_PATH = Path("datasets/splits/subsampled_with_folds.csv")
RESULTS_DIR = Path("results/baselines")
N_FOLDS = 5


def run_naive_baseline() -> dict:
    df = pd.read_csv(SPLITS_PATH, low_memory=False)

    fold_paucs = []
    for fold in range(N_FOLDS):
        val_df = df[df["fold"] == fold]
        y_true = val_df["target"].values
        y_pred = np.zeros(len(val_df))

        metrics = compute_all_metrics(y_true, y_pred)
        fold_paucs.append(metrics["pauc_above_80tpr"])
        logger.info(f"Fold {fold}: pAUC = {metrics['pauc_above_80tpr']:.4f}")

    summary = bootstrap_confidence_interval(np.array(fold_paucs))
    summary["fold_paucs"] = fold_paucs
    summary["baseline_name"] = "naive"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "naive_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))

    logger.info(f"Naive baseline: mean pAUC = {summary['mean']:.4f}, saved to {output_path}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_naive_baseline()