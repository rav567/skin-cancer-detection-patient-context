"""
Evaluation metrics for binary classification: partial AUC (pAUC), standard AUC, sensitivity at specificity, and F1 at threshold.

"""

import logging
from typing import Dict

import numpy as np
from sklearn.metrics import auc, f1_score, roc_curve

logger = logging.getLogger(__name__)

MIN_TPR = 0.80


def partial_auc_above_tpr(y_true: np.ndarray, y_pred: np.ndarray, min_tpr: float = MIN_TPR) -> float:
    """Compute partial AUC from a minimum TPR (default 0.80) to 1.0, using inverted ROC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2:
        return 0.0

    inverted_truth = np.abs(y_true - 1)
    inverted_pred = 1.0 - y_pred

    max_fpr = 1.0 - min_tpr
    fpr, tpr, _ = roc_curve(inverted_truth, inverted_pred, sample_weight=None)

    stop_index = np.searchsorted(fpr, max_fpr, side="right")
    fpr_partial = np.append(fpr[:stop_index], max_fpr)
    tpr_partial = np.append(tpr[:stop_index], np.interp(max_fpr, fpr, tpr))

    partial_auc_value = auc(fpr_partial, tpr_partial)
    return float(partial_auc_value)


def standard_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute standard ROC‑AUC; returns 0.5 if only one class present."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    return float(auc(fpr, tpr))


def sensitivity_at_specificity(y_true: np.ndarray, y_pred: np.ndarray, target_specificity: float = 0.90) -> float:
    """Return sensitivity (recall) at a given specificity threshold (default 0.90)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    target_fpr = 1.0 - target_specificity
    return float(np.interp(target_fpr, fpr, tpr))


def f1_at_threshold(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Compute F1 score after binarizing predictions at a fixed threshold (default 0.5)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_pred_binary = (y_pred >= threshold).astype(int)
    return float(f1_score(y_true, y_pred_binary, zero_division=0))


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return dictionary with all four key metrics: pAUC, AUC, sensitivity, and F1."""
    return {
        "pauc_above_80tpr": partial_auc_above_tpr(y_true, y_pred),
        "auc": standard_auc(y_true, y_pred),
        "sensitivity_at_90spec": sensitivity_at_specificity(y_true, y_pred, target_specificity=0.90),
        "f1_at_0.5": f1_at_threshold(y_true, y_pred, threshold=0.5),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = np.random.default_rng(42)
    n_samples = 10000
    y_true = rng.binomial(n=1, p=0.01, size=n_samples)
    perfect_pred = y_true.astype(float)
    random_pred = rng.uniform(0, 1, size=n_samples)
    decent_pred = np.clip(y_true * 0.4 + rng.uniform(0, 0.6, size=n_samples), 0, 1)

    print("\nMetric verification on synthetic data:")
    print(f"  Perfect predictions:")
    print(f"    {compute_all_metrics(y_true, perfect_pred)}")
    print(f"  Random predictions:")
    print(f"    {compute_all_metrics(y_true, random_pred)}")
    print(f"  Decent predictions:")
    print(f"    {compute_all_metrics(y_true, decent_pred)}")