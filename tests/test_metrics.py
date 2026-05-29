"""
Unit tests for classification metrics (pAUC, AUC, sensitivity, F1) used in model evaluation.

"""

import numpy as np
import pytest
from training.metrics import (
    compute_all_metrics,
    f1_at_threshold,
    partial_auc_above_tpr,
    sensitivity_at_specificity,
    standard_auc,
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    n_samples = 10000
    y_true = rng.binomial(n=1, p=0.01, size=n_samples)
    return y_true


def test_perfect_predictions_give_maximum_pauc(synthetic_data):
    y_true = synthetic_data
    y_pred = y_true.astype(float)
    score = partial_auc_above_tpr(y_true, y_pred)
    assert abs(score - 0.20) < 1e-6, f"Perfect predictions should score 0.20, got {score}"


def test_random_predictions_give_low_pauc(synthetic_data):
    y_true = synthetic_data
    rng = np.random.default_rng(99)
    y_pred = rng.uniform(0, 1, size=len(y_true))
    score = partial_auc_above_tpr(y_true, y_pred)
    assert score < 0.10, f"Random predictions should score below 0.10, got {score}"


def test_pauc_score_is_within_valid_range(synthetic_data):
    y_true = synthetic_data
    rng = np.random.default_rng(42)
    y_pred = np.clip(y_true * 0.4 + rng.uniform(0, 0.6, size=len(y_true)), 0, 1)
    score = partial_auc_above_tpr(y_true, y_pred)
    assert 0.0 <= score <= 0.20, f"pAUC score {score} is outside valid range [0.0, 0.20]"


def test_pauc_higher_than_random_for_decent_predictions(synthetic_data):
    y_true = synthetic_data
    rng = np.random.default_rng(42)
    y_pred_decent = np.clip(y_true * 0.4 + rng.uniform(0, 0.6, size=len(y_true)), 0, 1)
    y_pred_random = rng.uniform(0, 1, size=len(y_true))
    score_decent = partial_auc_above_tpr(y_true, y_pred_decent)
    score_random = partial_auc_above_tpr(y_true, y_pred_random)
    assert score_decent > score_random, f"Decent predictions ({score_decent}) should beat random ({score_random})"


def test_pauc_handles_single_class_gracefully():
    y_true_all_benign = np.zeros(100)
    y_pred = np.random.uniform(0, 1, size=100)
    score = partial_auc_above_tpr(y_true_all_benign, y_pred)
    assert score == 0.0, f"Single class input should return 0.0, got {score}"


def test_standard_auc_perfect_predictions(synthetic_data):
    y_true = synthetic_data
    y_pred = y_true.astype(float)
    score = standard_auc(y_true, y_pred)
    assert abs(score - 1.0) < 1e-6, f"Perfect AUC should be 1.0, got {score}"


def test_standard_auc_random_predictions(synthetic_data):
    y_true = synthetic_data
    rng = np.random.default_rng(99)
    y_pred = rng.uniform(0, 1, size=len(y_true))
    score = standard_auc(y_true, y_pred)
    assert 0.45 <= score <= 0.55, f"Random AUC should be near 0.5, got {score}"


def test_sensitivity_at_high_specificity(synthetic_data):
    y_true = synthetic_data
    y_pred = y_true.astype(float)
    sensitivity = sensitivity_at_specificity(y_true, y_pred, target_specificity=0.90)
    assert sensitivity == 1.0, f"Perfect predictor sensitivity should be 1.0, got {sensitivity}"


def test_f1_at_threshold_perfect(synthetic_data):
    y_true = synthetic_data
    y_pred = y_true.astype(float)
    f1 = f1_at_threshold(y_true, y_pred, threshold=0.5)
    assert abs(f1 - 1.0) < 1e-6, f"Perfect F1 should be 1.0, got {f1}"


def test_compute_all_metrics_returns_all_keys(synthetic_data):
    y_true = synthetic_data
    y_pred = y_true.astype(float)
    metrics = compute_all_metrics(y_true, y_pred)
    expected_keys = {"pauc_above_80tpr", "auc", "sensitivity_at_90spec", "f1_at_0.5"}
    assert set(metrics.keys()) == expected_keys, f"Missing keys: {expected_keys - set(metrics.keys())}"


def test_pauc_is_primary_metric_in_output(synthetic_data):
    y_true = synthetic_data
    y_pred = y_true.astype(float)
    metrics = compute_all_metrics(y_true, y_pred)
    assert "pauc_above_80tpr" in metrics
    assert metrics["pauc_above_80tpr"] == pytest.approx(0.20, abs=1e-6)