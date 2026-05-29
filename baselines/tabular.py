"""
Run LightGBM tabular baseline over multiple seeds and folds.

"""
import json
import logging
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd

from datasets.isic_dataset import build_tabular_encoder, encode_tabular_features
from training.metrics import compute_all_metrics
from training.stats import bootstrap_confidence_interval

logger = logging.getLogger(__name__)

SPLITS_PATH = Path("datasets/splits/subsampled_with_folds.csv")
RESULTS_DIR = Path("results/baselines")
N_FOLDS = 5
SEEDS = [42, 123, 7]


def encode_dataframe(df: pd.DataFrame, encoder_columns: List[str]) -> np.ndarray:
    """Encode tabular features for an entire DataFrame."""
    return np.stack([encode_tabular_features(row, encoder_columns) for _, row in df.iterrows()])


def train_lightgbm_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, encoder_columns: List[str], seed: int) -> float:
    """Train LightGBM on one fold and return the pAUC."""
    X_train = encode_dataframe(train_df, encoder_columns)
    y_train = train_df["target"].values
    X_val = encode_dataframe(val_df, encoder_columns)
    y_val = val_df["target"].values

    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1,
        "seed": seed,
    }

    train_dataset = lgb.Dataset(X_train, label=y_train)
    val_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=200,
        valid_sets=[val_dataset],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)],
    )

    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    metrics = compute_all_metrics(y_val, y_pred)
    return metrics["pauc_above_80tpr"]


def run_tabular_baseline() -> dict:
    """Run LightGBM over seeds/folds, bootstrap, and save summary JSON."""
    df = pd.read_csv(SPLITS_PATH, low_memory=False)
    encoder_columns, _ = build_tabular_encoder(df)

    all_paucs = []
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            train_df = df[df["fold"] != fold]
            val_df = df[df["fold"] == fold]

            pauc = train_lightgbm_fold(train_df, val_df, encoder_columns, seed=seed)
            all_paucs.append(pauc)
            logger.info(f"Seed {seed}, Fold {fold}: pAUC = {pauc:.4f}")

    summary = bootstrap_confidence_interval(np.array(all_paucs))
    summary["all_paucs"] = all_paucs
    summary["baseline_name"] = "tabular_lightgbm"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "tabular_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))

    logger.info(f"Tabular baseline: mean pAUC = {summary['mean']:.4f}, saved to {output_path}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_tabular_baseline()