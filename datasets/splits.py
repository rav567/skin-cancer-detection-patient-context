import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

DATA_DIR = Path("datasets")
SPLITS_DIR = DATA_DIR / "splits"
RANDOM_SEED = 42
TARGET_BENIGN_PATIENTS = 100
N_FOLDS = 5


def subsample_patients(df: pd.DataFrame, target_benign_patients: int = TARGET_BENIGN_PATIENTS, max_lesions_per_patient: int = 200, seed: int = RANDOM_SEED) -> pd.DataFrame:
    patients_with_malignancy = df.loc[df["target"] == 1, "patient_id"].unique()
    logger.info(f"Keeping all {len(patients_with_malignancy)} patients with malignancies")

    benign_only_patients = df.loc[~df["patient_id"].isin(patients_with_malignancy), "patient_id"].unique()
    logger.info(f"Pool of benign-only patients: {len(benign_only_patients)}")

    rng = np.random.default_rng(seed)
    n_to_sample = min(target_benign_patients, len(benign_only_patients))
    sampled_benign_patients = rng.choice(benign_only_patients, size=n_to_sample, replace=False)
    logger.info(f"Sampled {n_to_sample} benign-only patients")

    selected_patients = np.concatenate([patients_with_malignancy, sampled_benign_patients])
    subsampled = df[df["patient_id"].isin(selected_patients)].reset_index(drop=True)

    logger.info(f"Before per-patient cap: {len(subsampled):,} lesions")

    capped_parts = []
    for patient_id, group in subsampled.groupby("patient_id"):
        malignant_in_group = group[group["target"] == 1]
        benign_in_group = group[group["target"] == 0]
        n_benign_to_keep = max(0, max_lesions_per_patient - len(malignant_in_group))
        if len(benign_in_group) > n_benign_to_keep:
            benign_in_group = benign_in_group.sample(n=n_benign_to_keep, random_state=seed)
        capped_parts.append(pd.concat([malignant_in_group, benign_in_group]))
    subsampled = pd.concat(capped_parts).reset_index(drop=True)

    logger.info(
        f"Subsample: {len(subsampled):,} lesions, "
        f"{subsampled['patient_id'].nunique()} patients, "
        f"{subsampled['target'].sum()} malignant ({subsampled['target'].mean()*100:.3f}%)"
    )
    return subsampled


def create_patient_grouped_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    df = df.copy()
    df["fold"] = -1

    patient_targets = df.groupby("patient_id")["target"].max().reset_index()
    patient_targets.columns = ["patient_id", "patient_has_malignancy"]

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = splitter.split(
        X=patient_targets["patient_id"],
        y=patient_targets["patient_has_malignancy"],
        groups=patient_targets["patient_id"],
    )

    for fold_idx, (_, val_patient_indices) in enumerate(splits):
        val_patients = patient_targets.iloc[val_patient_indices]["patient_id"].values
        df.loc[df["patient_id"].isin(val_patients), "fold"] = fold_idx

    assert (df["fold"] != -1).all(), "Some lesions were not assigned to a fold"
    return df


def assert_no_patient_leakage(df: pd.DataFrame) -> None:
    patient_fold_counts = df.groupby("patient_id")["fold"].nunique()
    leaked_patients = patient_fold_counts[patient_fold_counts > 1]
    if len(leaked_patients) > 0:
        raise AssertionError(
            f"Patient leakage detected: {len(leaked_patients)} patients appear in multiple folds. "
            f"Examples: {leaked_patients.head().to_dict()}"
        )
    logger.info("Leakage check passed: every patient is in exactly one fold")


def log_fold_statistics(df: pd.DataFrame) -> None:
    logger.info("Fold composition:")
    fold_stats = df.groupby("fold").agg(
        n_lesions=("isic_id", "count"),
        n_patients=("patient_id", "nunique"),
        n_malignant=("target", "sum"),
    )
    fold_stats["malignancy_rate"] = fold_stats["n_malignant"] / fold_stats["n_lesions"]
    for fold_idx, row in fold_stats.iterrows():
        logger.info(
            f"  Fold {fold_idx}: {row['n_lesions']:,} lesions, "
            f"{row['n_patients']} patients, "
            f"{int(row['n_malignant'])} malignant ({row['malignancy_rate']*100:.3f}%)"
        )


def save_splits(df: pd.DataFrame, splits_dir: Path = SPLITS_DIR) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    output_path = splits_dir / "subsampled_with_folds.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df):,} rows with fold assignments to {output_path}")


def prepare_splits(metadata_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Subsampling patients")
    subsampled = subsample_patients(metadata_df)

    logger.info("Creating patient-grouped 5-fold splits")
    with_folds = create_patient_grouped_folds(subsampled)

    logger.info("Verifying no patient leakage across folds")
    assert_no_patient_leakage(with_folds)

    log_fold_statistics(with_folds)

    save_splits(with_folds)
    return with_folds


if __name__ == "__main__":
    from schema import validate_metadata

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    metadata_df = validate_metadata()
    prepare_splits(metadata_df)