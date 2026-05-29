"""
Main entry point for the full experimental pipeline: data preparation, baseline training,
configuration experiments, ablation studies, and result visualisation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

from datasets.schema import validate_metadata
from datasets.splits import prepare_splits
from datasets.exploration import generate_all_eda_figures
from baselines.naive import run_naive_baseline
from baselines.tabular import run_tabular_baseline
from baselines.compare import compare_baselines
from analysis.master_results import main as plot_master
from analysis.ablation_summary import main as plot_ablation
from analysis.variance_analysis import main as plot_variance
from analysis.umap_analysis import main as plot_umap
from analysis.training_curves import plot_training_curves
from utils.seed import set_seed

SEED = 42
META = Path("datasets/train-metadata.csv")
IMGS = Path("datasets/train-image/image")
SPLITS = Path("datasets/splits/subsampled_with_folds.csv")
LOG_PATH = Path("results/training.log")
SITES = [
    "head/neck", "upper extremity", "lower extremity",
    "anterior torso", "posterior torso",
]

DEMO_CFG = dict(
    epochs=1, patience=1, batch_size=16, num_workers=0, image_size=96,
    seeds=[SEED], n_folds=1, use_amp=False,
    save_predictions=True, save_best_model=False,
    swa_start_epoch=99, swa_lr=5e-5,
    learning_rate=1e-4, weight_decay=1e-4,
    loss="focal", loss_gamma=2.0, ohem_ratio=0.5,
    sampling="balanced_1to3", augmentation="standard",
    backbone="efficientnet_b0", pretrained=False,
    splits_path=str(SPLITS),
)


def main():
    """
    Run the complete ISIC 2024 skin cancer detection pipeline.

    Attempts to download the competition dataset via the Kaggle API; falls back
    to synthetic data if credentials are unavailable. The pipeline then runs:
      1. Data validation and patient-grouped cross-validation splits
      2. Exploratory data analysis figures (class distribution, lesions per patient, etc.)
      3. Naive and tabular (LightGBM) baselines
      4. Image-only EfficientNet-B0 baseline
      5. Multimodal configurations A-D (image, tabular, patient context, combined)
      6. Ablation studies over context aggregation and fusion strategies
      7. Result visualisations: training curves, master dot plot, ablation summary,
         variance analysis, and UMAP embedding plot
    All outputs (metrics, plots, model checkpoints) are saved to disk under results/.
    Execution is fully deterministic via fixed random seeds and requires no user input.
    """
    os.makedirs("results", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        ],
    )
    logger = logging.getLogger(__name__)

    set_seed(SEED)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)

    try:
        from datasets.download import download_dataset
        download_dataset()
    except Exception as exc:
        logger.warning("Kaggle download skipped (%s). Continuing with available or synthetic data.", exc)

    if META.exists() and IMGS.exists() and any(IMGS.iterdir()):
        df = prepare_splits(validate_metadata())
        generate_all_eda_figures(df)
        run_naive_baseline()
        run_tabular_baseline()
        train_full()
    else:
        synthesise()
        df = pd.read_csv(SPLITS, low_memory=False)
        generate_all_eda_figures(df)
        run_naive_baseline()
        run_tabular_baseline()
        train_demo()

    compare_baselines()
    plot_training_curves(LOG_PATH, "results/figures/training_curves.pdf")
    plot_master()
    plot_ablation()
    plot_variance()
    plot_umap()


def synthesise():
    """Generate synthetic metadata and dummy images when real competition data is not available."""
    rng = np.random.default_rng(SEED)
    IMGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(8):
        pid = "IP_SYN{:04d}".format(idx)
        for k in range(3):
            iid = "ISIC_SYN_{:07d}".format(int(rng.integers(0, 10000000)))
            rows.append(dict(
                isic_id=iid,
                patient_id=pid,
                target=1 if (idx < 18 and k == 0) else 0,
                age_approx=float(rng.integers(20, 80)),
                sex=str(rng.choice(["male", "female"])),
                anatom_site_general=str(rng.choice(SITES)),
                clin_size_long_diam_mm=float(round(rng.uniform(1, 12), 2)),
            ))
            Image.fromarray(
                rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
            ).save(IMGS / "{}.jpg".format(iid), quality=80)
    df = pd.DataFrame(rows)
    df.to_csv(META, index=False)
    fold_map = {p: i % 5 for i, p in enumerate(sorted(df["patient_id"].unique()))}
    df["fold"] = df["patient_id"].map(fold_map)
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS, index=False)


def train_full():
    """Run full training: image-only baseline, all configuration variants, and ablations."""
    from baselines.image_only import run_image_only_baseline
    from experiments.run_configurations import run_all_configurations
    from experiments.run_ablations import run_all_ablations
    run_image_only_baseline()
    run_all_configurations()
    run_all_ablations()


def train_demo():
    """Run a minimal demo training for fast testing using reduced epochs and folds."""
    from training.trainer import run_multi_seed_experiment
    from models.full_pipeline import pipeline_factory
    from models.backbones import BackboneClassifier

    def backbone(cfg, tabular_dim=0):
        return BackboneClassifier(
            backbone_name=cfg["backbone"],
            pretrained=cfg["pretrained"],
            tabular_dim=0,
        )

    base = dict(DEMO_CFG)
    base.update(dict(
        experiment_name="image_only_efficientnet_b0",
        results_dir="results/baselines",
        return_tabular=False,
        return_patient_context=False,
    ))
    run_multi_seed_experiment(overrides=base, model_factory=backbone)

    demo_runs = [
        ("config_a_image_only",    "results/configurations", False, False, "mean", "residual"),
        ("config_b_image_tabular", "results/configurations", True,  False, "mean", "residual"),
        ("config_c_image_context", "results/configurations", False, True,  "mean", "residual"),
        ("config_d_combined",      "results/configurations", True,  True,  "mean", "residual"),
        ("agg_mean",               "results/ablations",      False, True,  "mean", "residual"),
        ("fuse_h",                 "results/ablations",      False, False, "mean", "none"),
    ]
    for name, out, tab, ctx, agg, fuse in demo_runs:
        cfg = dict(DEMO_CFG)
        cfg.update(dict(
            experiment_name=name,
            results_dir=out,
            return_tabular=tab,
            return_patient_context=ctx,
            context_aggregation=agg,
            context_fusion=fuse,
        ))
        run_multi_seed_experiment(overrides=cfg, model_factory=pipeline_factory)


if __name__ == "__main__":
    main()
