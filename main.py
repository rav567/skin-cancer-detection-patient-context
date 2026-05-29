"""
Main entry point for the full experimental pipeline: data preparation, baseline training, configuration experiments, ablation studies, and result visualisation.

"""

from __future__ import annotations
 
from datasets.schema import validate_metadata
from datasets.splits import prepare_splits
from baselines.naive import run_naive_baseline
from baselines.tabular import run_tabular_baseline
from baselines.compare import compare_baselines
from analysis.master_results import main as plot_master
from analysis.ablation_summary import main as plot_ablation
from analysis.variance_analysis import main as plot_variance
from analysis.umap_analysis import main as plot_umap
from utils.seed import set_seed
 
import os
import subprocess
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
from PIL import Image
 
SEED = 42
META = Path("datasets/train-metadata.csv")
IMGS = Path("datasets/train-image/image")
SPLITS = Path("datasets/splits/subsampled_with_folds.csv")
SITES = ["head/neck", "upper extremity", "lower extremity",
         "anterior torso", "posterior torso"]
 
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
    This function must execute the complete experimental workflow developed
    for the selected computer vision competition and research hypothesis.

    The automated grading system will call this function. Therefore:
    - The function signature must not be changed.
    - It must not require any user input.
    - It must run deterministically (fixed random seeds).
    - All outputs (metrics, logs, plots) must be saved to disk.

    The workflow should include:

        1. Dataset loading and preparation
           - Download or load the competition dataset
           - Apply preprocessing and data augmentation (if applicable)
           - Create training / validation / test splits

        2. Model construction
           - Build the baseline model
           - Build the proposed model(s) used to test the hypothesis

        3. Training
           - Train model(s) using defined hyperparameters
           - Log training and validation performance

        4. Evaluation
           - Evaluate on validation/test data
           - Compute relevant metrics (e.g., accuracy, F1, etc.)
           - Compare models if testing a hypothesis

        5. Analysis and visualisation
           - Generate and save plots used in the report
           - Save final metrics to disk (e.g., JSON/CSV)

    The purpose of this function is to reproduce all experimental evidence
    presented in the report in a fully automated and reproducible manner.

    After implementing this method, replace this docstring with a concise
    description of what your main function does.
    """

    set_seed(SEED)
    os.makedirs("results", exist_ok=True)
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], check=True)
 
    if META.exists() and IMGS.exists() and any(IMGS.iterdir()):
        prepare_splits(validate_metadata())
        run_naive_baseline()
        run_tabular_baseline()
        train_full()
    else:
        synthesise()
        run_naive_baseline()
        run_tabular_baseline()
        train_demo()
 
    compare_baselines()
    plot_master()
    plot_ablation()
    plot_variance()
    plot_umap()
 
 
def synthesise():
    """Generate synthetic metadata and dummy images when real competition data is not available due to storage and computation issues."""
    rng = np.random.default_rng(SEED)
    IMGS.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(8):
        pid = f"IP_SYN{idx:04d}"
        for k in range(3):
            iid = f"ISIC_SYN_{rng.integers(0, 10_000_000):07d}"
            rows.append(dict(isic_id=iid, patient_id=pid,
                             target=1 if (idx < 18 and k == 0) else 0,
                             age_approx=float(rng.integers(20, 80)),
                             sex=str(rng.choice(["male", "female"])),
                             anatom_site_general=str(rng.choice(SITES)),
                             clin_size_long_diam_mm=float(round(rng.uniform(1, 12), 2))))
            Image.fromarray(rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)) \
                 .save(IMGS / f"{iid}.jpg", quality=80)
    df = pd.DataFrame(rows)
    df.to_csv(META, index=False)
    df["fold"] = df["patient_id"].map(
        {p: i % 5 for i, p in enumerate(sorted(df["patient_id"].unique()))})
    SPLITS.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS, index=False)
 
 
def train_full():
    """Run full training: image‑only baseline, all configuration variants, and all ablation experiments."""
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
        return BackboneClassifier(backbone_name=cfg["backbone"],
                                  pretrained=cfg["pretrained"], tabular_dim=0)
 
    run_multi_seed_experiment(
        overrides={**DEMO_CFG, "experiment_name": "image_only_efficientnet_b0",
                   "results_dir": "results/baselines",
                   "return_tabular": False, "return_patient_context": False},
        model_factory=backbone)
 
    for name, out, tab, ctx, agg, fuse in [
        ("config_a_image_only",    "results/configurations", False, False, "mean", "residual"),
        ("config_b_image_tabular", "results/configurations", True,  False, "mean", "residual"),
        ("config_c_image_context", "results/configurations", False, True,  "mean", "residual"),
        ("config_d_combined",      "results/configurations", True,  True,  "mean", "residual"),
        ("agg_mean",               "results/ablations",      False, True,  "mean", "residual"),
        ("fuse_h",                 "results/ablations",      False, False, "mean", "none"),
    ]:
        run_multi_seed_experiment(
            overrides={**DEMO_CFG, "experiment_name": name, "results_dir": out,
                       "return_tabular": tab, "return_patient_context": ctx,
                       "context_aggregation": agg, "context_fusion": fuse},
            model_factory=pipeline_factory)
 
 
if __name__ == "__main__":
    main()