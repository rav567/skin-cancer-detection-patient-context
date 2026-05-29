"""
Run the ablation experiments: aggregation function and fusion structure variants.

"""

import logging

from models.full_pipeline import pipeline_factory
from training.trainer import DEFAULT_CONFIG, run_multi_seed_experiment

logger = logging.getLogger(__name__)


def base_config(experiment_name):
    """Return a default config dict for an ablation run with the given experiment name."""
    return {
        **DEFAULT_CONFIG,
        "experiment_name": experiment_name,
        "splits_path": "datasets/splits/subsampled_with_folds.csv",
        "image_size": 96,
        "batch_size": 96,
        "num_workers": 0,
        "epochs": 6,
        "patience": 2,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "augmentation": "standard",
        "loss": "focal",
        "loss_gamma": 2.0,
        "ohem_ratio": 0.5,
        "sampling": "balanced_1to3",
        "swa_start_epoch": 5,
        "swa_lr": 5e-5,
        "use_amp": True,
        "save_predictions": True,
        "save_best_model": False,
        "results_dir": "results/ablations",
        "seeds": [42, 123],
        "n_folds": 3,
        "backbone": "efficientnet_b0",
        "pretrained": True,
    }


def run_condition(name, return_patient_context, context_aggregation, context_fusion):
    """Run a single ablation condition with the specified context settings."""
    config = base_config(name)
    config["return_tabular"] = False
    config["return_patient_context"] = return_patient_context
    config["context_aggregation"] = context_aggregation
    config["context_fusion"] = context_fusion
    return run_multi_seed_experiment(overrides=config, model_factory=pipeline_factory)


def run_all_ablations():
    """Run all five ablation conditions and log their mean pAUC results."""
    logger.info("PHASE E: Running ablation studies")

    results = {
        "agg_mean": run_condition("agg_mean", True, "mean", "residual"),
        "agg_attn": run_condition("agg_attn", True, "attention", "residual"),
        "fuse_h": run_condition("fuse_h", False, "mean", "none"),
        "fuse_hc": run_condition("fuse_hc", True, "mean", "concat"),
        "fuse_hcr": run_condition("fuse_hcr", True, "mean", "residual"),
    }

    logger.info("Ablations complete:")
    for key, summary in results.items():
        logger.info(f"  {key}: mean pAUC = {summary['mean']:.4f}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_all_ablations()