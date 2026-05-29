"""
Run configurations A–D: image‑only, image+tabular, image+context, combined.

"""

import logging

from models.full_pipeline import pipeline_factory
from training.trainer import run_multi_seed_experiment

logger = logging.getLogger(__name__)


SHARED_CONFIG = {
    "backbone": "efficientnet_b0",
    "pretrained": True,
    "loss": "focal",
    "loss_gamma": 2.0,
    "sampling": "balanced_1to3",
    "augmentation": "standard",
    "epochs": 6,
    "patience": 2,
    "batch_size": 96,
    "num_workers": 0,
    "image_size": 96,
    "learning_rate": 1e-4,
    "swa_start_epoch": 5,
    "results_dir": "results/configurations",
    "save_best_model": True,
    "save_predictions": True,
    "seeds": [42, 123],
    "n_folds": 3,
    "tabular_dim": 9,
}


def run_configuration_a():
    """Train the image‑only pipeline (Config A)."""
    config = {
        **SHARED_CONFIG,
        "experiment_name": "config_a_image_only",
        "return_tabular": False,
        "return_patient_context": False,
    }
    summary = run_multi_seed_experiment(overrides=config, model_factory=pipeline_factory)
    summary["configuration"] = "A: Image-only"
    logger.info(f"Config A: mean pAUC = {summary['mean']:.4f}")
    return summary


def run_configuration_b():
    """Train the image + tabular features pipeline (Config B)."""
    config = {
        **SHARED_CONFIG,
        "experiment_name": "config_b_image_tabular",
        "return_tabular": True,
        "return_patient_context": False,
    }
    summary = run_multi_seed_experiment(overrides=config, model_factory=pipeline_factory)
    summary["configuration"] = "B: Image + tabular"
    logger.info(f"Config B: mean pAUC = {summary['mean']:.4f}")
    return summary


def run_configuration_c():
    """Train the image + patient context pipeline (Config C)."""
    config = {
        **SHARED_CONFIG,
        "experiment_name": "config_c_image_context",
        "return_tabular": False,
        "return_patient_context": True,
        "context_aggregation": "mean",
    }
    summary = run_multi_seed_experiment(overrides=config, model_factory=pipeline_factory)
    summary["configuration"] = "C: Image + context"
    logger.info(f"Config C: mean pAUC = {summary['mean']:.4f}")
    return summary


def run_configuration_d():
    """Train the full combined pipeline (Config D)."""
    config = {
        **SHARED_CONFIG,
        "experiment_name": "config_d_combined",
        "return_tabular": True,
        "return_patient_context": True,
        "context_aggregation": "mean",
    }
    summary = run_multi_seed_experiment(overrides=config, model_factory=pipeline_factory)
    summary["configuration"] = "D: Combined"
    logger.info(f"Config D: mean pAUC = {summary['mean']:.4f}")
    return summary


def run_all_configurations():
    """Run all four configurations A-D and log results."""
    logger.info("PHASE D: Running all configurations A, B, C, D")

    summaries = {
        "A": run_configuration_a(),
        "B": run_configuration_b(),
        "C": run_configuration_c(),
        "D": run_configuration_d(),
    }

    logger.info("Configurations complete:")
    for key, summary in summaries.items():
        logger.info(f"  Config {key}: mean pAUC = {summary['mean']:.4f}")

    return summaries


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_all_configurations()