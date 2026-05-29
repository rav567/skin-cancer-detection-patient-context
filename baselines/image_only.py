"""
Run the image‑only EfficientNet‑B0 baseline experiment.
"""
import logging
from models.backbones import backbone_factory
from training.trainer import run_multi_seed_experiment

logger = logging.getLogger(__name__)


def run_image_only_baseline() -> dict:
    """Train the image‑only model across seeds/folds and return the summary."""
    config = {
        "backbone": "efficientnet_b0",
        "experiment_name": "image_only_efficientnet_b0",
        "pretrained": True,
        "loss": "focal",
        "loss_gamma": 2.0,
        "sampling": "balanced_1to3",
        "augmentation": "standard",
        "return_tabular": False,
        "return_patient_context": False,
        "epochs": 6,
        "patience": 2,
        "batch_size": 64,
        "image_size": 96,
        "learning_rate": 1e-4,
        "swa_start_epoch": 5,
        "results_dir": "results/baselines",
        "save_best_model": True,
        "save_predictions": True,
        "seeds": [42, 123],
        "n_folds": 5,
    }

    summary = run_multi_seed_experiment(overrides=config, model_factory=backbone_factory)
    summary["baseline_name"] = "image_only_efficientnet_b0"

    logger.info(f"Image-only baseline: mean pAUC = {summary['mean']:.4f}")
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_image_only_baseline()