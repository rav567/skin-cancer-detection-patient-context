"""
Multi‑seed, cross‑validated training loop with SWA, AMP, patient context caching, and experiment tracking.

"""

import copy
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.data import DataLoader

from datasets.isic_dataset import ISICDataset, build_tabular_encoder, collate_with_padding
from datasets.samplers import get_sampler
from datasets.transforms import get_transforms
from training.losses import compute_pos_weight, get_loss_function
from training.metrics import compute_all_metrics
from training.stats import summarise_configuration
from utils.seed import seed_dataloader_worker, set_seed

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "experiment_name": "default",
    "splits_path": "datasets/splits/subsampled_with_folds.csv",
    "image_size": 128,
    "batch_size": 32,
    "num_workers": 4,
    "epochs": 15,
    "patience": 5,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "augmentation": "standard",
    "loss": "focal",
    "loss_gamma": 2.0,
    "ohem_ratio": 0.5,
    "sampling": "balanced_1to3",
    "return_tabular": False,
    "return_patient_context": False,
    "swa_start_epoch": 10,
    "swa_lr": 5e-5,
    "use_amp": True,
    "save_predictions": True,
    "save_best_model": True,
    "results_dir": "results",
    "seeds": [42, 123, 7],
    "n_folds": 5,
}


def merge_config(overrides: Optional[Dict] = None) -> Dict:
    """Merge user overrides into a deep copy of the default configuration."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    if overrides:
        config.update(overrides)
    return config


def update_embedding_cache(model, dataset: ISICDataset, device: str, batch_size: int = 64) -> Dict[str, np.ndarray]:
    """Precompute image embeddings for all lesions in a dataset to use as patient context."""
    model.eval()
    cache = {}
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_with_padding)

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            embeddings = model.encode(images).cpu().numpy()
            for i, isic_id in enumerate(batch["isic_id"]):
                cache[isic_id] = embeddings[i]

    return cache


def evaluate_model(model, loader, device: str) -> Dict[str, float]:
    """Run inference on a validation loader and return metrics, true labels, and predictions."""
    model.eval()
    all_targets, all_preds = [], []

    with torch.no_grad():
        for batch in loader:
            batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits = model(batch_device)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_targets.append(batch["target"].numpy())
            all_preds.append(preds)

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    return compute_all_metrics(y_true, y_pred), y_true, y_pred


def train_single_run(config: Dict, model_factory: Callable, seed: int, fold: int, df: pd.DataFrame) -> Dict:
    """Train a single (seed, fold) combination. Returns metrics, history, and predictions."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = df[df["fold"] != fold].reset_index(drop=True)
    val_df = df[df["fold"] == fold].reset_index(drop=True)

    encoder_columns, tabular_dim = build_tabular_encoder(df) if config["return_tabular"] else ([], 0)

    train_transform = get_transforms(config["augmentation"], image_size=config["image_size"], training=True)
    eval_transform = get_transforms(config["augmentation"], image_size=config["image_size"], training=False)

    train_dataset = ISICDataset(
        train_df, transform=train_transform,
        return_tabular=config["return_tabular"],
        return_patient_context=config["return_patient_context"],
        encoder_columns=encoder_columns,
    )
    val_dataset = ISICDataset(
        val_df, transform=eval_transform,
        return_tabular=config["return_tabular"],
        return_patient_context=config["return_patient_context"],
        encoder_columns=encoder_columns,
    )

    sampler = get_sampler(
        labels=train_df["target"].values,
        sampling_strategy=config["sampling"],
        batch_size=config["batch_size"],
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler, num_workers=config["num_workers"],
        collate_fn=collate_with_padding, worker_init_fn=seed_dataloader_worker,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], collate_fn=collate_with_padding,
    )

    model = model_factory(config, tabular_dim=tabular_dim).to(device)
    optimiser = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scaler = torch.amp.GradScaler("cuda") if config["use_amp"] and device == "cuda" else None

    n_negative = int((train_df["target"] == 0).sum())
    n_positive = int((train_df["target"] == 1).sum())
    pos_weight = compute_pos_weight(n_negative, n_positive) if config["loss"] != "bce" else 1.0

    loss_fn = get_loss_function(
        config["loss"], pos_weight=pos_weight,
        gamma=config["loss_gamma"], ohem_ratio=config["ohem_ratio"],
    )

    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimiser, swa_lr=config["swa_lr"])

    history = {"train_loss": [], "val_pauc": []}
    best_pauc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(config["epochs"]):
        if config["return_patient_context"] and epoch > 0:
            train_cache = update_embedding_cache(model, train_dataset, device)
            val_cache = update_embedding_cache(model, val_dataset, device)
            cache = {**train_cache, **val_cache}
            train_dataset.update_embedding_cache(cache)
            val_dataset.update_embedding_cache(cache)

        model.train()
        epoch_losses = []

        for batch in train_loader:
            batch_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimiser.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(batch_device)
                    loss = loss_fn(logits, batch_device["target"])
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                logits = model(batch_device)
                loss = loss_fn(logits, batch_device["target"])
                loss.backward()
                optimiser.step()

            epoch_losses.append(loss.item())

        if epoch >= config["swa_start_epoch"]:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        val_metrics, _, _ = evaluate_model(model, val_loader, device)
        avg_loss = float(np.mean(epoch_losses))

        history["train_loss"].append(avg_loss)
        history["val_pauc"].append(val_metrics["pauc_above_80tpr"])

        logger.info(
            f"  Epoch {epoch+1}/{config['epochs']}: train_loss={avg_loss:.4f}, "
            f"val_pauc={val_metrics['pauc_above_80tpr']:.4f}"
        )

        if val_metrics["pauc_above_80tpr"] > best_pauc:
            best_pauc = val_metrics["pauc_above_80tpr"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, y_true, y_pred = evaluate_model(model, val_loader, device)

    return {
        "seed": seed,
        "fold": fold,
        "metrics": final_metrics,
        "history": history,
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "best_state": best_state,
    }


def run_multi_seed_experiment(
    overrides: Dict,
    model_factory: Callable,
) -> Dict:
    """Run cross‑validation across multiple seeds, aggregate results, and save best model."""
    config = merge_config(overrides)
    df = pd.read_csv(config["splits_path"], low_memory=False)

    results_dir = Path(config["results_dir"]) / config["experiment_name"]
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting experiment: {config['experiment_name']}")
    logger.info(f"  Seeds: {config['seeds']}, Folds: {config['n_folds']}")
    logger.info(f"  Total runs: {len(config['seeds']) * config['n_folds']}")

    all_paucs = []
    all_runs = []
    best_run_pauc = -1.0
    best_run_state = None
    start_time = time.time()

    for seed in config["seeds"]:
        for fold in range(config["n_folds"]):
            logger.info(f"Run: seed={seed}, fold={fold}")
            run = train_single_run(config, model_factory, seed, fold, df)

            pauc = run["metrics"]["pauc_above_80tpr"]
            all_paucs.append(pauc)

            if config["save_predictions"]:
                pred_file = results_dir / f"predictions_seed{seed}_fold{fold}.json"
                pred_file.write_text(json.dumps({
                    "seed": seed, "fold": fold,
                    "y_true": run["y_true"], "y_pred": run["y_pred"],
                    "metrics": run["metrics"],
                }))

            run_summary = {k: v for k, v in run.items() if k != "best_state"}
            all_runs.append(run_summary)

            if pauc > best_run_pauc and config["save_best_model"]:
                best_run_pauc = pauc
                best_run_state = run["best_state"]

    if best_run_state is not None and config["save_best_model"]:
        torch.save(best_run_state, results_dir / "best_model.pt")
        logger.info(f"Saved best model with pAUC={best_run_pauc:.4f}")

    summary = summarise_configuration(config["experiment_name"], np.array(all_paucs))
    summary["all_paucs"] = all_paucs
    summary["elapsed_seconds"] = time.time() - start_time

    summary_file = results_dir / "summary.json"
    summary_file.write_text(json.dumps({"config": config, "summary": summary, "runs": all_runs}, default=str))

    logger.info(f"Experiment complete: {config['experiment_name']} in {summary['elapsed_seconds']:.1f}s")
    return summary