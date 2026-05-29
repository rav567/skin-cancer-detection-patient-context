"""
Extract embeddings, run UMAP, and plot malignancy vs context‑residual norm.

"""

import logging
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets.isic_dataset import ISICDataset, collate_with_padding
from datasets.transforms import get_transforms
from models.full_pipeline import FullPipeline

logger = logging.getLogger(__name__)

# Paths and settings for the UMAP analysis
SPLITS_PATH = "datasets/splits/subsampled_with_folds.csv"
CHECKPOINT_PATH = "results/configurations/config_c_image_context/best_model.pt"
OUTPUT_PATH = "results/figures/umap_analysis.pdf"
IMAGE_SIZE = 96
SEED = 42
MAX_BENIGN_PLOT = 6000

mpl.rcParams.update({
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_encoder(device):
    """Load the trained Config C pipeline, or fall back to the pretrained backbone."""
    model = FullPipeline(
        backbone_name="efficientnet_b0", pretrained=True,
        use_patient_context=True, context_aggregation="mean", context_fusion="residual",
    )
    if os.path.exists(CHECKPOINT_PATH):
        state = torch.load(CHECKPOINT_PATH, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Loaded trained Config C checkpoint ({len(missing)} missing, "
                    f"{len(unexpected)} unexpected keys)")
    else:
        logger.warning(f"Checkpoint not found at {CHECKPOINT_PATH} - "
                       f"using ImageNet-pretrained backbone instead.")
    return model.to(device).eval()


def extract_embeddings(model, df, device, batch_size=128):
    """Return (embeddings, targets, patient_ids) for every lesion in df."""
    transform = get_transforms("standard", image_size=IMAGE_SIZE, training=False)
    dataset = ISICDataset(df, transform=transform, return_tabular=False,
                          return_patient_context=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_with_padding)

    embeddings, targets, patients = [], [], []
    with torch.no_grad():
        for batch in loader:
            emb = model.encode(batch["image"].to(device)).cpu().numpy()
            embeddings.append(emb)
            targets.append(batch["target"].numpy())
            patients.extend(batch["patient_id"])
    return np.concatenate(embeddings), np.concatenate(targets), np.array(patients)


def context_residual_norms(embeddings, patients):
    """For each lesion, where c is the mean of the patient's OTHER lesions."""
    residuals = np.full(len(embeddings), np.nan)
    index_by_patient = {}
    for i, p in enumerate(patients):
        index_by_patient.setdefault(p, []).append(i)
    for i, p in enumerate(patients):
        others = [j for j in index_by_patient[p] if j != i]
        if others:
            context = embeddings[others].mean(axis=0)
            residuals[i] = np.linalg.norm(embeddings[i] - context)
    return residuals


def plot_umap(xy, targets, residuals, output_path):
    """Two-panel UMAP: malignancy (left) and context-residual norm (right)."""
    rng = np.random.default_rng(SEED)
    mal_idx = np.where(targets == 1)[0]
    ben_idx = np.where(targets == 0)[0]
    if len(ben_idx) > MAX_BENIGN_PLOT:
        ben_idx = rng.choice(ben_idx, MAX_BENIGN_PLOT, replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.5))

    ax1.scatter(xy[ben_idx, 0], xy[ben_idx, 1], s=4, c="#bbbbbb", alpha=0.5,
                edgecolors="none", label="Benign")
    ax1.scatter(xy[mal_idx, 0], xy[mal_idx, 1], s=10, c="#c1121f", alpha=0.9,
                edgecolors="none", label="Malignant")
    ax1.set_title("(a) Coloured by malignancy")
    ax1.set_xlabel("UMAP-1")
    ax1.set_ylabel("UMAP-2")
    ax1.legend(loc="best", frameon=False, fontsize=8, markerscale=1.5)

    plot_idx = np.concatenate([ben_idx, mal_idx])
    valid = ~np.isnan(residuals[plot_idx])
    sc = ax2.scatter(xy[plot_idx][valid, 0], xy[plot_idx][valid, 1],
                     s=5, c=residuals[plot_idx][valid], cmap="viridis", alpha=0.7,
                     edgecolors="none")
    ax2.set_title("(b) Coloured by context residual $\\|h-c\\|$")
    ax2.set_xlabel("UMAP-1")
    ax2.set_ylabel("UMAP-2")
    fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)

    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved figure to {output_path}")


def main():
    """Extract embeddings, run UMAP, and save the two-panel representation figure."""
    os.makedirs("results/figures", exist_ok=True)
    import umap  # imported here so the rest of the repo does not require umap-learn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    df = pd.read_csv(SPLITS_PATH, low_memory=False)
    model = load_encoder(device)

    logger.info(f"Extracting embeddings for {len(df):,} lesions...")
    embeddings, targets, patients = extract_embeddings(model, df, device)

    logger.info("Computing patient-context residual norms...")
    residuals = context_residual_norms(embeddings, patients)

    logger.info("Running UMAP (this can take a minute)...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=SEED)
    xy = reducer.fit_transform(embeddings)

    plot_umap(xy, targets, residuals, OUTPUT_PATH)

    # A compact numeric summary for the report text.
    mal = residuals[(targets == 1) & ~np.isnan(residuals)]
    ben = residuals[(targets == 0) & ~np.isnan(residuals)]
    logger.info(f"Mean residual ||h-c||: malignant={mal.mean():.3f}, benign={ben.mean():.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()