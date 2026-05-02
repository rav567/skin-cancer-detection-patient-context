import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)

DATA_DIR = Path("datasets")
IMAGE_DIR = DATA_DIR / "train-image" / "image"
RESULTS_DIR = Path("results") / "eda"
RANDOM_SEED = 42

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.dpi"] = 300


def plot_class_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["target"].value_counts().sort_index()
    labels = ["Benign", "Malignant"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"])
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Number of lesions")
    ax.set_title(f"Class distribution ({counts[1]/len(df)*100:.3f}% malignant)")
    ax.set_yscale("log")
    fig.savefig(output_dir / "class_distribution.pdf")
    plt.close(fig)
    logger.info("Saved class_distribution.pdf")


def plot_lesions_per_patient(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df.groupby("patient_id").size()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(counts.values, bins=50, color="#4C72B0", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Lesions per patient (log scale)")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Lesions per patient (median={int(counts.median())})")
    fig.savefig(output_dir / "lesions_per_patient.pdf")
    plt.close(fig)
    logger.info("Saved lesions_per_patient.pdf")


def plot_malignancy_by_anatomical_site(df: pd.DataFrame, output_dir: Path) -> None:
    grouped = df.groupby("anatom_site_general").agg(n=("target", "count"), n_malignant=("target", "sum"))
    grouped["malignancy_rate"] = grouped["n_malignant"] / grouped["n"]
    grouped = grouped.sort_values("malignancy_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(grouped.index, grouped["malignancy_rate"] * 100, color="#C44E52")
    ax.set_xlabel("Malignancy rate (%)")
    ax.set_title("Malignancy rate by anatomical site")
    for i, (rate, n) in enumerate(zip(grouped["malignancy_rate"], grouped["n"])):
        ax.text(rate * 100, i, f" n={n:,}", va="center", fontsize=8)
    fig.savefig(output_dir / "malignancy_by_anatomical_site.pdf")
    plt.close(fig)
    logger.info("Saved malignancy_by_anatomical_site.pdf")


def plot_sample_images(df: pd.DataFrame, image_dir: Path, output_dir: Path, n_per_class: int = 6) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    benign_ids = df.loc[df["target"] == 0, "isic_id"].sample(n=n_per_class, random_state=RANDOM_SEED).values
    malignant_ids = df.loc[df["target"] == 1, "isic_id"].sample(n=n_per_class, random_state=RANDOM_SEED).values

    fig, axes = plt.subplots(2, n_per_class, figsize=(n_per_class * 1.8, 4))
    for ax, isic_id in zip(axes[0], benign_ids):
        ax.imshow(Image.open(image_dir / f"{isic_id}.jpg"))
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_ylabel("Benign", fontsize=11)

    for ax, isic_id in zip(axes[1], malignant_ids):
        ax.imshow(Image.open(image_dir / f"{isic_id}.jpg"))
        ax.set_xticks([])
        ax.set_yticks([])
    axes[1, 0].set_ylabel("Malignant", fontsize=11)

    fig.suptitle("Sample lesion images")
    fig.savefig(output_dir / "sample_images.pdf")
    plt.close(fig)
    logger.info("Saved sample_images.pdf")


def plot_leakage_verification(df: pd.DataFrame, output_dir: Path) -> None:
    matrix = df.groupby(["patient_id", "fold"]).size().unstack(fill_value=0)
    binary_matrix = (matrix > 0).astype(int)

    patients_per_fold_count = binary_matrix.sum(axis=1).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(patients_per_fold_count.index, patients_per_fold_count.values, color="#4C72B0")
    ax.set_xlabel("Number of folds a patient appears in")
    ax.set_ylabel("Number of patients")
    ax.set_title("Patient-fold leakage check")
    ax.set_xticks([1, 2, 3, 4, 5])
    for i, v in enumerate(patients_per_fold_count.values):
        ax.text(patients_per_fold_count.index[i], v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    fig.savefig(output_dir / "leakage_verification.pdf")
    plt.close(fig)
    logger.info("Saved leakage_verification.pdf")


def generate_all_eda_figures(df: pd.DataFrame, image_dir: Path = IMAGE_DIR, output_dir: Path = RESULTS_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating EDA figures in {output_dir}")

    plot_class_distribution(df, output_dir)
    plot_lesions_per_patient(df, output_dir)
    plot_malignancy_by_anatomical_site(df, output_dir)
    plot_sample_images(df, image_dir, output_dir)
    plot_leakage_verification(df, output_dir)

    logger.info("All EDA figures generated")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    df = pd.read_csv(DATA_DIR / "splits" / "subsampled_with_folds.csv", low_memory=False)
    generate_all_eda_figures(df)