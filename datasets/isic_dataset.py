"""
PyTorch Dataset for ISIC 2024 images with optional tabular/patient‑context features.

"""

import logging
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATA_DIR = Path("datasets")
IMAGE_DIR = DATA_DIR / "train-image" / "image"

NUMERIC_FEATURES = ["age_approx", "clin_size_long_diam_mm"]
CATEGORICAL_FEATURES = ["sex", "anatom_site_general"]


def build_tabular_encoder(df: pd.DataFrame):
    """Return one‑hot encoder column names and total feature dimension."""
    encoder = pd.get_dummies(df[CATEGORICAL_FEATURES], dummy_na=False).columns.tolist()
    feature_dim = len(NUMERIC_FEATURES) + len(encoder)
    return encoder, feature_dim


def encode_tabular_features(row: pd.Series, encoder_columns: list) -> np.ndarray:
    """Convert a single row’s numeric and categorical fields into a float32 array."""
    numeric = [0.0 if pd.isna(row[c]) else float(row[c]) for c in NUMERIC_FEATURES]
    one_hot_row = pd.get_dummies(pd.DataFrame([row[CATEGORICAL_FEATURES]]), dummy_na=False)
    one_hot = [float(one_hot_row[c].iloc[0]) if c in one_hot_row.columns else 0.0 for c in encoder_columns]
    return np.array(numeric + one_hot, dtype=np.float32)


class ISICDataset(Dataset):
    """Dataset that loads images and optionally returns tabular data and patient‑context embeddings."""
    def __init__(self, df, image_dir=IMAGE_DIR, transform=None, return_tabular=False,
                 return_patient_context=False, patient_embedding_cache=None,
                 max_context_lesions=50, encoder_columns=None):
        """Set up the dataset, image directory, transformations, and cache."""
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_tabular = return_tabular
        self.return_patient_context = return_patient_context
        self.cache = patient_embedding_cache or {}
        self.max_context = max_context_lesions
        self.encoder_columns = encoder_columns or []
        self.patient_to_lesions = self.df.groupby("patient_id")["isic_id"].apply(list).to_dict()

        logger.info(f"ISICDataset: {len(self.df):,} lesions, {len(self.patient_to_lesions)} patients, "
                    f"tabular={return_tabular}, context={return_patient_context}")

    def update_embedding_cache(self, cache: Dict[str, np.ndarray]) -> None:
        """Replace the in‑memory cache of lesion embeddings."""
        self.cache = cache
        logger.info(f"Updated cache: {len(cache):,} embeddings")

    def _get_context(self, patient_id: str, current_id: str):
        """Return other lesions’ cached embeddings for the same patient (padded)."""
        others = [lid for lid in self.patient_to_lesions.get(patient_id, []) if lid != current_id]
        if len(others) > self.max_context:
            others = list(np.random.default_rng().choice(others, self.max_context, replace=False))

        embeds = [self.cache[lid] for lid in others if lid in self.cache]
        if not embeds:
            return torch.zeros(1, 1, dtype=torch.float32), torch.zeros(1, dtype=torch.bool)

        return torch.from_numpy(np.stack(embeds)).float(), torch.ones(len(embeds), dtype=torch.bool)

    def __len__(self) -> int:
        """Number of lesions in the dataset."""
        return len(self.df)

    def __getitem__(self, idx: int):
        """Load the image, target, and optional tabular/context data for one lesion."""
        row = self.df.iloc[idx]
        image = np.array(Image.open(self.image_dir / f"{row['isic_id']}.jpg").convert("RGB"))
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        sample = {
            "image": image,
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "isic_id": row["isic_id"],
            "patient_id": row["patient_id"],
        }

        if self.return_tabular:
            sample["tabular"] = torch.from_numpy(encode_tabular_features(row, self.encoder_columns))

        if self.return_patient_context:
            ctx, mask = self._get_context(row["patient_id"], row["isic_id"])
            sample["patient_context"] = ctx
            sample["patient_context_mask"] = mask

        return sample


def collate_with_padding(batch):
    """Stack images/targets and pad patient‑context sequences to the same length."""
    output = {
        "image": torch.stack([b["image"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "isic_id": [b["isic_id"] for b in batch],
        "patient_id": [b["patient_id"] for b in batch],
    }

    if "tabular" in batch[0]:
        output["tabular"] = torch.stack([b["tabular"] for b in batch])

    if "patient_context" in batch[0]:
        max_len = max(b["patient_context"].shape[0] for b in batch)
        embed_dim = max(b["patient_context"].shape[1] for b in batch)
        contexts = torch.zeros(len(batch), max_len, embed_dim, dtype=torch.float32)
        masks = torch.zeros(len(batch), max_len, dtype=torch.bool)
        for i, b in enumerate(batch):
            n, d = b["patient_context"].shape
            contexts[i, :n, :d] = b["patient_context"]
            masks[i, :n] = b["patient_context_mask"]
        output["patient_context"] = contexts
        output["patient_context_mask"] = masks

    return output


if __name__ == "__main__":
    from transforms import get_transforms

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    df = pd.read_csv("datasets/splits/subsampled_with_folds.csv", low_memory=False)
    sample_df = df[df["fold"] != 0].head(100)

    encoder_columns, feature_dim = build_tabular_encoder(df)
    transform = get_transforms("standard", image_size=128, training=True)
    fake_cache = {r["isic_id"]: np.random.randn(64).astype(np.float32) for _, r in sample_df.iterrows()}

    dataset = ISICDataset(
        sample_df, transform=transform, return_tabular=True,
        return_patient_context=True, patient_embedding_cache=fake_cache,
        max_context_lesions=10, encoder_columns=encoder_columns,
    )

    sample = dataset[0]
    print(f"Image: {sample['image'].shape}, Tabular: {sample['tabular'].shape}, Context: {sample['patient_context'].shape}")
    print(f"Tabular feature dim: {feature_dim}")