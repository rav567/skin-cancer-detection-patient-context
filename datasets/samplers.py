"""
A batch sampler that enforces a fixed ratio of positive to negative samples per batch.

"""
import logging
from typing import Iterator, List

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class BalancedBatchSampler(Sampler[List[int]]):
    """Sampler that yields batches with a configurable positive/negative ratio."""
    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int = 32,
        positive_ratio: float = 0.25,
        seed: int = 42,
    ):
        super().__init__()

        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.seed = seed

        self.positive_indices = np.where(self.labels == 1)[0]
        self.negative_indices = np.where(self.labels == 0)[0]

        self.n_positive_per_batch = max(1, int(round(batch_size * positive_ratio)))
        self.n_negative_per_batch = batch_size - self.n_positive_per_batch

        if len(self.positive_indices) == 0:
            raise ValueError("No positive samples found in labels")
        if len(self.negative_indices) == 0:
            raise ValueError("No negative samples found in labels")

        self.n_batches = len(self.negative_indices) // self.n_negative_per_batch

        logger.info(
            f"BalancedBatchSampler: batch_size={batch_size}, "
            f"positive_ratio={positive_ratio:.2f}, "
            f"{self.n_positive_per_batch} pos + {self.n_negative_per_batch} neg per batch, "
            f"{self.n_batches} batches per epoch"
        )

    def __iter__(self) -> Iterator[List[int]]:
        """Yield shuffled batches, each with the specified class ratio."""
        rng = np.random.default_rng(self.seed)

        shuffled_negatives = rng.permutation(self.negative_indices)

        for batch_idx in range(self.n_batches):
            negative_start = batch_idx * self.n_negative_per_batch
            negative_end = negative_start + self.n_negative_per_batch
            batch_negatives = shuffled_negatives[negative_start:negative_end]

            batch_positives = rng.choice(
                self.positive_indices,
                size=self.n_positive_per_batch,
                replace=True,
            )

            batch = np.concatenate([batch_positives, batch_negatives])
            rng.shuffle(batch)
            yield batch.tolist()

    def __len__(self) -> int:
        return self.n_batches


def get_sampler(
    labels: np.ndarray,
    sampling_strategy: str = "balanced_1to3",
    batch_size: int = 32,
    seed: int = 42,
) -> BalancedBatchSampler:
    """Create a BalancedBatchSampler from a named strategy (e.g., 'balanced_1to3')."""
    ratios = {
        "balanced_1to1": 0.50,
        "balanced_1to3": 0.25,
        "balanced_1to5": 0.167,
    }

    if sampling_strategy not in ratios:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}. Choose from: {list(ratios.keys())}")

    return BalancedBatchSampler(
        labels=labels,
        batch_size=batch_size,
        positive_ratio=ratios[sampling_strategy],
        seed=seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    splits_path = "datasets/splits/subsampled_with_folds.csv"
    df = pd.read_csv(splits_path, low_memory=False)
    train_df = df[df["fold"] != 0]
    labels = train_df["target"].values

    print(f"\nDataset: {len(labels):,} samples, {labels.sum()} positive ({labels.mean()*100:.3f}%)")

    for strategy in ["balanced_1to1", "balanced_1to3", "balanced_1to5"]:
        sampler = get_sampler(labels, sampling_strategy=strategy, batch_size=32, seed=42)

        first_batch = next(iter(sampler))
        n_pos_in_batch = labels[first_batch].sum()

        print(f"  {strategy}: first batch has {n_pos_in_batch}/32 positive, total {len(sampler)} batches")