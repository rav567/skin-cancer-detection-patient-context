"""
Bootstrap confidence intervals and paired hypothesis testing for comparing model performance metrics.

"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CI_LEVEL = 0.95
DEFAULT_SEED = 42


def bootstrap_confidence_interval(
    measurements: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Compute bootstrap CI for the mean of a set of measurements (e.g., AUROC across folds)."""
    measurements = np.asarray(measurements)

    if len(measurements) == 0:
        raise ValueError("Cannot compute CI on empty measurements")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(n_bootstrap)
    n = len(measurements)

    for i in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_means[i] = measurements[sample_indices].mean()

    alpha = 1.0 - ci_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return {
        "mean": float(measurements.mean()),
        "std": float(measurements.std(ddof=1)) if len(measurements) > 1 else 0.0,
        "ci_low": float(np.percentile(bootstrap_means, lower_percentile)),
        "ci_high": float(np.percentile(bootstrap_means, upper_percentile)),
        "n_measurements": int(len(measurements)),
    }


def paired_bootstrap_test(
    measurements_a: np.ndarray,
    measurements_b: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> Dict[str, float]:
    """Paired bootstrap test for difference in means between two matched measurements sets."""
    measurements_a = np.asarray(measurements_a)
    measurements_b = np.asarray(measurements_b)

    if len(measurements_a) != len(measurements_b):
        raise ValueError(f"Paired test requires equal lengths, got {len(measurements_a)} vs {len(measurements_b)}")

    differences = measurements_a - measurements_b
    n = len(differences)
    rng = np.random.default_rng(seed)

    bootstrap_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample_indices = rng.integers(0, n, size=n)
        bootstrap_diffs[i] = differences[sample_indices].mean()

    p_value_two_sided = 2 * min(
        (bootstrap_diffs >= 0).mean(),
        (bootstrap_diffs <= 0).mean(),
    )

    return {
        "mean_diff": float(differences.mean()),
        "ci_low": float(np.percentile(bootstrap_diffs, 2.5)),
        "ci_high": float(np.percentile(bootstrap_diffs, 97.5)),
        "p_value": float(p_value_two_sided),
        "is_significant_at_05": bool(p_value_two_sided < 0.05),
        "n_pairs": int(n),
    }


def format_result_with_ci(stats: Dict[str, float], decimals: int = 4) -> str:
    """Format mean with 95% bootstrap CI as 'mean [low, high]'."""
    return f"{stats['mean']:.{decimals}f} [{stats['ci_low']:.{decimals}f}, {stats['ci_high']:.{decimals}f}]"


def format_result_with_std(stats: Dict[str, float], decimals: int = 4) -> str:
    """Format mean with standard deviation as 'mean ± std'."""
    return f"{stats['mean']:.{decimals}f} ± {stats['std']:.{decimals}f}"


def summarise_configuration(name: str, measurements: np.ndarray) -> Dict[str, float]:
    """Log and return bootstrap stats for a single model configuration."""
    stats = bootstrap_confidence_interval(measurements)
    logger.info(
        f"{name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"95% CI=[{stats['ci_low']:.4f}, {stats['ci_high']:.4f}], n={stats['n_measurements']}"
    )
    return stats


def compare_configurations(name_a: str, measurements_a: np.ndarray, name_b: str, measurements_b: np.ndarray) -> Dict[str, float]:
    """Log and return paired bootstrap test results comparing two configurations."""
    test_result = paired_bootstrap_test(measurements_a, measurements_b)
    significance = "significant" if test_result["is_significant_at_05"] else "not significant"
    logger.info(
        f"{name_a} vs {name_b}: diff={test_result['mean_diff']:+.4f}, "
        f"95% CI=[{test_result['ci_low']:+.4f}, {test_result['ci_high']:+.4f}], "
        f"p={test_result['p_value']:.4f} ({significance})"
    )
    return test_result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    rng = np.random.default_rng(42)

    config_a_results = rng.normal(0.130, 0.010, size=15)
    config_c_results = rng.normal(0.145, 0.010, size=15)
    config_c_no_improvement = rng.normal(0.131, 0.010, size=15)

    print("\nTest 1: Single configuration confidence interval")
    summarise_configuration("Configuration A (image-only)", config_a_results)
    summarise_configuration("Configuration C (with patient context)", config_c_results)

    print("\nTest 2: Significant comparison")
    compare_configurations("Configuration C", config_c_results, "Configuration A", config_a_results)

    print("\nTest 3: Non-significant comparison")
    compare_configurations("Configuration C-no-improvement", config_c_no_improvement, "Configuration A", config_a_results)

    print("\nTest 4: Formatted output strings")
    stats = bootstrap_confidence_interval(config_c_results)
    print(f"  With CI: {format_result_with_ci(stats)}")
    print(f"  With std: {format_result_with_std(stats)}")