"""Prespecified, training-schema-only feature mapping for DD N2 preparation.

This module does not score a reaction. It makes the otherwise incompatible
historical 7-column and K1 9-column checkpoint inputs explicit and testable.
The N2 cross-inference experiment still requires a separate approved runner.
"""

from __future__ import annotations

import numpy as np

from rerank.tuning import (
    AUGMENTED_COLUMNS as HISTORICAL_AUGMENTED,
    BASELINE_COLUMNS as HISTORICAL_BASELINE,
)
from rerank.pool_streaming import (
    AUGMENTED_COLUMNS as AIZ_AUGMENTED,
    BASELINE_COLUMNS as AIZ_BASELINE,
    FULL_FEATURE_NAMES,
)


HISTORICAL_NAMES = FULL_FEATURE_NAMES[:7]
AIZ_NAMES = FULL_FEATURE_NAMES
TRAINING_POOLS = ("historical_cap10", "aizynthfinder_only")


def training_arm_features(
    full_features: np.ndarray, *, training_pool: str, evaluation_pool: str, arm: str
) -> np.ndarray:
    """Map frozen evaluation features into the *training* checkpoint schema.

    The historical generator is AiZynthFinder, so its rows have deterministic
    source flags (1, 0). A final-K1 AiZ-only row must itself retain these flags.
    No prior transformation or normalizer refit is performed here.
    """

    if training_pool not in TRAINING_POOLS or evaluation_pool not in TRAINING_POOLS:
        raise ValueError("N2 initially supports historical and AiZ-only pools only.")
    if arm not in {"baseline", "augmented"}:
        raise ValueError("arm must be baseline or augmented.")
    features = np.asarray(full_features, dtype=np.float32)
    expected = 7 if evaluation_pool == "historical_cap10" else 9
    if features.ndim != 2 or features.shape[1] != expected or not np.isfinite(features).all():
        raise ValueError(f"Expected a finite [n,{expected}] {evaluation_pool} feature matrix.")

    if evaluation_pool == "aizynthfinder_only" and (
        not np.array_equal(features[:, 7], np.ones(len(features), dtype=np.float32))
        or not np.array_equal(features[:, 8], np.zeros(len(features), dtype=np.float32))
    ):
        raise ValueError("AiZ-only source indicators differ from the frozen 1/0 definition.")

    if training_pool == "historical_cap10":
        mapped = features[:, :7]
        columns = HISTORICAL_BASELINE if arm == "baseline" else HISTORICAL_AUGMENTED
    else:
        mapped = features if evaluation_pool == "aizynthfinder_only" else np.column_stack(
            (features, np.ones(len(features), dtype=np.float32),
             np.zeros(len(features), dtype=np.float32))
        )
        columns = AIZ_BASELINE if arm == "baseline" else AIZ_AUGMENTED
    return np.asarray(mapped[:, columns], dtype=np.float32)


def clipping_audit(
    arm_features: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_sigma: float
) -> dict[str, float | bool]:
    """Audit the frozen normalizer's clipping without fitting or transforming it."""

    values = np.asarray(arm_features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if values.ndim != 2 or not len(values) or mean.shape != (values.shape[1],) or std.shape != mean.shape:
        raise ValueError("Normalizer and feature matrix dimensions differ.")
    if not np.isfinite(values).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Clipping audit requires finite values.")
    if np.any(std <= 0) or not np.isfinite(clip_sigma) or clip_sigma <= 0:
        raise ValueError("Invalid frozen normalizer scale or clipping threshold.")
    standardized = (values - mean) / std
    prior_lower = float(np.mean(standardized[:, 0] < -clip_sigma))
    prior_upper = float(np.mean(standardized[:, 0] > clip_sigma))
    return {
        "prior_lower_fraction": prior_lower,
        "prior_upper_fraction": prior_upper,
        "any_feature_clipped_fraction": float(np.mean(np.any(np.abs(standardized) > clip_sigma, axis=1))),
        "prior_total_clipped_fraction": prior_lower + prior_upper,
        "uninterpretable_prior_clipping_gt_10pct": prior_lower + prior_upper > 0.1,
    }
