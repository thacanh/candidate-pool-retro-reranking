"""Prespecified validation-only tuning primitives for ``cap10-tuned-v1``.

This module is deliberately independent of the legacy fixed-50 runner.  In
particular, a selection bundle contains training and official-validation data
only; post-selection test data have no representation in its schema.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROTOCOL_ID = "cap10-tuned-v1"
CAPACITY_CONTROL_ID = "D-CAPACITY"
CAPACITY_PLAN_SCHEMA = 1
CAPACITY_TRIAL_SCHEMA = 1
SELECTION_BUNDLE_SCHEMA = 1
TRIAL_SCHEMA = 1
SEEDS = (42, 43, 44, 45, 46)
PRIOR_TRANSFORMS = ("raw", "log", "rank")
BASELINE_COLUMNS = (0, 1, 5, 6)
AUGMENTED_COLUMNS = tuple(range(7))
GRID_TIE_EPSILON = 1e-12
MIN_IMPROVEMENT = 1e-5
MAX_EPOCHS = 200
PATIENCE = 20


@dataclass(frozen=True)
class GridConfig:
    """One point in the exact D1 enumeration."""

    index: int
    hidden_width: int
    dropout: float
    learning_rate: float
    margin: float


@dataclass(frozen=True)
class PreparedValidation:
    """Normalized validation rows and fixed reaction boundaries/matches."""

    features: np.ndarray
    offsets: tuple[int, ...]
    match_masks: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class CapacitySettings:
    """Explicitly frozen non-width settings for the separate D3 control."""

    dropout: float
    learning_rate: float
    margin: float


def validate_capacity_settings(settings: CapacitySettings) -> None:
    """Fail closed unless D3 uses values already admitted by the D1 plan."""

    if settings.dropout not in (0.0, 0.1, 0.3):
        raise ValueError("D-CAPACITY dropout must be explicitly selected from 0, 0.1, 0.3.")
    if settings.learning_rate not in (1e-4, 3e-4, 1e-3):
        raise ValueError("D-CAPACITY learning rate must be 1e-4, 3e-4, or 1e-3.")
    if settings.margin not in (0.0, 0.1, 0.3):
        raise ValueError("D-CAPACITY margin must be explicitly selected from 0, 0.1, 0.3.")


def capacity_arm_config(settings: CapacitySettings, arm: str) -> GridConfig:
    """Return the sole D3 configuration for one capacity-matched arm."""

    validate_capacity_settings(settings)
    if arm == "baseline":
        width = 48
    elif arm == "augmented":
        width = 32
    else:
        raise ValueError("D-CAPACITY arm must be baseline or augmented.")
    return GridConfig(
        index=-1,
        hidden_width=width,
        dropout=settings.dropout,
        learning_rate=settings.learning_rate,
        margin=settings.margin,
    )


def capacity_settings_fingerprint(settings: CapacitySettings) -> str:
    validate_capacity_settings(settings)
    return "sha256:" + hashlib.sha256(
        json.dumps(asdict(settings), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def enumerate_d1_grid() -> tuple[GridConfig, ...]:
    """Return all 81 configurations in the signed-off nested-loop order."""

    values = itertools.product(
        (32, 64, 128),
        (0.0, 0.1, 0.3),
        (1e-4, 3e-4, 1e-3),
        (0.0, 0.1, 0.3),
    )
    grid = tuple(
        GridConfig(index, width, dropout, learning_rate, margin)
        for index, (width, dropout, learning_rate, margin) in enumerate(values)
    )
    if len(grid) != 81 or [item.index for item in grid] != list(range(81)):
        raise AssertionError("D1 grid must contain indices 0..80 exactly once.")
    return grid


def config_fingerprint(config: GridConfig) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def shard_config_indices(shard_index: int, shard_count: int) -> tuple[int, ...]:
    """Partition the exact enumeration without changing its order."""

    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("Require shard_count >= 1 and 0 <= shard_index < shard_count.")
    return tuple(range(shard_index, 81, shard_count))


def descending_midrank_score(priors: Sequence[float]) -> np.ndarray:
    """D2 descending normalized midrank, with exact ties sharing a score."""

    values = np.asarray(priors, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Priors must be a finite one-dimensional sequence.")
    n_items = len(values)
    if n_items == 0:
        return np.empty(0, dtype=np.float32)
    if n_items == 1:
        return np.ones(1, dtype=np.float32)

    order = np.argsort(-values, kind="stable")
    ranks = np.empty(n_items, dtype=np.float64)
    start = 0
    while start < n_items:
        end = start + 1
        while end < n_items and values[order[end]] == values[order[start]]:
            end += 1
        # One-based ranks start+1 ... end; their arithmetic mean is below.
        midrank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = midrank
        start = end
    return (1.0 - (ranks - 1.0) / (n_items - 1.0)).astype(np.float32)


def transform_prior_values(priors: Sequence[float], transform: str) -> np.ndarray:
    values = np.asarray(priors, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Priors must be a finite one-dimensional sequence.")
    if transform == "raw":
        result = values
    elif transform == "log":
        if np.any(values < 0.0):
            raise ValueError("log(prior + 1e-12) is undefined for negative priors.")
        result = np.log(values + 1e-12)
    elif transform == "rank":
        return descending_midrank_score(values)
    else:
        raise ValueError(f"Unknown prior transform: {transform!r}.")
    if not np.isfinite(result).all():
        raise ValueError(f"Prior transform {transform!r} produced non-finite values.")
    return result.astype(np.float32)


def transform_feature_matrix(
    matrix: np.ndarray, transform: str, columns: Sequence[int]
) -> np.ndarray:
    """Apply D2 to column zero, then select a frozen feature view."""

    source = np.asarray(matrix, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 7:
        raise ValueError(f"Expected a seven-column compact feature matrix, got {source.shape}.")
    full = source.copy()
    full[:, 0] = transform_prior_values(full[:, 0], transform)
    return full[:, tuple(columns)].copy()


def feature_columns_for_arm(arm: str) -> tuple[int, ...]:
    if arm in {"baseline", "capacity-baseline"}:
        return BASELINE_COLUMNS
    if arm in {"augmented", "capacity-augmented"}:
        return AUGMENTED_COLUMNS
    raise ValueError(f"Unsupported arm: {arm!r}.")


def transform_selection_cache(bundle: Mapping, arm: str, prior_transform: str) -> dict:
    """Build an isolated view; never mutate the retained compact bundle."""

    validate_selection_bundle(bundle)
    columns = feature_columns_for_arm(arm)
    train_products = []
    for product in bundle["train_products"]:
        copied = dict(product)
        copied["features"] = transform_feature_matrix(
            product["features"], prior_transform, columns
        )
        train_products.append(copied)
    validation = bundle["validation_payload"]
    return {
        "train_products": train_products,
        "eval_pwc": validation["eval_pwc"],
        "eval_ground_truths": validation["eval_ground_truths"],
        "eval_metadata": validation["eval_metadata"],
        "eval_features": [
            transform_feature_matrix(matrix, prior_transform, columns)
            for matrix in validation["eval_features"]
        ],
    }


def validate_selection_bundle(bundle: Mapping) -> None:
    if bundle.get("selection_bundle_schema") != SELECTION_BUNDLE_SCHEMA:
        raise ValueError("Unsupported selection-bundle schema.")
    if bundle.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Selection bundle has the wrong protocol ID.")
    if tuple(bundle.get("seeds", ())) != SEEDS:
        raise ValueError("Selection bundle must lock seeds 42--46.")
    provenance = bundle.get("representation_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Selection bundle lacks representation provenance.")
    if provenance.get("kind") == "encoder_control_without_conformer":
        if (
            provenance.get("encoder_control_has_conformer") is not False
            or provenance.get("conformer_seed") is not None
            or provenance.get(
                "prepare_conformer_seed_argument_is_scientific_provenance"
            )
            is not False
        ):
            raise ValueError("Encoder-control provenance incorrectly claims a conformer.")
    elif provenance.get("kind") == "indexed_conformer":
        if (
            provenance.get("encoder_control_has_conformer") is not True
            or not isinstance(provenance.get("conformer_seed"), int)
            or provenance.get(
                "prepare_conformer_seed_argument_is_scientific_provenance"
            )
            is not True
        ):
            raise ValueError("Indexed-conformer provenance is incomplete.")
    elif provenance.get("kind") == "multi_conformer_scalar_average":
        conformer_seeds = tuple(provenance.get("conformer_seeds", ()))
        if (
            provenance.get("encoder_control_has_conformer") is not True
            or provenance.get("conformer_seed") is not None
            or conformer_seeds != tuple(range(42, 52))
            or provenance.get("aggregation")
            != "arithmetic mean of each pair-level scalar after fragment handling"
            or provenance.get("atom_embeddings_averaged") is not False
        ):
            raise ValueError("Multi-conformer-average provenance is incomplete.")
    else:
        raise ValueError("Selection bundle has an unsupported representation kind.")
    forbidden = {
        "test",
        "eval_pwc",
        "eval_features",
        "eval_ground_truths",
        "eval_metadata",
        "post_selection_test",
    }
    leaked = forbidden.intersection(bundle)
    if leaked:
        raise PermissionError(f"Selection bundle contains forbidden test-like keys: {sorted(leaked)}")
    validation = bundle.get("validation_payload")
    if not isinstance(validation, Mapping):
        raise ValueError("Selection bundle lacks its official-validation payload.")
    required = {"eval_pwc", "eval_features", "eval_ground_truths", "eval_metadata"}
    if not required.issubset(validation):
        raise ValueError("Selection bundle has an incomplete validation payload.")
    lengths = {len(validation[key]) for key in required}
    if len(lengths) != 1 or lengths == {0}:
        raise ValueError("Official-validation payload is empty or misaligned.")
    metadata = validation["eval_metadata"]
    if any(str(row.get("source_split")) != "valid" for row in metadata):
        raise PermissionError("Selection payload contains a non-validation reaction.")
    for (_, candidates), matrix in zip(
        validation["eval_pwc"], validation["eval_features"]
    ):
        array = np.asarray(matrix, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 7 or len(array) != len(candidates):
            raise ValueError("Validation features must be candidate-aligned and seven-column.")
    train_products = bundle.get("train_products")
    if not isinstance(train_products, Sequence) or not train_products:
        raise ValueError("Selection bundle contains no training products.")
    for product in train_products:
        matrix = np.asarray(product.get("features"), dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != 7:
            raise ValueError("Selection training features must have exactly seven columns.")


def _unwrap_legacy_blob(blob: Mapping) -> Mapping:
    payload = blob.get("payload", blob)
    if not isinstance(payload, Mapping):
        raise ValueError("Legacy compact cache has no mapping payload.")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": file_sha256(path),
    }


def atomic_json_dump(payload: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def atomic_pickle_dump(payload: Mapping, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def prepare_selection_bundle(
    train_test_cache_path: str | Path,
    validation_cache_path: str | Path,
    output_path: str | Path,
    conformer_seed: int,
) -> dict:
    """Strip test before search and combine retained train with official valid.

    This preparation phase is the only pre-freeze operation allowed to open the
    legacy train/test artifact.  It copies ``train_products`` and intentionally
    discards all legacy ``eval_*`` values.  Search commands accept only the
    resulting restricted bundle.
    """

    with open(train_test_cache_path, "rb") as handle:
        legacy_blob = pickle.load(handle)
    with open(validation_cache_path, "rb") as handle:
        validation_blob = pickle.load(handle)
    legacy = _unwrap_legacy_blob(legacy_blob)
    validation = validation_blob.get("payload", validation_blob)
    if legacy.get("feature_mode") != "3d+prior" or validation_blob.get(
        "feature_mode"
    ) != "3d+prior":
        raise ValueError("Tuned revision requires seven-column 3d+prior compact caches.")
    expected_encoder_compatibility = {
        "cache_layout": "seven-column 3d+prior",
        "conformer_seed_field_omitted": True,
        "prepare_conformer_seed_argument_is_not_encoder_provenance": True,
    }
    legacy_encoder_control = legacy_blob.get("encoder_control_has_conformer") is False
    validation_encoder_control = (
        validation_blob.get("encoder_control_has_conformer") is False
    )
    if legacy_encoder_control or validation_encoder_control:
        if not (legacy_encoder_control and validation_encoder_control):
            raise ValueError("Encoder-control train/test and validation artifacts disagree.")
        if "conformer_seed" in legacy_blob or "conformer_seed" in validation_blob:
            raise ValueError("A no-conformer encoder control must omit conformer_seed.")
        if validation_blob.get("tuned_runner_compatibility") != expected_encoder_compatibility:
            raise ValueError("Encoder-control validation artifact lacks the exact runner contract.")
        if legacy_blob.get("tuned_runner_compatibility") != expected_encoder_compatibility:
            raise ValueError("Encoder-control train/test artifact lacks the exact runner contract.")
        representation_provenance = {
            "kind": "encoder_control_without_conformer",
            "encoder_control_has_conformer": False,
            "conformer_seed": None,
            "prepare_conformer_seed_argument": int(conformer_seed),
            "prepare_conformer_seed_argument_is_scientific_provenance": False,
            "tuned_runner_compatibility": expected_encoder_compatibility,
            "source_protocol_id": validation_blob.get("protocol_id"),
        }
    else:
        if "conformer_seed" not in validation_blob:
            raise ValueError("A conformer validation artifact must declare conformer_seed.")
        if int(validation_blob["conformer_seed"]) != int(conformer_seed):
            raise ValueError(
                "Validation artifact conformer seed does not match the requested seed."
            )
        representation_provenance = {
            "kind": "indexed_conformer",
            "encoder_control_has_conformer": True,
            "conformer_seed": int(conformer_seed),
            "prepare_conformer_seed_argument_is_scientific_provenance": True,
            "source_protocol_id": validation_blob.get("protocol_id"),
        }
    if "train_products" not in legacy:
        raise ValueError("Legacy compact cache lacks retained training products.")

    bundle = {
        "selection_bundle_schema": SELECTION_BUNDLE_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "comparator": "validation-tuned prior+2D baseline",
        "single_intended_change": "validation-only hyperparameter selection",
        "representation_provenance": representation_provenance,
        "seeds": list(SEEDS),
        "feature_mode": "3d+prior",
        "train_split": "train",
        "validation_split": "valid",
        "train_products": legacy["train_products"],
        "validation_payload": validation,
        "audit": {
            "train": legacy.get("audit"),
            "validation": validation_blob.get("audit"),
            "test_fields_discarded": sorted(
                key for key in legacy if key.startswith("eval_")
            ),
        },
        "input_fingerprints": {
            "retained_train_test_cache": file_fingerprint(train_test_cache_path),
            "official_validation_cache": file_fingerprint(validation_cache_path),
        },
    }
    validate_selection_bundle(bundle)
    atomic_pickle_dump(bundle, output_path)
    return bundle


def load_selection_bundle(path: str | Path) -> dict:
    with open(path, "rb") as handle:
        bundle = pickle.load(handle)
    validate_selection_bundle(bundle)
    return bundle


def select_shared_config(trials: Iterable[Mapping]) -> dict:
    """Select one D1 config by mean best validation MRR across all five seeds."""

    grouped: dict[int, dict[int, Mapping]] = {}
    for trial in trials:
        if trial.get("status") != "completed":
            continue
        grouped.setdefault(int(trial["config_index"]), {})[int(trial["seed"])] = trial

    complete = []
    for config in enumerate_d1_grid():
        by_seed = grouped.get(config.index, {})
        if set(by_seed) != set(SEEDS):
            continue
        scores = [float(by_seed[seed]["best_validation_mrr"]) for seed in SEEDS]
        complete.append(
            {
                "config": asdict(config),
                "config_fingerprint": config_fingerprint(config),
                "mean_best_validation_mrr": float(np.mean(scores)),
                "per_seed_best_validation_mrr": {
                    str(seed): float(by_seed[seed]["best_validation_mrr"])
                    for seed in SEEDS
                },
                "per_seed_best_epoch": {
                    str(seed): int(by_seed[seed]["best_epoch"]) for seed in SEEDS
                },
                "trials": {str(seed): dict(by_seed[seed]) for seed in SEEDS},
            }
        )
    if len(complete) != 81:
        missing = sorted(set(range(81)).difference(item["config"]["index"] for item in complete))
        raise RuntimeError(f"Selection requires all 81 x 5 trials; incomplete configs: {missing}")

    best = complete[0]
    for candidate in complete[1:]:
        if (
            candidate["mean_best_validation_mrr"]
            > best["mean_best_validation_mrr"] + GRID_TIE_EPSILON
        ):
            best = candidate
    return best


def select_prior_transform(trials_by_transform: Mapping[str, Iterable[Mapping]]) -> dict:
    """D2 baseline-only outer selection, tied raw then log then rank."""

    selected = {}
    for transform in PRIOR_TRANSFORMS:
        if transform not in trials_by_transform:
            raise RuntimeError(f"Missing baseline trials for prior transform {transform!r}.")
        selected[transform] = select_shared_config(trials_by_transform[transform])
    winner = PRIOR_TRANSFORMS[0]
    for transform in PRIOR_TRANSFORMS[1:]:
        if (
            selected[transform]["mean_best_validation_mrr"]
            > selected[winner]["mean_best_validation_mrr"] + GRID_TIE_EPSILON
        ):
            winner = transform
    return {
        "selected_prior_transform": winner,
        "selected_baseline": selected[winner],
        "all_transforms": selected,
        "tie_epsilon": GRID_TIE_EPSILON,
        "tie_order": list(PRIOR_TRANSFORMS),
    }


def trainable_parameter_count(input_dim: int, hidden_width: int) -> int:
    """Parameter count for one hidden affine layer and one affine output."""

    return input_dim * hidden_width + hidden_width + hidden_width + 1


def assert_d3_capacity_match() -> dict:
    from rerank.model import RankerMLP

    baseline_model = RankerMLP(
        input_dim=4, hidden_dims=[48], dropout=0.0, use_batch_norm=False
    )
    augmented_model = RankerMLP(
        input_dim=7, hidden_dims=[32], dropout=0.0, use_batch_norm=False
    )
    baseline = sum(parameter.numel() for parameter in baseline_model.parameters())
    augmented = sum(parameter.numel() for parameter in augmented_model.parameters())
    if baseline != trainable_parameter_count(4, 48):
        raise AssertionError("D3 baseline model no longer matches the declared architecture.")
    if augmented != trainable_parameter_count(7, 32):
        raise AssertionError("D3 augmented model no longer matches the declared architecture.")
    if baseline != 289 or augmented != 289 or baseline != augmented:
        raise AssertionError(
            f"D3 capacity mismatch: baseline={baseline}, augmented={augmented}."
        )
    return {
        "baseline": {"input_dim": 4, "hidden_width": 48, "parameters": baseline},
        "augmented": {"input_dim": 7, "hidden_width": 32, "parameters": augmented},
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def validation_mrr(
    model,
    normalizer,
    eval_pwc: Sequence,
    eval_features: Sequence[np.ndarray],
    ground_truths: Sequence[str],
    device: str,
) -> float:
    """Official-validation conditional MRR with stable candidate-score ties."""

    prepared = prepare_validation(
        normalizer, eval_pwc, eval_features, ground_truths
    )
    return validation_mrr_prepared(model, prepared, device)


def prepare_validation(
    normalizer,
    eval_pwc: Sequence,
    eval_features: Sequence[np.ndarray],
    ground_truths: Sequence[str],
) -> PreparedValidation:
    """Normalize and canonical-match the fixed validation partition once."""

    from rerank.evaluate import _is_match

    if not (len(eval_pwc) == len(eval_features) == len(ground_truths)):
        raise ValueError("Validation products, features, and ground truths are misaligned.")
    normalized_parts = []
    offsets = [0]
    match_masks = []
    for (_, candidates), features, ground_truth in zip(
        eval_pwc, eval_features, ground_truths
    ):
        matrix = np.asarray(features, dtype=np.float32)
        if len(matrix) != len(candidates):
            raise ValueError("A validation feature matrix is not candidate-aligned.")
        normalized_parts.append(normalizer.transform(matrix))
        match_masks.append(
            np.asarray(
                [_is_match(str(candidate["smiles"]), ground_truth) for candidate in candidates],
                dtype=bool,
            )
        )
        offsets.append(offsets[-1] + len(candidates))
    if not normalized_parts:
        raise ValueError("Official validation partition has no covered reactions.")
    if offsets[-1] == 0:
        raise ValueError("Official validation partition has no candidate rows.")
    return PreparedValidation(
        features=np.concatenate(normalized_parts, axis=0).astype(np.float32, copy=False),
        offsets=tuple(offsets),
        match_masks=tuple(match_masks),
    )


def validation_mrr_prepared(
    model, prepared: PreparedValidation, device: str, batch_size: int = 65_536
) -> float:
    """Score validation rows in large batches, then rank within each reaction."""

    import torch

    if batch_size < 1:
        raise ValueError("Validation batch size must be positive.")
    score_parts = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prepared.features), batch_size):
            tensor = torch.as_tensor(
                prepared.features[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            score_parts.append(model.score(tensor).detach().cpu().numpy())
    scores = np.concatenate(score_parts)
    reciprocal_ranks = []
    for index, matches in enumerate(prepared.match_masks):
        start, stop = prepared.offsets[index : index + 2]
        order = np.argsort(-scores[start:stop], kind="stable")
        ranked_matches = matches[order]
        positions = np.flatnonzero(ranked_matches)
        reciprocal_ranks.append(0.0 if len(positions) == 0 else 1.0 / (positions[0] + 1))
    return float(np.mean(reciprocal_ranks))


def train_validation_trial(
    selection_cache: Mapping,
    config: GridConfig,
    seed: int,
    device: str,
    checkpoint_path: str | Path,
    normalizer_path: str | Path,
    *,
    max_epochs: int = MAX_EPOCHS,
    patience: int = PATIENCE,
    min_improvement: float = MIN_IMPROVEMENT,
    hidden_width_override: int | None = None,
    allowed_seeds: Sequence[int] = SEEDS,
) -> dict:
    """Fit one trial, selecting its earliest checkpoint by validation MRR."""

    import torch
    from rerank.features import fit_normalizer_from_dataset
    from rerank.loss import PairwiseRankingLoss
    from rerank.model import RankerMLP
    from rerank.study_data import make_pairwise_dataset

    allowed_seeds = tuple(int(value) for value in allowed_seeds)
    if not allowed_seeds or len(set(allowed_seeds)) != len(allowed_seeds):
        raise ValueError("Trial seed allowlist must be non-empty and unique.")
    if seed not in allowed_seeds:
        raise ValueError(
            f"Seed {seed} is outside the explicit trial seed allowlist "
            f"{allowed_seeds}."
        )
    dataset = make_pairwise_dataset(
        selection_cache,
        seed=seed,
        max_neg_per_pos=5,
        negative_mining="random",
    )
    normalizer = fit_normalizer_from_dataset(dataset)
    Path(normalizer_path).parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(str(normalizer_path))
    # The sharded runner owns the single user-facing trial progress bar.
    # Suppress this 150k-pair inner bar so remote/Vast logs stay compact.
    dataset.apply_normalizer(normalizer, show_progress=False)

    seed_everything(seed)
    width = config.hidden_width if hidden_width_override is None else hidden_width_override
    model = RankerMLP(
        input_dim=dataset.feature_dim,
        hidden_dims=[width],
        dropout=config.dropout,
        use_batch_norm=False,
    ).to(device)
    criterion = PairwiseRankingLoss(margin=config.margin, reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(max_epochs - 5, 1), eta_min=config.learning_rate * 0.01
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    pos = dataset._tensor_pos
    neg = dataset._tensor_neg
    if pos is None or neg is None:
        raise AssertionError("Normalized pair tensors were not materialized.")
    prepared_validation = prepare_validation(
        normalizer,
        selection_cache["eval_pwc"],
        selection_cache["eval_features"],
        selection_cache["eval_ground_truths"],
    )

    best_mrr = float("-inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(dataset), generator=generator)
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(dataset), 256):
            indices = order[start : start + 256]
            x_pos = pos[indices].to(device)
            x_neg = neg[indices].to(device)
            optimizer.zero_grad()
            loss = criterion(model.score(x_pos), model.score(x_neg))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1

        mrr = validation_mrr_prepared(model, prepared_validation, device)
        improved = best_epoch == 0 or mrr > best_mrr + min_improvement
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(batches, 1),
                "validation_mrr": mrr,
                "selected_improvement": improved,
            }
        )
        if improved:
            best_mrr = mrr
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
        if epoch > 5:
            scheduler.step()
        if epochs_no_improve >= patience:
            break
    return {
        "best_validation_mrr": best_mrr,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < max_epochs,
        "history": history,
        "n_train_pairs": len(dataset),
        "n_validation_reactions": len(selection_cache["eval_pwc"]),
        "n_validation_candidate_rows": len(prepared_validation.features),
        "checkpoint_sha256": "sha256:" + file_sha256(checkpoint_path),
        "normalizer_sha256": "sha256:" + file_sha256(normalizer_path),
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
