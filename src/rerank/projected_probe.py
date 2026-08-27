"""Utilities for the prespecified C2 pooled-embedding upper-bound probe.

The probe deliberately keeps the four frozen 2D/base columns and replaces the
three pair-level Uni-Mol scalars with two learned 512 -> 16 linear projections:
one for the product and one for the aggregate reactants.  It is an upper-bound
probe, not a parameter-matched encoder control.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rerank.revision_tuning import BASELINE_COLUMNS


PROTOCOL_ID = "cap10-tuned-unimol-projected32-v1"
CONTROL_ID = "C-PROJECTED"
SCHEMA_VERSION = 1
POOLED_DIM = 512
PROJECTION_DIM = 16
BASE_DIM = 4
HEAD_INPUT_DIM = BASE_DIM + 2 * PROJECTION_DIM
RAW_INPUT_DIM = BASE_DIM + 2 * POOLED_DIM


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_fingerprint(record: Mapping) -> str:
    encoded = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def selected_base_features(matrix: np.ndarray) -> np.ndarray:
    source = np.asarray(matrix, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 7:
        raise ValueError(f"Expected candidate-aligned seven-column features, got {source.shape}.")
    result = source[:, BASELINE_COLUMNS].copy()
    if not np.isfinite(result).all():
        raise ValueError("Base features contain non-finite values.")
    return result


class ProjectedRanker:
    """Factory namespace kept importable when torch is unavailable."""

    @staticmethod
    def build(hidden_width: int, dropout: float):
        import torch
        from torch import nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.product_projection = nn.Linear(POOLED_DIM, PROJECTION_DIM)
                self.reactant_projection = nn.Linear(POOLED_DIM, PROJECTION_DIM)
                layers: list[nn.Module] = [
                    nn.Linear(HEAD_INPUT_DIM, hidden_width),
                    nn.ReLU(),
                ]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden_width, 1))
                self.head = nn.Sequential(*layers)
                self._initialize()

            def _initialize(self) -> None:
                for module in self.modules():
                    if isinstance(module, nn.Linear):
                        if module.out_features == 1:
                            nn.init.xavier_uniform_(module.weight)
                        else:
                            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                        nn.init.zeros_(module.bias)

            def forward(self, values):
                if values.ndim == 1:
                    values = values.unsqueeze(0)
                if values.shape[-1] != RAW_INPUT_DIM:
                    raise ValueError(
                        f"C2 expects {RAW_INPUT_DIM} normalized raw inputs, got {values.shape[-1]}."
                    )
                base = values[:, :BASE_DIM]
                product = values[:, BASE_DIM : BASE_DIM + POOLED_DIM]
                reactant = values[:, BASE_DIM + POOLED_DIM :]
                merged = torch.cat(
                    [base, self.product_projection(product), self.reactant_projection(reactant)],
                    dim=1,
                )
                return self.head(merged)

            def score(self, values):
                return self.forward(values).squeeze(-1)

        return _Model()


def expected_parameter_count(hidden_width: int = 128) -> int:
    projections = 2 * (POOLED_DIM * PROJECTION_DIM + PROJECTION_DIM)
    head = HEAD_INPUT_DIM * hidden_width + hidden_width + hidden_width + 1
    return projections + head


@dataclass
class ProjectionNormalizer:
    mean: np.ndarray
    std: np.ndarray
    clip_sigma: float = 5.0

    def __post_init__(self) -> None:
        self.mean = np.asarray(self.mean, dtype=np.float32)
        self.std = np.asarray(self.std, dtype=np.float32)
        if self.mean.shape != (RAW_INPUT_DIM,) or self.std.shape != (RAW_INPUT_DIM,):
            raise ValueError("C2 normalizer has the wrong dimensionality.")
        if np.any(self.std < 1e-6) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all():
            raise ValueError("C2 normalizer is invalid.")

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        return np.clip((array - self.mean) / self.std, -self.clip_sigma, self.clip_sigma)

    def save(self, path: str | Path) -> None:
        np.savez(path, mean=self.mean, std=self.std, clip_sigma=[self.clip_sigma])

    @classmethod
    def load(cls, path: str | Path) -> "ProjectionNormalizer":
        blob = np.load(path)
        return cls(blob["mean"], blob["std"], float(blob["clip_sigma"][0]))


def validate_projected_selection(bundle: Mapping, *, require_complete: bool = True) -> None:
    if bundle.get("schema_version") != SCHEMA_VERSION or bundle.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("Unsupported C2 selection cache.")
    if bundle.get("control_id") != CONTROL_ID:
        raise ValueError("C2 control identity is missing.")
    if bundle.get("test_partition_loaded") is not False:
        raise PermissionError("C2 selection cache must never contain test data.")
    if require_complete and bundle.get("status") != "complete":
        raise ValueError("A partial C2 cache cannot enter scientific fitting.")
    train = bundle.get("train_products")
    validation = bundle.get("validation_payload")
    if not isinstance(train, Sequence) or not train or not isinstance(validation, Mapping):
        raise ValueError("C2 selection cache is empty or incomplete.")
    for item in train:
        _validate_product(item)
    required = {
        "eval_pwc", "eval_ground_truths", "eval_metadata", "base_features",
        "product_embeddings", "reactant_embeddings",
    }
    if not required.issubset(validation):
        raise ValueError("C2 validation payload is incomplete.")
    lengths = {len(validation[key]) for key in ("eval_pwc", "eval_ground_truths", "eval_metadata")}
    lengths.add(len(validation["base_features"]))
    lengths.add(len(validation["product_embeddings"]))
    lengths.add(len(validation["reactant_embeddings"]))
    if len(lengths) != 1:
        raise ValueError("C2 validation payload is misaligned.")
    if any(str(row.get("source_split")) != "valid" for row in validation["eval_metadata"]):
        raise PermissionError("C2 selection contains a non-validation reaction.")
    for index, ((_, candidates), base, product, reactants) in enumerate(zip(
        validation["eval_pwc"], validation["base_features"],
        validation["product_embeddings"], validation["reactant_embeddings"]
    )):
        if np.asarray(base).shape != (len(candidates), BASE_DIM):
            raise ValueError(f"Validation base rows are misaligned at reaction {index}.")
        if np.asarray(product).shape != (POOLED_DIM,) or np.asarray(reactants).shape != (len(candidates), POOLED_DIM):
            raise ValueError(f"Validation pooled rows are misaligned at reaction {index}.")


def _validate_product(item: Mapping) -> None:
    candidates = item.get("candidates", ())
    base = np.asarray(item.get("base_features"), dtype=np.float32)
    product = np.asarray(item.get("product_embedding"), dtype=np.float32)
    reactants = np.asarray(item.get("reactant_embeddings"), dtype=np.float32)
    if base.shape != (len(candidates), BASE_DIM):
        raise ValueError("C2 training base features are candidate-misaligned.")
    if product.shape != (POOLED_DIM,) or reactants.shape != (len(candidates), POOLED_DIM):
        raise ValueError("C2 training pooled embeddings are candidate-misaligned.")
    positives = tuple(int(value) for value in item.get("positive_indices", ()))
    negatives = tuple(int(value) for value in item.get("negative_indices", ()))
    if not positives or not negatives or set(positives).intersection(negatives):
        raise ValueError("C2 training labels are invalid.")


def make_pair_indices(train_products: Sequence[Mapping], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match D1 seeded random negative sampling without materializing features."""

    rng = random.Random(int(seed))
    product_indices: list[int] = []
    positive_indices: list[int] = []
    negative_indices: list[int] = []
    for product_index, item in enumerate(train_products):
        negatives = list(item["negative_indices"])
        for positive in item["positive_indices"]:
            selected = negatives
            if len(negatives) > 5:
                selected = rng.sample(negatives, 5)
            for negative in selected:
                product_indices.append(product_index)
                positive_indices.append(int(positive))
                negative_indices.append(int(negative))
    if not product_indices:
        raise ValueError("No C2 pairwise training samples could be constructed.")
    return (
        np.asarray(product_indices, dtype=np.int32),
        np.asarray(positive_indices, dtype=np.int16),
        np.asarray(negative_indices, dtype=np.int16),
    )


def materialize_rows(
    train_products: Sequence[Mapping], product_indices: np.ndarray, candidate_indices: np.ndarray
) -> np.ndarray:
    if product_indices.shape != candidate_indices.shape:
        raise ValueError("C2 row indices are misaligned.")
    result = np.empty((len(product_indices), RAW_INPUT_DIM), dtype=np.float32)
    for position, (product_index, candidate_index) in enumerate(
        zip(product_indices.tolist(), candidate_indices.tolist())
    ):
        item = train_products[product_index]
        result[position, :BASE_DIM] = item["base_features"][candidate_index]
        result[position, BASE_DIM : BASE_DIM + POOLED_DIM] = item["product_embedding"]
        result[position, BASE_DIM + POOLED_DIM :] = item["reactant_embeddings"][candidate_index]
    return result


def fit_pair_normalizer(positive: np.ndarray, negative: np.ndarray) -> ProjectionNormalizer:
    if positive.shape != negative.shape or positive.ndim != 2 or positive.shape[1] != RAW_INPUT_DIM:
        raise ValueError("C2 positive/negative matrices are misaligned.")
    count = 2 * len(positive)
    sums = positive.sum(axis=0, dtype=np.float64) + negative.sum(axis=0, dtype=np.float64)
    squares = np.square(positive, dtype=np.float64).sum(axis=0) + np.square(
        negative, dtype=np.float64
    ).sum(axis=0)
    mean = sums / count
    variance = np.maximum(squares / count - mean * mean, 0.0)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return ProjectionNormalizer(mean.astype(np.float32), std.astype(np.float32))


def materialize_validation(validation: Mapping) -> tuple[np.ndarray, tuple[int, ...]]:
    lengths = [len(candidates) for _, candidates in validation["eval_pwc"]]
    offsets = tuple(np.cumsum([0] + lengths, dtype=np.int64).tolist())
    result = np.empty((offsets[-1], RAW_INPUT_DIM), dtype=np.float32)
    for index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        result[start:stop, :BASE_DIM] = validation["base_features"][index]
        result[start:stop, BASE_DIM : BASE_DIM + POOLED_DIM] = validation["product_embeddings"][index]
        result[start:stop, BASE_DIM + POOLED_DIM :] = validation["reactant_embeddings"][index]
    return result, offsets

