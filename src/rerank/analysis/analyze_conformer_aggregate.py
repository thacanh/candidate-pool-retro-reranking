#!/usr/bin/env python
"""Aggregate the prespecified WS-B conformer analyses without regenerating embeddings.

The runner consumes only completed, checksummed ``seed_42`` through ``seed_51``
folders.  It implements B1 feature stability, B2's crossed 5 x 5 analysis, and
B3's arithmetic mean of the three already-computed pair-level representation
features.  It never averages atom embeddings.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import subprocess
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

from rerank.study_data import STUDY_CACHE_SCHEMA


ALL_CONFORMER_SEEDS = tuple(range(42, 52))
CONFIRMATORY_CONFORMER_SEEDS = tuple(range(42, 47))
TRAINING_SEEDS = tuple(range(42, 47))
SCALAR_NAMES = (
    "atom_set_similarity",
    "reaction_distance",
    "cosine_reaction_vec",
)
SCALAR_COLUMNS = (2, 3, 4)
NON_SCALAR_COLUMNS = (0, 1, 5, 6)
FEATURE_DIMENSION = 7
BOOTSTRAP_SEED = 2026
BOOTSTRAP_SAMPLES = 10_000
CV_MEAN_EPSILON = 1e-8
QUANTILE_PROBABILITIES = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
FEATURE_CACHE_NAME = f"official_3d_prior_schema{STUDY_CACHE_SCHEMA}.pkl"
VALIDATION_FEATURE_CACHE_NAME = (
    f"official_valid_3d_prior_schema{STUDY_CACHE_SCHEMA}.pkl"
)


def _sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_load(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}.")
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None) -> None:
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer CSV columns for empty output {path}.")
        fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _safe_checksum_path(seed_root: Path, relative: str) -> Path:
    candidate = (seed_root / Path(relative)).resolve()
    try:
        candidate.relative_to(seed_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Checksum path escapes seed folder: {relative!r}.") from exc
    return candidate


def verify_seed_run(seed_root: str | Path, expected_seed: int) -> dict:
    """Fail closed unless a completed seed folder and every retained checksum pass."""
    seed_root = Path(seed_root).resolve()
    checksum_path = seed_root / "checksums.sha256"
    required_top_level = {
        "COMPLETED.json",
        "manifest.json",
        "result_summary.json",
        "checksums.sha256",
    }
    missing_top_level = sorted(
        name for name in required_top_level if not (seed_root / name).is_file()
    )
    if missing_top_level:
        raise RuntimeError(
            f"Seed {expected_seed} is incomplete; missing {missing_top_level}."
        )

    checksum_entries: dict[str, str] = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise RuntimeError(
                f"Malformed checksum line {line_number} in {checksum_path}."
            ) from exc
        relative = Path(relative).as_posix()
        if relative in checksum_entries:
            raise RuntimeError(f"Duplicate checksum entry {relative!r}.")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RuntimeError(f"Invalid SHA-256 digest for {relative!r}.")
        target = _safe_checksum_path(seed_root, relative)
        if not target.is_file():
            raise RuntimeError(f"Missing checksummed artifact: {target}.")
        observed = _sha256(target)
        if observed != digest:
            raise RuntimeError(
                f"Checksum mismatch for {target}: {observed} != {digest}."
            )
        checksum_entries[relative] = digest

    critical = {
        "manifest.json",
        "result_summary.json",
        "embedding_summary.json",
        "embedding_non_ok.csv",
        f"features/{FEATURE_CACHE_NAME}",
        f"features/{VALIDATION_FEATURE_CACHE_NAME}",
        "ranking_legacy_fixed50/manifest.json",
        "ranking_legacy_fixed50/per_seed_metrics.json",
        *{
            f"ranking_legacy_fixed50/eval_seed{seed}.csv"
            for seed in TRAINING_SEEDS
        },
    }
    omitted = sorted(critical.difference(checksum_entries))
    if omitted:
        raise RuntimeError(
            f"Seed {expected_seed} checksum manifest omitted critical artifacts: {omitted}."
        )

    completed = _json_load(seed_root / "COMPLETED.json")
    manifest = _json_load(seed_root / "manifest.json")
    result_summary = _json_load(seed_root / "result_summary.json")
    inner_manifest = _json_load(seed_root / "ranking_legacy_fixed50" / "manifest.json")
    if completed.get("status") != "complete":
        raise RuntimeError(f"Seed {expected_seed} is not marked complete.")
    identities = {
        int(completed.get("seed", -1)),
        int(manifest.get("conformer_seed", -1)),
        int(result_summary.get("conformer_seed", -1)),
    }
    if identities != {int(expected_seed)}:
        raise RuntimeError(
            f"Conformer seed identity mismatch in {seed_root}: {sorted(identities)}."
        )
    if manifest.get("protocol_id") != "legacy-cap10-fixed50-v1":
        raise RuntimeError(
            f"Unexpected protocol in seed {expected_seed}: {manifest.get('protocol_id')!r}."
        )
    if manifest.get("training_seeds") != list(TRAINING_SEEDS):
        raise RuntimeError(f"Seed {expected_seed} does not contain training seeds 42--46.")
    if inner_manifest.get("feature_mode") != "3d+prior":
        raise RuntimeError(f"Seed {expected_seed} inner ranking arm is not 3d+prior.")
    if inner_manifest.get("seeds") != list(TRAINING_SEEDS):
        raise RuntimeError(f"Seed {expected_seed} inner ranking seeds are not 42--46.")
    if completed.get("manifest_sha256") != _sha256(seed_root / "manifest.json"):
        raise RuntimeError(f"COMPLETED.json manifest hash mismatch for seed {expected_seed}.")
    if completed.get("result_summary_sha256") != _sha256(
        seed_root / "result_summary.json"
    ):
        raise RuntimeError(
            f"COMPLETED.json result-summary hash mismatch for seed {expected_seed}."
        )
    return {
        "seed": expected_seed,
        "seed_root": seed_root,
        "manifest": manifest,
        "result_summary": result_summary,
        "checksums": checksum_entries,
        "feature_cache": seed_root / "features" / FEATURE_CACHE_NAME,
        "validation_features": seed_root
        / "features"
        / VALIDATION_FEATURE_CACHE_NAME,
        "metrics": seed_root / "ranking_legacy_fixed50" / "per_seed_metrics.json",
        "ranking": seed_root / "ranking_legacy_fixed50",
        "embedding_summary": seed_root / "embedding_summary.json",
        "embedding_non_ok": seed_root / "embedding_non_ok.csv",
    }


def verify_cross_seed_inputs(records: Sequence[dict]) -> dict:
    """Require scientific input hashes to agree across all conformer folders."""
    if not records:
        raise ValueError("No seed records were supplied.")
    reference: dict[str, tuple[int, str]] = {}
    for record in records:
        fingerprints = record["manifest"].get("input_fingerprints")
        if not isinstance(fingerprints, dict):
            raise RuntimeError(f"Seed {record['seed']} has no input fingerprints.")
        current: dict[str, tuple[int, str]] = {}
        for name, value in fingerprints.items():
            if not isinstance(value, dict) or "sha256" not in value:
                raise RuntimeError(
                    f"Seed {record['seed']} fingerprint {name!r} has no SHA-256."
                )
            current[name] = (int(value.get("size_bytes", -1)), str(value["sha256"]))
        if not reference:
            reference = current
        elif current != reference:
            raise RuntimeError(
                f"Scientific input fingerprints differ for conformer seed {record['seed']}."
            )
    return {
        name: {"size_bytes": size, "sha256": digest}
        for name, (size, digest) in reference.items()
    }


def audit_embedding_fallbacks(records: Sequence[dict]) -> dict:
    """Preserve every package fallback as an indexed conformer observation."""
    per_seed: dict[str, dict] = {}
    reference_required: tuple[int, str] | None = None
    for record in records:
        seed = int(record["seed"])
        summary = _json_load(record["embedding_summary"])
        if int(summary.get("conformer_seed", -1)) != seed or not summary.get("complete"):
            raise RuntimeError(f"Embedding summary is incomplete or mislabeled for seed {seed}.")
        required = (
            int(summary.get("required_items", -1)),
            str(summary.get("required_keys_sha256", "")),
        )
        if required[0] < 1 or len(required[1]) != 64:
            raise RuntimeError(f"Invalid required-molecule audit for seed {seed}.")
        if reference_required is None:
            reference_required = required
        elif required != reference_required:
            raise RuntimeError(f"Required molecule inventory differs for seed {seed}.")
        status_counts = {
            str(name): int(count)
            for name, count in summary.get("status_counts", {}).items()
        }
        if sum(status_counts.values()) != int(summary.get("stored_items", -1)):
            raise RuntimeError(f"Embedding status counts do not sum to stored items for seed {seed}.")
        with open(record["embedding_non_ok"], encoding="utf-8", newline="") as handle:
            non_ok_rows = list(csv.DictReader(handle))
        expected_non_ok = sum(
            count for status, count in status_counts.items() if status != "ok"
        )
        if len(non_ok_rows) != expected_non_ok:
            raise RuntimeError(
                f"Non-OK embedding row count mismatch for seed {seed}: "
                f"{len(non_ok_rows)} != {expected_non_ok}."
            )
        per_seed[str(seed)] = {
            "conformer_label": f"C{seed - 41}",
            "required_items": required[0],
            "status_counts": status_counts,
            "null_embedding_items": int(summary.get("null_embedding_items", -1)),
            "non_ok_rows": len(non_ok_rows),
            "fallback_records_retained_as_replicate": True,
        }
    assert reference_required is not None
    return {
        "required_items_per_seed": reference_required[0],
        "required_keys_sha256": reference_required[1],
        "per_seed": per_seed,
        "rule": "package fallback is retained at its conformer index; no replicate omitted",
    }


def _load_feature_blob(path: Path, expected_seed: int, validation: bool = False) -> dict:
    with open(path, "rb") as handle:
        blob = pickle.load(handle)
    if (
        not isinstance(blob, dict)
        or int(blob.get("schema_version", -1)) != STUDY_CACHE_SCHEMA
    ):
        raise RuntimeError(f"Unsupported feature-cache schema in {path}.")
    if blob.get("feature_mode", "3d+prior") != "3d+prior":
        raise RuntimeError(f"Feature cache is not 3d+prior: {path}.")
    if validation and int(blob.get("conformer_seed", -1)) != expected_seed:
        raise RuntimeError(f"Validation feature seed mismatch in {path}.")
    if not isinstance(blob.get("payload"), dict):
        raise RuntimeError(f"Feature cache has no payload: {path}.")
    return blob


def _canonical_product(smiles: str) -> str:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise RuntimeError(f"Cannot canonicalize product SMILES {smiles!r}.")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _canonical_candidate(smiles: str) -> str:
    fragments = [_canonical_product(part) for part in str(smiles).split(".") if part]
    if not fragments:
        raise RuntimeError(f"Empty candidate SMILES {smiles!r}.")
    return ".".join(sorted(fragments))


@lru_cache(maxsize=None)
def _strict_rotatable_bonds(smiles: str) -> int:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise RuntimeError(f"Cannot calculate rotatable bonds for {smiles!r}.")
    return int(rdMolDescriptors.CalcNumRotatableBonds(molecule, strict=True))


def _stratum(product_smiles: str, candidate_smiles: str) -> tuple[str, int]:
    maximum = max(
        _strict_rotatable_bonds(product_smiles),
        _strict_rotatable_bonds(candidate_smiles),
    )
    if maximum == 0:
        return "rigid_zero", maximum
    if maximum < 5:
        return "intermediate_1_4", maximum
    return "flexible_ge5", maximum


def _validated_array(value, location: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != FEATURE_DIMENSION:
        raise RuntimeError(f"Unexpected feature shape {array.shape} at {location}.")
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite feature value at {location}.")
    return array


def _iter_segments(main_blob: dict, validation_blob: dict) -> Iterator[dict]:
    payload = main_blob["payload"]
    train = payload.get("train_products", [])
    for index, product in enumerate(train):
        array = _validated_array(product.get("features"), f"train[{index}]")
        candidates = product.get("candidates", [])
        if len(candidates) != len(array):
            raise RuntimeError(f"Candidate/feature mismatch at train[{index}].")
        yield {
            "segment": f"train:{index}",
            "split": "train",
            "product_identity": str(product.get("product_key")),
            "product_smiles": str(product.get("product_smiles")),
            "candidates": candidates,
            "array": array,
            "metadata": {
                "positive_indices": product.get("positive_indices"),
                "negative_indices": product.get("negative_indices"),
            },
        }

    def evaluation_segments(blob: dict, split: str) -> Iterator[dict]:
        eval_payload = blob["payload"]
        pwc = eval_payload.get("eval_pwc", [])
        features = eval_payload.get("eval_features", [])
        metadata = eval_payload.get("eval_metadata", [{} for _ in pwc])
        ground_truths = eval_payload.get("eval_ground_truths", [None for _ in pwc])
        if not (len(pwc) == len(features) == len(metadata) == len(ground_truths)):
            raise RuntimeError(f"Misaligned {split} evaluation payload.")
        for index, ((product_smiles, candidates), value, meta, ground_truth) in enumerate(
            zip(pwc, features, metadata, ground_truths)
        ):
            array = _validated_array(value, f"{split}[{index}]")
            if len(candidates) != len(array):
                raise RuntimeError(f"Candidate/feature mismatch at {split}[{index}].")
            product_identity = _canonical_product(str(product_smiles))
            yield {
                "segment": f"{split}:{index}",
                "split": split,
                "product_identity": product_identity,
                "product_smiles": str(product_smiles),
                "candidates": candidates,
                "array": array,
                "metadata": {"reaction": meta, "ground_truth": ground_truth},
            }

    yield from evaluation_segments(main_blob, "test")
    yield from evaluation_segments(validation_blob, "valid")


def _candidate_record(candidate: dict) -> tuple[str, float, str]:
    if not isinstance(candidate, dict) or "smiles" not in candidate or "prior" not in candidate:
        raise RuntimeError(f"Malformed candidate record: {candidate!r}.")
    smiles = str(candidate["smiles"])
    canonical = str(candidate.get("canonical_smiles") or _canonical_candidate(smiles))
    prior = float(candidate["prior"])
    if not math.isfinite(prior):
        raise RuntimeError(f"Non-finite candidate prior for {smiles!r}.")
    return smiles, prior, canonical


def feature_structure_digest(main_blob: dict, validation_blob: dict) -> str:
    """Fingerprint alignment, identities, priors and all non-Uni-Mol columns."""
    digest = hashlib.sha256()
    for segment in _iter_segments(main_blob, validation_blob):
        header = {
            "segment": segment["segment"],
            "split": segment["split"],
            "product_identity": segment["product_identity"],
            "product_smiles": segment["product_smiles"],
            "metadata": segment["metadata"],
        }
        digest.update(json.dumps(header, sort_keys=True, default=str).encode("utf-8"))
        array = segment["array"]
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(
            np.ascontiguousarray(array[:, NON_SCALAR_COLUMNS], dtype="<f4").tobytes()
        )
        for candidate in segment["candidates"]:
            digest.update(
                json.dumps(_candidate_record(candidate), separators=(",", ":")).encode(
                    "utf-8"
                )
            )
    return digest.hexdigest()


def _feature_array_refs(main_blob: dict, validation_blob: dict) -> list[np.ndarray]:
    return [segment["array"] for segment in _iter_segments(main_blob, validation_blob)]


def build_unique_pair_index(main_blob: dict, validation_blob: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened first-occurrence indices and prespecified strata metadata."""
    seen: set[tuple[str, str, str]] = set()
    indices: list[int] = []
    strata: list[str] = []
    offset = 0
    for segment in _iter_segments(main_blob, validation_blob):
        product_identity = segment["product_identity"]
        for row_index, candidate in enumerate(segment["candidates"]):
            candidate_smiles, _, candidate_identity = _candidate_record(candidate)
            key = (segment["split"], product_identity, candidate_identity)
            if key not in seen:
                seen.add(key)
                label, _ = _stratum(
                    segment["product_smiles"], candidate_smiles
                )
                indices.append(offset + row_index)
                strata.append(label)
        offset += len(segment["array"])
    return (
        np.asarray(indices, dtype=np.int64),
        np.asarray(strata, dtype=object),
    )


def flatten_scalar_features(
    main_blob: dict, validation_blob: dict, unique_indices: np.ndarray
) -> np.ndarray:
    blocks = [
        np.asarray(array[:, SCALAR_COLUMNS], dtype=np.float64)
        for array in _feature_array_refs(main_blob, validation_blob)
    ]
    if not blocks:
        raise RuntimeError("Feature artifacts contain no pair-level features.")
    flattened = np.concatenate(blocks, axis=0)
    if len(unique_indices) and int(unique_indices.max()) >= len(flattened):
        raise RuntimeError("Unique-pair index is outside the feature matrix.")
    return flattened[unique_indices]


def _stratum_masks(strata: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("all", np.ones(len(strata), dtype=bool)),
        (
            "rigid_0_4",
            np.isin(strata, ("rigid_zero", "intermediate_1_4")),
        ),
        ("rigid_zero", strata == "rigid_zero"),
        ("intermediate_1_4", strata == "intermediate_1_4"),
        ("flexible_ge5", strata == "flexible_ge5"),
    ]


def _correlation(left: np.ndarray, right: np.ndarray, spearman: bool = False) -> float | None:
    if len(left) < 2:
        return None
    if spearman:
        left = rankdata(left, method="average")
        right = rankdata(right, method="average")
    left_sd = float(np.std(left))
    right_sd = float(np.std(right))
    if left_sd == 0.0 or right_sd == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _distribution(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "min": None,
            "median": None,
            "q90": None,
            "q95": None,
            "q99": None,
            "max": None,
        }
    quantiles = np.quantile(values, [0.0, 0.5, 0.9, 0.95, 0.99, 1.0])
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q90": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
        "max": float(quantiles[5]),
    }


def analyze_b1(
    features_by_seed: dict[int, np.ndarray], strata: np.ndarray
) -> dict[str, list[dict]]:
    if tuple(features_by_seed) != CONFIRMATORY_CONFORMER_SEEDS:
        raise RuntimeError("B1 requires conformer seeds 42--46 in order.")
    shape = next(iter(features_by_seed.values())).shape
    if any(value.shape != shape for value in features_by_seed.values()):
        raise RuntimeError("B1 conformer feature matrices are not aligned.")
    if shape != (len(strata), len(SCALAR_NAMES)):
        raise RuntimeError("B1 strata are not aligned with scalar features.")

    stability_rows: list[dict] = []
    abs_quantile_rows: list[dict] = []
    for left_seed_index, left_seed in enumerate(CONFIRMATORY_CONFORMER_SEEDS):
        for right_seed in CONFIRMATORY_CONFORMER_SEEDS[left_seed_index + 1 :]:
            left = features_by_seed[left_seed]
            right = features_by_seed[right_seed]
            for scalar_index, scalar in enumerate(SCALAR_NAMES):
                for stratum, mask in _stratum_masks(strata):
                    left_values = left[mask, scalar_index]
                    right_values = right[mask, scalar_index]
                    absolute = np.abs(left_values - right_values)
                    stability_rows.append(
                        {
                            "conformer_seed_a": left_seed,
                            "conformer_seed_b": right_seed,
                            "conformer_a": f"C{left_seed - 41}",
                            "conformer_b": f"C{right_seed - 41}",
                            "scalar": scalar,
                            "stratum": stratum,
                            **{f"abs_diff_{key}": value for key, value in _distribution(absolute).items()},
                            "pearson": _correlation(left_values, right_values),
                            "spearman": _correlation(left_values, right_values, spearman=True),
                        }
                    )
                    if len(absolute):
                        quantiles = np.quantile(absolute, QUANTILE_PROBABILITIES)
                        for probability, value in zip(QUANTILE_PROBABILITIES, quantiles):
                            abs_quantile_rows.append(
                                {
                                    "conformer_seed_a": left_seed,
                                    "conformer_seed_b": right_seed,
                                    "scalar": scalar,
                                    "stratum": stratum,
                                    "probability": probability,
                                    "absolute_difference": float(value),
                                }
                            )

    stacked = np.stack(
        [features_by_seed[seed] for seed in CONFIRMATORY_CONFORMER_SEEDS], axis=0
    )
    means = np.mean(stacked, axis=0)
    sample_sd = np.std(stacked, axis=0, ddof=1)
    defined = np.abs(means) > CV_MEAN_EPSILON
    cv = np.full_like(means, np.nan, dtype=np.float64)
    cv[defined] = sample_sd[defined] / np.abs(means[defined])
    cv_rows: list[dict] = []
    cv_quantile_rows: list[dict] = []
    for scalar_index, scalar in enumerate(SCALAR_NAMES):
        for stratum, mask in _stratum_masks(strata):
            local_defined = defined[mask, scalar_index]
            values = cv[mask, scalar_index][local_defined]
            cv_rows.append(
                {
                    "scalar": scalar,
                    "stratum": stratum,
                    "n_pairs": int(mask.sum()),
                    "n_cv_defined": int(local_defined.sum()),
                    "n_cv_undefined_abs_mean_le_1e-8": int((~local_defined).sum()),
                    **{f"cv_{key}": value for key, value in _distribution(values).items()},
                }
            )
            if len(values):
                quantiles = np.quantile(values, QUANTILE_PROBABILITIES)
                for probability, value in zip(QUANTILE_PROBABILITIES, quantiles):
                    cv_quantile_rows.append(
                        {
                            "scalar": scalar,
                            "stratum": stratum,
                            "probability": probability,
                            "coefficient_of_variation": float(value),
                        }
                    )

    strata_rows = [
        {"stratum": label, "n_unique_pairs": int(mask.sum())}
        for label, mask in _stratum_masks(strata)
    ]
    return {
        "stability": stability_rows,
        "abs_quantiles": abs_quantile_rows,
        "cv": cv_rows,
        "cv_quantiles": cv_quantile_rows,
        "strata": strata_rows,
    }


def average_feature_blobs(
    base_main: dict,
    base_validation: dict,
    remaining: Iterable[tuple[int, dict, dict]],
    conformer_seeds: Sequence[int] = ALL_CONFORMER_SEEDS,
) -> tuple[dict, dict]:
    """Average only columns 2--4 of aligned pair-level feature arrays."""
    base_main = copy.deepcopy(base_main)
    base_validation = copy.deepcopy(base_validation)
    base_arrays = _feature_array_refs(base_main, base_validation)
    accumulators = [
        np.asarray(array[:, SCALAR_COLUMNS], dtype=np.float64).copy()
        for array in base_arrays
    ]
    observed_seeds = [int(conformer_seeds[0])]
    reference_digest = feature_structure_digest(base_main, base_validation)
    for seed, main_blob, validation_blob in remaining:
        if feature_structure_digest(main_blob, validation_blob) != reference_digest:
            raise RuntimeError(f"Feature structure/non-scalar mismatch for seed {seed}.")
        arrays = _feature_array_refs(main_blob, validation_blob)
        if len(arrays) != len(base_arrays):
            raise RuntimeError(f"Feature-array count mismatch for seed {seed}.")
        for index, (base, current, accumulator) in enumerate(
            zip(base_arrays, arrays, accumulators)
        ):
            if base.shape != current.shape:
                raise RuntimeError(f"Feature shape mismatch for seed {seed}, block {index}.")
            if not np.array_equal(base[:, NON_SCALAR_COLUMNS], current[:, NON_SCALAR_COLUMNS]):
                raise RuntimeError(
                    f"Non-scalar feature changed for seed {seed}, block {index}."
                )
            accumulator += np.asarray(current[:, SCALAR_COLUMNS], dtype=np.float64)
        observed_seeds.append(int(seed))
    if tuple(observed_seeds) != tuple(conformer_seeds):
        raise RuntimeError(
            f"B3 expected conformer seeds {list(conformer_seeds)}, got {observed_seeds}."
        )
    for array, accumulator in zip(base_arrays, accumulators):
        array[:, SCALAR_COLUMNS] = (accumulator / len(conformer_seeds)).astype(
            array.dtype, copy=False
        )

    aggregation = {
        "experiment_id": "B-AVG10",
        "protocol_id": "cap10-conformer-avg10-features-v1",
        "comparator": "single-conformer 3d+prior feature artifact",
        "single_intended_change": (
            "arithmetic mean of each of the three already-computed pair-level "
            "scalar features across conformer seeds 42--51"
        ),
        "conformer_seeds": list(conformer_seeds),
        "scalar_names": list(SCALAR_NAMES),
        "scalar_columns_zero_based": list(SCALAR_COLUMNS),
        "atom_embeddings_averaged": False,
        "accumulator_dtype": "float64",
        "stored_feature_dtype": str(base_arrays[0].dtype) if base_arrays else None,
    }
    base_main["artifact_kind"] = "official_train_test_conformer_features_avg10"
    base_main["protocol_id"] = aggregation["protocol_id"]
    base_main["representation_aggregation"] = aggregation
    base_validation["artifact_kind"] = "official_validation_conformer_features_avg10"
    base_validation["protocol_id"] = aggregation["protocol_id"]
    base_validation["conformer_seed"] = None
    base_validation["conformer_seeds"] = list(conformer_seeds)
    base_validation["representation_aggregation"] = aggregation
    return base_main, base_validation


def _prediction_signature_update(digest, row: dict, canonical_product: str) -> None:
    fields = (
        row["reaction_id"],
        canonical_product,
        row["ground_truth"],
        row.get("source_split", ""),
        row["baseline_hit@1"],
        row["baseline_rr"],
        row.get("baseline_rank", ""),
        row.get("baseline_candidates_json", ""),
    )
    digest.update(json.dumps(fields, separators=(",", ":")).encode("utf-8"))


def _load_prediction_arm(path: Path) -> dict:
    required = {
        "reaction_id",
        "product_smiles",
        "ground_truth",
        "baseline_hit@1",
        "reranked_hit@1",
        "baseline_rr",
        "reranked_rr",
    }
    reaction_ids: list[str] = []
    clusters: list[str] = []
    reranked_top1: list[float] = []
    reranked_rr: list[float] = []
    digest = hashlib.sha256()
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise RuntimeError(
                f"Prediction CSV {path} is missing {sorted(required.difference(reader.fieldnames or []))}."
            )
        for row in reader:
            reaction_id = str(row["reaction_id"])
            product = _canonical_product(row["product_smiles"])
            baseline_hit = float(row["baseline_hit@1"])
            reranked_hit = float(row["reranked_hit@1"])
            baseline_reciprocal = float(row["baseline_rr"])
            reranked_reciprocal = float(row["reranked_rr"])
            values = (baseline_hit, reranked_hit, baseline_reciprocal, reranked_reciprocal)
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"Non-finite prediction metric in {path}.")
            _prediction_signature_update(digest, row, product)
            reaction_ids.append(reaction_id)
            clusters.append(product)
            reranked_top1.append(reranked_hit)
            reranked_rr.append(reranked_reciprocal)
    if not reaction_ids or len(set(reaction_ids)) != len(reaction_ids):
        raise RuntimeError(f"Prediction CSV is empty or has duplicate reaction IDs: {path}.")
    return {
        "signature": digest.hexdigest(),
        "reaction_ids": reaction_ids,
        "clusters": np.asarray(clusters, dtype=object),
        "reranked_top1": np.asarray(reranked_top1, dtype=np.float64),
        "reranked_mrr": np.asarray(reranked_rr, dtype=np.float64),
    }


def load_paired_prediction_differences(
    baseline_2d_path: Path, augmented_path: Path
) -> dict:
    """Pair two trained arms; the CSV ``baseline_*`` columns are prior-only."""
    baseline = _load_prediction_arm(baseline_2d_path)
    augmented = _load_prediction_arm(augmented_path)
    if (
        baseline["signature"] != augmented["signature"]
        or baseline["reaction_ids"] != augmented["reaction_ids"]
        or not np.array_equal(baseline["clusters"], augmented["clusters"])
    ):
        raise RuntimeError(
            f"2D/augmented reaction or candidate-pool mismatch: "
            f"{baseline_2d_path} versus {augmented_path}."
        )
    return {
        "signature": augmented["signature"],
        "reaction_ids": augmented["reaction_ids"],
        "clusters": augmented["clusters"],
        "top1": augmented["reranked_top1"] - baseline["reranked_top1"],
        "mrr": augmented["reranked_mrr"] - baseline["reranked_mrr"],
        "baseline_top1": float(np.mean(baseline["reranked_top1"])),
        "top1_value": float(np.mean(augmented["reranked_top1"])),
        "baseline_mrr": float(np.mean(baseline["reranked_mrr"])),
        "mrr_value": float(np.mean(augmented["reranked_mrr"])),
    }


def _assert_close(observed: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(observed, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"{label} mismatch: prediction={observed}, metric={expected}.")


def load_b2_grid(
    records: Sequence[dict], shared_2d_root: str | Path
) -> tuple[list[dict], dict[str, np.ndarray], np.ndarray]:
    shared_2d_root = Path(shared_2d_root).resolve()
    completed = _json_load(shared_2d_root / "COMPLETED.json")
    shared_manifest = _json_load(shared_2d_root / "manifest.json")
    shared_ranking = shared_2d_root / "ranking"
    shared_inner = _json_load(shared_ranking / "manifest.json")
    shared_metrics = _json_load(shared_ranking / "per_seed_metrics.json")
    if completed.get("status") != "complete":
        raise RuntimeError("Shared trained 2D arm is not complete.")
    if completed.get("manifest_sha256") != _sha256(shared_2d_root / "manifest.json"):
        raise RuntimeError("Shared 2D COMPLETED.json manifest checksum failed.")
    checksum_entries: dict[str, str] = {}
    for line in (shared_2d_root / "checksums.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = _safe_checksum_path(shared_2d_root, relative)
        if not target.is_file() or _sha256(target) != digest:
            raise RuntimeError(f"Shared 2D checksum failed: {relative}")
        checksum_entries[Path(relative).as_posix()] = digest
    required_shared = {
        "manifest.json", "ranking/manifest.json", "ranking/per_seed_metrics.json",
        *{f"ranking/eval_seed{seed}.csv" for seed in TRAINING_SEEDS},
    }
    if not required_shared.issubset(checksum_entries):
        raise RuntimeError("Shared 2D checksum manifest omits a critical artifact.")
    if shared_manifest.get("protocol_id") != "legacy-cap10-fixed50-v1":
        raise RuntimeError("Shared 2D arm has the wrong protocol ID.")
    if shared_inner.get("feature_mode") != "2d+prior":
        raise RuntimeError("Shared comparator is not the trained 2d+prior arm.")
    expected_seed_keys = {str(seed) for seed in TRAINING_SEEDS}
    if set(shared_metrics) != expected_seed_keys or shared_inner.get("seeds") != list(TRAINING_SEEDS):
        raise RuntimeError("Shared 2D arm does not contain exactly training seeds 42--46.")
    by_seed = {int(record["seed"]): record for record in records}
    rows: list[dict] = []
    top1_cells: list[np.ndarray] = []
    mrr_cells: list[np.ndarray] = []
    reference_signature: str | None = None
    reference_ids: list[str] | None = None
    reference_clusters: np.ndarray | None = None
    for conformer_seed in CONFIRMATORY_CONFORMER_SEEDS:
        record = by_seed[conformer_seed]
        metrics = _json_load(record["metrics"])
        if set(metrics) != {str(seed) for seed in TRAINING_SEEDS}:
            raise RuntimeError(
                f"Metrics for conformer seed {conformer_seed} are not exactly seeds 42--46."
            )
        for training_seed in TRAINING_SEEDS:
            cell = metrics[str(training_seed)]
            prediction = load_paired_prediction_differences(
                shared_ranking / f"eval_seed{training_seed}.csv",
                record["ranking"] / f"eval_seed{training_seed}.csv",
            )
            if reference_signature is None:
                reference_signature = prediction["signature"]
                reference_ids = prediction["reaction_ids"]
                reference_clusters = prediction["clusters"]
            elif (
                prediction["signature"] != reference_signature
                or prediction["reaction_ids"] != reference_ids
                or not np.array_equal(prediction["clusters"], reference_clusters)
            ):
                raise RuntimeError(
                    f"B2 reaction/2D pairing mismatch at conformer={conformer_seed}, "
                    f"training_seed={training_seed}."
                )
            baseline_cell = shared_metrics[str(training_seed)]
            for metric_name in ("top1", "mrr"):
                if metric_name not in cell or metric_name not in baseline_cell:
                    raise RuntimeError(f"Missing trained-arm {metric_name} in B2 metric cell.")
                _assert_close(
                    prediction[f"{metric_name}_value"], float(cell[metric_name]),
                    f"augmented {metric_name} conformer={conformer_seed} training_seed={training_seed}",
                )
                _assert_close(
                    prediction[f"baseline_{metric_name}"], float(baseline_cell[metric_name]),
                    f"2D {metric_name} training_seed={training_seed}",
                )
            top1_cells.append(prediction["top1"])
            mrr_cells.append(prediction["mrr"])
            rows.append(
                {
                    "conformer_seed": conformer_seed,
                    "conformer_label": f"C{conformer_seed - 41}",
                    "training_seed": training_seed,
                    "n_reactions": len(prediction["top1"]),
                    "baseline_top1": float(baseline_cell["top1"]),
                    "augmented_top1": float(cell["top1"]),
                    "top1_gain": float(cell["top1"] - baseline_cell["top1"]),
                    "baseline_mrr": float(baseline_cell["mrr"]),
                    "augmented_mrr": float(cell["mrr"]),
                    "mrr_gain": float(cell["mrr"] - baseline_cell["mrr"]),
                }
            )
    assert reference_clusters is not None
    return (
        rows,
        {
            "top1": np.stack(top1_cells, axis=0),
            "mrr": np.stack(mrr_cells, axis=0),
        },
        reference_clusters,
    )


def fit_crossed_reml(values: np.ndarray) -> dict:
    """Fit y = mean + training seed + conformer + residual by exact REML."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (5, 5):
        raise ValueError("Crossed REML requires a 5 x 5 conformer-by-training grid.")
    y = values.reshape(-1)
    n = len(y)
    x = np.ones((n, 1), dtype=np.float64)
    training = np.tile(np.arange(5), 5)
    conformer = np.repeat(np.arange(5), 5)
    zs = np.eye(5, dtype=np.float64)[training]
    zc = np.eye(5, dtype=np.float64)[conformer]
    ss = zs @ zs.T
    cc = zc @ zc.T
    identity = np.eye(n, dtype=np.float64)

    def objective(log_variances: np.ndarray) -> float:
        if np.any(log_variances < -40.0) or np.any(log_variances > 20.0):
            return 1e100
        seed_var, conformer_var, residual_var = np.exp(log_variances)
        covariance = seed_var * ss + conformer_var * cc + residual_var * identity
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            return 1e100
        try:
            solved_x = np.linalg.solve(covariance, x)
            solved_y = np.linalg.solve(covariance, y)
            xt_vinv_x = x.T @ solved_x
            beta = np.linalg.solve(xt_vinv_x, x.T @ solved_y)
        except np.linalg.LinAlgError:
            return 1e100
        residual = y - x @ beta
        quadratic = float(residual @ np.linalg.solve(covariance, residual))
        _, logdet_fixed = np.linalg.slogdet(xt_vinv_x)
        return 0.5 * (
            (n - 1) * math.log(2.0 * math.pi)
            + logdet
            + float(logdet_fixed)
            + quadratic
        )

    scale = max(float(np.var(y, ddof=1)), 1e-10)
    starts = (
        np.log([scale / 3, scale / 3, scale / 3]),
        np.log([scale * 0.05, scale * 0.05, scale * 0.9]),
        np.log([scale * 0.45, scale * 0.45, scale * 0.1]),
    )
    fits = [
        minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
        )
        for start in starts
    ]
    fit = min(fits, key=lambda result: float(result.fun))
    variances = np.exp(fit.x)
    variances[variances < max(scale * 1e-8, 1e-14)] = 0.0
    total = float(variances.sum())
    components = {
        "training_seed": float(variances[0]),
        "conformer": float(variances[1]),
        "residual_seed_by_conformer": float(variances[2]),
    }
    return {
        "method": "restricted maximum likelihood for crossed random intercepts",
        "mean_gain": float(np.mean(y)),
        "variance_components": components,
        "variance_fractions": {
            name: (value / total if total > 0 else None)
            for name, value in components.items()
        },
        "restricted_negative_log_likelihood": float(fit.fun),
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
    }


def crossed_bootstrap(
    differences: np.ndarray,
    cluster_ids: np.ndarray,
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
    batch_size: int = 128,
) -> tuple[dict, np.ndarray]:
    """Independently resample product clusters, training seeds and conformers."""
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or differences.shape[0] != 25:
        raise ValueError("differences must have shape (25, n_reactions).")
    if len(cluster_ids) != differences.shape[1]:
        raise ValueError("cluster_ids must align with reaction differences.")
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")
    unique_clusters = list(dict.fromkeys(cluster_ids.tolist()))
    cluster_lookup = {cluster: index for index, cluster in enumerate(unique_clusters)}
    cluster_index = np.asarray([cluster_lookup[value] for value in cluster_ids], dtype=np.int64)
    n_clusters = len(unique_clusters)
    cluster_sizes = np.bincount(cluster_index, minlength=n_clusters).astype(np.float64)
    cluster_sums = np.zeros((25, n_clusters), dtype=np.float64)
    for cell in range(25):
        cluster_sums[cell] = np.bincount(
            cluster_index, weights=differences[cell], minlength=n_clusters
        )

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_samples, dtype=np.float64)
    complete = 0
    while complete < n_samples:
        size = min(batch_size, n_samples - complete)
        sampled_clusters = rng.integers(0, n_clusters, size=(size, n_clusters))
        cluster_weights = np.zeros((size, n_clusters), dtype=np.float64)
        np.add.at(
            cluster_weights,
            (np.repeat(np.arange(size), n_clusters), sampled_clusters.reshape(-1)),
            1.0,
        )
        sampled_seed = rng.integers(0, 5, size=(size, 5))
        sampled_conformer = rng.integers(0, 5, size=(size, 5))
        seed_weights = np.stack(
            [np.bincount(row, minlength=5) for row in sampled_seed]
        )
        conformer_weights = np.stack(
            [np.bincount(row, minlength=5) for row in sampled_conformer]
        )
        cell_weights = (
            conformer_weights[:, :, None] * seed_weights[:, None, :]
        ).reshape(size, 25)
        selected_cell_cluster_sums = cluster_weights @ cluster_sums.T
        numerator = np.sum(cell_weights * selected_cell_cluster_sums, axis=1)
        sampled_reactions = cluster_weights @ cluster_sizes
        estimates[complete : complete + size] = numerator / (25.0 * sampled_reactions)
        complete += size
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return (
        {
            "method": (
                "crossed paired bootstrap independently resampling canonical-product "
                "clusters, five training-seed levels, and five conformer levels"
            ),
            "n_samples": int(n_samples),
            "rng_seed": int(seed),
            "n_product_clusters": n_clusters,
            "n_reactions": int(differences.shape[1]),
            "point_estimate": float(np.mean(differences)),
            "ci95": [float(lower), float(upper)],
        },
        estimates,
    )


def analyze_b2(
    grid_rows: list[dict],
    differences: dict[str, np.ndarray],
    cluster_ids: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> tuple[dict, list[dict], list[dict]]:
    results: dict[str, dict] = {}
    variance_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    for metric_index, metric in enumerate(("top1", "mrr")):
        values = np.asarray([row[f"{metric}_gain"] for row in grid_rows]).reshape(5, 5)
        reml = fit_crossed_reml(values)
        bootstrap, estimates = crossed_bootstrap(
            differences[metric],
            cluster_ids,
            n_samples=n_bootstrap,
            seed=bootstrap_seed,
        )
        positive_cells = int(np.sum(values > 0.0))
        results[metric] = {
            "positive_cells": positive_cells,
            "zero_cells": int(np.sum(values == 0.0)),
            "negative_cells": int(np.sum(values < 0.0)),
            "reml": reml,
            "crossed_bootstrap": bootstrap,
        }
        for component, variance in reml["variance_components"].items():
            variance_rows.append(
                {
                    "metric": metric,
                    "component": component,
                    "variance": variance,
                    "fraction": reml["variance_fractions"][component],
                }
            )
        for index, estimate in enumerate(estimates):
            bootstrap_rows.append(
                {
                    "metric": metric,
                    "bootstrap_index": index,
                    "gain": float(estimate),
                }
            )
    top1_ci = results["top1"]["crossed_bootstrap"]["ci95"]
    robust = results["top1"]["positive_cells"] >= 20 and top1_ci[0] > 0.0
    results["top1_robustness_gate"] = {
        "requires_augmented_above_baseline_cells": 20,
        "observed_positive_cells": results["top1"]["positive_cells"],
        "requires_crossed_ci_excludes_zero_on_positive_side": True,
        "observed_ci95": top1_ci,
        "passed": bool(robust),
        "prespecified_label": "robust" if robust else "conformer-sensitive",
    }
    return results, variance_rows, bootstrap_rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _environment() -> dict:
    versions = {}
    for distribution in ("numpy", "scipy", "rdkit"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "b1_stability": output_dir / "b1_pairwise_stability.csv",
        "b1_abs_quantiles": output_dir / "b1_abs_diff_quantiles.csv",
        "b1_cv": output_dir / "b1_cv_summary.csv",
        "b1_cv_quantiles": output_dir / "b1_cv_quantiles.csv",
        "b1_strata": output_dir / "b1_strata_counts.csv",
        "b2_grid": output_dir / "b2_5x5_gain_grid.csv",
        "b2_variance": output_dir / "b2_reml_variance_components.csv",
        "b2_bootstrap": output_dir / "b2_crossed_bootstrap_draws.csv",
        "b2_analysis": output_dir / "b2_analysis.json",
        "b3_main": output_dir / "b3_avg10_train_test_features.pkl",
        "b3_validation": output_dir / "b3_avg10_validation_features.pkl",
        "manifest": output_dir / "manifest.json",
        "checksums": output_dir / "checksums.sha256",
    }


def run_aggregate(
    conformer_root: str | Path,
    output_dir: str | Path,
    shared_2d_root: str | Path = "outputs/jcheminform_revision/shared_2d_legacy_fixed50",
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    overwrite: bool = False,
) -> dict:
    conformer_root = Path(conformer_root).resolve()
    output_dir = Path(output_dir).resolve()
    paths = _output_paths(output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise RuntimeError(
            "Aggregate outputs already exist; choose a new --output-dir or pass "
            "--overwrite for these exact files: " + ", ".join(map(str, existing))
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    started_clock = time.perf_counter()

    records = [
        verify_seed_run(conformer_root / f"seed_{seed}", seed)
        for seed in ALL_CONFORMER_SEEDS
    ]
    scientific_inputs = verify_cross_seed_inputs(records)
    fallback_audit = audit_embedding_fallbacks(records)

    features_by_seed: dict[int, np.ndarray] = {}
    first_main: dict | None = None
    first_validation: dict | None = None
    unique_indices: np.ndarray | None = None
    strata: np.ndarray | None = None
    reference_structure: str | None = None
    input_feature_hashes: dict[str, dict] = {}
    for record in records:
        seed = int(record["seed"])
        main_blob = _load_feature_blob(record["feature_cache"], seed)
        valid_blob = _load_feature_blob(record["validation_features"], seed, validation=True)
        structure = feature_structure_digest(main_blob, valid_blob)
        if reference_structure is None:
            reference_structure = structure
            first_main = main_blob
            first_validation = valid_blob
            unique_indices, strata = build_unique_pair_index(main_blob, valid_blob)
        elif structure != reference_structure:
            raise RuntimeError(f"Feature pairing/non-scalar structure differs for seed {seed}.")
        assert unique_indices is not None
        if seed in CONFIRMATORY_CONFORMER_SEEDS:
            features_by_seed[seed] = flatten_scalar_features(
                main_blob, valid_blob, unique_indices
            )
        input_feature_hashes[str(seed)] = {
            "feature_cache_sha256": record["checksums"][
                record["feature_cache"].relative_to(record["seed_root"]).as_posix()
            ],
            "validation_features_sha256": record["checksums"][
                record["validation_features"].relative_to(record["seed_root"]).as_posix()
            ],
            "structure_digest": structure,
        }

    assert first_main is not None and first_validation is not None and strata is not None
    b1 = analyze_b1(features_by_seed, strata)
    _write_csv(paths["b1_stability"], b1["stability"])
    _write_csv(paths["b1_abs_quantiles"], b1["abs_quantiles"])
    _write_csv(paths["b1_cv"], b1["cv"])
    _write_csv(paths["b1_cv_quantiles"], b1["cv_quantiles"])
    _write_csv(paths["b1_strata"], b1["strata"])

    def remaining_for_average() -> Iterator[tuple[int, dict, dict]]:
        # Reload one seed at a time so ten compact caches are never resident at once.
        for record in records[1:]:
            seed = int(record["seed"])
            yield (
                seed,
                _load_feature_blob(record["feature_cache"], seed),
                _load_feature_blob(record["validation_features"], seed, validation=True),
            )

    averaged_main, averaged_validation = average_feature_blobs(
        first_main,
        first_validation,
        remaining_for_average(),
        conformer_seeds=ALL_CONFORMER_SEEDS,
    )
    _atomic_pickle(paths["b3_main"], averaged_main)
    _atomic_pickle(paths["b3_validation"], averaged_validation)

    grid_rows, differences, clusters = load_b2_grid(records, shared_2d_root)
    b2, variance_rows, bootstrap_rows = analyze_b2(
        grid_rows,
        differences,
        clusters,
        n_bootstrap=n_bootstrap,
        bootstrap_seed=bootstrap_seed,
    )
    _write_csv(paths["b2_grid"], grid_rows)
    _write_csv(paths["b2_variance"], variance_rows)
    _write_csv(paths["b2_bootstrap"], bootstrap_rows)
    _atomic_json(paths["b2_analysis"], b2)

    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_at_utc": started_at,
        "git_commit": _git_commit(),
        "environment": _environment(),
        "workstream": "WS-B conformer aggregate",
        "analyses": {
            "B1": {
                "experiment_id": "B-C1..C5",
                "protocol_id": "cap10-conformer-features-v1",
                "comparator": "same pair-level feature code across C1--C5",
                "single_intended_change": "RDKit conformer seed 42--46",
                "n_unique_pairs": int(len(strata)),
                "cv_undefined_rule": "abs(mean) <= 1e-8; counted, not imputed",
            },
            "B2": {
                "experiment_id": "B-25GRID",
                "protocol_id": "legacy-cap10-fixed50-v1",
                "comparator": "trained four-input 2D-only RankerMLP with matched training seed",
                "single_intended_change": "three Uni-Mol-derived scalar features",
                "training_seeds": list(TRAINING_SEEDS),
                "conformer_seeds": list(CONFIRMATORY_CONFORMER_SEEDS),
                "results": b2,
            },
            "B3": averaged_main["representation_aggregation"],
        },
        "scientific_input_fingerprints": scientific_inputs,
        "seed_feature_artifacts": input_feature_hashes,
        "shared_2d_manifest_sha256": _sha256(Path(shared_2d_root) / "manifest.json"),
        "input_seed_manifest_sha256": {
            str(record["seed"]): record["checksums"]["manifest.json"]
            for record in records
        },
        "pairing_gate": {
            "feature_structure_digest": reference_structure,
            "ranking_reaction_and_2d_alignment": "passed",
        },
        "runtime_seconds": time.perf_counter() - started_clock,
    }
    manifest["analyses"]["B3"]["fallback_audit"] = fallback_audit
    generated = [path for name, path in paths.items() if name not in {"manifest", "checksums"}]
    manifest["generated_artifacts"] = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in generated
    }
    _atomic_json(paths["manifest"], manifest)
    checksummed = [*generated, paths["manifest"]]
    with open(paths["checksums"], "w", encoding="utf-8", newline="\n") as handle:
        for path in sorted(checksummed):
            handle.write(f"{_sha256(path)}  {path.name}\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conformer-root",
        default="outputs/jcheminform_revision/conformers",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/jcheminform_revision/conformer_aggregate",
    )
    parser.add_argument(
        "--shared-2d-root",
        default="outputs/jcheminform_revision/shared_2d_legacy_fixed50",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_aggregate(
        args.conformer_root,
        args.output_dir,
        shared_2d_root=args.shared_2d_root,
        n_bootstrap=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        overwrite=args.overwrite,
    )
    print(json.dumps({"status": "complete", "analyses": list(manifest["analyses"])}, indent=2))


if __name__ == "__main__":
    main()
