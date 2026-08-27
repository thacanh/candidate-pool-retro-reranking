#!/usr/bin/env python
"""Compile scalar WS-C shards into tuned-runner compact cache schemas.

The compiler consumes only seven-column feature shards; atom representations
are never loaded or persisted.  Official train/valid/test labels and reaction
IDs are reconstructed from the verified source and metadata tables.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from rerank.encoder_controls import FULL_FEATURE_NAMES, file_fingerprint, file_sha256
from rerank.study_data import (
    STUDY_CACHE_SCHEMA,
    canonicalize_reactant_set,
    canonicalize_smiles,
    load_reactions,
)


def _atomic_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_feature_products(manifest_path: str | Path) -> tuple[dict[str, dict], dict]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Encoder-control feature manifest is not complete.")
    identity = manifest.get("identity", {})
    if identity.get("feature_names") != list(FULL_FEATURE_NAMES):
        raise ValueError("Feature manifest does not declare the frozen seven columns.")
    if identity.get("global_atom_cache") is not False:
        raise ValueError("Feature manifest does not attest scalar-only streaming.")

    products: dict[str, dict] = {}
    query_total = 0
    pair_total = 0
    required_arrays = {
        "feature_names",
        "query_ids",
        "product_smiles",
        "canonical_product_identities",
        "query_offsets",
        "candidate_smiles",
        "canonical_candidate_identities",
        "priors",
        "ranks",
        "features",
    }
    for shard in manifest.get("shards", []):
        shard_path = manifest_path.parent / str(shard.get("path", ""))
        if not shard_path.is_file() or file_sha256(shard_path) != shard.get("sha256"):
            raise ValueError(f"Feature shard is missing or changed: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as payload:
            if not required_arrays.issubset(payload.files):
                raise ValueError(f"Feature shard has an incomplete schema: {shard_path}")
            names = tuple(str(value) for value in payload["feature_names"].tolist())
            if names != FULL_FEATURE_NAMES:
                raise ValueError("Feature shard column order differs from the frozen order.")
            query_ids = [str(value) for value in payload["query_ids"].tolist()]
            product_smiles = [str(value) for value in payload["product_smiles"].tolist()]
            product_keys = [
                str(value)
                for value in payload["canonical_product_identities"].tolist()
            ]
            offsets = np.asarray(payload["query_offsets"], dtype=np.int64)
            candidate_smiles = [
                str(value) for value in payload["candidate_smiles"].tolist()
            ]
            candidate_keys = [
                str(value)
                for value in payload["canonical_candidate_identities"].tolist()
            ]
            priors = np.asarray(payload["priors"], dtype=np.float32)
            ranks = np.asarray(payload["ranks"], dtype=np.int32)
            features = np.asarray(payload["features"], dtype=np.float32)
        if not (
            len(query_ids)
            == len(product_smiles)
            == len(product_keys)
            == len(offsets) - 1
        ):
            raise ValueError("Query-level shard arrays are misaligned.")
        if not (
            len(candidate_smiles)
            == len(candidate_keys)
            == len(priors)
            == len(ranks)
            == len(features)
            == int(offsets[-1])
        ):
            raise ValueError("Candidate-level shard arrays are misaligned.")
        if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
            raise ValueError("Query offsets are invalid.")
        if features.ndim != 2 or features.shape[1] != len(FULL_FEATURE_NAMES):
            raise ValueError("Feature shard does not contain seven-column rows.")
        if not np.isfinite(features).all() or not np.isfinite(priors).all():
            raise ValueError("Feature shard contains non-finite data.")
        if not np.array_equal(features[:, 0], priors):
            raise ValueError("Feature prior column is not aligned to retained priors.")

        for query_index, (query_id, product, product_key) in enumerate(
            zip(query_ids, product_smiles, product_keys)
        ):
            start, stop = (int(value) for value in offsets[query_index : query_index + 2])
            raw_candidates = candidate_smiles[start:stop]
            canonical_candidates = candidate_keys[start:stop]
            query_priors = priors[start:stop]
            query_ranks = ranks[start:stop]
            query_features = features[start:stop]
            if product_key in products:
                raise ValueError(f"Canonical product appears more than once: {product_key}")
            if canonicalize_smiles(product) != product_key:
                raise ValueError(f"Product identity mismatch for {query_id}.")
            if len(set(canonical_candidates)) != len(canonical_candidates):
                raise ValueError(f"Duplicate candidate identity remains in {query_id}.")
            if [canonicalize_reactant_set(value) for value in raw_candidates] != canonical_candidates:
                raise ValueError(f"Candidate identity mismatch for {query_id}.")
            if not np.array_equal(query_ranks, np.arange(stop - start, dtype=np.int32)):
                raise ValueError(f"Candidate ranks are not contiguous for {query_id}.")
            if np.any(query_priors[1:] > query_priors[:-1]):
                raise ValueError(f"Candidate priors are not decreasing for {query_id}.")
            products[product_key] = {
                "query_id": query_id,
                "product_smiles": product,
                "product_key": product_key,
                "candidates": [
                    {
                        "smiles": raw,
                        "prior": float(prior),
                        "canonical_smiles": canonical,
                    }
                    for raw, prior, canonical in zip(
                        raw_candidates, query_priors, canonical_candidates
                    )
                ],
                "features": query_features.copy(),
            }
        if len(query_ids) != int(shard.get("query_count", -1)):
            raise ValueError("Manifest query count disagrees with a feature shard.")
        if len(features) != int(shard.get("pair_count", -1)):
            raise ValueError("Manifest pair count disagrees with a feature shard.")
        query_total += len(query_ids)
        pair_total += len(features)

    if query_total != int(manifest.get("completed_query_count", -1)):
        raise ValueError("Manifest completed-query count is inconsistent.")
    if pair_total != int(manifest.get("completed_pair_count", -1)):
        raise ValueError("Manifest completed-pair count is inconsistent.")
    return products, manifest


def _build_training_payload(reactions, products: dict[str, dict]) -> tuple[list[dict], dict]:
    non_train_products = {
        reaction.product_key for reaction in reactions if reaction.source_split != "train"
    }
    ground_truths: dict[str, set[str]] = {}
    reaction_ids: dict[str, list[int]] = {}
    source_smiles: dict[str, str] = {}
    overlap_excluded = 0
    for reaction in reactions:
        if reaction.source_split != "train":
            continue
        if reaction.product_key in non_train_products:
            overlap_excluded += 1
            continue
        ground_truths.setdefault(reaction.product_key, set()).add(
            reaction.ground_truth_key
        )
        reaction_ids.setdefault(reaction.product_key, []).append(reaction.reaction_id)
        source_smiles.setdefault(reaction.product_key, reaction.product_smiles)

    retained: list[dict] = []
    uncovered = 0
    without_negative = 0
    for product_key in sorted(ground_truths):
        product = products.get(product_key)
        if product is None:
            uncovered += 1
            continue
        candidates = product["candidates"]
        positive_indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["canonical_smiles"] in ground_truths[product_key]
        ]
        if not positive_indices:
            uncovered += 1
            continue
        positive_set = set(positive_indices)
        negative_indices = [
            index for index in range(len(candidates)) if index not in positive_set
        ]
        if not negative_indices:
            without_negative += 1
            continue
        retained.append(
            {
                "product_key": product_key,
                "product_smiles": source_smiles[product_key],
                "candidates": candidates,
                "positive_indices": positive_indices,
                "negative_indices": negative_indices,
                "labels": np.asarray(
                    [index in positive_set for index in range(len(candidates))],
                    dtype=np.int8,
                ),
                "source_reaction_ids": reaction_ids[product_key],
                "features": product["features"].copy(),
            }
        )
    return retained, {
        "train_products": len(retained),
        "train_overlap_reactions_excluded": overlap_excluded,
        "train_products_uncovered": uncovered,
        "train_products_without_negative": without_negative,
    }


def _build_evaluation_payload(
    reactions, products: dict[str, dict], split: str
) -> tuple[dict, dict]:
    split_reactions = [
        reaction for reaction in reactions if reaction.source_split == split
    ]
    eval_pwc = []
    eval_ground_truths = []
    eval_metadata = []
    eval_features = []
    eval_labels = []
    uncovered = 0
    covered_products: set[str] = set()
    uncovered_products: set[str] = set()
    for reaction in split_reactions:
        product = products.get(reaction.product_key)
        candidates = [] if product is None else product["candidates"]
        matches = [
            candidate["canonical_smiles"] == reaction.ground_truth_key
            for candidate in candidates
        ]
        positions = [index for index, match in enumerate(matches) if match]
        if not positions:
            uncovered += 1
            uncovered_products.add(reaction.product_key)
            continue
        covered_products.add(reaction.product_key)
        clean_pool = [
            {"smiles": candidate["smiles"], "prior": candidate["prior"]}
            for candidate in candidates
        ]
        eval_pwc.append((reaction.product_smiles, clean_pool))
        eval_ground_truths.append(reaction.ground_truth)
        metadata = reaction.metadata()
        metadata.update(
            {"candidate_count": len(candidates), "coverage_rank": positions[0] + 1}
        )
        eval_metadata.append(metadata)
        eval_features.append(product["features"].copy())
        eval_labels.append(np.asarray(matches, dtype=np.int8))
    all_products = {reaction.product_key for reaction in split_reactions}
    payload = {
        "source_split": split,
        "eval_pwc": eval_pwc,
        "eval_ground_truths": eval_ground_truths,
        "eval_metadata": eval_metadata,
        "eval_features": eval_features,
        "eval_labels": eval_labels,
    }
    audit = {
        "source_split": split,
        "reactions_total": len(split_reactions),
        "reactions_covered": len(eval_pwc),
        "reactions_uncovered": uncovered,
        "unique_products_total": len(all_products),
        "unique_products_covered": len(covered_products),
        "unique_products_with_uncovered_reactions": len(uncovered_products),
        "unique_products_fully_uncovered": len(all_products.difference(covered_products)),
    }
    return payload, audit


def compile_encoder_control_caches(
    feature_manifest: str | Path,
    source_csv: str | Path,
    metadata_csv: str | Path,
    train_test_output: str | Path,
    validation_output: str | Path,
) -> dict:
    train_test_path = Path(train_test_output).resolve()
    validation_path = Path(validation_output).resolve()
    if train_test_path == validation_path:
        raise ValueError("Train/test and validation outputs must be separate files.")
    existing_targets = [
        path for path in (train_test_path, validation_path) if path.exists()
    ]
    if existing_targets:
        raise FileExistsError(
            "Refusing to overwrite existing encoder-control cache target(s): "
            + ", ".join(str(path) for path in existing_targets)
        )
    staging_paths = [
        path.with_suffix(path.suffix + ".tmp")
        for path in (train_test_path, validation_path)
    ]
    existing_staging = [path for path in staging_paths if path.exists()]
    if existing_staging:
        raise FileExistsError(
            "Refusing to overwrite existing staging artifact(s): "
            + ", ".join(str(path) for path in existing_staging)
        )

    products, feature_manifest_payload = _load_feature_products(feature_manifest)
    reactions = load_reactions(source_csv, metadata_csv)
    train_products, train_audit = _build_training_payload(reactions, products)
    test_payload, test_audit = _build_evaluation_payload(reactions, products, "test")
    validation_payload, validation_audit = _build_evaluation_payload(
        reactions, products, "valid"
    )
    if not train_products or not test_payload["eval_pwc"] or not validation_payload["eval_pwc"]:
        raise ValueError("Compiled official train/valid/test payload is unexpectedly empty.")

    protocol_id = str(feature_manifest_payload["identity"]["protocol_id"])
    common = {
        "schema_version": STUDY_CACHE_SCHEMA,
        "protocol_id": protocol_id,
        "feature_mode": "3d+prior",
        "feature_names": list(FULL_FEATURE_NAMES),
        "encoder_control_has_conformer": False,
        "tuned_runner_compatibility": {
            "cache_layout": "seven-column 3d+prior",
            "conformer_seed_field_omitted": True,
            "prepare_conformer_seed_argument_is_not_encoder_provenance": True,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_fingerprints": {
            "feature_manifest": file_fingerprint(feature_manifest),
            "source_csv": file_fingerprint(source_csv),
            "metadata_csv": file_fingerprint(metadata_csv),
        },
        "encoder": feature_manifest_payload["identity"]["encoder"],
        "scalar_only_feature_shards": True,
        "atom_embeddings_written": False,
    }
    legacy_audit = {
        "schema_version": STUDY_CACHE_SCHEMA,
        "feature_mode": "3d+prior",
        "train_split": "train",
        "eval_split": "test",
        "exclude_cross_split_train_products": True,
        **train_audit,
        "eval_reactions_total": test_audit["reactions_total"],
        "eval_reactions_covered": test_audit["reactions_covered"],
        "eval_reactions_uncovered": test_audit["reactions_uncovered"],
        "eval_unique_products_covered": test_audit["unique_products_covered"],
        "compiler_alignment": {
            "canonical_product_count": len(products),
            "test": test_audit,
            "validation": validation_audit,
        },
    }
    train_test_payload = {
        "schema_version": STUDY_CACHE_SCHEMA,
        "feature_mode": "3d+prior",
        "train_products": train_products,
        "eval_pwc": test_payload["eval_pwc"],
        "eval_ground_truths": test_payload["eval_ground_truths"],
        "eval_metadata": test_payload["eval_metadata"],
        "eval_features": test_payload["eval_features"],
        "eval_labels": test_payload["eval_labels"],
        "audit": legacy_audit,
    }
    train_test_blob = {
        **common,
        "artifact_kind": "encoder_control_train_test_compact_cache",
        "audit": legacy_audit,
        "payload": train_test_payload,
    }
    validation_blob = {
        **common,
        "artifact_kind": "official_validation_encoder_control_features",
        "audit": validation_audit,
        "payload": validation_payload,
    }
    _atomic_pickle(train_test_path, train_test_blob)
    _atomic_pickle(validation_path, validation_blob)
    return {
        "protocol_id": protocol_id,
        "train_test_output": file_fingerprint(train_test_path),
        "validation_output": file_fingerprint(validation_path),
        "train_audit": train_audit,
        "test_audit": test_audit,
        "validation_audit": validation_audit,
        "atom_embeddings_written": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    parser.add_argument("--train-test-output", required=True)
    parser.add_argument("--validation-output", required=True)
    args = parser.parse_args()
    result = compile_encoder_control_caches(
        args.feature_manifest,
        args.source_csv,
        args.metadata_csv,
        args.train_test_output,
        args.validation_output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
