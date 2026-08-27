"""Compile and run the frozen WS-E three-pool ranking comparison.

Phases are intentionally separate:

1. ``join-shard`` reuses the merged-union scalar features for one pool.
2. ``finalize-features`` freezes a compact feature manifest.
3. ``prepare-selection`` reads train/official-valid labels only and writes
   deterministic seed-specific pairs plus a validation payload.
4. ``fit-validation`` fits one frozen primary configuration/seed/arm.
5. ``freeze`` validates all ten checkpoints for a pool.
6. ``evaluate-test`` is the only command allowed to read official-test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rerank.features import FeatureNormalizer
from rerank.study_data import (
    ReactionRecord,
    canonicalize_reactant_set,
    canonicalize_smiles,
    load_reactions,
)
from rerank.ws_e_streaming import (
    AUGMENTED_COLUMNS,
    BASELINE_COLUMNS,
    FEATURE_PROTOCOL_ID,
    FULL_FEATURE_NAMES,
    POOL_PROTOCOL_ID,
    RANKING_PROTOCOL_ID,
    arm_view,
    atomic_json,
    atomic_npz,
    digest_array,
    digest_key,
    fingerprint,
    full_feature_matrix,
    identity_digest,
    load_pool_index,
    load_products,
    raw_identity_digest,
    read_product_records,
    sha256_file,
    shard_bounds,
    utc_now,
)


SEEDS = tuple(range(42, 47))
MAX_EPOCHS = 200
PATIENCE = 20
MIN_IMPROVEMENT = 1e-5


def _parse_seed_list(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique comma list")
    return seeds


def _selection_seeds(selection: Mapping) -> tuple[int, ...]:
    seeds = tuple(int(seed) for seed in selection.get("seeds", SEEDS))
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("WS-E selection has an invalid seed set.")
    return seeds


def _candidate_cap(selection: Mapping) -> int | None:
    value = selection.get("candidate_cap")
    if value is None:
        return None
    cap = int(value)
    if cap <= 0:
        raise ValueError("candidate_cap must be positive")
    return cap


def _truncate_feature_lookup(
    lookup: Mapping[int, tuple[np.ndarray, np.ndarray]], cap: int | None
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if cap is None:
        return dict(lookup)
    return {
        rank: (digests[:cap].copy(), features[:cap].copy())
        for rank, (digests, features) in lookup.items()
    }


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_absent(*paths: str | Path) -> None:
    existing = [str(Path(path)) for path in paths if Path(path).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite immutable WS-E output(s): {existing}")


def _resolve_frozen_path(record: Mapping, *local_dirs: str | Path) -> Path:
    """Resolve a checksummed artifact after moving a frozen result bundle.

    Vast manifests intentionally retain the absolute path used during the
    scientific run.  Imported archives therefore first try that recorded path
    and, when it is absent, relocate by basename inside an explicitly supplied
    local artifact directory.  Every successful relocation is still guarded by
    the frozen byte size and SHA-256; no manifest is rewritten.
    """

    recorded = Path(str(record["path"]))
    candidates = [recorded, *(Path(directory) / recorded.name for directory in local_dirs)]
    resolved = next((path for path in candidates if path.is_file()), None)
    if resolved is None:
        raise FileNotFoundError(
            f"Frozen WS-E artifact is absent at its recorded or relocated path: {recorded}"
        )
    expected_size = int(record.get("size_bytes", -1))
    if expected_size >= 0 and resolved.stat().st_size != expected_size:
        raise ValueError(f"Frozen WS-E artifact size mismatch: {resolved}")
    if sha256_file(resolved) != str(record["sha256"]):
        raise ValueError(f"Frozen WS-E artifact checksum mismatch: {resolved}")
    return resolved.resolve()


def _scalar_manifest_paths(freeze: Mapping, freeze_path: str | Path) -> list[Path]:
    records = list(freeze.get("shard_manifests", []))
    if len(records) != int(freeze.get("shard_count", -1)):
        raise ValueError("WS-E scalar freeze has incomplete shard manifests.")
    local_root = Path(freeze_path).resolve().parent
    return [
        _resolve_frozen_path(record, local_root / "shards", local_root)
        for record in records
    ]


@lru_cache(maxsize=8)
def _load_pool_index_verified_cached(
    index_npz: str,
    index_manifest: str,
    products_jsonl: str,
    pool_jsonl: str,
    file_state: tuple[tuple[int, int], ...],
):
    # ``file_state`` is deliberately part of the cache key: any size or mtime
    # change forces the full fingerprint gate to run again.
    del file_state
    return load_pool_index(index_npz, index_manifest, products_jsonl, pool_jsonl)


def _load_pool_index_once(
    index_npz: str | Path,
    index_manifest: str | Path,
    products_jsonl: str | Path,
    pool_jsonl: str | Path,
):
    paths = tuple(
        Path(value).resolve()
        for value in (index_npz, index_manifest, products_jsonl, pool_jsonl)
    )
    state = tuple((path.stat().st_size, path.stat().st_mtime_ns) for path in paths)
    return _load_pool_index_verified_cached(
        *(str(path) for path in paths), file_state=state
    )


def join_shard(args: argparse.Namespace) -> dict:
    scalar_freeze = _load_json(args.scalar_freeze)
    if scalar_freeze.get("protocol_id") != FEATURE_PROTOCOL_ID or not scalar_freeze.get("complete"):
        raise ValueError("A complete frozen WS-E scalar manifest is required.")
    scalar_manifests = _scalar_manifest_paths(scalar_freeze, args.scalar_freeze)
    if args.shard_count != int(scalar_freeze["shard_count"]):
        raise ValueError("Pool-feature shard count must equal scalar shard count.")
    scalar_record = _load_json(scalar_manifests[args.shard_index])
    scalar_path = _resolve_frozen_path(
        scalar_record["output"], scalar_manifests[args.shard_index].parent
    )
    scalar = np.load(scalar_path, allow_pickle=False)

    pool_index = _load_pool_index_once(
        args.pool_index_npz,
        args.pool_index_manifest,
        args.products_jsonl,
        args.pool_jsonl,
    )
    merged_index = _load_pool_index_once(
        args.merged_index_npz,
        args.merged_index_manifest,
        args.products_jsonl,
        args.merged_jsonl,
    )
    start_rank, stop_rank = shard_bounds(
        pool_index.product_count, args.shard_index, args.shard_count
    )
    if start_rank != int(scalar_record["product_rank_start"]) or stop_rank != int(
        scalar_record["product_rank_stop"]
    ):
        raise ValueError("Pool and scalar shard product ranges differ.")
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    _require_absent(output, manifest_path)

    scalar_product_ranks = scalar["product_rank"]
    scalar_digests = scalar["candidate_sha256"]
    scalar_features = scalar["core_features"]
    stored_rows: dict[int, list[tuple[bytes, np.ndarray]]] = defaultdict(list)
    for product_rank, digest, features in zip(
        scalar_product_ranks, scalar_digests, scalar_features
    ):
        stored_rows[int(product_rank)].append(
            (digest_key(digest), np.asarray(features, dtype=np.float32))
        )

    # The remote scalar archive predates canonical identity digests and stores
    # the exact merged-pool candidate spelling.  Re-anchor every row to the
    # checksummed merged pool in order, verify that legacy digest byte-for-byte,
    # then key the join by the prespecified fragment-order-invariant identity.
    product_rows: dict[int, dict[bytes, np.ndarray]] = defaultdict(dict)
    for product_rank in range(start_rank, stop_rank):
        merged_records = read_product_records(
            args.merged_jsonl, merged_index, product_rank
        )
        rows = stored_rows.get(product_rank, [])
        if len(merged_records) != len(rows):
            raise ValueError(
                f"Merged pool/scalar row count mismatch at product {product_rank}."
            )
        for record, (stored_digest, features) in zip(merged_records, rows):
            candidate = str(record["reactant"])
            canonical_digest = identity_digest(candidate)
            if stored_digest not in {
                raw_identity_digest(candidate),
                canonical_digest,
            }:
                raise ValueError(
                    f"Merged pool order differs from scalar archive at product {product_rank}."
                )
            key = canonical_digest
            if key in product_rows[product_rank]:
                raise ValueError(
                    f"Merged pool duplicates canonical identity at product {product_rank}."
                )
            product_rows[product_rank][key] = features

    output_product_ranks: list[int] = []
    output_candidates: list[str] = []
    feature_parts: list[np.ndarray] = []
    for product_rank in range(start_rank, stop_rank):
        records = read_product_records(args.pool_jsonl, pool_index, product_rank)
        if not records:
            continue
        lookup = product_rows.get(product_rank, {})
        core_rows = []
        for record in records:
            candidate = str(record["reactant"])
            key = identity_digest(candidate)
            if key not in lookup:
                raise ValueError(
                    f"Pool candidate is absent from merged scalars at product {product_rank}."
                )
            core_rows.append(lookup[key])
            output_product_ranks.append(product_rank)
            output_candidates.append(candidate)
        feature_parts.append(full_feature_matrix(records, np.stack(core_rows)))

    features = (
        np.concatenate(feature_parts, axis=0)
        if feature_parts
        else np.empty((0, len(FULL_FEATURE_NAMES)), dtype=np.float32)
    )
    atomic_npz(
        output,
        product_rank=np.asarray(output_product_ranks, dtype=np.int32),
        candidate_sha256=digest_array(output_candidates),
        features=features,
    )
    manifest = {
        "schema_version": 1,
        "record_kind": "ws_e_pool_feature_shard",
        "protocol_id": RANKING_PROTOCOL_ID,
        "source_pool_protocol_id": POOL_PROTOCOL_ID,
        "pool_name": args.pool_name,
        "feature_names": list(FULL_FEATURE_NAMES),
        "baseline_columns": list(BASELINE_COLUMNS),
        "augmented_columns": list(AUGMENTED_COLUMNS),
        "source_indicators_in_both_arms": True,
        "single_intended_arm_change": "add three Uni-Mol-derived pair-level scalars",
        "product_rank_start": start_rank,
        "product_rank_stop": stop_rank,
        "candidate_rows": len(features),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "inputs": {
            "pool": pool_index.pool_fingerprint,
            "pool_index": fingerprint(args.pool_index_npz),
            "merged_pool": merged_index.pool_fingerprint,
            "merged_pool_index": fingerprint(args.merged_index_npz),
            "scalar_freeze": fingerprint(args.scalar_freeze),
            "scalar_shard": scalar_record["output"],
        },
        "output": fingerprint(output),
        "test_partition_loaded": False,
        "ground_truth_loaded": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def finalize_features(args: argparse.Namespace) -> dict:
    root = Path(args.shard_root).resolve()
    manifests = []
    shards = []
    total_rows = 0
    expected_start = 0
    for shard_index in range(args.shard_count):
        path = root / f"shard_{shard_index:03d}_of_{args.shard_count:03d}.json"
        record = _load_json(path)
        if record.get("protocol_id") != RANKING_PROTOCOL_ID:
            raise ValueError("Pool feature shard has the wrong WS-E protocol.")
        if record.get("pool_name") != args.pool_name:
            raise ValueError("Pool feature shards mix pool identities.")
        if int(record["product_rank_start"]) != expected_start:
            raise ValueError("Pool feature shards have a product-range gap.")
        expected_start = int(record["product_rank_stop"])
        output = Path(record["output"]["path"])
        if sha256_file(output) != record["output"]["sha256"]:
            raise ValueError("Pool feature shard checksum mismatch.")
        arrays = np.load(output, allow_pickle=False)
        rows = len(arrays["product_rank"])
        if arrays["candidate_sha256"].shape != (rows, 32):
            raise ValueError("Pool candidate digest array is malformed.")
        if arrays["features"].shape != (rows, len(FULL_FEATURE_NAMES)):
            raise ValueError("Pool feature array is malformed.")
        if rows != int(record["candidate_rows"]):
            raise ValueError("Pool feature row count differs from its manifest.")
        total_rows += rows
        manifests.append(fingerprint(path))
        shards.append(record["output"])
    pool_index_manifest = _load_json(args.pool_index_manifest)
    if expected_start != int(pool_index_manifest["product_count"]):
        raise ValueError("Pool feature shards do not span the product inventory.")
    if total_rows != int(pool_index_manifest["candidate_count"]):
        raise ValueError("Pool feature shards do not span the candidate pool.")
    result = {
        "schema_version": 1,
        "record_kind": "ws_e_pool_feature_freeze",
        "protocol_id": RANKING_PROTOCOL_ID,
        "source_pool_protocol_id": POOL_PROTOCOL_ID,
        "pool_name": args.pool_name,
        "product_count": expected_start,
        "candidate_rows": total_rows,
        "feature_names": list(FULL_FEATURE_NAMES),
        "baseline_columns": list(BASELINE_COLUMNS),
        "augmented_columns": list(AUGMENTED_COLUMNS),
        "source_indicators_in_both_arms": True,
        "single_intended_arm_change": "add three Uni-Mol-derived pair-level scalars",
        "shard_count": args.shard_count,
        "shards": shards,
        "shard_manifests": manifests,
        "pool_index_manifest": fingerprint(args.pool_index_manifest),
        "test_partition_loaded": False,
        "ground_truth_loaded": False,
        "complete": True,
        "created_at_utc": utc_now(),
    }
    _require_absent(args.output)
    atomic_json(args.output, result)
    return result


def _feature_shard_records(feature_freeze: Mapping) -> list[tuple[dict, np.lib.npyio.NpzFile]]:
    result = []
    for item in feature_freeze["shard_manifests"]:
        record = _load_json(item["path"])
        output = Path(record["output"]["path"])
        if sha256_file(output) != record["output"]["sha256"]:
            raise ValueError("Frozen pool feature shard checksum mismatch.")
        result.append((record, np.load(output, allow_pickle=False)))
    return result


def _product_feature_lookup(feature_freeze: Mapping) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for _, arrays in _feature_shard_records(feature_freeze):
        product_ranks = arrays["product_rank"]
        digests = arrays["candidate_sha256"]
        features = arrays["features"]
        if len(product_ranks) == 0:
            continue
        boundaries = np.flatnonzero(np.r_[True, product_ranks[1:] != product_ranks[:-1], True])
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            rank = int(product_ranks[start])
            if rank in lookup:
                raise ValueError("Product features appear in multiple WS-E shards.")
            lookup[rank] = (digests[start:stop].copy(), features[start:stop].copy())
    return lookup


def _reaction_maps(source_csv: str | Path, metadata_csv: str | Path, products_jsonl: str | Path):
    reactions = load_reactions(source_csv, metadata_csv)
    products = load_products(products_jsonl)
    rank_by_canonical = {
        str(item["canonical_product"]): int(item["product_rank"]) for item in products
    }
    if len(rank_by_canonical) != len(products):
        raise ValueError("Canonical WS-E product identities are not unique.")
    return reactions, products, rank_by_canonical


def _load_selection_reactions(
    source_csv: str | Path, metadata_csv: str | Path
) -> tuple[list[ReactionRecord], set[str], dict]:
    """Read train/valid labels while using test *product identity* only.

    The test reactant field is never accessed or canonicalized.  Test product
    identities are required solely for the frozen cross-split leakage rule.
    """

    classes: dict[int, str | None] = {}
    with open(metadata_csv, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            reaction_id = int(row["reaction_id"])
            value = row.get("reaction_class")
            classes[reaction_id] = None if value in (None, "", "nan") else str(value)
    reactions: list[ReactionRecord] = []
    non_train_products: set[str] = set()
    split_counts: dict[str, int] = defaultdict(int)
    malformed_products = malformed_ground_truths = 0
    with open(source_csv, "r", encoding="utf-8-sig", newline="") as handle:
        for reaction_id, row in enumerate(csv.DictReader(handle)):
            split = str(row["set"])
            split_counts[split] += 1
            product = str(row["products_smiles"])
            product_key = canonicalize_smiles(product)
            if product_key is None:
                malformed_products += 1
                continue
            if split != "train":
                non_train_products.add(product_key)
            if split not in {"train", "valid"}:
                # Deliberately do not access row["reactants_smiles"] here.
                continue
            ground_truth = str(row["reactants_smiles"])
            ground_truth_key = canonicalize_reactant_set(ground_truth)
            if ground_truth_key is None:
                malformed_ground_truths += 1
                continue
            reactions.append(
                ReactionRecord(
                    reaction_id=reaction_id,
                    source_split=split,
                    product_smiles=product,
                    product_key=product_key,
                    ground_truth=ground_truth,
                    ground_truth_key=ground_truth_key,
                    reaction_class=classes.get(reaction_id),
                )
            )
    audit = {
        "split_counts": dict(sorted(split_counts.items())),
        "selection_reactions_loaded": len(reactions),
        "malformed_products": malformed_products,
        "malformed_selection_ground_truths": malformed_ground_truths,
        "test_product_identities_used_for_leakage_exclusion": True,
        "test_ground_truth_loaded": False,
    }
    return reactions, non_train_products, audit


def _pool_candidate_digests(
    pool_jsonl: str | Path, pool_index, product_rank: int
) -> tuple[list[dict], list[bytes]]:
    records = read_product_records(pool_jsonl, pool_index, product_rank)
    return records, [identity_digest(str(item["reactant"])) for item in records]


def prepare_selection(args: argparse.Namespace) -> dict:
    feature_freeze = _load_json(args.feature_freeze)
    if feature_freeze.get("protocol_id") != RANKING_PROTOCOL_ID or not feature_freeze.get("complete"):
        raise ValueError("A complete frozen WS-E pool-feature manifest is required.")
    if (
        feature_freeze["pool_index_manifest"]["sha256"]
        != fingerprint(args.pool_index_manifest)["sha256"]
    ):
        raise ValueError("WS-E feature freeze belongs to a different pool index.")
    pool_index = _load_pool_index_once(
        args.pool_index_npz,
        args.pool_index_manifest,
        args.products_jsonl,
        args.pool_jsonl,
    )
    reactions, non_train_products, selection_load_audit = _load_selection_reactions(
        args.source_csv, args.metadata_csv
    )
    products = load_products(args.products_jsonl)
    rank_by_canonical = {
        str(item["canonical_product"]): int(item["product_rank"]) for item in products
    }
    train_truths: dict[str, set[bytes]] = defaultdict(set)
    train_overlap_excluded = 0
    for reaction in reactions:
        if reaction.source_split != "train":
            continue
        if reaction.product_key in non_train_products:
            train_overlap_excluded += 1
            continue
        train_truths[reaction.product_key].add(identity_digest(reaction.ground_truth_key))

    seeds = tuple(getattr(args, "seeds", SEEDS))
    candidate_cap = getattr(args, "candidate_cap", None)
    if candidate_cap is not None and int(candidate_cap) <= 0:
        raise ValueError("candidate_cap must be positive")
    output_protocol_id = str(
        getattr(args, "output_protocol_id", None) or RANKING_PROTOCOL_ID
    )
    feature_lookup = _truncate_feature_lookup(
        _product_feature_lookup(feature_freeze), candidate_cap
    )
    output_root = Path(args.output_root).resolve()
    _require_absent(output_root / "manifest.json")
    output_root.mkdir(parents=True, exist_ok=True)

    train_audits = {}
    pair_outputs = {}
    for seed in seeds:
        rng = random.Random(seed)
        positive_rows: list[np.ndarray] = []
        negative_rows: list[np.ndarray] = []
        uncovered = without_negative = products_used = 0
        for product_key in sorted(train_truths):
            product_rank = rank_by_canonical.get(product_key)
            if product_rank is None:
                uncovered += 1
                continue
            shard_digests, features = feature_lookup.get(
                product_rank, (np.empty((0, 32), dtype=np.uint8), np.empty((0, 9)))
            )
            digests = [digest_key(value) for value in shard_digests]
            if not digests:
                uncovered += 1
                continue
            positive_keys = train_truths[product_key]
            positive_indices = [i for i, value in enumerate(digests) if value in positive_keys]
            if not positive_indices:
                uncovered += 1
                continue
            positive_index_set = set(positive_indices)
            negative_indices = [i for i in range(len(digests)) if i not in positive_index_set]
            if not negative_indices:
                without_negative += 1
                continue
            products_used += 1
            for positive_index in positive_indices:
                selected = negative_indices
                if len(selected) > 5:
                    selected = rng.sample(selected, 5)
                positive_row = features[positive_index]
                for negative_index in selected:
                    positive_rows.append(positive_row)
                    negative_rows.append(features[negative_index])
        pairs_path = output_root / f"train_pairs_seed_{seed}.npz"
        positive = np.stack(positive_rows).astype(np.float32, copy=False)
        negative = np.stack(negative_rows).astype(np.float32, copy=False)
        atomic_npz(pairs_path, positive=positive, negative=negative)
        pair_outputs[str(seed)] = fingerprint(pairs_path)
        train_audits[str(seed)] = {
            "products": products_used,
            "pairs": len(positive),
            "products_uncovered": uncovered,
            "products_without_negative": without_negative,
        }

    validation_features: list[np.ndarray] = []
    validation_masks: list[np.ndarray] = []
    reaction_ids: list[int] = []
    product_ranks: list[int] = []
    offsets = [0]
    validation_total = validation_uncovered = 0
    for reaction in reactions:
        if reaction.source_split != "valid":
            continue
        validation_total += 1
        product_rank = rank_by_canonical.get(reaction.product_key)
        if product_rank is None or product_rank not in feature_lookup:
            validation_uncovered += 1
            continue
        shard_digests, features = feature_lookup[product_rank]
        digests = [digest_key(value) for value in shard_digests]
        target = identity_digest(reaction.ground_truth_key)
        mask = np.asarray([value == target for value in digests], dtype=np.bool_)
        if not mask.any():
            validation_uncovered += 1
            continue
        validation_features.append(features)
        validation_masks.append(mask)
        reaction_ids.append(reaction.reaction_id)
        product_ranks.append(product_rank)
        offsets.append(offsets[-1] + len(features))
    validation_path = output_root / "official_valid.npz"
    atomic_npz(
        validation_path,
        features=np.concatenate(validation_features, axis=0).astype(np.float32, copy=False),
        offsets=np.asarray(offsets, dtype=np.int64),
        match_mask=np.concatenate(validation_masks).astype(np.bool_, copy=False),
        reaction_id=np.asarray(reaction_ids, dtype=np.int64),
        product_rank=np.asarray(product_ranks, dtype=np.int32),
    )
    manifest = {
        "schema_version": 1,
        "record_kind": "ws_e_train_validation_selection",
        "protocol_id": output_protocol_id,
        "source_feature_protocol_id": RANKING_PROTOCOL_ID,
        "source_pool_protocol_id": POOL_PROTOCOL_ID,
        "pool_name": feature_freeze["pool_name"],
        "comparator": "same pool, frozen 2D baseline configuration",
        "single_intended_arm_change": "add three Uni-Mol-derived pair-level scalars",
        "seeds": list(seeds),
        "candidate_cap": candidate_cap,
        "candidate_ordering": (
            "first rows of the frozen pool order; no re-sorting"
            if candidate_cap is not None
            else "complete frozen pool order"
        ),
        "feature_names": list(FULL_FEATURE_NAMES),
        "baseline_columns": list(BASELINE_COLUMNS),
        "augmented_columns": list(AUGMENTED_COLUMNS),
        "source_indicators_in_both_arms": True,
        "negative_sampling": "seeded random, at most five negatives per positive",
        "train_overlap_reactions_excluded": train_overlap_excluded,
        "train_audit": train_audits,
        "validation_audit": {
            "reactions_total": validation_total,
            "reactions_covered": len(reaction_ids),
            "reactions_uncovered": validation_uncovered,
            "candidate_rows": offsets[-1],
        },
        "selection_load_audit": selection_load_audit,
        "train_pairs": pair_outputs,
        "validation": fingerprint(validation_path),
        "inputs": {
            "feature_freeze": fingerprint(args.feature_freeze),
            "source_csv": fingerprint(args.source_csv),
            "metadata_csv": fingerprint(args.metadata_csv),
            "pool": pool_index.pool_fingerprint,
            "products": pool_index.products_fingerprint,
        },
        "test_partition_loaded": False,
        "test_ground_truth_loaded": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(output_root / "manifest.json", manifest)
    return manifest


def _config_for_arm(primary_freeze: Mapping, arm: str) -> dict:
    key = "selected_baseline" if arm == "baseline" else "selected_augmented"
    return dict(primary_freeze[key]["config"])


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validation_mrr(model, features, offsets, masks, device: str) -> float:
    import torch

    scores = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), 65_536):
            tensor = torch.as_tensor(features[start : start + 65_536], device=device)
            scores.append(model.score(tensor).detach().cpu().numpy())
    scores = np.concatenate(scores)
    reciprocal = []
    for index in range(len(offsets) - 1):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        order = np.argsort(-scores[start:stop], kind="stable")
        positions = np.flatnonzero(masks[start:stop][order])
        reciprocal.append(1.0 / (int(positions[0]) + 1))
    return float(np.mean(reciprocal))


def fit_validation(args: argparse.Namespace) -> dict:
    import torch

    from rerank.loss import PairwiseRankingLoss
    from rerank.model import RankerMLP

    selection = _load_json(Path(args.selection_root) / "manifest.json")
    seeds = _selection_seeds(selection)
    if args.seed not in seeds:
        raise ValueError(f"WS-E validation seed must be one of {seeds}.")
    if args.arm not in {"baseline", "augmented"}:
        raise ValueError("arm must be baseline or augmented")
    primary = _load_json(args.primary_freeze)
    if primary.get("protocol_id") != "cap10-tuned-v1":
        raise ValueError("WS-E requires the frozen cap10-tuned-v1 primary configuration.")
    config = _config_for_arm(primary, args.arm)
    output_root = Path(args.output_root).resolve() / "validation" / args.arm / f"seed_{args.seed}"
    trial_path = output_root / "trial.json"
    checkpoint_path = output_root / "best_checkpoint.pt"
    normalizer_path = output_root / "normalizer.npz"
    _require_absent(trial_path, checkpoint_path, normalizer_path)
    output_root.mkdir(parents=True, exist_ok=True)

    pairs_path = Path(selection["train_pairs"][str(args.seed)]["path"])
    if sha256_file(pairs_path) != selection["train_pairs"][str(args.seed)]["sha256"]:
        raise ValueError("WS-E training-pair checksum mismatch.")
    pairs = np.load(pairs_path, allow_pickle=False)
    positive = arm_view(pairs["positive"], args.arm)
    negative = arm_view(pairs["negative"], args.arm)
    normalizer = FeatureNormalizer().fit(np.concatenate((positive, negative), axis=0))
    normalizer.save(str(normalizer_path))
    positive = normalizer.transform(positive)
    negative = normalizer.transform(negative)

    valid_path = Path(selection["validation"]["path"])
    if sha256_file(valid_path) != selection["validation"]["sha256"]:
        raise ValueError("WS-E validation checksum mismatch.")
    valid = np.load(valid_path, allow_pickle=False)
    valid_features = normalizer.transform(arm_view(valid["features"], args.arm))
    offsets = valid["offsets"]
    masks = valid["match_mask"]

    _seed_everything(args.seed)
    model = RankerMLP(
        input_dim=positive.shape[1],
        hidden_dims=[int(config["hidden_width"])],
        dropout=float(config["dropout"]),
        use_batch_norm=False,
    ).to(args.device)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    criterion = PairwiseRankingLoss(margin=float(config["margin"]), reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]), weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(MAX_EPOCHS - 5, 1),
        eta_min=float(config["learning_rate"]) * 0.01,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    positive_tensor = torch.as_tensor(positive, dtype=torch.float32)
    negative_tensor = torch.as_tensor(negative, dtype=torch.float32)
    best_mrr = float("-inf")
    best_epoch = 0
    no_improve = 0
    history = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(positive_tensor), generator=generator)
        loss_sum = 0.0
        batches = 0
        for start in range(0, len(order), 256):
            indices = order[start : start + 256]
            pos = positive_tensor[indices].to(args.device)
            neg = negative_tensor[indices].to(args.device)
            optimizer.zero_grad()
            loss = criterion(model.score(pos), model.score(neg))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item())
            batches += 1
        mrr = _validation_mrr(model, valid_features, offsets, masks, args.device)
        improved = best_epoch == 0 or mrr > best_mrr + MIN_IMPROVEMENT
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
            no_improve = 0
            temporary = checkpoint_path.with_suffix(".tmp.pt")
            torch.save(model.state_dict(), temporary)
            os.replace(temporary, checkpoint_path)
        else:
            no_improve += 1
        if epoch > 5:
            scheduler.step()
        print(
            f"\rWS-E {selection['pool_name']} {args.arm} seed {args.seed}: "
            f"epoch {epoch}/{MAX_EPOCHS} | best MRR {best_mrr:.5f} @ {best_epoch}",
            end="",
            flush=True,
        )
        if no_improve >= PATIENCE:
            break
    print(flush=True)
    result = {
        "schema_version": 1,
        "record_kind": "ws_e_validation_fit",
        "protocol_id": selection.get("protocol_id", RANKING_PROTOCOL_ID),
        "pool_name": selection["pool_name"],
        "arm": args.arm,
        "seed": args.seed,
        "comparator": "same pool and seed; source indicators retained in both arms",
        "single_intended_change": "add three Uni-Mol-derived pair-level scalars",
        "config_source": fingerprint(args.primary_freeze),
        "config": config,
        "model_parameters": model_parameters,
        "candidate_cap": _candidate_cap(selection),
        "feature_columns": list(BASELINE_COLUMNS if args.arm == "baseline" else AUGMENTED_COLUMNS),
        "source_indicators_in_both_arms": True,
        "best_validation_mrr": best_mrr,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "early_stopped": len(history) < MAX_EPOCHS,
        "history": history,
        "train_pairs": len(positive),
        "validation_reactions": len(offsets) - 1,
        "checkpoint": fingerprint(checkpoint_path),
        "normalizer": fingerprint(normalizer_path),
        "selection_manifest": fingerprint(Path(args.selection_root) / "manifest.json"),
        "test_partition_loaded": False,
        "runtime_seconds": time.perf_counter() - started,
        "created_at_utc": utc_now(),
    }
    atomic_json(trial_path, result)
    return result


def freeze(args: argparse.Namespace) -> dict:
    selection = _load_json(Path(args.selection_root) / "manifest.json")
    seeds = _selection_seeds(selection)
    primary = _load_json(args.primary_freeze)
    trials = {}
    for arm in ("baseline", "augmented"):
        trials[arm] = {}
        expected_config = _config_for_arm(primary, arm)
        for seed in seeds:
            trial_path = Path(args.output_root) / "validation" / arm / f"seed_{seed}" / "trial.json"
            trial = _load_json(trial_path)
            if trial.get("pool_name") != selection["pool_name"] or trial.get("arm") != arm:
                raise ValueError("WS-E trial identity mismatch.")
            if trial.get("config") != expected_config:
                raise ValueError("WS-E trial did not use the frozen primary configuration.")
            for artifact in ("checkpoint", "normalizer"):
                if sha256_file(trial[artifact]["path"]) != trial[artifact]["sha256"]:
                    raise ValueError(f"WS-E {artifact} checksum mismatch.")
            trials[arm][str(seed)] = fingerprint(trial_path)
    result = {
        "schema_version": 1,
        "record_kind": "ws_e_pool_model_freeze",
        "protocol_id": selection.get("protocol_id", RANKING_PROTOCOL_ID),
        "source_pool_protocol_id": POOL_PROTOCOL_ID,
        "pool_name": selection["pool_name"],
        "seeds": list(seeds),
        "candidate_cap": _candidate_cap(selection),
        "candidate_ordering": selection.get("candidate_ordering"),
        "baseline_config": _config_for_arm(primary, "baseline"),
        "augmented_config": _config_for_arm(primary, "augmented"),
        "selected_prior_transform": "raw",
        "selection_metric": "official-validation conditional MRR for checkpoint epoch only",
        "config_selection_performed_for_ws_e": False,
        "source_indicators_in_both_arms": True,
        "single_intended_arm_change": "add three Uni-Mol-derived pair-level scalars",
        "trials": trials,
        "selection_manifest": fingerprint(Path(args.selection_root) / "manifest.json"),
        "primary_freeze": fingerprint(args.primary_freeze),
        "test_partition_loaded": False,
        "complete": True,
        "created_at_utc": utc_now(),
    }
    _require_absent(args.output)
    atomic_json(args.output, result)
    return result


def _rank_metrics(ranks: Sequence[int], denominator: int | None = None) -> dict:
    values = np.asarray(ranks, dtype=np.int32)
    denom = len(values) if denominator is None else int(denominator)
    if denom < len(values):
        raise ValueError("Metric denominator cannot be smaller than covered ranks.")
    result = {f"top{k}": float(np.sum((values > 0) & (values <= k)) / denom) for k in (1, 3, 5, 10, 20, 50)}
    result["mrr"] = float(np.sum(np.where(values > 0, 1.0 / values, 0.0)) / denom)
    result["denominator"] = denom
    result["covered"] = int(np.sum(values > 0))
    return result


def evaluate_test(args: argparse.Namespace) -> dict:
    import torch

    from rerank.model import RankerMLP

    model_freeze = _load_json(args.model_freeze)
    if not model_freeze.get("complete"):
        raise PermissionError("Official test requires a complete immutable model freeze.")
    selection = _load_json(Path(args.selection_root) / "manifest.json")
    selection_protocol = selection.get("protocol_id", RANKING_PROTOCOL_ID)
    if (
        model_freeze.get("protocol_id") != selection_protocol
    ):
        raise PermissionError("Official test requires a matching complete immutable model freeze.")
    seeds = tuple(int(seed) for seed in model_freeze.get("seeds", ()))
    if seeds != _selection_seeds(selection):
        raise PermissionError("Model-freeze and selection seed sets differ.")
    if _candidate_cap(model_freeze) != _candidate_cap(selection):
        raise PermissionError("Model-freeze and selection candidate caps differ.")
    freeze_hash_before = sha256_file(args.model_freeze)
    feature_freeze = _load_json(args.feature_freeze)
    if (
        feature_freeze["pool_index_manifest"]["sha256"]
        != fingerprint(args.pool_index_manifest)["sha256"]
    ):
        raise ValueError("WS-E feature freeze belongs to a different pool index.")
    pool_index = _load_pool_index_once(
        args.pool_index_npz,
        args.pool_index_manifest,
        args.products_jsonl,
        args.pool_jsonl,
    )
    reactions, _, rank_by_canonical = _reaction_maps(
        args.source_csv, args.metadata_csv, args.products_jsonl
    )
    feature_lookup = _truncate_feature_lookup(
        _product_feature_lookup(feature_freeze), _candidate_cap(selection)
    )
    test_reactions = [item for item in reactions if item.source_split == "test"]
    covered = []
    prior_ranks = []
    for reaction in test_reactions:
        product_rank = rank_by_canonical.get(reaction.product_key)
        if product_rank is None or product_rank not in feature_lookup:
            continue
        shard_digests, features = feature_lookup[product_rank]
        digests = [digest_key(value) for value in shard_digests]
        target = identity_digest(reaction.ground_truth_key)
        positions = [i for i, value in enumerate(digests) if value == target]
        if not positions:
            continue
        covered.append(
            (
                reaction,
                product_rank,
                features,
                np.asarray([value == target for value in digests]),
            )
        )
        prior_ranks.append(positions[0] + 1)

    per_seed = {"baseline": {}, "augmented": {}}
    reaction_rows = {
        reaction.reaction_id: {
            "reaction_id": reaction.reaction_id,
            "reaction_class": reaction.reaction_class,
            "source_split": "test",
            "pool_name": selection["pool_name"],
            "covered": True,
            "candidate_count": len(features),
            "prior_rank": prior_rank,
        }
        for (reaction, _, features, _), prior_rank in zip(covered, prior_ranks)
    }
    primary = _load_json(args.primary_freeze)
    for arm in ("baseline", "augmented"):
        config = _config_for_arm(primary, arm)
        for seed in seeds:
            trial_path = Path(args.output_root) / "validation" / arm / f"seed_{seed}" / "trial.json"
            trial = _load_json(trial_path)
            normalizer = FeatureNormalizer.load(trial["normalizer"]["path"])
            columns = BASELINE_COLUMNS if arm == "baseline" else AUGMENTED_COLUMNS
            model = RankerMLP(
                input_dim=len(columns),
                hidden_dims=[int(config["hidden_width"])],
                dropout=float(config["dropout"]),
                use_batch_norm=False,
            ).to(args.device)
            state = torch.load(trial["checkpoint"]["path"], map_location=args.device)
            model.load_state_dict(state)
            model.eval()
            ranks = []
            with torch.no_grad():
                for reaction, _, features, mask in covered:
                    normalized = normalizer.transform(arm_view(features, arm))
                    scores = model.score(
                        torch.as_tensor(normalized, dtype=torch.float32, device=args.device)
                    ).detach().cpu().numpy()
                    order = np.argsort(-scores, kind="stable")
                    positions = np.flatnonzero(mask[order])
                    rank = int(positions[0]) + 1
                    ranks.append(rank)
                    reaction_rows[reaction.reaction_id][f"{arm}_rank_seed_{seed}"] = rank
            per_seed[arm][str(seed)] = {
                "within_pool": _rank_metrics(ranks),
                "end_to_end": _rank_metrics(ranks, denominator=len(test_reactions)),
            }

    for reaction in test_reactions:
        if reaction.reaction_id not in reaction_rows:
            reaction_rows[reaction.reaction_id] = {
                "reaction_id": reaction.reaction_id,
                "reaction_class": reaction.reaction_class,
                "source_split": "test",
                "pool_name": selection["pool_name"],
                "covered": False,
                "candidate_count": 0,
                "prior_rank": 0,
            }
    output_root = Path(args.result_dir).resolve()
    manifest_path = output_root / "manifest.json"
    predictions_path = output_root / "reaction_ranks.jsonl"
    _require_absent(manifest_path, predictions_path)
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = predictions_path.with_name(f".{predictions_path.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        for reaction_id in sorted(reaction_rows):
            handle.write(json.dumps(reaction_rows[reaction_id], sort_keys=True) + "\n")
    os.replace(temporary, predictions_path)
    if sha256_file(args.model_freeze) != freeze_hash_before:
        raise RuntimeError("WS-E model freeze changed during official-test evaluation.")
    result = {
        "schema_version": 1,
        "record_kind": "ws_e_official_test_results",
        "protocol_id": selection_protocol,
        "source_pool_protocol_id": POOL_PROTOCOL_ID,
        "pool_name": selection["pool_name"],
        "test_reactions_total": len(test_reactions),
        "test_reactions_covered": len(covered),
        "coverage": len(covered) / len(test_reactions),
        "candidate_cap": _candidate_cap(selection),
        "candidate_ordering": selection.get("candidate_ordering"),
        "prior_metrics": {
            "within_pool": _rank_metrics(prior_ranks),
            "end_to_end": _rank_metrics(prior_ranks, denominator=len(test_reactions)),
        },
        "per_seed_metrics": per_seed,
        "predictions": fingerprint(predictions_path),
        "model_freeze": fingerprint(args.model_freeze),
        "feature_freeze": fingerprint(args.feature_freeze),
        "source_csv": fingerprint(args.source_csv),
        "metadata_csv": fingerprint(args.metadata_csv),
        "test_partition_loaded_only_after_model_freeze": True,
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    join = sub.add_parser("join-shard")
    join.add_argument("--pool-name", required=True)
    join.add_argument("--pool-jsonl", required=True)
    join.add_argument("--products-jsonl", required=True)
    join.add_argument("--pool-index-npz", required=True)
    join.add_argument("--pool-index-manifest", required=True)
    join.add_argument("--merged-jsonl", required=True)
    join.add_argument("--merged-index-npz", required=True)
    join.add_argument("--merged-index-manifest", required=True)
    join.add_argument("--scalar-freeze", required=True)
    join.add_argument("--shard-index", type=int, required=True)
    join.add_argument("--shard-count", type=int, required=True)
    join.add_argument("--output", required=True)
    join.add_argument("--manifest", required=True)

    final = sub.add_parser("finalize-features")
    final.add_argument("--pool-name", required=True)
    final.add_argument("--shard-root", required=True)
    final.add_argument("--shard-count", type=int, required=True)
    final.add_argument("--pool-index-manifest", required=True)
    final.add_argument("--output", required=True)

    prepare = sub.add_parser("prepare-selection")
    prepare.add_argument("--feature-freeze", required=True)
    prepare.add_argument("--pool-jsonl", required=True)
    prepare.add_argument("--products-jsonl", required=True)
    prepare.add_argument("--pool-index-npz", required=True)
    prepare.add_argument("--pool-index-manifest", required=True)
    prepare.add_argument("--source-csv", required=True)
    prepare.add_argument("--metadata-csv", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--candidate-cap", type=int)
    prepare.add_argument("--seeds", type=_parse_seed_list, default=SEEDS)
    prepare.add_argument("--output-protocol-id", default=RANKING_PROTOCOL_ID)

    fit = sub.add_parser("fit-validation")
    fit.add_argument("--selection-root", required=True)
    fit.add_argument("--primary-freeze", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--arm", choices=("baseline", "augmented"), required=True)
    fit.add_argument("--seed", type=int, required=True)
    fit.add_argument("--device", default="cpu")

    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--selection-root", required=True)
    freeze_parser.add_argument("--primary-freeze", required=True)
    freeze_parser.add_argument("--output-root", required=True)
    freeze_parser.add_argument("--output", required=True)

    test = sub.add_parser("evaluate-test")
    test.add_argument("--model-freeze", required=True)
    test.add_argument("--feature-freeze", required=True)
    test.add_argument("--selection-root", required=True)
    test.add_argument("--primary-freeze", required=True)
    test.add_argument("--output-root", required=True)
    test.add_argument("--pool-jsonl", required=True)
    test.add_argument("--products-jsonl", required=True)
    test.add_argument("--pool-index-npz", required=True)
    test.add_argument("--pool-index-manifest", required=True)
    test.add_argument("--source-csv", required=True)
    test.add_argument("--metadata-csv", required=True)
    test.add_argument("--result-dir", required=True)
    test.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = {
        "join-shard": join_shard,
        "finalize-features": finalize_features,
        "prepare-selection": prepare_selection,
        "fit-validation": fit_validation,
        "freeze": freeze,
        "evaluate-test": evaluate_test,
    }
    result = commands[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
