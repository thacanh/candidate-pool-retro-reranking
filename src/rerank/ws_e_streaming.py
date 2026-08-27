"""Storage-bounded primitives for the frozen WS-E three-pool comparison.

The candidate pools are large JSONL streams, but are already canonical,
deduplicated, and grouped by product.  This module indexes those streams and
stores only compact pair-level scalar features.  Atom representations are
never persisted.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from rerank.features import (
    _atom_set_similarity,
    _cosine,
    _heavy_atom_ratio,
    _morgan_similarity,
    _reaction_distance,
)


POOL_PROTOCOL_ID = "ws-e-localretro-three-pools-filtered-v2"
FEATURE_PROTOCOL_ID = "ws-e-seed42-streaming-scalars-v1"
RANKING_PROTOCOL_ID = "ws-e-three-pool-frozen-ranker-v1"
CORE_FEATURE_NAMES = (
    "morgan_similarity",
    "atom_set_similarity",
    "reaction_distance",
    "cosine_reaction_vec",
    "n_fragments",
    "heavy_atom_ratio",
)
FULL_FEATURE_NAMES = (
    "prior",
    *CORE_FEATURE_NAMES,
    "source_aizynthfinder",
    "source_localretro",
)
BASELINE_COLUMNS = (0, 1, 5, 6, 7, 8)
AUGMENTED_COLUMNS = tuple(range(len(FULL_FEATURE_NAMES)))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(resolved),
    }


def atomic_json(path: str | Path, payload: Mapping) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, output)


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, output)


def raw_identity_digest(candidate: str) -> bytes:
    """Hash an exact candidate spelling (legacy scalar-archive identity)."""

    return hashlib.sha256(str(candidate).encode("utf-8")).digest()


def identity_digest(candidate: str) -> bytes:
    """Hash stereochemistry-preserving, fragment-order-invariant identity."""

    # Imported lazily to keep the low-level streaming module lightweight and
    # avoid loading study-data dependencies until candidate identity is used.
    from rerank.study_data import canonicalize_reactant_set

    canonical = canonicalize_reactant_set(str(candidate))
    if canonical is None:
        raise ValueError("Cannot construct canonical identity for invalid reactant SMILES.")
    return hashlib.sha256(canonical.encode("utf-8")).digest()


def digest_array(values: Iterable[str]) -> np.ndarray:
    blobs = b"".join(identity_digest(value) for value in values)
    if not blobs:
        return np.empty((0, 32), dtype=np.uint8)
    return np.frombuffer(blobs, dtype=np.uint8).reshape(-1, 32).copy()


def digest_key(row: np.ndarray) -> bytes:
    return np.asarray(row, dtype=np.uint8).tobytes()


def load_products(path: str | Path) -> list[dict]:
    products: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for expected_rank, raw in enumerate(handle):
            record = json.loads(raw)
            if int(record.get("product_rank", -1)) != expected_rank:
                raise ValueError("WS-E products are not in contiguous product-rank order.")
            product = str(record.get("canonical_product", ""))
            if not product:
                raise ValueError(f"Missing canonical product at rank {expected_rank}.")
            products.append(record)
    if not products:
        raise ValueError("WS-E product inventory is empty.")
    return products


@dataclass(frozen=True)
class PoolIndex:
    starts: np.ndarray
    counts: np.ndarray
    products: tuple[str, ...]
    canonical_products: tuple[str, ...]
    pool_fingerprint: dict
    products_fingerprint: dict

    @property
    def product_count(self) -> int:
        return len(self.products)

    @property
    def candidate_count(self) -> int:
        return int(self.counts.sum())


def build_pool_index(
    pool_path: str | Path,
    products_path: str | Path,
    output_npz: str | Path,
    output_manifest: str | Path,
) -> dict:
    """Index byte starts/counts while validating grouped canonical JSONL."""

    pool_path = Path(pool_path).resolve()
    products_path = Path(products_path).resolve()
    products = load_products(products_path)
    # Candidate records intentionally retain the source product spelling while
    # ``canonical_product`` is a separate identity used for reaction joins.
    product_keys = tuple(str(item["product"]) for item in products)
    rank_by_product = {value: index for index, value in enumerate(product_keys)}
    if len(rank_by_product) != len(product_keys):
        raise ValueError("WS-E product inventory contains duplicate canonical products.")

    starts = np.full(len(product_keys), -1, dtype=np.int64)
    counts = np.zeros(len(product_keys), dtype=np.int32)
    previous_rank = -1
    expected_candidate_rank = 0
    line_count = 0
    with open(pool_path, "rb") as handle:
        while True:
            position = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            if not raw.strip():
                continue
            record = json.loads(raw)
            if record.get("protocol_id") != POOL_PROTOCOL_ID:
                raise ValueError("Candidate record has the wrong WS-E protocol ID.")
            product = str(record.get("product", ""))
            try:
                product_rank = rank_by_product[product]
            except KeyError as exc:
                raise ValueError(f"Unexpected WS-E product {product!r}.") from exc
            if product_rank < previous_rank:
                raise ValueError("Candidate products reappear non-contiguously.")
            if product_rank != previous_rank:
                expected_candidate_rank = 1
                if starts[product_rank] != -1:
                    raise ValueError("Candidate product reappears non-contiguously.")
                starts[product_rank] = position
                previous_rank = product_rank
            else:
                expected_candidate_rank += 1
            if int(record.get("candidate_rank", -1)) != expected_candidate_rank:
                raise ValueError(
                    f"Non-contiguous candidate rank for product {product_rank}."
                )
            candidate = str(record.get("reactant", ""))
            if not candidate:
                raise ValueError("Candidate record has no canonical reactant.")
            if int(record.get("source_aizynthfinder", -1)) not in (0, 1):
                raise ValueError("Invalid AiZynthFinder source indicator.")
            if int(record.get("source_localretro", -1)) not in (0, 1):
                raise ValueError("Invalid LocalRetro source indicator.")
            counts[product_rank] += 1
            line_count += 1

    atomic_npz(output_npz, starts=starts, counts=counts)
    manifest = {
        "schema_version": 1,
        "record_kind": "ws_e_pool_byte_index",
        "protocol_id": FEATURE_PROTOCOL_ID,
        "source_pool_protocol_id": POOL_PROTOCOL_ID,
        "pool": fingerprint(pool_path),
        "products": fingerprint(products_path),
        "index": fingerprint(output_npz),
        "product_count": len(product_keys),
        "candidate_count": line_count,
        "products_without_candidates": int(np.sum(counts == 0)),
        "test_partition_loaded": False,
        "ground_truth_loaded": False,
        "created_at_utc": utc_now(),
    }
    atomic_json(output_manifest, manifest)
    return manifest


def load_pool_index(
    index_npz: str | Path,
    index_manifest: str | Path,
    products_path: str | Path,
    pool_path: str | Path,
) -> PoolIndex:
    manifest = json.loads(Path(index_manifest).read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != FEATURE_PROTOCOL_ID:
        raise ValueError("Wrong WS-E index protocol.")
    if fingerprint(pool_path)["sha256"] != manifest["pool"]["sha256"]:
        raise ValueError("WS-E pool differs from its byte-index manifest.")
    if fingerprint(products_path)["sha256"] != manifest["products"]["sha256"]:
        raise ValueError("WS-E products differ from their byte-index manifest.")
    if fingerprint(index_npz)["sha256"] != manifest["index"]["sha256"]:
        raise ValueError("WS-E byte index differs from its manifest.")
    arrays = np.load(index_npz, allow_pickle=False)
    product_records = load_products(products_path)
    products = tuple(str(item["product"]) for item in product_records)
    canonical_products = tuple(
        str(item["canonical_product"]) for item in product_records
    )
    starts = arrays["starts"].astype(np.int64, copy=False)
    counts = arrays["counts"].astype(np.int32, copy=False)
    if starts.shape != counts.shape or len(starts) != len(products):
        raise ValueError("WS-E byte index arrays are misaligned.")
    return PoolIndex(
        starts=starts,
        counts=counts,
        products=products,
        canonical_products=canonical_products,
        pool_fingerprint=manifest["pool"],
        products_fingerprint=manifest["products"],
    )


def read_product_records(
    pool_path: str | Path, index: PoolIndex, product_rank: int
) -> list[dict]:
    count = int(index.counts[product_rank])
    if count == 0:
        return []
    start = int(index.starts[product_rank])
    if start < 0:
        raise ValueError("Non-empty product has no byte offset.")
    records: list[dict] = []
    with open(pool_path, "rb") as handle:
        handle.seek(start)
        for expected_rank in range(1, count + 1):
            raw = handle.readline()
            if not raw:
                raise EOFError("WS-E pool ended inside an indexed product.")
            record = json.loads(raw)
            if str(record.get("product")) != index.products[product_rank]:
                raise ValueError("Indexed WS-E product identity mismatch.")
            if int(record.get("candidate_rank", -1)) != expected_rank:
                raise ValueError("Indexed WS-E candidate order mismatch.")
            records.append(record)
    return records


def shard_bounds(product_count: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("Invalid shard index/count.")
    start = product_count * shard_index // shard_count
    stop = product_count * (shard_index + 1) // shard_count
    return start, stop


def fragment_smiles(candidate: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in candidate.split(".") if value.strip())


def combine_embeddings(
    fragments: Sequence[str], embeddings: Mapping[str, np.ndarray | None]
) -> np.ndarray:
    arrays = [embeddings[value] for value in fragments if embeddings.get(value) is not None]
    if not arrays:
        return np.zeros((1, 512), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)


def core_scalar_row(
    product: str,
    candidate: str,
    product_embedding: np.ndarray,
    candidate_embedding: np.ndarray,
) -> np.ndarray:
    product_embedding = np.asarray(product_embedding, dtype=np.float32)
    candidate_embedding = np.asarray(candidate_embedding, dtype=np.float32)
    product_mean = product_embedding.mean(axis=0)
    reaction_vector = product_mean - candidate_embedding.mean(axis=0)
    return np.asarray(
        [
            _morgan_similarity(product, candidate),
            _atom_set_similarity(product_embedding, candidate_embedding),
            _reaction_distance(product_embedding, candidate_embedding),
            _cosine(product_mean, reaction_vector),
            float(len(fragment_smiles(candidate))),
            _heavy_atom_ratio(product, candidate),
        ],
        dtype=np.float32,
    )


def core_scalar_rows(
    product: str,
    candidates: Sequence[str],
    product_embedding: np.ndarray,
    candidate_embeddings: Sequence[np.ndarray],
) -> np.ndarray:
    """Compute a product group while reusing product-only intermediates.

    This is algebraically identical to calling :func:`core_scalar_row` for
    every candidate; it only avoids repeatedly normalizing and averaging the
    same product atom matrix.
    """

    if len(candidates) != len(candidate_embeddings):
        raise ValueError("Candidate strings and atom representations are misaligned.")
    product_embedding = np.asarray(product_embedding, dtype=np.float32)
    product_mean = product_embedding.mean(axis=0)
    product_norm = product_embedding / (
        np.linalg.norm(product_embedding, axis=1, keepdims=True) + 1e-10
    )
    rows = []
    for candidate, candidate_embedding in zip(candidates, candidate_embeddings):
        reactant = np.asarray(candidate_embedding, dtype=np.float32)
        reactant_mean = reactant.mean(axis=0)
        reactant_norm = reactant / (
            np.linalg.norm(reactant, axis=1, keepdims=True) + 1e-10
        )
        similarity = product_norm @ reactant_norm.T
        atom_set = float(
            0.5 * (similarity.max(axis=1).mean() + similarity.max(axis=0).mean())
        )
        reaction_vector = product_mean - reactant_mean
        rows.append(
            [
                _morgan_similarity(product, candidate),
                atom_set,
                float(np.linalg.norm(reaction_vector)),
                _cosine(product_mean, reaction_vector),
                float(len(fragment_smiles(candidate))),
                _heavy_atom_ratio(product, candidate),
            ]
        )
    if not rows:
        return np.empty((0, len(CORE_FEATURE_NAMES)), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def full_feature_matrix(records: Sequence[Mapping], core: np.ndarray) -> np.ndarray:
    if len(records) != len(core) or core.ndim != 2 or core.shape[1] != len(CORE_FEATURE_NAMES):
        raise ValueError("WS-E records and core features are misaligned.")
    result = np.empty((len(records), len(FULL_FEATURE_NAMES)), dtype=np.float32)
    result[:, 0] = np.asarray([float(item["prior"]) for item in records], dtype=np.float32)
    result[:, 1:7] = core
    result[:, 7] = np.asarray(
        [int(item["source_aizynthfinder"]) for item in records], dtype=np.float32
    )
    result[:, 8] = np.asarray(
        [int(item["source_localretro"]) for item in records], dtype=np.float32
    )
    if not np.isfinite(result).all():
        raise ValueError("WS-E feature matrix contains non-finite values.")
    return result


def arm_view(features: np.ndarray, arm: str) -> np.ndarray:
    columns = BASELINE_COLUMNS if arm == "baseline" else AUGMENTED_COLUMNS
    if arm not in {"baseline", "augmented"}:
        raise ValueError("arm must be baseline or augmented")
    return np.asarray(features[:, columns], dtype=np.float32)
