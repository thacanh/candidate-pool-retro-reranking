"""Build compact, resumable seed-42 WS-E pair-level scalar features.

The merged canonical union is encoded once.  Each product is processed in
isolation: live atom representations are used to compute six scalar features
and are then discarded.  Output shards contain only SHA-256 candidate IDs and
float32 scalars.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from rerank.data.build_conformer_sqlite import (
    SeededUniMolBackend,
    discover_weight_paths,
    validate_conformer_seed,
)
from rerank.ws_e_streaming import (
    CORE_FEATURE_NAMES,
    FEATURE_PROTOCOL_ID,
    atomic_json,
    atomic_npz,
    build_pool_index,
    combine_embeddings,
    core_scalar_rows,
    digest_array,
    fingerprint,
    fragment_smiles,
    load_pool_index,
    read_product_records,
    sha256_file,
    shard_bounds,
    utc_now,
)


def build_feature_shard(args: argparse.Namespace) -> dict:
    seed = validate_conformer_seed(args.conformer_seed)
    if seed != 42:
        raise ValueError("WS-E frozen representation uses conformer seed 42 only.")
    if args.lru_items < 256:
        raise ValueError("--lru-items must be at least 256 for one complete product.")
    index = load_pool_index(
        args.index_npz,
        args.index_manifest,
        args.products_jsonl,
        args.merged_jsonl,
    )
    start_rank, stop_rank = shard_bounds(
        index.product_count, args.shard_index, args.shard_count
    )
    if args.limit_products is not None:
        if args.limit_products < 1:
            raise ValueError("--limit-products must be positive.")
        stop_rank = min(stop_rank, start_rank + args.limit_products)
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a WS-E scalar shard or manifest.")

    checkpoint, dictionary = discover_weight_paths(args.checkpoint, args.dictionary)
    backend = SeededUniMolBackend(
        seed=seed,
        checkpoint=checkpoint,
        dictionary=dictionary,
        batch_size=args.batch_size,
        threads=args.threads,
        device=args.device,
    )
    started = time.perf_counter()
    product_ranks: list[int] = []
    candidates: list[str] = []
    scalar_parts: list[np.ndarray] = []
    status_counts: Counter[str] = Counter()
    failures: list[dict] = []
    embedding_cache: OrderedDict[str, np.ndarray | None] = OrderedDict()
    cache_hits = 0
    cache_misses = 0
    backend_seconds = 0.0
    scalar_seconds = 0.0

    for product_rank in range(start_rank, stop_rank):
        records = read_product_records(args.merged_jsonl, index, product_rank)
        if not records:
            continue
        product = index.canonical_products[product_rank]
        required = {product}
        for record in records:
            required.update(fragment_smiles(str(record["reactant"])))
        ordered_required = sorted(required)
        missing = [value for value in ordered_required if value not in embedding_cache]
        for value in ordered_required:
            if value in embedding_cache:
                embedding_cache.move_to_end(value)
        cache_hits += len(ordered_required) - len(missing)
        cache_misses += len(missing)
        backend_started = time.perf_counter()
        if args.verbose_backend:
            arrays, statuses, errors = backend.encode_batch(missing)
        else:
            # Uni-Mol emits one tqdm per product.  Keep Vast/remote output to a
            # single outer progress line while preserving raised exceptions.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                arrays, statuses, errors = backend.encode_batch(missing)
        backend_seconds += time.perf_counter() - backend_started
        for smiles, array in zip(missing, arrays):
            embedding_cache[smiles] = array
            embedding_cache.move_to_end(smiles)
        for smiles, status, error in zip(missing, statuses, errors):
            status_counts[status] += 1
            if error is not None or embedding_cache[smiles] is None:
                failures.append(
                    {
                        "product_rank": product_rank,
                        "smiles_sha256": sha256_file_bytes(smiles),
                        "status": status,
                        "error": error,
                    }
                )
        while len(embedding_cache) > args.lru_items:
            embedding_cache.popitem(last=False)
        embeddings = {value: embedding_cache[value] for value in ordered_required}
        product_embedding = combine_embeddings((product,), embeddings)
        candidate_values = [str(record["reactant"]) for record in records]
        candidate_embeddings = [
            combine_embeddings(fragment_smiles(candidate), embeddings)
            for candidate in candidate_values
        ]
        scalar_started = time.perf_counter()
        rows = core_scalar_rows(
            product, candidate_values, product_embedding, candidate_embeddings
        )
        scalar_seconds += time.perf_counter() - scalar_started
        for candidate in candidate_values:
            product_ranks.append(product_rank)
            candidates.append(candidate)
        scalar_parts.append(rows.astype(np.float32, copy=False))
        completed = product_rank - start_rank + 1
        total = max(stop_rank - start_rank, 1)
        if completed % 25 == 0 or product_rank + 1 == stop_rank:
            elapsed = time.perf_counter() - started
            rate = completed / max(elapsed, 1e-9)
            eta = (total - completed) / max(rate, 1e-9)
            print(
                f"\rWS-E shard {args.shard_index + 1}/{args.shard_count}: "
                f"{completed}/{total} products ({100*completed/total:.1f}%) | "
                f"ETA {eta/60:.1f} min",
                end="",
                flush=True,
            )
    print(flush=True)

    scalars = (
        np.concatenate(scalar_parts, axis=0)
        if scalar_parts
        else np.empty((0, len(CORE_FEATURE_NAMES)), dtype=np.float32)
    )
    atomic_npz(
        output,
        product_rank=np.asarray(product_ranks, dtype=np.int32),
        candidate_sha256=digest_array(candidates),
        core_features=scalars,
    )
    manifest = {
        "schema_version": 1,
        "record_kind": "ws_e_scalar_feature_shard",
        "protocol_id": FEATURE_PROTOCOL_ID,
        "source_pool_protocol_id": "ws-e-localretro-three-pools-filtered-v2",
        "comparator": "same candidate identity with no persisted atom representations",
        "single_intended_change": "compute six frozen pair-level scalars for WS-E candidates",
        "conformer_seed": seed,
        "feature_names": list(CORE_FEATURE_NAMES),
        "product_rank_start": start_rank,
        "product_rank_stop": stop_rank,
        "products_processed": stop_rank - start_rank,
        "candidate_rows": len(candidates),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "smoke_limit_products": args.limit_products,
        "status_counts": dict(sorted(status_counts.items())),
        "bounded_embedding_lru": {
            "maximum_items": args.lru_items,
            "final_items": len(embedding_cache),
            "hits": cache_hits,
            "misses": cache_misses,
            "persisted": False,
        },
        "failed_embedding_items": len(failures),
        "failures": failures,
        "atom_embeddings_written": False,
        "test_partition_loaded": False,
        "ground_truth_loaded": False,
        "inputs": {
            "merged_pool": index.pool_fingerprint,
            "products": index.products_fingerprint,
            "index": fingerprint(args.index_npz),
        },
        "backend": backend.metadata(),
        "output": fingerprint(output),
        "runtime_seconds": time.perf_counter() - started,
        "timing_breakdown_seconds": {
            "backend_encode": backend_seconds,
            "scalar_computation": scalar_seconds,
        },
        "host": {
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "created_at_utc": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def sha256_file_bytes(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def finalize(args: argparse.Namespace) -> dict:
    root = Path(args.shard_root).resolve()
    manifests = []
    expected_start = 0
    total_rows = 0
    outputs = []
    backend_identity = None
    for shard_index in range(args.shard_count):
        manifest_path = root / f"shard_{shard_index:03d}_of_{args.shard_count:03d}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing WS-E scalar manifest: {manifest_path}")
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        if record.get("protocol_id") != FEATURE_PROTOCOL_ID:
            raise ValueError("WS-E scalar shard has the wrong protocol.")
        if record.get("smoke_limit_products") is not None:
            raise ValueError("A smoke-limited shard cannot enter the scientific manifest.")
        if int(record["shard_index"]) != shard_index or int(record["shard_count"]) != args.shard_count:
            raise ValueError("WS-E scalar shard numbering mismatch.")
        if int(record["product_rank_start"]) != expected_start:
            raise ValueError("WS-E scalar shards do not cover contiguous product ranks.")
        expected_start = int(record["product_rank_stop"])
        output = Path(record["output"]["path"])
        if sha256_file(output) != record["output"]["sha256"]:
            raise ValueError("WS-E scalar shard checksum mismatch.")
        arrays = np.load(output, allow_pickle=False)
        rows = len(arrays["product_rank"])
        if arrays["candidate_sha256"].shape != (rows, 32):
            raise ValueError("WS-E candidate digest array is malformed.")
        if arrays["core_features"].shape != (rows, len(CORE_FEATURE_NAMES)):
            raise ValueError("WS-E core feature array is malformed.")
        if rows != int(record["candidate_rows"]):
            raise ValueError("WS-E scalar manifest row count mismatch.")
        if not np.isfinite(arrays["core_features"]).all():
            raise ValueError("WS-E scalar shard contains non-finite values.")
        backend = record["backend"]
        current_backend_identity = {
            "seed": backend["seed"],
            "device": backend["device"],
            "batch_size": backend["batch_size"],
            "threads": backend["threads"],
            "unimol_tools_version": backend["unimol_tools_version"],
            "rdkit_version": backend["rdkit_version"],
            "torch_version": backend["torch_version"],
            "torch_cuda_version": backend["torch_cuda_version"],
            "cuda_device": backend["cuda_device"],
            "conformer": backend["conformer"],
            "checkpoint_sha256": backend["checkpoint"]["sha256"],
            "dictionary_sha256": backend["dictionary"]["sha256"],
            "lru_items": record["bounded_embedding_lru"]["maximum_items"],
        }
        if backend_identity is None:
            backend_identity = current_backend_identity
        elif current_backend_identity != backend_identity:
            raise ValueError(
                "WS-E scalar shards were produced by different execution backends."
            )
        total_rows += rows
        outputs.append(record["output"])
        manifests.append(fingerprint(manifest_path))

    index_manifest = json.loads(Path(args.index_manifest).read_text(encoding="utf-8"))
    if expected_start != int(index_manifest["product_count"]):
        raise ValueError("WS-E scalar shards do not cover the full product inventory.")
    if total_rows != int(index_manifest["candidate_count"]):
        raise ValueError("WS-E scalar rows do not cover the merged candidate union.")
    result = {
        "schema_version": 1,
        "record_kind": "ws_e_scalar_feature_freeze",
        "protocol_id": FEATURE_PROTOCOL_ID,
        "source_pool_protocol_id": "ws-e-localretro-three-pools-filtered-v2",
        "conformer_seed": 42,
        "feature_names": list(CORE_FEATURE_NAMES),
        "product_count": expected_start,
        "candidate_rows": total_rows,
        "shard_count": args.shard_count,
        "shards": outputs,
        "shard_manifests": manifests,
        "backend_identity": backend_identity,
        "index_manifest": fingerprint(args.index_manifest),
        "atom_embeddings_written": False,
        "test_partition_loaded": False,
        "ground_truth_loaded": False,
        "complete": True,
        "created_at_utc": utc_now(),
    }
    atomic_json(args.output, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("--merged-jsonl", required=True)
    index_parser.add_argument("--products-jsonl", required=True)
    index_parser.add_argument("--output-npz", required=True)
    index_parser.add_argument("--output-manifest", required=True)

    shard_parser = subparsers.add_parser("build-shard")
    shard_parser.add_argument("--merged-jsonl", required=True)
    shard_parser.add_argument("--products-jsonl", required=True)
    shard_parser.add_argument("--index-npz", required=True)
    shard_parser.add_argument("--index-manifest", required=True)
    shard_parser.add_argument("--output", required=True)
    shard_parser.add_argument("--manifest", required=True)
    shard_parser.add_argument("--shard-index", type=int, required=True)
    shard_parser.add_argument("--shard-count", type=int, required=True)
    shard_parser.add_argument("--conformer-seed", type=int, default=42)
    shard_parser.add_argument("--checkpoint")
    shard_parser.add_argument("--dictionary")
    shard_parser.add_argument("--batch-size", type=int, default=16)
    shard_parser.add_argument("--threads", type=int, default=8)
    shard_parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    shard_parser.add_argument("--lru-items", type=int, default=5000)
    shard_parser.add_argument("--verbose-backend", action="store_true")
    shard_parser.add_argument("--limit-products", type=int)

    final_parser = subparsers.add_parser("finalize")
    final_parser.add_argument("--shard-root", required=True)
    final_parser.add_argument("--shard-count", type=int, required=True)
    final_parser.add_argument("--index-manifest", required=True)
    final_parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "index":
        result = build_pool_index(
            args.merged_jsonl,
            args.products_jsonl,
            args.output_npz,
            args.output_manifest,
        )
    elif args.command == "build-shard":
        result = build_feature_shard(args)
    else:
        result = finalize(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
