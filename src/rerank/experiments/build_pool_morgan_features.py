"""N4 atom-centred Morgan features on frozen WS-E cap-10 candidate lists.

This stage touches candidates but never official-test references.  Its output
uses the same seven feature definitions as the historical Morgan control;
source-indicator columns are deliberately excluded from *all* N4 pools.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from rerank.encoder_controls import MorganAtomEncoder, compute_query_control_features
from rerank.experiments.protocol_gate import fingerprint, require_approval, resolve_frozen
from rerank.pool_streaming import (
    POOL_PROTOCOL_ID, RANKING_PROTOCOL_ID, atomic_json, atomic_npz,
    digest_array, fingerprint as ws_fingerprint, load_pool_index, load_products,
    read_product_records, shard_bounds,
)


PROTOCOL_ID = "dd-n4-morgan-atom-transfer-v1"
FEATURE_NAMES = (
    "prior", "morgan_similarity", "atom_set_similarity", "reaction_distance",
    "cosine_reaction_vec", "n_fragments", "heavy_atom_ratio",
)
BASELINE_COLUMNS = (0, 1, 5, 6)
AUGMENTED_COLUMNS = tuple(range(7))


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def _verified_index(index_npz: str, index_manifest: str, products_jsonl: str,
                    pool_jsonl: str):
    """Verify the large pool once per persistent feature worker."""
    return load_pool_index(index_npz, index_manifest, products_jsonl, pool_jsonl)


@lru_cache(maxsize=2)
def _verified_products(products_jsonl: str):
    return load_products(products_jsonl)


def build_shard(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    raw = _load(args.raw_feature_freeze)
    if (raw.get("protocol_id") != RANKING_PROTOCOL_ID or not raw.get("complete")
            or raw.get("pool_name") != args.pool_name
            or raw.get("pool_index_manifest", {}).get("sha256")
            != fingerprint(args.pool_index_manifest)["sha256"]):
        raise ValueError("Wrong frozen WS-E source feature/index pair.")
    if args.shard_count != int(raw["shard_count"]) or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("N4 shard index/count must match the source feature freeze.")
    pool_index = _verified_index(args.pool_index_npz, args.pool_index_manifest,
                                 args.products_jsonl, args.pool_jsonl)
    products = _verified_products(args.products_jsonl)
    start, stop = shard_bounds(len(products), args.shard_index, args.shard_count)
    source_record = raw["shard_manifests"][args.shard_index]
    source_manifest_path = resolve_frozen(source_record,
                                           Path(args.raw_feature_freeze).parent / "shards")
    source_manifest = _load(source_manifest_path)
    if (int(source_manifest["product_rank_start"]) != start
            or int(source_manifest["product_rank_stop"]) != stop):
        raise ValueError("Source feature shard product bounds differ.")
    source_npz = resolve_frozen(source_manifest["output"], source_manifest_path.parent)
    with np.load(source_npz, allow_pickle=False) as arrays:
        source_ranks = arrays["product_rank"]
        source_digests = arrays["candidate_sha256"]
        source_features = arrays["features"]
        if source_features.ndim != 2 or source_features.shape[1] != 9:
            raise ValueError("Source WS-E shard must have nine columns.")
        source_ranks = source_ranks.copy()
        source_digests = source_digests.copy()
        source_features = source_features.copy()

    encoder = MorganAtomEncoder()
    encoder.initialize()
    result_ranks: list[int] = []
    result_digests: list[np.ndarray] = []
    result_features: list[np.ndarray] = []
    boundaries = np.flatnonzero(np.r_[True, source_ranks[1:] != source_ranks[:-1], True])
    slices = {int(source_ranks[left]): (int(left), int(right))
              for left, right in zip(boundaries[:-1], boundaries[1:])}
    for product_rank in range(start, stop):
        records = read_product_records(args.pool_jsonl, pool_index, product_rank)[:10]
        n_rows = len(records)
        if n_rows:
            left, right = slices[product_rank]
            if right - left < n_rows:
                raise ValueError("Source feature shard is shorter than the frozen candidate list.")
            end = left + n_rows
            if (not np.all(source_ranks[left:end] == product_rank)
                    or not np.array_equal(source_digests[left:end],
                                          digest_array(str(row["reactant"]) for row in records))):
                raise ValueError(f"N4 candidate order/digest differs at product {product_rank}.")
            source = source_features[left:end]
            controls = compute_query_control_features(
                encoder, str(products[product_rank]["product"]),
                [str(row["reactant"]) for row in records], args.encoder_batch_size,
            )
            full = np.column_stack((source[:, :2], controls, source[:, 5:7])).astype(np.float32)
            if full.shape != (n_rows, 7) or not np.isfinite(full).all():
                raise ValueError("N4 Morgan matrix is malformed/non-finite.")
            result_ranks.extend([product_rank] * n_rows)
            result_digests.append(source_digests[left:end])
            result_features.append(full)
        if product_rank % 25 == 0:
            print(f"\rN4 {args.pool_name} shard {args.shard_index + 1}/{args.shard_count}: "
                  f"product {product_rank - start + 1}/{stop - start}", end="", flush=True)
    print(flush=True)
    # Source shards contain full cap-50 or cap-100 rows. Check *every* source
    # product's first ten, not a flattened prefix of the source shard.
    output_root = Path(args.output_root).resolve()
    shard_root = output_root / "shards"
    stem = f"shard_{args.shard_index:03d}_of_{args.shard_count:03d}"
    output_npz = shard_root / f"{stem}.npz"
    manifest_path = shard_root / f"{stem}.json"
    if output_npz.exists() or manifest_path.exists():
        raise FileExistsError(f"N4 shard already exists: {stem}")
    atomic_npz(output_npz,
               product_rank=np.asarray(result_ranks, dtype=np.int32),
               candidate_sha256=(np.concatenate(result_digests) if result_digests
                                 else np.empty((0, 32), dtype=np.uint8)),
               features=(np.concatenate(result_features) if result_features
                         else np.empty((0, 7), dtype=np.float32)))
    manifest = {
        "schema_version": 1, "record_kind": "dd_n4_morgan_feature_shard",
        "protocol_id": RANKING_PROTOCOL_ID, "representation_control_protocol_id": PROTOCOL_ID,
        "pool_name": args.pool_name, "product_rank_start": start, "product_rank_stop": stop,
        "candidate_rows": len(result_ranks), "shard_index": args.shard_index,
        "shard_count": args.shard_count, "feature_names": list(FEATURE_NAMES),
        "encoder": encoder.metadata(), "candidate_cap": 10,
        "candidate_ordering": "first ten rows in each frozen pool list; no re-sorting",
        "inputs": {"source_feature_shard": fingerprint(source_npz),
                   "source_feature_manifest": fingerprint(source_manifest_path),
                   "pool": pool_index.pool_fingerprint,
                   "products": pool_index.products_fingerprint},
        "output": ws_fingerprint(output_npz), "runtime_seconds": time.perf_counter() - started,
        "test_ground_truth_loaded": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def freeze(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    root = Path(args.output_root).resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError("N4 feature freeze already exists.")
    index = _load(args.pool_index_manifest)
    previous_stop = 0
    shard_manifests = []
    candidate_rows = 0
    encoder = None
    for shard_index in range(args.shard_count):
        path = root / "shards" / f"shard_{shard_index:03d}_of_{args.shard_count:03d}.json"
        record = _load(path)
        if (record.get("representation_control_protocol_id") != PROTOCOL_ID
                or record.get("pool_name") != args.pool_name
                or record.get("shard_index") != shard_index
                or record.get("product_rank_start") != previous_stop):
            raise ValueError("N4 shard protocol, pool, or product sequence differs.")
        resolve_frozen(record["output"], path.parent)
        previous_stop = int(record["product_rank_stop"])
        candidate_rows += int(record["candidate_rows"])
        if encoder is not None and encoder != record["encoder"]:
            raise ValueError("N4 atom encoder metadata differs across shards.")
        encoder = record["encoder"]
        shard_manifests.append(fingerprint(path))
    if previous_stop != int(index["product_count"]):
        raise ValueError("N4 shards do not cover the frozen product inventory.")
    result = {
        "schema_version": 1, "record_kind": "dd_n4_morgan_feature_freeze",
        "protocol_id": RANKING_PROTOCOL_ID, "representation_control_protocol_id": PROTOCOL_ID,
        "source_pool_protocol_id": POOL_PROTOCOL_ID, "pool_name": args.pool_name,
        "product_count": previous_stop, "candidate_rows": candidate_rows,
        "feature_names": list(FEATURE_NAMES), "baseline_columns": list(BASELINE_COLUMNS),
        "augmented_columns": list(AUGMENTED_COLUMNS), "source_indicators_in_both_arms": False,
        "single_intended_arm_change": "add three atom-centred Morgan pair-level scalars",
        "encoder": encoder, "candidate_cap": 10,
        "shard_count": args.shard_count, "shard_manifests": shard_manifests,
        "pool_index_manifest": fingerprint(args.pool_index_manifest),
        "approval": fingerprint(args.approval_record),
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "test_partition_loaded": False, "ground_truth_loaded": False,
        "complete": True, "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(root / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-shard", "build-range", "freeze"))
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--pool-name", required=True)
    parser.add_argument("--pool-index-manifest", required=True)
    parser.add_argument("--shard-count", type=int, default=64)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--raw-feature-freeze")
    parser.add_argument("--pool-index-npz")
    parser.add_argument("--pool-jsonl")
    parser.add_argument("--products-jsonl")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-indices", help="Comma-separated indices for one persistent worker")
    parser.add_argument("--encoder-batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.command in {"build-shard", "build-range"}:
        required = ("raw_feature_freeze", "pool_index_npz", "pool_jsonl", "products_jsonl")
        if args.command == "build-shard":
            required += ("shard_index",)
        else:
            required += ("shard_indices",)
        for name in required:
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required for {args.command}")
        if args.command == "build-shard":
            result = build_shard(args)
        else:
            indices = tuple(int(value) for value in args.shard_indices.split(","))
            if not indices or len(set(indices)) != len(indices):
                parser.error("--shard-indices must list distinct shard indices")
            result = {"completed_shards": [build_shard(argparse.Namespace(
                **{**vars(args), "shard_index": index})
            )["shard_index"] for index in indices]}
    else:
        result = freeze(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
