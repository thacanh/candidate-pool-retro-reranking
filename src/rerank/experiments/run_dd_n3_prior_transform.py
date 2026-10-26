"""Build immutable N3 cap-10 prior-transform feature overlays (no labels).

Only column zero changes; all candidate digests, ordering and other feature
bytes are retained. The official-test partition is not used for selection.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from rerank.experiments.dd_remote_gate import fingerprint, require_approval, resolve_frozen
from rerank.tuning import transform_prior_values
from rerank.ws_e_streaming import RANKING_PROTOCOL_ID, atomic_json, atomic_npz


PROTOCOL_ID = "dd-n3-pool-specific-selection-v1"
TRANSFORMS = ("log", "rank")


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def transformed_shard(matrix: np.ndarray, ranks: np.ndarray, transform: str) -> np.ndarray:
    """Transform each product's first ten priors, preserving all other bytes."""

    source = np.asarray(matrix, dtype=np.float32)
    groups = np.asarray(ranks)
    if (source.ndim != 2 or source.shape[1] != 9 or groups.shape != (len(source),)
            or transform not in TRANSFORMS or not np.isfinite(source).all()):
        raise ValueError("Malformed frozen nine-column source shard/transform.")
    if np.any(groups[1:] < groups[:-1]):
        raise ValueError("Product ranks are not grouped in ascending order.")
    result = source.copy()
    boundaries = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        end = min(int(start) + 10, int(stop))
        result[start:end, 0] = transform_prior_values(source[start:end, 0], transform)
    if not np.array_equal(result[:, 1:], source[:, 1:]):
        raise AssertionError("N3 changed a non-prior feature.")
    return result


def build_shard(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    raw = _load(args.raw_feature_freeze)
    if (raw.get("protocol_id") != RANKING_PROTOCOL_ID or not raw.get("complete")
            or raw.get("pool_name") != args.pool_name
            or not 0 <= args.shard_index < int(raw["shard_count"])):
        raise ValueError("Wrong frozen WS-E source feature shard.")
    source_manifest_path = resolve_frozen(raw["shard_manifests"][args.shard_index],
                                           Path(args.raw_feature_freeze).parent / "shards")
    source_manifest = _load(source_manifest_path)
    if (source_manifest.get("pool_name") != args.pool_name
            or source_manifest.get("shard_index") != args.shard_index):
        raise ValueError("N3 source feature shard order/pool differs.")
    source_npz = resolve_frozen(source_manifest["output"], source_manifest_path.parent)
    with np.load(source_npz, allow_pickle=False) as arrays:
        ranks = arrays["product_rank"].copy()
        digests = arrays["candidate_sha256"].copy()
        features = transformed_shard(arrays["features"], ranks, args.transform)
    root = Path(args.output_root).resolve()
    shard_root = root / "shards"
    stem = f"shard_{args.shard_index:03d}_of_{raw['shard_count']:03d}"
    output = shard_root / f"{stem}.npz"
    manifest_path = shard_root / f"{stem}.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError("N3 transformed feature shard already exists.")
    atomic_npz(output, product_rank=ranks, candidate_sha256=digests, features=features)
    result = {
        "schema_version": 1, "record_kind": "dd_n3_transformed_feature_shard",
        "protocol_id": RANKING_PROTOCOL_ID, "selection_protocol_id": PROTOCOL_ID,
        "pool_name": args.pool_name, "prior_transform": args.transform,
        "product_rank_start": int(source_manifest["product_rank_start"]),
        "product_rank_stop": int(source_manifest["product_rank_stop"]),
        "candidate_rows": len(ranks), "shard_index": args.shard_index,
        "shard_count": int(raw["shard_count"]),
        "single_intended_change": "transform prior column within each first-ten candidate list",
        "input": fingerprint(source_npz), "output": fingerprint(output),
        "test_references_loaded": False, "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest_path, result)
    return result


def freeze(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    raw = _load(args.raw_feature_freeze)
    if raw.get("pool_name") != args.pool_name or not raw.get("complete"):
        raise ValueError("Wrong source feature freeze.")
    root = Path(args.output_root).resolve()
    if (root / "manifest.json").exists():
        raise FileExistsError("N3 transformed feature freeze already exists.")
    records = []
    previous_stop = 0
    candidate_rows = 0
    for shard_index in range(int(raw["shard_count"])):
        path = root / "shards" / f"shard_{shard_index:03d}_of_{raw['shard_count']:03d}.json"
        record = _load(path)
        if (record.get("selection_protocol_id") != PROTOCOL_ID
                or record.get("pool_name") != args.pool_name
                or record.get("prior_transform") != args.transform
                or int(record["product_rank_start"]) != previous_stop):
            raise ValueError("N3 transformed shard gap/protocol mismatch.")
        resolve_frozen(record["output"], path.parent)
        previous_stop = int(record["product_rank_stop"])
        candidate_rows += int(record["candidate_rows"])
        records.append(fingerprint(path))
    if previous_stop != int(raw["product_count"]) or candidate_rows != int(raw["candidate_rows"]):
        raise ValueError("N3 transformed features do not cover the frozen source.")
    result = {
        "schema_version": 1, "record_kind": "dd_n3_transformed_feature_freeze",
        "protocol_id": RANKING_PROTOCOL_ID, "selection_protocol_id": PROTOCOL_ID,
        "pool_name": args.pool_name, "prior_transform": args.transform,
        "product_count": previous_stop, "candidate_rows": candidate_rows,
        "candidate_cap": 10, "feature_names": raw["feature_names"],
        "baseline_columns": raw["baseline_columns"],
        "augmented_columns": raw["augmented_columns"],
        "source_indicators_in_both_arms": True,
        "single_intended_arm_change": raw["single_intended_arm_change"],
        "shard_count": int(raw["shard_count"]), "shard_manifests": records,
        "pool_index_manifest": raw["pool_index_manifest"],
        "source_feature_freeze": fingerprint(args.raw_feature_freeze),
        "approval": fingerprint(args.approval_record),
        "test_partition_loaded": False, "test_references_loaded": False,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "complete": True, "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(root / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-shard", "freeze"))
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--pool-name", required=True)
    parser.add_argument("--transform", choices=TRANSFORMS, required=True)
    parser.add_argument("--raw-feature-freeze", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--shard-index", type=int)
    args = parser.parse_args()
    if args.command == "build-shard":
        if args.shard_index is None:
            parser.error("--shard-index is required for build-shard")
        result = build_shard(args)
    else:
        result = freeze(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
