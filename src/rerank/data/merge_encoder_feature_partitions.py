#!/usr/bin/env python
"""Fail-closed merge of independent query-scoped encoder feature partitions."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from rerank.encoder_controls import FULL_FEATURE_NAMES, file_fingerprint, file_sha256


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def merge_feature_partitions(
    partition_manifests: Sequence[str | Path],
    output_manifest: str | Path,
    expected_query_count: int,
    expected_pair_count: int,
) -> dict:
    """Validate ordered partitions and expose their shards through one manifest."""
    if expected_query_count < 1 or expected_pair_count < 1:
        raise ValueError("Expected query/pair counts must be positive.")
    output = Path(output_manifest).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise FileExistsError(f"Refusing to overwrite merged manifest: {output}")
    paths = [Path(path).resolve() for path in partition_manifests]
    if not paths:
        raise ValueError("At least one partition manifest is required.")

    records: list[tuple[int, int, Path, dict]] = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"Partition manifest is not complete: {path}")
        identity = manifest.get("identity", {})
        query_range = identity.get("query_range", {})
        start = query_range.get("start_inclusive")
        stop = query_range.get("stop_exclusive")
        if not isinstance(start, int) or not isinstance(stop, int) or stop <= start:
            raise ValueError(f"Partition has an invalid frozen query range: {path}")
        if int(manifest.get("completed_query_count", -1)) != stop - start:
            raise ValueError(f"Partition query count differs from its range: {path}")
        if identity.get("parallelization") != (
            "independent_query_scoped_worker_partition"
        ):
            raise ValueError(f"Partition does not attest query-scoped encoding: {path}")
        records.append((start, stop, path, manifest))
    records.sort(key=lambda item: item[0])

    comparable_identity_fields = (
        "protocol_id",
        "single_intended_change",
        "input",
        "encoder",
        "feature_names",
        "encoder_batch_size",
        "queries_per_shard",
        "global_atom_cache",
        "candidate_normalization",
    )
    reference_identity = records[0][3]["identity"]
    next_query_index = 0
    pair_total = 0
    merged_shards: list[dict] = []
    source_partitions: list[dict] = []

    for partition_index, (start, stop, path, manifest) in enumerate(records):
        if start != next_query_index:
            raise ValueError(
                f"Partition ranges are not contiguous at {next_query_index}: {path}"
            )
        identity = manifest["identity"]
        for field in comparable_identity_fields:
            if identity.get(field) != reference_identity.get(field):
                raise ValueError(f"Partition identity differs at {field!r}: {path}")

        observed_partition_queries = 0
        observed_partition_pairs = 0
        expected_query_index = start
        for shard in manifest.get("shards", []):
            shard_path = path.parent / str(shard.get("path", ""))
            if not shard_path.is_file() or file_sha256(shard_path) != shard.get("sha256"):
                raise ValueError(f"Partition shard is missing or changed: {shard_path}")
            with np.load(shard_path, allow_pickle=False) as payload:
                names = tuple(str(value) for value in payload["feature_names"].tolist())
                query_ids = [str(value) for value in payload["query_ids"].tolist()]
                features = np.asarray(payload["features"], dtype=np.float32)
            if names != FULL_FEATURE_NAMES:
                raise ValueError(f"Partition shard feature order changed: {shard_path}")
            expected_ids = [
                f"query-{index:08d}"
                for index in range(expected_query_index, expected_query_index + len(query_ids))
            ]
            if query_ids != expected_ids:
                raise ValueError(f"Partition shard query order changed: {shard_path}")
            if features.ndim != 2 or features.shape[1] != len(FULL_FEATURE_NAMES):
                raise ValueError(f"Partition shard feature shape changed: {shard_path}")
            expected_query_index += len(query_ids)
            observed_partition_queries += len(query_ids)
            observed_partition_pairs += int(features.shape[0])

            merged_shard = deepcopy(shard)
            merged_shard["index"] = len(merged_shards)
            merged_shard["path"] = os.path.relpath(
                shard_path, output.parent
            ).replace("\\", "/")
            merged_shards.append(merged_shard)

        if observed_partition_queries != stop - start:
            raise ValueError(f"Partition shard query total is inconsistent: {path}")
        if observed_partition_pairs != int(manifest.get("completed_pair_count", -1)):
            raise ValueError(f"Partition shard pair total is inconsistent: {path}")
        if expected_query_index != stop:
            raise ValueError(f"Partition query IDs do not reach the frozen stop: {path}")
        pair_total += observed_partition_pairs
        next_query_index = stop
        source_partitions.append(
            {
                "partition_index": partition_index,
                "start_inclusive": start,
                "stop_exclusive": stop,
                "manifest": file_fingerprint(path),
            }
        )

    if next_query_index != expected_query_count:
        raise ValueError(
            f"Merged query total {next_query_index} != expected {expected_query_count}."
        )
    if pair_total != expected_pair_count:
        raise ValueError(f"Merged pair total {pair_total} != expected {expected_pair_count}.")

    merged_identity = deepcopy(reference_identity)
    merged_identity["query_range"] = {
        "start_inclusive": 0,
        "stop_exclusive": expected_query_count,
    }
    merged_identity["parallelization"] = (
        "independent_query_scoped_workers_no_cross_query_encoder_batches"
    )
    merged_identity["partition_count"] = len(records)
    merged = {
        "schema_version": 3,
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "identity": merged_identity,
        "completed_query_count": expected_query_count,
        "completed_pair_count": expected_pair_count,
        "shards": merged_shards,
        "source_partitions": source_partitions,
        "failures": [],
        "atom_embeddings_written": False,
    }
    _atomic_write_json(output, merged)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-manifest", action="append", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--expected-query-count", type=int, required=True)
    parser.add_argument("--expected-pair-count", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_feature_partitions(
        args.partition_manifest,
        args.output_manifest,
        args.expected_query_count,
        args.expected_pair_count,
    )
    print(
        json.dumps(
            {
                "status": merged["status"],
                "completed_query_count": merged["completed_query_count"],
                "completed_pair_count": merged["completed_pair_count"],
                "shard_count": len(merged["shards"]),
                "partition_count": len(merged["source_partitions"]),
                "atom_embeddings_written": merged["atom_embeddings_written"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
