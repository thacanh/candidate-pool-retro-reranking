#!/usr/bin/env python
"""Stream WS-C encoder controls into compact, resumable feature shards.

Candidate rows are grouped globally by canonical product in first-occurrence
order, matching the controlled-study candidate loader. Atom states are kept
only for the current query/candidate batch and never written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Sequence

import numpy as np

from rerank.encoder_controls import (
    FULL_FEATURE_NAMES,
    AtomEncoder,
    GroverAtomEncoder,
    MorganAtomEncoder,
    compute_full_query_features,
    file_fingerprint,
    file_sha256,
)


MANIFEST_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class CandidateQuery:
    query_id: str
    product_smiles: str
    canonical_product_identity: str
    candidate_smiles: list[str]
    canonical_candidate_identities: list[str]
    priors: list[float]
    ranks: list[int]


def iter_candidate_queries(path: str | Path) -> Iterator[CandidateQuery]:
    """Yield normalized queries from a flat JSONL.

    Within each query this mirrors the controlled-study pool identity: RDKit
    canonical product, fragment-order-invariant canonical reactant identity,
    maximum-prior duplicate winner (stable first exact tie), then stable
    decreasing-prior order. Products are grouped globally because distinct raw
    product strings can canonicalize to the same identity at non-contiguous
    positions. Invalid molecular records are excluded exactly as they are by
    the controlled candidate-pool loader.
    """
    from rerank.study_data import canonicalize_reactant_set, canonicalize_smiles

    pools: dict[str, dict[str, tuple[str, float]]] = {}

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSON at line {line_number}.") from exc
            product = str(record.get("product", "")).strip()
            candidate = str(record.get("reactant", "")).strip()
            if not product or not candidate:
                raise ValueError(f"Missing product/reactant at line {line_number}.")
            try:
                prior = float(record.get("prior", 0.0))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prior at line {line_number}.") from exc
            if not np.isfinite(prior):
                raise ValueError(f"Non-finite prior at line {line_number}.")
            product_key = canonicalize_smiles(product)
            candidate_key = canonicalize_reactant_set(candidate)
            if product_key is None or candidate_key is None:
                continue

            candidate_winners = pools.setdefault(product_key, {})
            existing = candidate_winners.get(candidate_key)
            if existing is None or prior > existing[1]:
                candidate_winners[candidate_key] = (candidate, prior)

    for query_index, (product_key, candidate_winners) in enumerate(pools.items()):
        winners = sorted(
            candidate_winners.items(), key=lambda item: item[1][1], reverse=True
        )
        yield CandidateQuery(
            query_id=f"query-{query_index:08d}",
            product_smiles=product_key,
            canonical_product_identity=product_key,
            candidate_smiles=[item[1][0] for item in winners],
            canonical_candidate_identities=[item[0] for item in winners],
            priors=[item[1][1] for item in winners],
            ranks=list(range(len(winners))),
        )


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_feature_shard(
    output_dir: Path,
    shard_index: int,
    query_rows: Sequence[tuple[CandidateQuery, np.ndarray]],
) -> dict:
    filename = f"features-{shard_index:05d}.npz"
    path = output_dir / filename
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {path}")
    query_ids: list[str] = []
    products: list[str] = []
    canonical_products: list[str] = []
    candidates: list[str] = []
    canonical_candidates: list[str] = []
    priors: list[float] = []
    ranks: list[int] = []
    features: list[np.ndarray] = []
    offsets = [0]
    for query, matrix in query_rows:
        if matrix.shape != (len(query.candidate_smiles), len(FULL_FEATURE_NAMES)):
            raise ValueError("Feature matrix shape is not aligned to its query.")
        query_ids.append(query.query_id)
        products.append(query.product_smiles)
        canonical_products.append(query.canonical_product_identity)
        candidates.extend(query.candidate_smiles)
        canonical_candidates.extend(query.canonical_candidate_identities)
        priors.extend(query.priors)
        ranks.extend(query.ranks)
        features.append(matrix.astype(np.float32, copy=False))
        offsets.append(offsets[-1] + len(query.candidate_smiles))
    feature_matrix = (
        np.concatenate(features, axis=0)
        if features
        else np.empty((0, len(FULL_FEATURE_NAMES)), dtype=np.float32)
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray([MANIFEST_SCHEMA_VERSION], dtype=np.int16),
            feature_names=np.asarray(FULL_FEATURE_NAMES),
            query_ids=np.asarray(query_ids),
            product_smiles=np.asarray(products),
            canonical_product_identities=np.asarray(canonical_products),
            query_offsets=np.asarray(offsets, dtype=np.int64),
            candidate_smiles=np.asarray(candidates),
            canonical_candidate_identities=np.asarray(canonical_candidates),
            priors=np.asarray(priors, dtype=np.float32),
            ranks=np.asarray(ranks, dtype=np.int32),
            features=feature_matrix,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "index": shard_index,
        "path": filename,
        "query_count": len(query_rows),
        "pair_count": int(feature_matrix.shape[0]),
        "first_query_id": query_ids[0] if query_ids else None,
        "last_query_id": query_ids[-1] if query_ids else None,
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _validate_resume_manifest(manifest: dict, identity: dict, output_dir: Path) -> None:
    if manifest.get("identity") != identity:
        raise ValueError("Resume manifest identity does not match this run.")
    completed = int(manifest.get("completed_query_count", -1))
    if completed < 0:
        raise ValueError("Resume manifest has an invalid completed-query count.")
    shard_queries = 0
    for shard in manifest.get("shards", []):
        path = output_dir / str(shard["path"])
        if not path.is_file() or file_sha256(path) != shard.get("sha256"):
            raise ValueError(f"Resume shard is missing or changed: {path}")
        shard_queries += int(shard["query_count"])
    if shard_queries != completed:
        raise ValueError("Resume manifest query count does not match its shards.")


def build_feature_shards(
    input_jsonl: str | Path,
    output_dir: str | Path,
    encoder: AtomEncoder,
    protocol_id: str,
    encoder_batch_size: int = 32,
    queries_per_shard: int = 128,
    resume: bool = False,
    max_queries: Optional[int] = None,
    query_start_index: int = 0,
    query_stop_index: Optional[int] = None,
) -> dict:
    """Build or resume compact scalar-only shards for one encoder control."""
    if encoder_batch_size < 1 or queries_per_shard < 1:
        raise ValueError("Batch and shard sizes must be positive.")
    if max_queries is not None and max_queries < 1:
        raise ValueError("max_queries must be positive when provided.")
    if query_start_index < 0:
        raise ValueError("query_start_index must be non-negative.")
    if query_stop_index is not None and query_stop_index <= query_start_index:
        raise ValueError("query_stop_index must be greater than query_start_index.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    encoder.initialize()
    encoder_metadata = encoder.metadata()
    identity = {
        "protocol_id": protocol_id,
        "single_intended_change": "Uni-Mol atom states -> encoder-control atom states",
        "input": file_fingerprint(input_jsonl),
        "encoder": encoder_metadata,
        "feature_names": list(FULL_FEATURE_NAMES),
        "encoder_batch_size": encoder_batch_size,
        "queries_per_shard": queries_per_shard,
        "query_range": {
            "start_inclusive": query_start_index,
            "stop_exclusive": query_stop_index,
        },
        "parallelization": "independent_query_scoped_worker_partition",
        "global_atom_cache": False,
        "candidate_normalization": {
            "product_identity": "RDKit canonical isomeric SMILES",
            "product_grouping": "global canonical identity in first-occurrence order",
            "reactant_identity": "canonical fragments sorted and dot-joined",
            "duplicate_rule": "maximum prior; stable first exact tie",
            "ordering": "stable decreasing prior",
            "invalid_molecular_records": "excluded",
        },
    }

    if manifest_path.exists():
        if not resume:
            raise FileExistsError("Manifest exists; pass resume=True to continue safely.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_resume_manifest(manifest, identity, output)
    else:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "running",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "identity": identity,
            "completed_query_count": 0,
            "completed_pair_count": 0,
            "shards": [],
            "failures": [],
            "atom_embeddings_written": False,
        }
        _atomic_write_json(manifest_path, manifest)

    completed_before = int(manifest["completed_query_count"])
    processed_this_run = 0
    pending: list[tuple[CandidateQuery, np.ndarray]] = []
    exhausted = True
    range_boundary_reached = False
    started = time.perf_counter()

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        shard = _write_feature_shard(output, len(manifest["shards"]), pending)
        manifest["shards"].append(shard)
        manifest["completed_query_count"] += shard["query_count"]
        manifest["completed_pair_count"] += shard["pair_count"]
        manifest["status"] = "running"
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(manifest_path, manifest)
        pending = []

    for query_index, query in enumerate(iter_candidate_queries(input_jsonl)):
        if query_index < query_start_index + completed_before:
            continue
        if query_stop_index is not None and query_index >= query_stop_index:
            range_boundary_reached = True
            break
        if max_queries is not None and processed_this_run >= max_queries:
            exhausted = False
            break
        try:
            features = compute_full_query_features(
                encoder,
                query.product_smiles,
                query.candidate_smiles,
                query.priors,
                encoder_batch_size,
            )
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["failures"].append(
                {
                    "query_index": query_index,
                    "query_id": query.query_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
            _atomic_write_json(manifest_path, manifest)
            raise
        pending.append((query, features))
        processed_this_run += 1
        if len(pending) >= queries_per_shard:
            flush_pending()

    flush_pending()
    total_completed = int(manifest["completed_query_count"])
    expected_range_count = (
        query_stop_index - query_start_index
        if query_stop_index is not None
        else None
    )
    if expected_range_count is not None and total_completed > expected_range_count:
        raise ValueError("Partition completed-query count exceeds its frozen range.")
    if (
        expected_range_count is not None
        and exhausted
        and not range_boundary_reached
        and total_completed < expected_range_count
    ):
        manifest["status"] = "failed"
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write_json(manifest_path, manifest)
        raise ValueError("Candidate query stream ended before the frozen range stop.")
    range_complete = (
        total_completed == expected_range_count
        if expected_range_count is not None
        else exhausted
    )
    manifest["status"] = "complete" if range_complete else "partial"
    manifest["last_run_seconds"] = time.perf_counter() - started
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder", choices=("morgan_atom", "grover"), required=True)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--query-start-index", type=int, default=0)
    parser.add_argument("--query-stop-index", type=int)
    parser.add_argument("--queries-per-shard", type=int, default=128)
    parser.add_argument(
        "--max-queries",
        type=int,
        help="Infrastructure benchmark limit; omit for a complete scientific run.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--grover-repo")
    parser.add_argument("--grover-checkpoint")
    parser.add_argument("--grover-backend-factory")
    parser.add_argument(
        "--grover-atom-state-choice",
        choices=("atom_from_atom", "atom_from_bond", "concatenation"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.encoder == "morgan_atom":
        if args.device not in {"auto", "cpu"}:
            raise SystemExit("Morgan-atom is a CPU control; --device cuda is invalid.")
        encoder: AtomEncoder = MorganAtomEncoder()
        protocol_id = "C-MORGAN-ATOM"
    else:
        if args.grover_atom_state_choice is None:
            raise SystemExit(
                "--grover-atom-state-choice is required for GROVER."
            )
        if args.grover_atom_state_choice != "concatenation":
            raise SystemExit(
                "The approved C1a protocol is frozen to GROVER concatenation; "
                "other atom-state choices require a new prespecified protocol."
            )
        encoder = GroverAtomEncoder(
            repo_path=args.grover_repo,
            checkpoint_path=args.grover_checkpoint,
            device=args.device,
            batch_size=args.encoder_batch_size,
            atom_state_choice=args.grover_atom_state_choice,
            backend_factory_spec=args.grover_backend_factory,
        )
        protocol_id = "cap10-tuned-grover-concatenation-v1"
    manifest = build_feature_shards(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        encoder=encoder,
        protocol_id=protocol_id,
        encoder_batch_size=args.encoder_batch_size,
        queries_per_shard=args.queries_per_shard,
        resume=args.resume,
        max_queries=args.max_queries,
        query_start_index=args.query_start_index,
        query_stop_index=args.query_stop_index,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "completed_query_count": manifest["completed_query_count"],
                "completed_pair_count": manifest["completed_pair_count"],
                "shard_count": len(manifest["shards"]),
                "atom_embeddings_written": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
