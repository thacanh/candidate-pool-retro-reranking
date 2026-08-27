#!/usr/bin/env python
"""Build the compact, resumable seed-42 pooled Uni-Mol cache required by C2.

Unlike the B-CONFORMER scratch cache, this artifact stores only one 512-vector
and its atom count per canonical molecular fragment.  This is sufficient to
reconstruct exact atom-count-weighted means for multi-fragment reactants while
reducing retained storage from about 15 GB to well below 1 GB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rerank.benchmarks.benchmark_conformer_timing import collect_official_feature_workload
from rerank.data.build_conformer_sqlite import (
    EMBEDDING_DIM, SeededUniMolBackend, _backend_identity, discover_weight_paths,
    validate_conformer_seed,
)
from rerank.study_data import file_fingerprint


CACHE_SCHEMA_VERSION = 1
CACHE_KIND = "unimol_atom_mean_with_atom_count"


def _read_metadata(connection: sqlite3.Connection, key: str, default=None):
    row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    return json.loads(row[0])


def _write_metadata(connection: sqlite3.Connection, key: str, value) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
        (key, json.dumps(value, sort_keys=True)),
    )


def _atomic_json(payload: dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, target)


def _open_database(path: Path, seed: int, required: list[str]) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "smiles TEXT PRIMARY KEY, atom_count INTEGER NOT NULL, data BLOB, "
        "status TEXT NOT NULL, error TEXT) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )
    required_hash = hashlib.sha256("\n".join(required).encode("utf-8")).hexdigest()
    if existed:
        checks = {
            "cache_kind": CACHE_KIND,
            "schema_version": CACHE_SCHEMA_VERSION,
            "conformer_seed": seed,
            "required_items": len(required),
            "required_keys_sha256": required_hash,
        }
        for key, expected in checks.items():
            observed = _read_metadata(connection, key)
            if observed not in (None, expected):
                raise RuntimeError(f"Existing pooled cache has incompatible {key}: {observed!r}.")
    for key, value in {
        "cache_kind": CACHE_KIND,
        "schema_version": CACHE_SCHEMA_VERSION,
        "conformer_seed": seed,
        "required_items": len(required),
        "required_keys_sha256": required_hash,
        "complete": 0,
    }.items():
        _write_metadata(connection, key, value)
    connection.commit()
    return connection


def build(args: argparse.Namespace) -> dict:
    seed = validate_conformer_seed(args.seed)
    inventory = collect_official_feature_workload(
        args.source_csv, args.metadata_csv, args.candidate_jsonl
    )
    required = sorted(set(inventory.smiles))
    scientific_complete = True
    inventory_audit = dict(inventory.audit)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        required = required[: args.limit]
        scientific_complete = False
        inventory_audit["smoke_limit"] = args.limit
        inventory_audit["required_key_count"] = len(required)
        inventory_audit["required_keys_sha256"] = hashlib.sha256(
            "\n".join(required).encode("utf-8")
        ).hexdigest()

    checkpoint, dictionary = discover_weight_paths(args.checkpoint, args.dictionary)
    backend = SeededUniMolBackend(
        seed=seed, checkpoint=checkpoint, dictionary=dictionary,
        batch_size=args.batch_size, threads=args.threads, device=args.device,
    )
    output = Path(args.output).resolve()
    connection = _open_database(output, seed, required)
    stored = {row[0] for row in connection.execute("SELECT smiles FROM embeddings")}
    pending = [smiles for smiles in required if smiles not in stored]
    backend_metadata = backend.metadata() if pending else _read_metadata(connection, "backend", {})
    identity = _backend_identity(backend_metadata)
    previous = _read_metadata(connection, "backend_identity")
    if stored and previous is None:
        raise RuntimeError("Partial pooled cache lacks backend identity; refusing unsafe resume.")
    if previous is not None and previous != identity:
        raise RuntimeError("Partial pooled cache was produced by a different backend.")
    _write_metadata(connection, "backend_identity", identity)
    connection.commit()

    started = time.perf_counter()
    print(
        f"C2 pooled seed {seed}: {len(stored):,} cached, {len(pending):,} pending, "
        f"{len(required):,} required.", flush=True
    )
    for batch_index, start in enumerate(range(0, len(pending), args.batch_size)):
        batch = pending[start : start + args.batch_size]
        arrays, statuses, errors = backend.encode_batch(batch)
        rows = []
        for smiles, array, status, error in zip(batch, arrays, statuses, errors):
            if array is None:
                rows.append((smiles, 0, None, status, error))
            else:
                values = np.asarray(array, dtype=np.float32)
                pooled = values.mean(axis=0, dtype=np.float64).astype(np.float32)
                rows.append((smiles, int(values.shape[0]), pooled.tobytes(order="C"), status, error))
        connection.executemany(
            "INSERT INTO embeddings(smiles,atom_count,data,status,error) VALUES(?,?,?,?,?)",
            rows,
        )
        if (batch_index + 1) % 32 == 0 or start + len(batch) == len(pending):
            connection.commit()
        completed = len(stored) + start + len(batch)
        if completed % 1000 < len(batch) or completed == len(required):
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = (completed - len(stored)) / elapsed if completed > len(stored) else 0.0
            eta = (len(required) - completed) / rate if rate > 0 else None
            progress = {
                "status": "running", "protocol_id": "C-PROJECTED",
                "cache_kind": CACHE_KIND, "conformer_seed": seed,
                "completed_items": completed, "required_items": len(required),
                "percent": 100.0 * completed / len(required),
                "rate_items_per_second": rate, "eta_seconds": eta,
                "output": str(output), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_json(progress, args.progress_json)
            print(
                f"C2 pooled seed {seed}: {completed:,}/{len(required):,} "
                f"({progress['percent']:.1f}%) | {rate:.2f} mol/s | "
                f"ETA {0 if eta is None else eta / 3600:.2f} h",
                flush=True,
            )

    row_count = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    null_count = int(connection.execute("SELECT COUNT(*) FROM embeddings WHERE data IS NULL").fetchone()[0])
    bad = int(connection.execute(
        "SELECT COUNT(*) FROM embeddings WHERE data IS NOT NULL AND "
        "(atom_count < 1 OR length(data) != ?)", (EMBEDDING_DIM * 4,)
    ).fetchone()[0])
    if row_count != len(required) or bad:
        raise RuntimeError(f"Invalid pooled cache: rows={row_count}, required={len(required)}, bad={bad}.")
    backend_metadata = backend.metadata() if pending else backend_metadata
    inputs = {
        "source_csv": file_fingerprint(args.source_csv),
        "metadata_csv": file_fingerprint(args.metadata_csv),
        "candidate_jsonl": file_fingerprint(args.candidate_jsonl),
    }
    complete_flag = 1 if scientific_complete else 0
    for key, value in {
        "complete": complete_flag, "scientific_complete": scientific_complete,
        "null_embedding_items": null_count, "backend": backend_metadata,
        "input_fingerprints": inputs, "inventory_audit": inventory_audit,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }.items():
        _write_metadata(connection, key, value)
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    summary = {
        "status": "complete" if scientific_complete else "partial_benchmark",
        "protocol_id": "C-PROJECTED", "cache_kind": CACHE_KIND,
        "conformer_seed": seed, "required_items": len(required),
        "stored_items": row_count, "null_embedding_items": null_count,
        "cache_path": str(output), "cache_size_bytes": output.stat().st_size,
        "inventory_audit": inventory_audit, "input_fingerprints": inputs,
        "backend": backend_metadata,
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(summary, args.summary_json)
    _atomic_json({
        "status": summary["status"], "protocol_id": "C-PROJECTED",
        "conformer_seed": seed, "completed_items": row_count,
        "required_items": len(required), "percent": 100.0,
        "output": str(output), "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, args.progress_json)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    parser.add_argument("--candidate-jsonl", default="outputs/rerank_dataset.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--progress-json", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dictionary")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.threads < 1:
        raise SystemExit("--batch-size and --threads must be positive.")
    print(json.dumps(build(args), indent=2), flush=True)


if __name__ == "__main__":
    main()

