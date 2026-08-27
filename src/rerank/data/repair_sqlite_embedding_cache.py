#!/usr/bin/env python
"""Build a sidecar for rows present in SQLite but unreachable by key lookup.

This is a recovery utility for a cache produced by an interrupted conversion
with SQLite journaling disabled.  It never changes the primary database.  The
disk-backed encoder automatically consults ``<cache>.repair.sqlite`` when the
primary equality lookup misses.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

from rerank.data.convert_embedding_cache import required_smiles_from_feature_cache


def repair(primary: str | Path, output: str | Path, required: set[str]) -> dict:
    primary = Path(primary)
    output = Path(output)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Repair output already exists: {output} or {temporary}")

    source = sqlite3.connect(primary)
    stored_keys = {row[0] for row in source.execute("SELECT smiles FROM embeddings")}
    lookup_failures = set()
    for index, key in enumerate(required, start=1):
        if source.execute(
            "SELECT 1 FROM embeddings WHERE smiles=?", (key,)
        ).fetchone() is None:
            lookup_failures.add(key)
        if index % 25_000 == 0:
            print(
                f"Checked {index:,}/{len(required):,} keys; "
                f"lookup failures={len(lookup_failures):,}",
                flush=True,
            )
    repairable = lookup_failures.intersection(stored_keys)
    truly_missing = lookup_failures.difference(stored_keys)
    print(
        f"Repairable rows={len(repairable):,}; truly missing={len(truly_missing):,}",
        flush=True,
    )

    target = sqlite3.connect(temporary)
    target.execute("PRAGMA journal_mode=DELETE")
    target.execute("PRAGMA synchronous=NORMAL")
    target.execute(
        "CREATE TABLE embeddings ("
        "smiles TEXT PRIMARY KEY, n_rows INTEGER, n_cols INTEGER, data BLOB"
        ") WITHOUT ROWID"
    )
    target.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )
    inserted = 0
    batch = []
    for key, n_rows, n_cols, data in source.execute(
        "SELECT smiles, n_rows, n_cols, data FROM embeddings"
    ):
        if key not in repairable:
            continue
        batch.append((key, n_rows, n_cols, data))
        if len(batch) >= 1_000:
            target.executemany(
                "INSERT INTO embeddings(smiles,n_rows,n_cols,data) VALUES(?,?,?,?)",
                batch,
            )
            inserted += len(batch)
            batch.clear()
            target.commit()
            print(f"Copied {inserted:,}/{len(repairable):,} repair rows", flush=True)
    if batch:
        target.executemany(
            "INSERT INTO embeddings(smiles,n_rows,n_cols,data) VALUES(?,?,?,?)", batch
        )
        inserted += len(batch)
    if inserted != len(repairable):
        raise RuntimeError(
            f"Repair scan copied {inserted} rows, expected {len(repairable)}."
        )
    metadata = {
        "complete": 1,
        "required_keys": len(required),
        "primary_rows": len(stored_keys),
        "lookup_failures": len(lookup_failures),
        "repair_rows": inserted,
        "truly_missing": sorted(truly_missing),
    }
    target.executemany(
        "INSERT INTO metadata(key,value) VALUES(?,?)",
        [
            (key, json.dumps(value) if isinstance(value, list) else str(value))
            for key, value in metadata.items()
        ],
    )
    target.commit()
    target.close()
    source.close()
    os.replace(temporary, output)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--primary", default="outputs/study_cache/study_atom_embeddings.sqlite"
    )
    parser.add_argument(
        "--feature-cache", default="outputs/study_cache/official_2d_prior_schema1.pkl"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = args.output or f"{args.primary}.repair.sqlite"
    required = required_smiles_from_feature_cache(args.feature_cache)
    metadata = repair(args.primary, output, required)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
