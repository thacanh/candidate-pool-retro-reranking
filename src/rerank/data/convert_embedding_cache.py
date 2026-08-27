#!/usr/bin/env python
"""Stream the large atom-embedding pickle into a study-scoped SQLite cache.

The source pickle is one giant dictionary and cannot be loaded on a 16 GB
machine.  This converter overrides the pure-Python unpickler's outer-dict
``SETITEMS`` operation, writes each batch directly to SQLite, and discards
memoized array payloads after their construction.  Only molecules required by
the official-split feature cache are retained.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from rerank.cached_encoder import canon
from rerank.study_data import file_fingerprint


def required_smiles_from_feature_cache(path: str | Path) -> set[str]:
    with open(path, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob["payload"]
    required: set[str] = set()

    def add(smiles: str) -> None:
        for fragment in str(smiles).split("."):
            fragment = fragment.strip()
            if not fragment:
                continue
            canonical = canon(fragment)
            if canonical is not None:
                required.add(canonical)

    for product in payload["train_products"]:
        add(product["product_smiles"])
        for candidate in product["candidates"]:
            add(candidate["smiles"])
    for product_smiles, candidates in payload["eval_pwc"]:
        add(product_smiles)
        for candidate in candidates:
            add(candidate["smiles"])
    return required


class StreamingOuterDictUnpickler(pickle._Unpickler):
    """Unpickle values in batches without materializing the outer dictionary."""

    dispatch = pickle._Unpickler.dispatch.copy()

    def __init__(self, file, consumer: Callable[[list], None]):
        super().__init__(file)
        self.consumer = consumer
        self.outer_dict = None
        self.memo_index = 0

    def load_empty_dictionary(self):
        value = {}
        self.append(value)
        if self.outer_dict is None:
            self.outer_dict = value

    def load_memoize(self):
        # Protocol 5 assigns monotonically increasing memo indices.  Future
        # values in this NumPy pickle only GET the shared reconstruction
        # callable, dtype, and order string established in the first value
        # (indices 4, 11, and 15).  Keep the complete bootstrap memo and use
        # lightweight placeholders for subsequent per-array payloads.
        value = self.stack[-1]
        self.memo[self.memo_index] = (
            value
            if self.memo_index < 18 or self._safe_to_retain(value)
            else None
        )
        self.memo_index += 1

    @classmethod
    def _safe_to_retain(cls, value) -> bool:
        """Keep shared metadata but never retain large array storage."""
        if value is None or isinstance(value, (str, int, float, bool, np.dtype)):
            return True
        if callable(value):
            return True
        if isinstance(value, tuple):
            return all(cls._safe_to_retain(item) for item in value)
        return False

    def _memo_get(self, index: int):
        value = self.memo[index]
        if value is None:
            raise pickle.UnpicklingError(
                f"Unexpected reference to discarded memo index {index}."
            )
        self.append(value)

    def load_binget(self):
        self._memo_get(self.read(1)[0])

    def load_long_binget(self):
        self._memo_get(int.from_bytes(self.read(4), "little", signed=False))

    def load_setitem(self):
        value = self.stack.pop()
        key = self.stack.pop()
        target = self.stack[-1]
        if target is self.outer_dict:
            self.consumer([key, value])
        else:
            target[key] = value

    def load_setitems(self):
        items = self.pop_mark()
        target = self.stack[-1]
        if target is self.outer_dict:
            self.consumer(items)
        else:
            for index in range(0, len(items), 2):
                target[items[index]] = items[index + 1]


StreamingOuterDictUnpickler.dispatch[pickle.EMPTY_DICT[0]] = (
    StreamingOuterDictUnpickler.load_empty_dictionary
)
StreamingOuterDictUnpickler.dispatch[pickle.MEMOIZE[0]] = (
    StreamingOuterDictUnpickler.load_memoize
)
StreamingOuterDictUnpickler.dispatch[pickle.BINGET[0]] = (
    StreamingOuterDictUnpickler.load_binget
)
StreamingOuterDictUnpickler.dispatch[pickle.LONG_BINGET[0]] = (
    StreamingOuterDictUnpickler.load_long_binget
)
StreamingOuterDictUnpickler.dispatch[pickle.SETITEM[0]] = (
    StreamingOuterDictUnpickler.load_setitem
)
StreamingOuterDictUnpickler.dispatch[pickle.SETITEMS[0]] = (
    StreamingOuterDictUnpickler.load_setitems
)


def convert(
    source_pickle: str | Path,
    output_sqlite: str | Path,
    required: set[str],
    resume: bool = False,
) -> dict:
    output_sqlite = Path(output_sqlite)
    temporary = output_sqlite.with_suffix(output_sqlite.suffix + ".incomplete")
    if output_sqlite.exists():
        raise FileExistsError(f"Output already exists: {output_sqlite}")
    if temporary.exists() and not resume:
        raise FileExistsError(
            f"Incomplete cache already exists; inspect or remove it first: {temporary}"
        )

    resuming = temporary.exists()
    connection = sqlite3.connect(temporary)
    # A previous interrupted conversion with journal_mode=OFF left a B-tree
    # that could be full-scanned but failed indexed equality lookups.  Use a
    # recoverable journal so a failed conversion can be resumed safely.
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-131072")
    if not resuming:
        connection.execute(
            "CREATE TABLE embeddings ("
            "smiles TEXT PRIMARY KEY, n_rows INTEGER, n_cols INTEGER, data BLOB"
            ") WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
        )
    stored_keys = {
        row[0] for row in connection.execute("SELECT smiles FROM embeddings")
    }
    failed_existing = int(
        connection.execute("SELECT COUNT(*) FROM embeddings WHERE data IS NULL").fetchone()[0]
    )
    state = {
        "source_items": 0,
        "stored_items": len(stored_keys),
        "failed_items": failed_existing,
    }
    if resuming:
        print(f"Resuming with {len(stored_keys):,} embeddings already stored.", flush=True)

    def consume(items: list) -> None:
        rows = []
        for index in range(0, len(items), 2):
            key = str(items[index])
            value: Optional[np.ndarray] = items[index + 1]
            state["source_items"] += 1
            if key not in required or key in stored_keys:
                continue
            if value is None:
                rows.append((key, 0, 0, None))
                state["failed_items"] += 1
            else:
                array = np.asarray(value, dtype=np.float32)
                if array.ndim != 2:
                    raise ValueError(f"Unexpected embedding shape for {key}: {array.shape}")
                rows.append(
                    (key, int(array.shape[0]), int(array.shape[1]), array.tobytes(order="C"))
                )
            state["stored_items"] += 1
            stored_keys.add(key)
        if rows:
            connection.executemany(
                "INSERT INTO embeddings(smiles,n_rows,n_cols,data) VALUES(?,?,?,?)", rows
            )
        if state["source_items"] % 10_000 == 0:
            connection.commit()
            print(
                f"Processed {state['source_items']:,} source entries; "
                f"stored {state['stored_items']:,}/{len(required):,}",
                flush=True,
            )

    with open(source_pickle, "rb") as handle:
        StreamingOuterDictUnpickler(handle, consume).load()

    missing = len(required) - state["stored_items"]
    metadata = {
        **state,
        "required_items": len(required),
        "missing_required_items": missing,
        "source_fingerprint": file_fingerprint(source_pickle, include_sha256=False),
        "complete": 1,
    }
    connection.executemany(
        "INSERT INTO metadata(key,value) VALUES(?,?)",
        [(key, json.dumps(value) if isinstance(value, dict) else str(value)) for key, value in metadata.items()],
    )
    connection.commit()
    connection.execute("PRAGMA optimize")
    connection.close()
    os.replace(temporary, output_sqlite)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pickle", default="outputs/atom_embeddings.pkl")
    parser.add_argument(
        "--feature-cache",
        default="outputs/study_cache/official_2d_prior_schema1.pkl",
        help="Official-split 2D feature cache used to define the required molecule set.",
    )
    parser.add_argument(
        "--output", default="outputs/study_cache/study_atom_embeddings.sqlite"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    required = required_smiles_from_feature_cache(args.feature_cache)
    print(f"Required canonical fragment embeddings: {len(required):,}", flush=True)
    metadata = convert(args.source_pickle, args.output, required, resume=args.resume)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
