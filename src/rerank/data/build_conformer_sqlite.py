#!/usr/bin/env python
"""Build one resumable, seed-specific Uni-Mol atom cache in SQLite.

The cache is deliberately a temporary work artifact.  It is consumed by
``run_conformer_seed.py`` to create compact pair-feature and ranking artifacts,
then deleted only after those retained artifacts pass validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from rerank.benchmarks.benchmark_conformer_timing import collect_official_feature_workload
from rerank.study_data import file_fingerprint


ALLOWED_CONFORMER_SEEDS = tuple(range(42, 52))
EMBEDDING_DIM = 512
PINNED_UNIMOL_VERSION = "0.1.3"
PINNED_RDKIT_VERSION = "2025.09.6"
PINNED_CHECKPOINT_NAME = "mol_pre_no_h_220816.pt"
PINNED_CHECKPOINT_SHA256 = (
    "da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9"
)
PINNED_DICTIONARY_NAME = "mol.dict.txt"
PINNED_DICTIONARY_SHA256 = (
    "94135cb9a9198f988de684cb61e2c372882a3bd59b8320effbae704c38057127"
)
CACHE_SCHEMA_VERSION = 1


def _sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_metadata(connection: sqlite3.Connection, key: str, value) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES(?,?)",
        (str(key), json.dumps(value, sort_keys=True)),
    )


def _read_metadata(connection: sqlite3.Connection, key: str, default=None):
    row = connection.execute(
        "SELECT value FROM metadata WHERE key=?", (str(key),)
    ).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return row[0]


def _backend_identity(metadata: dict) -> dict:
    """Fields that must not change while resuming one partially built cache."""
    checkpoint = metadata.get("checkpoint") or {}
    dictionary = metadata.get("dictionary") or {}
    return {
        "name": metadata.get("name"),
        "seed": metadata.get("seed"),
        "device": metadata.get("device"),
        "cuda_device": metadata.get("cuda_device"),
        "unimol_tools_version": metadata.get("unimol_tools_version"),
        "rdkit_version": metadata.get("rdkit_version"),
        "torch_version": metadata.get("torch_version"),
        "torch_cuda_version": metadata.get("torch_cuda_version"),
        "checkpoint_sha256": checkpoint.get("sha256"),
        "dictionary_sha256": dictionary.get("sha256"),
        "conformer": metadata.get("conformer"),
    }


def validate_conformer_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in ALLOWED_CONFORMER_SEEDS:
        raise ValueError(
            f"Conformer seed must be one of {ALLOWED_CONFORMER_SEEDS}; got {seed}."
        )
    return seed


def discover_weight_paths(
    checkpoint: Optional[str | Path] = None,
    dictionary: Optional[str | Path] = None,
) -> tuple[Path, Path]:
    if checkpoint is None:
        configured = os.environ.get("UNIMOL_WEIGHT_DIR")
        if configured:
            weight_dir = Path(configured)
        else:
            spec = importlib.util.find_spec("unimol_tools")
            if spec is None or not spec.submodule_search_locations:
                raise RuntimeError("unimol-tools is not installed in this Python environment.")
            weight_dir = Path(next(iter(spec.submodule_search_locations))) / "weights"
        checkpoint_path = weight_dir / PINNED_CHECKPOINT_NAME
    else:
        checkpoint_path = Path(checkpoint)
    dictionary_path = (
        checkpoint_path.parent / PINNED_DICTIONARY_NAME
        if dictionary is None
        else Path(dictionary)
    )
    return checkpoint_path.resolve(), dictionary_path.resolve()


def validate_weight_paths(checkpoint: Path, dictionary: Path) -> dict:
    if checkpoint.name != PINNED_CHECKPOINT_NAME or not checkpoint.is_file():
        raise FileNotFoundError(
            f"Expected checkpoint {PINNED_CHECKPOINT_NAME}, got {checkpoint}."
        )
    if dictionary.name != PINNED_DICTIONARY_NAME or not dictionary.is_file():
        raise FileNotFoundError(
            f"Expected dictionary {PINNED_DICTIONARY_NAME}, got {dictionary}."
        )
    if checkpoint.parent != dictionary.parent:
        raise RuntimeError("Checkpoint and dictionary must share one weight directory.")
    checkpoint_hash = _sha256(checkpoint)
    dictionary_hash = _sha256(dictionary)
    if checkpoint_hash != PINNED_CHECKPOINT_SHA256:
        raise RuntimeError("Pinned Uni-Mol checkpoint SHA-256 mismatch.")
    if dictionary_hash != PINNED_DICTIONARY_SHA256:
        raise RuntimeError("Pinned Uni-Mol dictionary SHA-256 mismatch.")
    return {
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
        },
        "dictionary": {
            "path": str(dictionary),
            "size_bytes": dictionary.stat().st_size,
            "sha256": dictionary_hash,
        },
    }


class SeededUniMolBackend:
    """Pinned Uni-Mol backend with one explicit RDKit conformer seed."""

    def __init__(
        self,
        seed: int,
        checkpoint: Path,
        dictionary: Path,
        batch_size: int,
        threads: int,
        device: str = "cpu",
    ) -> None:
        self.seed = validate_conformer_seed(seed)
        self.checkpoint = checkpoint
        self.dictionary = dictionary
        self.batch_size = int(batch_size)
        self.threads = int(threads)
        self.device = str(device).lower()
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'.")
        self._repr = None
        self._conformer = None
        self._trainer = None
        self._dataset_class = None
        self._metadata: dict = {}
        self._oom_split_count = 0
        self._smallest_inference_chunk = self.batch_size

    def initialize(self) -> None:
        if self._repr is not None:
            return
        installed_unimol = importlib.metadata.version("unimol-tools")
        installed_rdkit = importlib.metadata.version("rdkit")
        if installed_unimol != PINNED_UNIMOL_VERSION:
            raise RuntimeError(
                f"Expected unimol-tools {PINNED_UNIMOL_VERSION}, got {installed_unimol}."
            )
        if installed_rdkit.replace(".9.6", ".09.6") != PINNED_RDKIT_VERSION:
            raise RuntimeError(
                f"Expected RDKit {PINNED_RDKIT_VERSION}, got {installed_rdkit}."
            )

        assets = validate_weight_paths(self.checkpoint, self.dictionary)
        if any(name.startswith("unimol_tools") for name in sys.modules):
            raise RuntimeError(
                "unimol_tools was imported before UNIMOL_WEIGHT_DIR was pinned; "
                "start the seed runner in a fresh Python process."
            )
        os.environ["UNIMOL_WEIGHT_DIR"] = str(self.checkpoint.parent)

        import torch
        from unimol_tools import UniMolRepr
        from unimol_tools.data.conformer import ConformerGen
        from unimol_tools.predictor import MolDataset
        from unimol_tools.tasks import Trainer

        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        use_cuda = self.device == "cuda"
        if use_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was selected but this PyTorch build/driver cannot use CUDA. "
                "Run CHECK_MACHINE.cmd or select --device cpu."
            )
        if use_cuda:
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self._repr = UniMolRepr(
            data_type="molecule",
            batch_size=self.batch_size,
            remove_hs=True,
            model_name="unimolv1",
            model_size="84m",
            use_cuda=use_cuda,
            use_ddp=False,
            use_gpu="0",
        )
        self._conformer = ConformerGen(
            seed=self.seed,
            max_atoms=256,
            data_type="molecule",
            method="rdkit_random",
            mode="fast",
            remove_hs=True,
            multi_process=False,
        )
        self._trainer = Trainer(task="repr", **self._repr.params)
        self._dataset_class = MolDataset
        self._metadata = {
            "seed": self.seed,
            "device": self.device,
            "batch_size": self.batch_size,
            "threads": self.threads,
            "unimol_tools_version": installed_unimol,
            "rdkit_version": installed_rdkit,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_device": (
                {
                    "name": torch.cuda.get_device_name(0),
                    "total_memory_bytes": int(
                        torch.cuda.get_device_properties(0).total_memory
                    ),
                    "capability": list(torch.cuda.get_device_capability(0)),
                }
                if use_cuda
                else None
            ),
            "conformer": {
                "method": "rdkit_random",
                "mode": "fast",
                "remove_hs": True,
                "max_atoms": 256,
                "seed": self.seed,
            },
            **assets,
        }

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        return "out of memory" in str(exc).lower() and "cuda" in str(exc).lower()

    def _infer_payloads(self, payloads: Sequence[dict]) -> list[np.ndarray]:
        """Run inference, recursively reducing only a CUDA batch that OOMs."""
        dataset = self._dataset_class(list(payloads))
        try:
            raw = self._trainer.inference(
                self._repr.model,
                dataset,
                return_repr=True,
                return_atomic_reprs=True,
                feature_name=None,
            )
        except Exception as exc:
            if self.device == "cuda" and self._is_cuda_oom(exc) and len(payloads) > 1:
                import torch

                self._oom_split_count += 1
                torch.cuda.empty_cache()
                midpoint = len(payloads) // 2
                return self._infer_payloads(payloads[:midpoint]) + self._infer_payloads(
                    payloads[midpoint:]
                )
            raise RuntimeError(
                f"Uni-Mol inference failed for batch size {len(payloads)}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        atomic = raw.get("atomic_reprs", raw.get("atom_repr"))
        if not isinstance(atomic, (list, tuple)) or len(atomic) != len(payloads):
            raise RuntimeError("Uni-Mol returned misaligned atomic representations.")
        self._smallest_inference_chunk = min(
            self._smallest_inference_chunk, len(payloads)
        )
        return list(atomic)

    def encode_batch(
        self, smiles: Sequence[str]
    ) -> tuple[list[Optional[np.ndarray]], list[str], list[Optional[str]]]:
        self.initialize()
        payloads = []
        statuses: list[str] = []
        errors: list[Optional[str]] = []
        for value in smiles:
            try:
                feature = self._conformer.single_process(value)
                coordinates = np.asarray(feature["src_coord"])
                if np.all(coordinates == 0.0):
                    status = "fallback_zero"
                elif coordinates.ndim == 2 and np.all(coordinates[:, 2] == 0.0):
                    status = "fallback_2d"
                else:
                    status = "ok"
                payloads.append(feature)
                statuses.append(status)
                errors.append(None)
            except Exception as exc:
                payloads.append(None)
                statuses.append("failure_preprocess")
                errors.append(f"{type(exc).__name__}: {exc}")

        valid_indices = [index for index, item in enumerate(payloads) if item is not None]
        aligned: list[Optional[np.ndarray]] = [None] * len(payloads)
        if valid_indices:
            atomic = self._infer_payloads(
                [payloads[index] for index in valid_indices]
            )
            for index, representation in zip(valid_indices, atomic):
                array = np.asarray(representation, dtype=np.float32)
                if array.ndim != 2 or array.shape[1] != EMBEDDING_DIM:
                    raise RuntimeError(
                        f"Unexpected atom representation shape {array.shape} for {smiles[index]}."
                    )
                if not np.isfinite(array).all():
                    raise RuntimeError(f"Non-finite atom representation for {smiles[index]}.")
                aligned[index] = array
        return aligned, statuses, errors

    def metadata(self) -> dict:
        self.initialize()
        return {
            **self._metadata,
            "cuda_oom_split_count": self._oom_split_count,
            "smallest_successful_inference_chunk": self._smallest_inference_chunk,
        }


def _create_or_validate_database(
    path: Path,
    seed: int,
    required_count: int,
    required_sha256: str,
) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-131072")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "smiles TEXT PRIMARY KEY, n_rows INTEGER NOT NULL, n_cols INTEGER NOT NULL, "
        "data BLOB, status TEXT NOT NULL, error TEXT) WITHOUT ROWID"
    )
    connection.execute(
        "CREATE TABLE IF NOT EXISTS metadata ("
        "key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )
    if existed:
        existing_seed = _read_metadata(connection, "conformer_seed")
        existing_count = _read_metadata(connection, "required_items")
        existing_hash = _read_metadata(connection, "required_keys_sha256")
        if existing_seed not in (None, seed):
            raise RuntimeError(
                f"Existing cache seed {existing_seed} does not match requested seed {seed}."
            )
        if existing_count not in (None, required_count) or existing_hash not in (
            None,
            required_sha256,
        ):
            raise RuntimeError("Existing cache workload fingerprint does not match inputs.")
    _json_metadata(connection, "schema_version", CACHE_SCHEMA_VERSION)
    _json_metadata(connection, "conformer_seed", seed)
    _json_metadata(connection, "required_items", required_count)
    _json_metadata(connection, "required_keys_sha256", required_sha256)
    _json_metadata(connection, "complete", 0)
    connection.commit()
    return connection


def build_sqlite_cache(
    required_smiles: Sequence[str],
    inventory_audit: dict,
    output_path: str | Path,
    seed: int,
    backend,
    batch_size: int,
    status_csv_path: str | Path,
    summary_json_path: str | Path,
    input_fingerprints: Optional[dict] = None,
) -> dict:
    """Build or resume a cache.  ``backend`` needs ``encode_batch``/``metadata``."""
    seed = validate_conformer_seed(seed)
    required = sorted(set(str(item) for item in required_smiles))
    required_hash = hashlib.sha256("\n".join(required).encode("utf-8")).hexdigest()
    if inventory_audit.get("required_key_count") != len(required):
        raise ValueError("Inventory audit count does not match required SMILES.")
    output = Path(output_path).resolve()
    connection = _create_or_validate_database(
        output, seed, len(required), required_hash
    )
    stored = {
        row[0] for row in connection.execute("SELECT smiles FROM embeddings")
    }
    pending = [value for value in required if value not in stored]
    started = time.perf_counter()
    print(
        f"Seed {seed}: {len(stored):,} cached, {len(pending):,} pending, "
        f"{len(required):,} required.",
        flush=True,
    )
    backend_metadata = backend.metadata() if pending else {}
    if pending:
        identity = _backend_identity(backend_metadata)
        previous_identity = _read_metadata(connection, "backend_identity")
        if previous_identity is None and stored:
            raise RuntimeError(
                "Partial cache has no execution-backend identity; refusing to mix "
                "embeddings from an unverifiable earlier process."
            )
        if previous_identity is not None and previous_identity != identity:
            raise RuntimeError(
                "Partial cache was created with a different execution backend; "
                "resume it on the original device/environment."
            )
        _json_metadata(connection, "backend_identity", identity)
        connection.commit()
    commit_interval_batches = 32
    for batch_index, start in enumerate(range(0, len(pending), batch_size)):
        batch = pending[start : start + batch_size]
        arrays, statuses, errors = backend.encode_batch(batch)
        rows = []
        for smiles, array, status, error in zip(batch, arrays, statuses, errors):
            if array is None:
                rows.append((smiles, 0, EMBEDDING_DIM, None, status, error))
            else:
                rows.append(
                    (
                        smiles,
                        int(array.shape[0]),
                        int(array.shape[1]),
                        array.tobytes(order="C"),
                        status,
                        error,
                    )
                )
        connection.executemany(
            "INSERT INTO embeddings(smiles,n_rows,n_cols,data,status,error) "
            "VALUES(?,?,?,?,?,?)",
            rows,
        )
        if (
            (batch_index + 1) % commit_interval_batches == 0
            or start + len(batch) == len(pending)
        ):
            connection.commit()
        completed = len(stored) + start + len(batch)
        if completed % 1000 < len(batch) or completed == len(required):
            print(
                f"Seed {seed}: {completed:,}/{len(required):,} "
                f"({100.0 * completed / len(required):.1f}%)",
                flush=True,
            )

    if pending:
        backend_metadata = backend.metadata()

    row_count = int(connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    if row_count != len(required):
        raise RuntimeError(f"Cache row count {row_count} != required {len(required)}.")
    status_counts = dict(
        connection.execute(
            "SELECT status, COUNT(*) FROM embeddings GROUP BY status ORDER BY status"
        ).fetchall()
    )
    null_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM embeddings WHERE data IS NULL"
        ).fetchone()[0]
    )
    bad_shape_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM embeddings "
            "WHERE data IS NOT NULL AND (n_rows < 1 OR n_cols != ? OR length(data) != n_rows*n_cols*4)",
            (EMBEDDING_DIM,),
        ).fetchone()[0]
    )
    if bad_shape_count:
        raise RuntimeError(f"Cache contains {bad_shape_count} invalid embedding blobs.")

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "protocol_id": f"B-CONFORMER-SEED-{seed}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conformer_seed": seed,
        "conformer_label": f"C{seed - 41}",
        "required_items": len(required),
        "required_keys_sha256": required_hash,
        "stored_items": row_count,
        "null_embedding_items": null_count,
        "status_counts": status_counts,
        "elapsed_seconds_this_invocation": elapsed,
        "cache_path": str(output),
        "cache_size_bytes": output.stat().st_size,
        "inventory_audit": inventory_audit,
        "input_fingerprints": input_fingerprints or {},
        "backend": backend_metadata,
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
        },
        "complete": True,
    }
    for key, value in {
        "complete": 1,
        "status_counts": status_counts,
        "null_embedding_items": null_count,
        "backend": backend_metadata,
        "input_fingerprints": input_fingerprints or {},
        "completed_at_utc": summary["created_at_utc"],
    }.items():
        _json_metadata(connection, key, value)
    connection.commit()
    connection.execute("PRAGMA optimize")

    status_csv = Path(status_csv_path)
    status_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(status_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["smiles", "status", "error"])
        writer.writerows(
            connection.execute(
                "SELECT smiles,status,error FROM embeddings "
                "WHERE status != 'ok' OR data IS NULL ORDER BY smiles"
            )
        )
    connection.close()

    summary_path = Path(summary_json_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    parser.add_argument("--candidate-jsonl", default="outputs/rerank_dataset.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--status-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--dictionary")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--limit",
        type=int,
        help="Infrastructure smoke test only; never use for a scientific run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = validate_conformer_seed(args.seed)
    if args.batch_size < 1 or args.threads < 1:
        raise SystemExit("--batch-size and --threads must be positive.")
    inventory = collect_official_feature_workload(
        args.source_csv, args.metadata_csv, args.candidate_jsonl
    )
    required_smiles = inventory.smiles
    inventory_audit = inventory.audit
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive.")
        required_smiles = inventory.smiles[: args.limit]
        inventory_audit = dict(inventory.audit)
        inventory_audit["required_key_count"] = len(required_smiles)
        inventory_audit["smoke_limit"] = args.limit
    checkpoint, dictionary = discover_weight_paths(args.checkpoint, args.dictionary)
    backend = SeededUniMolBackend(
        seed=seed,
        checkpoint=checkpoint,
        dictionary=dictionary,
        batch_size=args.batch_size,
        threads=args.threads,
        device=args.device,
    )
    inputs = {
        "source_csv": file_fingerprint(args.source_csv),
        "metadata_csv": file_fingerprint(args.metadata_csv),
        "candidate_jsonl": file_fingerprint(args.candidate_jsonl),
    }
    summary = build_sqlite_cache(
        required_smiles,
        inventory_audit,
        args.output,
        seed,
        backend,
        args.batch_size,
        args.status_csv,
        args.summary_json,
        inputs,
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
