#!/usr/bin/env python
"""Run one conformer seed from embedding generation through paper-ready outputs.

Each seed is isolated below ``outputs/jcheminform_revision/conformers``.  The
large atom cache is resumable and seed-local; it is deleted only after compact
features, five ranking runs, reaction-level predictions, metrics, manifests and
checksums have all passed validation.

This runner intentionally labels the ranking stage as the frozen
``legacy-cap10-fixed50-v1`` sensitivity protocol.  The saved feature cache can
later be reused by the separately prespecified tuned protocol without
regenerating the conformer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from rerank.data.build_conformer_sqlite import (
    ALLOWED_CONFORMER_SEEDS,
    PINNED_CHECKPOINT_SHA256,
    PINNED_DICTIONARY_SHA256,
    discover_weight_paths,
    validate_conformer_seed,
    validate_weight_paths,
)
from rerank.features import FEATURE_NAMES_MAP
from rerank.study_data import STUDY_CACHE_SCHEMA, file_fingerprint


PROTOCOL_ID = "legacy-cap10-fixed50-v1"
TRAINING_SEEDS = (42, 43, 44, 45, 46)
EXPECTED_INPUT_HASHES = {
    "source_csv": "688c5b8ea7c3269b53ae15ffca9ec98f51fd29ea3fc25edc8fa66cabe9042d6a",
    "metadata_csv": "c9f3c5f01e64cc9192f20f9889694b85d83986f4e991d2ee5bb52b5db8e08e52",
    "candidate_jsonl": "9ec1cf192c49eeb7d74a320dd721287fabdef9863cc06f95d0f13baab8c3ff85",
}
EXPECTED_TRAIN_PRODUCTS = 31_513
EXPECTED_VALID_REACTIONS = 5_004
EXPECTED_VALID_COVERED = 4_039
EXPECTED_TEST_REACTIONS = 5_004
EXPECTED_TEST_COVERED = 3_985
EXPECTED_FEATURE_DIM = len(FEATURE_NAMES_MAP["3d+prior"])


def resolve_runtime(
    requested_device: str = "auto",
    embedding_batch_size: int = 0,
    embedding_threads: int = 0,
    *,
    torch_module=None,
    logical_cpu_count: Optional[int] = None,
    physical_cpu_count: Optional[int] = None,
) -> dict:
    """Resolve a conservative CPU/GPU configuration for a heterogeneous host."""
    requested = requested_device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda.")
    if embedding_batch_size < 0 or embedding_threads < 0:
        raise ValueError("batch size and threads must be zero (auto) or positive.")
    if torch_module is None:
        import torch as torch_module

    logical = int(logical_cpu_count or os.cpu_count() or 1)
    if physical_cpu_count is None:
        try:
            import psutil

            physical_cpu_count = psutil.cpu_count(logical=False)
        except Exception:
            physical_cpu_count = None
    physical = int(physical_cpu_count or max(1, logical // 2))
    cuda_available = bool(torch_module.cuda.is_available())
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was explicitly requested, but PyTorch cannot use it on this machine."
        )
    resolved_device = "cuda" if cuda_available and requested != "cpu" else "cpu"
    gpu = None
    if resolved_device == "cuda":
        properties = torch_module.cuda.get_device_properties(0)
        vram = int(properties.total_memory)
        if vram >= 16 * 1024**3:
            automatic_batch_size = 32
        elif vram >= 8 * 1024**3:
            automatic_batch_size = 16
        elif vram >= 4 * 1024**3:
            automatic_batch_size = 8
        else:
            automatic_batch_size = 4
        gpu = {
            "index": 0,
            "name": torch_module.cuda.get_device_name(0),
            "total_memory_bytes": vram,
            "total_memory_gib": vram / 1024**3,
            "capability": list(torch_module.cuda.get_device_capability(0)),
        }
    else:
        automatic_batch_size = 8
    threads = embedding_threads or max(1, min(16, physical))
    batch_size = embedding_batch_size or automatic_batch_size
    return {
        "requested_device": requested,
        "resolved_device": resolved_device,
        "cuda_available": cuda_available,
        "torch_version": torch_module.__version__,
        "torch_cuda_version": torch_module.version.cuda,
        "logical_cpu_count": logical,
        "physical_cpu_count": physical,
        "embedding_threads": threads,
        "embedding_batch_size": batch_size,
        "embedding_threads_source": "explicit" if embedding_threads else "auto",
        "embedding_batch_size_source": (
            "explicit" if embedding_batch_size else "auto"
        ),
        "ranking_device": resolved_device,
        "gpu": gpu,
        "cuda_oom_policy": "recursively halve only the failing inference batch",
        "rdkit_device": "cpu",
    }


def conformer_label(seed: int) -> str:
    return f"C{validate_conformer_seed(seed) - 41}"


def seed_run_dir(output_root: str | Path, seed: int) -> Path:
    return Path(output_root).resolve() / f"seed_{validate_conformer_seed(seed)}"


def build_layout(output_root: str | Path, seed: int) -> dict[str, Path]:
    root = seed_run_dir(output_root, seed)
    return {
        "root": root,
        "logs": root / "logs",
        "scratch": root / "scratch",
        "atom_cache": root / "scratch" / f"atom_embeddings_seed_{seed}.sqlite",
        "features": root / "features",
        "feature_cache": root
        / "features"
        / f"official_3d_prior_schema{STUDY_CACHE_SCHEMA}.pkl",
        "validation_features": root
        / "features"
        / f"official_valid_3d_prior_schema{STUDY_CACHE_SCHEMA}.pkl",
        "ranking": root / "ranking_legacy_fixed50",
        "embedding_summary": root / "embedding_summary.json",
        "embedding_status": root / "embedding_non_ok.csv",
        "run_status": root / "RUN_STATUS.json",
        "manifest": root / "manifest.json",
        "result_summary": root / "result_summary.json",
        "checksums": root / "checksums.sha256",
        "completed": root / "COMPLETED.json",
        "lock": root / "RUNNING.lock",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _capture_environment(runtime: Optional[dict] = None) -> dict:
    requested = (
        "numpy",
        "pandas",
        "torch",
        "rdkit",
        "unimol-tools",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "psutil",
    )
    versions = {}
    records = {}
    for name in requested:
        try:
            distribution = importlib.metadata.distribution(name)
            versions[name] = distribution.version
            record = distribution.read_text("RECORD") or ""
            records[name] = hashlib.sha256(record.encode("utf-8")).hexdigest()
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            records[name] = None
    freeze = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "logical_cpu_count": os.cpu_count(),
        "package_versions": versions,
        "distribution_record_sha256": records,
        "installed_snapshot_sha256": hashlib.sha256(
            "\n".join(freeze).encode("utf-8")
        ).hexdigest(),
        "installed_snapshot": freeze,
        "runtime_selection": runtime,
    }


def validate_inputs(
    source_csv: Path,
    metadata_csv: Path,
    candidate_jsonl: Path,
    checkpoint: Path,
    dictionary: Path,
) -> dict:
    assets = validate_weight_paths(checkpoint, dictionary)
    inputs = {}
    for key, path in {
        "source_csv": source_csv,
        "metadata_csv": metadata_csv,
        "candidate_jsonl": candidate_jsonl,
    }.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required input: {path}")
        fingerprint = file_fingerprint(path)
        expected = EXPECTED_INPUT_HASHES[key]
        if fingerprint["sha256"] != expected:
            raise RuntimeError(
                f"{key} SHA-256 mismatch: {fingerprint['sha256']} != {expected}."
            )
        inputs[key] = fingerprint
    return {**inputs, **assets}


def _is_process_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(int(pid))
    except Exception:
        return False


def acquire_seed_lock(lock_path: Path, seed: int) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            previous = json.loads(lock_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
        same_host = previous.get("hostname") == socket.gethostname()
        if same_host and not _is_process_alive(int(previous.get("pid", -1))):
            lock_path.unlink()
        else:
            raise RuntimeError(
                f"Seed {seed} already has a live or unverified lock: {lock_path}."
            )
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        payload = json.dumps(
            {
                "seed": seed,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ).encode("utf-8")
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _run_logged(command: list[str], log_path: Path, repo_root: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    source_path = str((repo_root / "src").resolve())
    env["PYTHONPATH"] = (
        source_path
        if not env.get("PYTHONPATH")
        else source_path + os.pathsep + env["PYTHONPATH"]
    )
    with open(log_path, "a", encoding="utf-8", errors="replace") as log_handle:
        log_handle.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        log_handle.flush()
        process = subprocess.Popen(
            command,
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_handle.write(line)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(
            f"Command failed with exit code {return_code}; see {log_path}."
        )


def _iter_feature_arrays(payload: dict) -> Iterable[np.ndarray]:
    for product in payload["train_products"]:
        yield np.asarray(product["features"])
    for features in payload["eval_features"]:
        yield np.asarray(features)


def retained_results_exist(layout: dict[str, Path]) -> bool:
    required = [
        layout["feature_cache"],
        layout["validation_features"],
        layout["ranking"] / "per_seed_metrics.json",
        layout["ranking"] / "manifest.json",
    ]
    required.extend(
        layout["ranking"] / f"eval_seed{seed}.csv" for seed in TRAINING_SEEDS
    )
    return all(path.is_file() for path in required)


def validate_retained_results(layout: dict[str, Path]) -> dict:
    feature_path = layout["feature_cache"]
    if not feature_path.is_file():
        raise RuntimeError(f"Missing retained feature cache: {feature_path}")
    with open(feature_path, "rb") as handle:
        blob = pickle.load(handle)
    if blob.get("schema_version") != STUDY_CACHE_SCHEMA:
        raise RuntimeError("Feature-cache schema mismatch.")
    if blob.get("feature_mode") != "3d+prior":
        raise RuntimeError("Feature cache is not the 3d+prior arm.")
    payload = blob.get("payload")
    if not isinstance(payload, dict):
        raise RuntimeError("Feature cache has no payload.")
    audit = payload.get("audit", {})
    expected_audit = {
        "train_products": EXPECTED_TRAIN_PRODUCTS,
        "train_overlap_reactions_excluded": 233,
        "eval_reactions_total": EXPECTED_TEST_REACTIONS,
        "eval_reactions_covered": EXPECTED_TEST_COVERED,
    }
    for key, expected in expected_audit.items():
        if int(audit.get(key, -1)) != expected:
            raise RuntimeError(
                f"Feature-cache audit {key}={audit.get(key)}; expected {expected}."
            )
    feature_rows = 0
    for array in _iter_feature_arrays(payload):
        if array.ndim != 2 or array.shape[1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(f"Unexpected feature array shape {array.shape}.")
        if not np.isfinite(array).all():
            raise RuntimeError("Retained feature cache contains non-finite values.")
        feature_rows += int(array.shape[0])

    validation_path = layout["validation_features"]
    if not validation_path.is_file():
        raise RuntimeError(f"Missing official-validation feature artifact: {validation_path}")
    with open(validation_path, "rb") as handle:
        validation_blob = pickle.load(handle)
    if validation_blob.get("schema_version") != STUDY_CACHE_SCHEMA:
        raise RuntimeError("Validation-feature schema mismatch.")
    validation_audit = validation_blob.get("audit", {})
    if int(validation_audit.get("reactions_total", -1)) != EXPECTED_VALID_REACTIONS:
        raise RuntimeError(
            f"Validation artifact does not contain {EXPECTED_VALID_REACTIONS:,} reactions."
        )
    if int(validation_audit.get("reactions_covered", -1)) != EXPECTED_VALID_COVERED:
        raise RuntimeError(
            f"Validation artifact does not contain {EXPECTED_VALID_COVERED:,} covered reactions."
        )
    validation_payload = validation_blob.get("payload", {})
    validation_rows = 0
    for array in validation_payload.get("eval_features", []):
        array = np.asarray(array)
        if array.ndim != 2 or array.shape[1] != EXPECTED_FEATURE_DIM:
            raise RuntimeError(f"Unexpected validation feature shape {array.shape}.")
        if not np.isfinite(array).all():
            raise RuntimeError("Validation feature artifact contains non-finite values.")
        validation_rows += int(array.shape[0])

    metrics_path = layout["ranking"] / "per_seed_metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("Missing per_seed_metrics.json.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if set(metrics) != {str(seed) for seed in TRAINING_SEEDS}:
        raise RuntimeError("Ranking metrics do not contain exactly seeds 42-46.")
    metric_names = ("top1", "top3", "top5", "top10", "mrr")
    for training_seed in TRAINING_SEEDS:
        row = metrics[str(training_seed)]
        for name in metric_names:
            value = float(row[name])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise RuntimeError(
                    f"Invalid {name}={value} for training seed {training_seed}."
                )
        eval_path = layout["ranking"] / f"eval_seed{training_seed}.csv"
        if not eval_path.is_file():
            raise RuntimeError(f"Missing reaction predictions: {eval_path}")
        with open(eval_path, encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
        if row_count != EXPECTED_TEST_COVERED:
            raise RuntimeError(
                f"{eval_path.name} has {row_count} rows; expected {EXPECTED_TEST_COVERED}."
            )
    if not (layout["ranking"] / "manifest.json").is_file():
        raise RuntimeError("Missing inner ranking manifest.")
    return {
        "feature_cache": file_fingerprint(feature_path),
        "feature_rows": feature_rows,
        "validation_feature_artifact": file_fingerprint(validation_path),
        "validation_feature_rows": validation_rows,
        "validation_feature_audit": validation_audit,
        "feature_cache_audit": audit,
        "training_seeds": list(TRAINING_SEEDS),
        "per_seed_metrics": metrics,
        "reaction_prediction_rows_per_seed": EXPECTED_TEST_COVERED,
        "passed": True,
    }


def safe_cleanup_seed_cache(layout: dict[str, Path], seed: int) -> list[str]:
    root = layout["root"].resolve()
    scratch = layout["scratch"].resolve()
    try:
        scratch.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("Scratch directory is outside the seed run root.") from exc
    cache = layout["atom_cache"].resolve()
    expected_name = f"atom_embeddings_seed_{seed}.sqlite"
    if cache.parent != scratch or cache.name != expected_name:
        raise RuntimeError("Refusing cleanup: atom-cache target is not the exact seed path.")
    removed = []
    for target in (
        cache,
        Path(str(cache) + "-journal"),
        Path(str(cache) + "-wal"),
        Path(str(cache) + "-shm"),
    ):
        if target.exists():
            if target.parent.resolve() != scratch:
                raise RuntimeError(f"Refusing cleanup outside seed scratch: {target}")
            target.unlink()
            removed.append(str(target))
    return removed


def _write_checksums(layout: dict[str, Path]) -> dict[str, str]:
    root = layout["root"]
    excluded = {
        layout["checksums"].resolve(),
        layout["lock"].resolve(),
        layout["run_status"].resolve(),
    }
    checksums = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        if layout["scratch"].resolve() in path.resolve().parents:
            continue
        relative = path.relative_to(root).as_posix()
        checksums[relative] = _sha256(path)
    with open(layout["checksums"], "w", encoding="utf-8", newline="\n") as handle:
        for relative, digest in checksums.items():
            handle.write(f"{digest}  {relative}\n")
    return checksums


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--output-root",
        default="outputs/jcheminform_revision/conformers",
    )
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    parser.add_argument("--candidate-jsonl", default="outputs/rerank_dataset.jsonl")
    parser.add_argument("--checkpoint")
    parser.add_argument("--dictionary")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Auto uses CUDA when the installed PyTorch build and driver support it.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=0,
        help="Zero selects a conservative batch from device VRAM.",
    )
    parser.add_argument(
        "--embedding-threads",
        type=int,
        default=0,
        help="Zero selects up to 16 physical CPU cores.",
    )
    parser.add_argument("--min-free-gib", type=float, default=20.0)
    parser.add_argument(
        "--keep-atom-cache",
        action="store_true",
        help="Retain the large seed-local SQLite cache after successful validation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print paths/commands without scientific computation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed = validate_conformer_seed(args.seed)
    repo_root = Path(__file__).resolve().parents[3]
    layout = build_layout(args.output_root, seed)
    for key in ("root", "logs", "scratch", "features", "ranking"):
        layout[key].mkdir(parents=True, exist_ok=True)
    checkpoint, dictionary = discover_weight_paths(args.checkpoint, args.dictionary)
    source_csv = Path(args.source_csv).resolve()
    metadata_csv = Path(args.metadata_csv).resolve()
    candidate_jsonl = Path(args.candidate_jsonl).resolve()
    inputs = validate_inputs(
        source_csv, metadata_csv, candidate_jsonl, checkpoint, dictionary
    )
    runtime = resolve_runtime(
        args.device, args.embedding_batch_size, args.embedding_threads
    )
    if layout["completed"].is_file():
        validation = validate_retained_results(layout)
        completed = json.loads(layout["completed"].read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    **completed,
                    "already_complete": True,
                    "retained_validation_passed": validation["passed"],
                },
                indent=2,
            )
        )
        return
    free_bytes = shutil.disk_usage(layout["root"]).free
    required_free = int(args.min_free_gib * 1024**3)
    if free_bytes < required_free and not layout["atom_cache"].exists():
        raise RuntimeError(
            f"Only {free_bytes / 1024**3:.1f} GiB free; "
            f"seed start requires at least {args.min_free_gib:.1f} GiB."
        )

    builder_command = [
        sys.executable,
        str(repo_root / "src" / "build_conformer_sqlite.py"),
        "--seed",
        str(seed),
        "--source-csv",
        str(source_csv),
        "--metadata-csv",
        str(metadata_csv),
        "--candidate-jsonl",
        str(candidate_jsonl),
        "--output",
        str(layout["atom_cache"]),
        "--status-csv",
        str(layout["embedding_status"]),
        "--summary-json",
        str(layout["embedding_summary"]),
        "--checkpoint",
        str(checkpoint),
        "--dictionary",
        str(dictionary),
        "--batch-size",
        str(runtime["embedding_batch_size"]),
        "--threads",
        str(runtime["embedding_threads"]),
        "--device",
        runtime["resolved_device"],
    ]
    ranking_command = [
        sys.executable,
        str(repo_root / "src" / "run_controlled_study.py"),
        "--candidate-jsonl",
        str(candidate_jsonl),
        "--source-csv",
        str(source_csv),
        "--metadata-csv",
        str(metadata_csv),
        "--atom-cache",
        str(layout["atom_cache"]),
        "--feature-mode",
        "3d+prior",
        "--seeds",
        ",".join(str(value) for value in TRAINING_SEEDS),
        "--device",
        runtime["ranking_device"],
        "--output-dir",
        str(layout["ranking"]),
        "--feature-cache-dir",
        str(layout["features"]),
    ]
    validation_command = [
        sys.executable,
        str(repo_root / "src" / "export_conformer_validation_features.py"),
        "--source-csv",
        str(source_csv),
        "--metadata-csv",
        str(metadata_csv),
        "--candidate-jsonl",
        str(candidate_jsonl),
        "--atom-cache",
        str(layout["atom_cache"]),
        "--output",
        str(layout["validation_features"]),
        "--conformer-seed",
        str(seed),
    ]
    retained_ready = retained_results_exist(layout)
    if not retained_ready:
        ranking_command.append("--force-rebuild-feature-cache")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "seed": seed,
                    "conformer_label": conformer_label(seed),
                    "run_root": str(layout["root"]),
                    "free_gib": free_bytes / 1024**3,
                    "inputs": inputs,
                    "runtime_selection": runtime,
                    "builder_command": builder_command,
                    "ranking_command": ranking_command,
                    "validation_feature_command": validation_command,
                    "scientific_computation_started": False,
                },
                indent=2,
            )
        )
        return

    acquire_seed_lock(layout["lock"], seed)
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    status = {
        "seed": seed,
        "conformer_label": conformer_label(seed),
        "status": "running",
        "started_at_utc": started_at,
        "stage": "embedding",
        "runtime_selection": runtime,
    }
    _write_json(layout["run_status"], status)
    try:
        if not retained_ready and (
            not layout["embedding_summary"].exists()
            or not layout["atom_cache"].exists()
        ):
            _run_logged(
                builder_command,
                layout["logs"] / "01_embedding.log",
                repo_root,
            )
        status["stage"] = "feature_and_ranking"
        _write_json(layout["run_status"], status)
        if not retained_ready:
            _run_logged(
                ranking_command,
                layout["logs"] / "02_feature_and_ranking.log",
                repo_root,
            )
            _run_logged(
                validation_command,
                layout["logs"] / "03_validation_features.log",
                repo_root,
            )
        status["stage"] = "validation"
        _write_json(layout["run_status"], status)
        validation = validate_retained_results(layout)

        metrics = validation["per_seed_metrics"]
        result_summary = {
            "conformer_seed": seed,
            "conformer_label": conformer_label(seed),
            "protocol_id": PROTOCOL_ID,
            "confirmatory_role": (
                "B1/B2 conformer replicate"
                if seed <= 46
                else "B3 avg10 input; individual ranking is exploratory diagnostic"
            ),
            "training_seeds": list(TRAINING_SEEDS),
            "top1_mean": float(np.mean([metrics[str(value)]["top1"] for value in TRAINING_SEEDS])),
            "top1_std": float(np.std([metrics[str(value)]["top1"] for value in TRAINING_SEEDS], ddof=1)),
            "mrr_mean": float(np.mean([metrics[str(value)]["mrr"] for value in TRAINING_SEEDS])),
            "mrr_std": float(np.std([metrics[str(value)]["mrr"] for value in TRAINING_SEEDS], ddof=1)),
            "per_seed_metrics": metrics,
            "reaction_prediction_rows_per_seed": EXPECTED_TEST_COVERED,
        }
        _write_json(layout["result_summary"], result_summary)

        removed = []
        if not args.keep_atom_cache:
            status["stage"] = "safe_cleanup"
            _write_json(layout["run_status"], status)
            removed = safe_cleanup_seed_cache(layout, seed)
        elapsed = time.perf_counter() - start
        manifest = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "experiment_id": f"B-{conformer_label(seed)}",
            "comparator": "same frozen protocol; only RDKit conformer seed changes",
            "single_intended_change": f"RDKit conformer seed={seed}",
            "confirmatory_role": result_summary["confirmatory_role"],
            "conformer_seed": seed,
            "conformer_label": conformer_label(seed),
            "training_seeds": list(TRAINING_SEEDS),
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": elapsed,
            "git_commit": _git_commit(repo_root),
            "input_fingerprints": inputs,
            "environment": _capture_environment(runtime),
            "validation": validation,
            "result_summary": result_summary,
            "large_atom_cache_retained": bool(args.keep_atom_cache),
            "safe_cleanup_removed": removed,
            "retained_paths": {
                "feature_cache": str(layout["feature_cache"]),
                "validation_features": str(layout["validation_features"]),
                "ranking": str(layout["ranking"]),
                "embedding_summary": str(layout["embedding_summary"]),
                "embedding_non_ok": str(layout["embedding_status"]),
            },
        }
        _write_json(layout["manifest"], manifest)
        checksums = _write_checksums(layout)
        completed = {
            "seed": seed,
            "conformer_label": conformer_label(seed),
            "status": "complete",
            "completed_at_utc": manifest["completed_at_utc"],
            "runtime_seconds": elapsed,
            "large_atom_cache_retained": bool(args.keep_atom_cache),
            "retained_file_count": len(checksums),
            "result_summary_sha256": _sha256(layout["result_summary"]),
            "manifest_sha256": _sha256(layout["manifest"]),
        }
        _write_json(layout["completed"], completed)
        status.update({"status": "complete", "stage": "complete", **completed})
        _write_json(layout["run_status"], status)
        print(json.dumps(completed, indent=2), flush=True)
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "large_atom_cache_preserved_for_resume": layout["atom_cache"].exists(),
            }
        )
        _write_json(layout["run_status"], status)
        raise
    finally:
        if layout["lock"].exists():
            layout["lock"].unlink()


if __name__ == "__main__":
    main()
