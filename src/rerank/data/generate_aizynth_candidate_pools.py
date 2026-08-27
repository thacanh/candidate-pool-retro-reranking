"""Resumable one-pass AiZynthFinder cap-10 reproduction and cap-50 generation.

The expensive expansion-policy call is performed once per source row.  Up to
50 raw outcomes are retained in policy/action order.  The historical cap-10
view is derived from the first ten *raw* outcomes and the cap-50 view from the
first fifty, applying exact-string deduplication only after truncation.  This
matches the historical generator's ordering semantics and avoids running the
policy twice.

Chunk files are immutable, independently validatable resume units.  Completed
chunks are merged in source-row order, so worker completion order cannot alter
the scientific artifact.  The official historical comparison is fail-closed:
cap-50 is not released when the regenerated cap-10 prefix differs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
GENERATOR_PROTOCOL_ID = "A-CAP50-ONEPASS-v1"
CAP10_PROTOCOL_ID = "A-CAP10-REPRO"
CAP50_PROTOCOL_ID = "A-CAP50"
DEFAULT_ASSETS = (
    "modelchem/uspto_model.onnx",
    "modelchem/uspto_templates.csv.gz",
    "modelchem/uspto_ringbreaker_model.onnx",
    "modelchem/uspto_ringbreaker_templates.csv.gz",
    "modelchem/uspto_filter_model.onnx",
    "modelchem/zinc_stock.hdf5",
    "aizynth_jobs/requirements-aizynth-linux-py310.lock",
)

_WORKER_POLICY: Any = None
_WORKER_MAX_ACTIONS = 50
_WORKER_RAW_CAP = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(resolved),
    }


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, target)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def read_source_products(
    path: str | Path,
    product_column: str = "products_smiles",
    max_products: int | None = None,
) -> list[tuple[int, str]]:
    products: list[tuple[int, str]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or product_column not in reader.fieldnames:
            raise ValueError(
                f"Source CSV lacks required column {product_column!r}: {reader.fieldnames}"
            )
        for source_index, row in enumerate(reader):
            if max_products is not None and len(products) >= max_products:
                break
            product = str(row.get(product_column, "")).strip()
            if not product:
                raise ValueError(f"Empty product at zero-based source row {source_index}.")
            products.append((source_index, product))
    if not products:
        raise ValueError("Source CSV yielded no products.")
    return products


def exact_string_deduplicate(
    raw_outcomes: Sequence[Mapping[str, Any]], cap: int
) -> list[dict[str, Any]]:
    """Truncate raw outcomes, then deduplicate like the historical generator."""
    if cap <= 0:
        raise ValueError("Candidate cap must be positive.")
    ordered: dict[str, dict[str, Any]] = {}
    for raw_position, outcome in enumerate(raw_outcomes[:cap]):
        reactant = str(outcome["reactant"])
        prior = float(outcome["prior"])
        candidate = {
            "reactant": reactant,
            "prior": prior,
            "raw_outcome_index": int(outcome.get("raw_outcome_index", raw_position)),
            "action_index": int(outcome.get("action_index", -1)),
        }
        existing = ordered.get(reactant)
        if existing is None:
            ordered[reactant] = candidate
        elif prior > float(existing["prior"]):
            # Assignment to an existing key preserves its first insertion order.
            ordered[reactant] = candidate
    return list(ordered.values())


def flat_candidate_records(
    product_record: Mapping[str, Any], cap: int
) -> list[dict[str, Any]]:
    candidates = exact_string_deduplicate(product_record.get("raw_outcomes", ()), cap)
    return [
        {
            "product": str(product_record["product"]),
            "reactant": candidate["reactant"],
            "label": 0,
            "prior": candidate["prior"],
            "source_row_index": int(product_record["source_row_index"]),
            "raw_outcome_index": candidate["raw_outcome_index"],
            "action_index": candidate["action_index"],
        }
        for candidate in candidates
    ]


def _normalise_actions(actions: Any, priors: Any) -> tuple[list[Any], list[Any]]:
    if actions is None or priors is None:
        return [], []
    actions_list = list(actions)
    priors_list = list(priors)
    if actions_list and isinstance(actions_list[0], (list, tuple)):
        actions_list = list(actions_list[0])
        priors_list = list(priors_list[0])
    return actions_list, priors_list


def _outcome_to_smiles(outcome: Any) -> str:
    if hasattr(outcome, "smiles"):
        return str(outcome.smiles)
    molecules = list(outcome)
    return ".".join(str(molecule.smiles) for molecule in molecules)


def _initialise_worker(config_path: str, max_actions: int, raw_cap: int) -> None:
    global _WORKER_POLICY, _WORKER_MAX_ACTIONS, _WORKER_RAW_CAP
    from aizynthfinder.context.config import Configuration

    config = Configuration.from_file(config_path)
    policy = config.expansion_policy
    policy.select_all()
    _WORKER_POLICY = policy
    _WORKER_MAX_ACTIONS = int(max_actions)
    _WORKER_RAW_CAP = int(raw_cap)


def _expand_product(task: tuple[int, str]) -> dict[str, Any]:
    source_index, smiles = task
    if _WORKER_POLICY is None:
        raise RuntimeError("AiZynthFinder worker was not initialized.")

    from aizynthfinder.chem import TreeMolecule

    record: dict[str, Any] = {
        "source_row_index": int(source_index),
        "product": smiles,
        "raw_outcomes": [],
        "actions_returned": 0,
        "actions_visited": 0,
        "errors": [],
        "status": "running",
    }
    try:
        molecule = TreeMolecule(parent=None, transform=0, smiles=smiles)
    except Exception as error:  # pragma: no cover - exercised with real backend
        record["status"] = "product_parse_error"
        record["errors"].append(
            {"stage": "product_parse", "type": type(error).__name__, "message": str(error)}
        )
        return record

    try:
        actions, priors = _WORKER_POLICY.get_actions([molecule])
        actions, priors = _normalise_actions(actions, priors)
    except Exception as error:  # pragma: no cover - exercised with real backend
        record["status"] = "policy_error"
        record["errors"].append(
            {"stage": "policy", "type": type(error).__name__, "message": str(error)}
        )
        return record

    record["actions_returned"] = len(actions)
    for action_index, (action, prior_value) in enumerate(zip(actions, priors)):
        if action_index >= _WORKER_MAX_ACTIONS or len(record["raw_outcomes"]) >= _WORKER_RAW_CAP:
            break
        record["actions_visited"] += 1
        try:
            outcomes = action.reactants
            if not outcomes:
                outcomes = action.apply(molecule)
            for outcome_index, outcome in enumerate(outcomes or ()):
                try:
                    reactant = _outcome_to_smiles(outcome)
                    prior = float(prior_value)
                    if not reactant or not math.isfinite(prior):
                        raise ValueError("empty reactant or non-finite prior")
                    record["raw_outcomes"].append(
                        {
                            "reactant": reactant,
                            "prior": prior,
                            "action_index": action_index,
                            "outcome_index": outcome_index,
                            "raw_outcome_index": len(record["raw_outcomes"]),
                        }
                    )
                except Exception as error:
                    record["errors"].append(
                        {
                            "stage": "outcome",
                            "action_index": action_index,
                            "outcome_index": outcome_index,
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                    )
                if len(record["raw_outcomes"]) >= _WORKER_RAW_CAP:
                    break
        except Exception as error:
            record["errors"].append(
                {
                    "stage": "action",
                    "action_index": action_index,
                    "type": type(error).__name__,
                    "message": str(error),
                }
            )

    record["status"] = "ok" if record["raw_outcomes"] else "empty"
    return record


def _process_chunk(tasks: list[tuple[int, str]]) -> list[dict[str, Any]]:
    return [_expand_product(task) for task in tasks]


def chunk_path(chunks_dir: Path, start: int, stop: int) -> Path:
    return chunks_dir / f"chunk_{start:06d}_{stop:06d}.jsonl"


def validate_chunk(
    path: str | Path, expected_tasks: Sequence[tuple[int, str]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed chunk line {line_number}: {path}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Non-object chunk line {line_number}: {path}")
            records.append(record)
    if len(records) != len(expected_tasks):
        raise ValueError(
            f"Chunk {path} has {len(records)} rows; expected {len(expected_tasks)}."
        )
    for record, (expected_index, expected_product) in zip(records, expected_tasks):
        if int(record.get("source_row_index", -1)) != expected_index:
            raise ValueError(f"Chunk source index mismatch: {path}")
        if str(record.get("product", "")) != expected_product:
            raise ValueError(f"Chunk product mismatch: {path}")
        if not isinstance(record.get("raw_outcomes"), list):
            raise ValueError(f"Chunk lacks raw outcome list: {path}")
    return records


def write_chunk(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(
        json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        for record in records
    )
    atomic_write_text(path, content)


def _iter_chunk_specs(
    products: Sequence[tuple[int, str]], chunk_size: int
) -> Iterable[tuple[int, int, list[tuple[int, str]]]]:
    for start in range(0, len(products), chunk_size):
        stop = min(start + chunk_size, len(products))
        yield start, stop, list(products[start:stop])


def _write_jsonl_atomic(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def merge_chunks(
    products: Sequence[tuple[int, str]],
    chunks_dir: str | Path,
    chunk_size: int,
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    chunks_root = Path(chunks_dir)
    all_records: list[dict[str, Any]] = []
    for start, stop, tasks in _iter_chunk_specs(products, chunk_size):
        all_records.extend(validate_chunk(chunk_path(chunks_root, start, stop), tasks))

    raw_output = root / "raw_policy_outcomes_top50.jsonl"
    cap10_output = root / "candidates_cap10.jsonl"
    cap50_output = root / "candidates_cap50.jsonl"
    raw_count = _write_jsonl_atomic(raw_output, all_records)
    cap10_count = _write_jsonl_atomic(
        cap10_output,
        (
            candidate
            for record in all_records
            for candidate in flat_candidate_records(record, 10)
        ),
    )
    cap50_count = _write_jsonl_atomic(
        cap50_output,
        (
            candidate
            for record in all_records
            for candidate in flat_candidate_records(record, 50)
        ),
    )
    return {
        "raw_product_records": raw_count,
        "raw_outcomes": sum(len(record["raw_outcomes"]) for record in all_records),
        "cap10_candidate_records": cap10_count,
        "cap50_candidate_records": cap50_count,
        "empty_products": sum(record["status"] == "empty" for record in all_records),
        "product_parse_errors": sum(
            record["status"] == "product_parse_error" for record in all_records
        ),
        "policy_errors": sum(record["status"] == "policy_error" for record in all_records),
        "records_with_errors": sum(bool(record["errors"]) for record in all_records),
        "action_or_outcome_errors": sum(len(record["errors"]) for record in all_records),
        "output_paths": {
            "raw": raw_output,
            "cap10": cap10_output,
            "cap50": cap50_output,
        },
    }


def environment_record() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for package in (
        "aizynthfinder",
        "numpy",
        "onnxruntime",
        "pandas",
        "rdkit",
        "reaction-utils",
        "networkx",
        "dask",
        "tables",
        "tqdm",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED", "not-set"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "not-set"),
    }


def build_manifest(
    *,
    protocol_id: str,
    comparator: str,
    intended_change: str,
    input_fingerprints: Mapping[str, Any],
    output_fingerprints: Mapping[str, Any],
    settings: Mapping[str, Any],
    counts: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol_id,
        "comparator": comparator,
        "single_intended_change": intended_change,
        "input_fingerprints": dict(input_fingerprints),
        "settings": dict(settings),
        "environment": environment_record(),
        "failures": {
            key: counts[key]
            for key in (
                "empty_products",
                "product_parse_errors",
                "policy_errors",
                "records_with_errors",
                "action_or_outcome_errors",
            )
        },
        "counts": {key: value for key, value in counts.items() if key != "output_paths"},
        "output": dict(output_fingerprints),
        "runtime": dict(runtime),
        "created_at_utc": utc_now(),
    }


def _run_cap10_gate(
    legacy_reference: Path,
    regenerated: Path,
    manifest_path: Path,
    output_root: Path,
) -> bool:
    from rerank.analysis.compare_candidate_pools import (
        compare_candidate_pools,
        validate_manifest_schema,
        write_reports,
    )

    summary, discrepancies = compare_candidate_pools(legacy_reference, regenerated)
    manifest_errors = validate_manifest_schema(_load_json(manifest_path))
    summary["manifest_validation"] = {
        "path": str(manifest_path.resolve()),
        "valid": not manifest_errors,
        "errors": manifest_errors,
    }
    summary["passed"] = bool(summary["comparison_passed"] and not manifest_errors)
    if manifest_errors:
        discrepancies.append(
            {
                "product_key": None,
                "issues": [{"type": "manifest_schema_invalid", "errors": manifest_errors}],
            }
        )
    summary["discrepancy_record_count"] = len(discrepancies)
    write_reports(
        summary,
        discrepancies,
        output_root / "cap10_comparison_summary.json",
        output_root / "cap10_discrepancies.jsonl",
    )
    gate = {
        "schema_version": 1,
        "protocol_id": CAP10_PROTOCOL_ID,
        "passed": bool(summary["passed"]),
        "cap50_released": bool(summary["passed"]),
        "rule": "cap-50 downstream work is forbidden unless regenerated cap-10 passes",
        "summary_sha256": sha256_file(output_root / "cap10_comparison_summary.json"),
        "checked_at_utc": utc_now(),
    }
    atomic_write_json(output_root / "CAP50_RELEASE_GATE.json", gate)
    return bool(summary["passed"])


def run_generation(args: argparse.Namespace) -> int:
    if args.cap10 != 10 or args.cap50 != 50 or args.max_actions != 50:
        raise ValueError("Approved A-CAP10/A-CAP50 settings require caps 10/50 and 50 actions.")
    if args.chunk_size <= 0 or args.workers <= 0:
        raise ValueError("workers and chunk-size must be positive.")
    if args.max_products is not None and args.max_products <= 0:
        raise ValueError("max-products must be positive when supplied.")

    output_root = Path(args.output_root).resolve()
    chunks_dir = output_root / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    products = read_source_products(args.input_csv, args.product_column, args.max_products)

    asset_paths = [Path(path).resolve() for path in args.asset]
    for path in (Path(args.input_csv).resolve(), Path(args.config).resolve(), *asset_paths):
        if not path.is_file():
            raise FileNotFoundError(path)
    legacy_reference = Path(args.legacy_reference).resolve() if args.legacy_reference else None
    if args.max_products is None and legacy_reference is None:
        raise ValueError("A full scientific run requires --legacy-reference for the fail-closed gate.")
    if legacy_reference is not None and not legacy_reference.is_file():
        raise FileNotFoundError(legacy_reference)

    fingerprints = {
        "source_csv": file_fingerprint(args.input_csv),
        "configuration": file_fingerprint(args.config),
    }
    for index, path in enumerate(asset_paths):
        fingerprints[f"asset_{index:02d}_{path.name}"] = file_fingerprint(path)
    if legacy_reference is not None:
        fingerprints["legacy_cap10_reference"] = file_fingerprint(legacy_reference)

    identity = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": GENERATOR_PROTOCOL_ID,
        "input_fingerprints": fingerprints,
        "settings": {
            "product_column": args.product_column,
            "max_actions_visited_per_product": args.max_actions,
            "raw_outcome_cap": args.cap50,
            "cap10": args.cap10,
            "cap50": args.cap50,
            "chunk_size": args.chunk_size,
            "max_products": args.max_products,
            "selected_expansion_policies": "all configured policies",
            "filter_policy_called": False,
            "deduplication": "exact reactant string after raw-outcome truncation",
            "merge_order": "source CSV row order",
        },
    }
    identity_path = output_root / "run_identity.json"
    if identity_path.exists():
        if _load_json(identity_path) != identity:
            raise PermissionError(
                "Output root belongs to different inputs/settings; use a new output root."
            )
    else:
        atomic_write_json(identity_path, identity)

    state_path = output_root / "run_state.json"
    state = _load_json(state_path) if state_path.exists() else {
        "started_at_utc": utc_now(),
        "active_runtime_seconds": 0.0,
        "sessions": 0,
    }
    session_start = time.perf_counter()
    state["sessions"] = int(state.get("sessions", 0)) + 1

    specs = list(_iter_chunk_specs(products, args.chunk_size))
    pending: list[tuple[int, int, list[tuple[int, str]], Path]] = []
    completed_products = 0
    resumed_chunks = 0
    for start, stop, tasks in specs:
        path = chunk_path(chunks_dir, start, stop)
        if path.exists():
            try:
                validate_chunk(path, tasks)
            except ValueError as error:
                raise RuntimeError(
                    f"Existing chunk is partial/tampered; preserve it for investigation: {path}"
                ) from error
            completed_products += len(tasks)
            resumed_chunks += 1
        else:
            pending.append((start, stop, tasks, path))

    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    progress = (
        tqdm(
            total=len(products),
            initial=completed_products,
            unit="product",
            desc="AiZynth Top-50",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )
    if pending:
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_initialise_worker,
            initargs=(str(Path(args.config).resolve()), args.max_actions, args.cap50),
        ) as executor:
            future_map = {
                executor.submit(_process_chunk, tasks): (start, stop, tasks, path)
                for start, stop, tasks, path in pending
            }
            for future in as_completed(future_map):
                start, stop, tasks, path = future_map[future]
                records = future.result()
                if len(records) != len(tasks):
                    raise RuntimeError(f"Worker returned incomplete chunk {start}:{stop}.")
                write_chunk(path, records)
                validate_chunk(path, tasks)
                completed_products += len(tasks)
                if progress is not None:
                    progress.update(len(tasks))
                    progress.set_postfix(workers=args.workers, chunks=f"{completed_products}/{len(products)}")
    if progress is not None:
        progress.close()

    merged = merge_chunks(products, chunks_dir, args.chunk_size, output_root)
    active_seconds = float(state.get("active_runtime_seconds", 0.0)) + (
        time.perf_counter() - session_start
    )
    state.update(
        {
            "active_runtime_seconds": active_seconds,
            "completed_products": len(products),
            "completed_chunks": len(specs),
            "status": "generated",
            "updated_at_utc": utc_now(),
        }
    )
    atomic_write_json(state_path, state)

    output_paths = merged["output_paths"]
    output_fingerprints = {
        name: file_fingerprint(path) for name, path in output_paths.items()
    }
    runtime = {
        "started_at_utc": state["started_at_utc"],
        "completed_at_utc": utc_now(),
        "active_runtime_seconds": active_seconds,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
        "chunk_count": len(specs),
        "resumed_chunk_count_at_last_session": resumed_chunks,
    }
    settings = identity["settings"]
    common_kwargs = {
        "input_fingerprints": fingerprints,
        "output_fingerprints": output_fingerprints,
        "settings": settings,
        "counts": merged,
        "runtime": runtime,
    }
    generation_manifest = build_manifest(
        protocol_id=GENERATOR_PROTOCOL_ID,
        comparator="one policy expansion stream feeding both candidate caps",
        intended_change="record up to 50 raw outcomes once and derive both approved caps",
        **common_kwargs,
    )
    cap10_manifest = build_manifest(
        protocol_id=CAP10_PROTOCOL_ID,
        comparator="legacy-cap10-fixed50-v1 historical candidate JSONL",
        intended_change="clean-environment reproduction only; no intended scientific change",
        **common_kwargs,
    )
    cap50_manifest = build_manifest(
        protocol_id=CAP50_PROTOCOL_ID,
        comparator="regenerated A-CAP10-REPRO prefix from the identical raw policy stream",
        intended_change="maximum stored raw outcomes per product changes from 10 to 50",
        **common_kwargs,
    )
    atomic_write_json(output_root / "generation_manifest.json", generation_manifest)
    atomic_write_json(output_root / "cap10_manifest.json", cap10_manifest)
    atomic_write_json(output_root / "cap50_manifest.json", cap50_manifest)

    if args.max_products is not None:
        pilot = {
            "status": "timing-pilot-only",
            "scientific_result": False,
            "products": len(products),
            "active_runtime_seconds": active_seconds,
            "products_per_second": len(products) / max(active_seconds, 1e-9),
            "full_50037_eta_seconds": 50037 * active_seconds / len(products),
        }
        atomic_write_json(output_root / "pilot_summary.json", pilot)
        print(json.dumps(pilot, indent=2))
        return 0

    assert legacy_reference is not None
    gate_passed = _run_cap10_gate(
        legacy_reference,
        Path(output_paths["cap10"]),
        output_root / "cap10_manifest.json",
        output_root,
    )
    state["status"] = "complete" if gate_passed else "blocked-cap10-mismatch"
    state["cap10_gate_passed"] = gate_passed
    atomic_write_json(state_path, state)
    print(
        json.dumps(
            {
                "status": state["status"],
                "products": len(products),
                "cap10_records": merged["cap10_candidate_records"],
                "cap50_records": merged["cap50_candidate_records"],
                "active_runtime_seconds": active_seconds,
                "cap50_released": gate_passed,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )
    return 0 if gate_passed else 2


def run_status(args: argparse.Namespace) -> int:
    root = Path(args.output_root).resolve()
    state = _load_json(root / "run_state.json") if (root / "run_state.json").exists() else {}
    identity = _load_json(root / "run_identity.json") if (root / "run_identity.json").exists() else {}
    chunk_files = sorted((root / "chunks").glob("chunk_*.jsonl")) if (root / "chunks").exists() else []
    payload = {
        "output_root": str(root),
        "status": state.get("status", "not-started"),
        "completed_products": state.get("completed_products", 0),
        "completed_chunks": len(chunk_files),
        "active_runtime_seconds": state.get("active_runtime_seconds", 0.0),
        "cap10_gate_passed": state.get("cap10_gate_passed"),
        "protocol_id": identity.get("protocol_id"),
        "artifacts": {
            name: (root / name).is_file()
            for name in (
                "raw_policy_outcomes_top50.jsonl",
                "candidates_cap10.jsonl",
                "candidates_cap50.jsonl",
                "generation_manifest.json",
                "CAP50_RELEASE_GATE.json",
            )
        },
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--input-csv", default="data/uspto_smiles.csv")
    generate.add_argument("--product-column", default="products_smiles")
    generate.add_argument("--config", default="modelchem/config.yml")
    generate.add_argument("--legacy-reference", default="outputs/rerank_dataset.jsonl")
    generate.add_argument(
        "--asset", action="append", default=None,
        help="Fingerprint one generator asset; repeat as needed.",
    )
    generate.add_argument(
        "--output-root",
        default="outputs/jcheminform_revision/candidate_pools/aizynth_onepass",
    )
    generate.add_argument("--workers", type=int, default=4)
    generate.add_argument("--chunk-size", type=int, default=128)
    generate.add_argument("--max-actions", type=int, default=50)
    generate.add_argument("--cap10", type=int, default=10)
    generate.add_argument("--cap50", type=int, default=50)
    generate.add_argument(
        "--max-products", type=int,
        help="Timing pilot only. Supplying this disables the scientific comparison gate.",
    )
    status = subparsers.add_parser("status")
    status.add_argument(
        "--output-root",
        default="outputs/jcheminform_revision/candidate_pools/aizynth_onepass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        if args.asset is None:
            args.asset = list(DEFAULT_ASSETS)
        return run_generation(args)
    if args.command == "status":
        return run_status(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
