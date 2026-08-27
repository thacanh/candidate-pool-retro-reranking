#!/usr/bin/env python
"""Deterministic CPU timing pilot for the C1 Uni-Mol conformer pipeline.

This is infrastructure, not a scientific experiment. It times a stratified
sample and writes only compact JSON/CSV summaries; atom representations are
serialized in memory for timing and immediately discarded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence

import numpy as np
from packaging.version import InvalidVersion, Version


C1_CONFORMER_SEED = 42
PINNED_UNIMOL_VERSION = "0.1.3"
PINNED_RDKIT_VERSION = "2025.09.6"
PINNED_CHECKPOINT_SHA256 = (
    "da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9"
)
REQUIRED_CHECKPOINT_BASENAME = "mol_pre_no_h_220816.pt"
REQUIRED_DICTIONARY_BASENAME = "mol.dict.txt"
REPORT_SCHEMA_VERSION = 1

HEAVY_ATOM_STRATA = (
    ("01_1_10", 1, 10),
    ("02_11_20", 11, 20),
    ("03_21_30", 21, 30),
    ("04_31_40", 31, 40),
    ("05_41_50", 41, 50),
    ("06_51_60", 51, 60),
    ("07_61_80", 61, 80),
    ("08_81_128", 81, 128),
)
TAIL_CENSUS_STRATUM = "08_81_128"
WARMUP_STRATA = ("02_11_20", "03_21_30")
DEFAULT_SAMPLE_PER_STRATUM = 64
DEFAULT_WARMUP_COUNT = 16


@dataclass(frozen=True)
class RequiredSmilesInventory:
    smiles: list[str]
    audit: dict


@dataclass
class PreparedBatch:
    payloads: list[Any]
    statuses: list[str]


@dataclass
class InferenceBatch:
    atom_representations: list[Optional[np.ndarray]]


class TimingBackend(Protocol):
    def initialize(self) -> None: ...

    def preprocess(self, smiles: Sequence[str], conformer_seed: int) -> PreparedBatch: ...

    def infer(self, prepared: PreparedBatch) -> InferenceBatch: ...

    def metadata(self) -> dict: ...


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: str | Path) -> dict:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": file_sha256(resolved),
    }


def _normalized_versions_equal(installed: str, expected: str) -> bool:
    try:
        return Version(installed) == Version(expected)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"Cannot compare package versions {installed!r} and {expected!r}."
        ) from exc


def _canonicalize_fragment(smiles: str) -> Optional[str]:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def collect_required_smiles(candidate_jsonl: str | Path) -> RequiredSmilesInventory:
    """Collect exact canonical product/fragment keys required by the pool."""
    required: set[str] = set()
    audit = {
        "physical_lines": 0,
        "candidate_records": 0,
        "malformed_json_lines": 0,
        "empty_fields": 0,
        "invalid_molecular_smiles": 0,
    }
    with open(candidate_jsonl, encoding="utf-8") as handle:
        for raw in handle:
            audit["physical_lines"] += 1
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                audit["malformed_json_lines"] += 1
                continue
            audit["candidate_records"] += 1
            values = [str(record.get("product", ""))]
            values.extend(str(record.get("reactant", "")).split("."))
            for value in values:
                value = value.strip()
                if not value:
                    audit["empty_fields"] += 1
                    continue
                canonical = _canonicalize_fragment(value)
                if canonical is None:
                    audit["invalid_molecular_smiles"] += 1
                    continue
                required.add(canonical)
    smiles = sorted(required)
    audit["required_key_count"] = len(smiles)
    audit["required_keys_sha256"] = hashlib.sha256(
        "\n".join(smiles).encode("utf-8")
    ).hexdigest()
    return RequiredSmilesInventory(smiles=smiles, audit=audit)


def collect_official_feature_workload(
    source_csv: str | Path,
    metadata_csv: str | Path,
    candidate_jsonl: str | Path,
    train_split: str = "train",
    eval_splits: Sequence[str] = ("valid", "test"),
    exclude_cross_split_train_products: bool = True,
) -> RequiredSmilesInventory:
    """Reconstruct the exact molecules eligible for official feature caches.

    This mirrors the product eligibility rules in ``rerank.study_data`` but
    stops before feature extraction. Training products must be covered, have a
    negative candidate, and (by default) not overlap another official split.
    Evaluation products enter only through covered reactions. Every retained
    product and candidate is then split into the canonical molecular fragments
    consumed by the atom-embedding cache.
    """
    from rerank.study_data import (
        candidate_rank,
        canonicalize_smiles,
        load_candidate_pools,
        load_reactions,
    )

    reactions = load_reactions(source_csv, metadata_csv)
    pools, pool_audit = load_candidate_pools(candidate_jsonl)
    non_train_products = {
        reaction.product_key
        for reaction in reactions
        if reaction.source_split != train_split
    }
    train_ground_truths: dict[str, set[str]] = {}
    train_overlap_reactions_excluded = 0
    for reaction in reactions:
        if reaction.source_split != train_split:
            continue
        if (
            exclude_cross_split_train_products
            and reaction.product_key in non_train_products
        ):
            train_overlap_reactions_excluded += 1
            continue
        train_ground_truths.setdefault(reaction.product_key, set()).add(
            reaction.ground_truth_key
        )

    eligible_train_products: set[str] = set()
    train_products_uncovered = 0
    train_products_without_negative = 0
    for product_key, positive_keys in train_ground_truths.items():
        pool = pools.get(product_key, [])
        positive_indices = {
            index
            for index, candidate in enumerate(pool)
            if candidate["canonical_smiles"] in positive_keys
        }
        if not positive_indices:
            train_products_uncovered += 1
            continue
        if len(positive_indices) == len(pool):
            train_products_without_negative += 1
            continue
        eligible_train_products.add(product_key)

    eval_audit: dict[str, dict] = {}
    eligible_eval_products: set[str] = set()
    for split in eval_splits:
        split_reactions = [
            reaction for reaction in reactions if reaction.source_split == split
        ]
        covered = [
            reaction
            for reaction in split_reactions
            if candidate_rank(reaction, pools.get(reaction.product_key, [])) > 0
        ]
        covered_products = {reaction.product_key for reaction in covered}
        eligible_eval_products.update(covered_products)
        eval_audit[str(split)] = {
            "reactions_total": len(split_reactions),
            "reactions_covered": len(covered),
            "reactions_uncovered": len(split_reactions) - len(covered),
            "unique_products_covered": len(covered_products),
        }

    required: set[str] = set()
    invalid_retained_fragments = 0

    def add_molecular_fragments(smiles: str) -> None:
        nonlocal invalid_retained_fragments
        for fragment in str(smiles).split("."):
            fragment = fragment.strip()
            if not fragment:
                continue
            canonical = canonicalize_smiles(fragment)
            if canonical is None:
                invalid_retained_fragments += 1
            else:
                required.add(canonical)

    eligible_products = eligible_train_products | eligible_eval_products
    for product_key in sorted(eligible_products):
        add_molecular_fragments(product_key)
        for candidate in pools.get(product_key, []):
            add_molecular_fragments(str(candidate["smiles"]))

    smiles = sorted(required)
    audit = {
        "inventory_method": "official_feature_cache_eligibility_v1",
        "source_csv": file_fingerprint(source_csv),
        "metadata_csv": file_fingerprint(metadata_csv),
        "candidate_jsonl": file_fingerprint(candidate_jsonl),
        "candidate_pool_audit": pool_audit,
        "train_split": train_split,
        "eval_splits": list(eval_splits),
        "exclude_cross_split_train_products": exclude_cross_split_train_products,
        "eligible_train_products": len(eligible_train_products),
        "train_overlap_reactions_excluded": train_overlap_reactions_excluded,
        "train_products_uncovered": train_products_uncovered,
        "train_products_without_negative": train_products_without_negative,
        "evaluation": eval_audit,
        "eligible_unique_products": len(eligible_products),
        "invalid_retained_fragments": invalid_retained_fragments,
        "required_key_count": len(smiles),
        "required_keys_sha256": hashlib.sha256(
            "\n".join(smiles).encode("utf-8")
        ).hexdigest(),
    }
    return RequiredSmilesInventory(smiles=smiles, audit=audit)


def heavy_atom_count(smiles: str) -> int:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(str(smiles))
    return 0 if molecule is None else int(molecule.GetNumHeavyAtoms())


def heavy_atom_stratum(count: int) -> str:
    for name, lower, upper in HEAVY_ATOM_STRATA:
        if count >= lower and (upper is None or count <= upper):
            return name
    raise ValueError(
        f"Heavy-atom count {count} is outside the prespecified 1-128 workload."
    )


def _stable_sample_key(smiles: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\0{smiles}".encode("utf-8")).digest()


def stratified_sample(
    required_smiles: Sequence[str],
    sample_per_stratum: int,
    sample_seed: int,
    atom_counter: Callable[[str], int] = heavy_atom_count,
) -> tuple[list[dict], dict]:
    if sample_per_stratum < 1:
        raise ValueError("sample_per_stratum must be positive")
    population: dict[str, list[tuple[str, int]]] = {
        name: [] for name, _, _ in HEAVY_ATOM_STRATA
    }
    for smiles in sorted(set(required_smiles)):
        count = int(atom_counter(smiles))
        population[heavy_atom_stratum(count)].append((smiles, count))

    selected: list[dict] = []
    audit: dict[str, dict] = {}
    for name, _, _ in HEAVY_ATOM_STRATA:
        members = sorted(
            population[name],
            key=lambda item: (_stable_sample_key(item[0], sample_seed), item[0]),
        )
        sample = members if name == TAIL_CENSUS_STRATUM else members[:sample_per_stratum]
        selected.extend(
            {"smiles": smiles, "heavy_atoms": count, "stratum": name}
            for smiles, count in sample
        )
        audit[name] = {
            "population_count": len(members),
            "sample_count": len(sample),
            "population_heavy_atom_min": min((item[1] for item in members), default=None),
            "population_heavy_atom_max": max((item[1] for item in members), default=None),
            "sampling_mode": (
                "census" if name == TAIL_CENSUS_STRATUM else "fixed_size"
            ),
        }
    return selected, audit


def select_disjoint_warmup(
    required_smiles: Sequence[str],
    timed_sample: Sequence[dict],
    warmup_count: int,
    sample_seed: int,
    atom_counter: Callable[[str], int] = heavy_atom_count,
) -> list[dict]:
    """Select balanced warm-ups from the two central strata, disjointly."""
    if warmup_count < 0:
        raise ValueError("warmup_count cannot be negative")
    if warmup_count == 0:
        return []
    timed_keys = {item["smiles"] for item in timed_sample}
    available: dict[str, list[dict]] = {name: [] for name in WARMUP_STRATA}
    for smiles in sorted(set(required_smiles)):
        if smiles in timed_keys:
            continue
        count = int(atom_counter(smiles))
        stratum = heavy_atom_stratum(count)
        if stratum in available:
            available[stratum].append(
                {"smiles": smiles, "heavy_atoms": count, "stratum": stratum}
            )
    for stratum in WARMUP_STRATA:
        available[stratum].sort(
            key=lambda item: (
                _stable_sample_key(item["smiles"], sample_seed + 1),
                item["smiles"],
            )
        )

    selected: list[dict] = []
    offsets = {name: 0 for name in WARMUP_STRATA}
    while len(selected) < warmup_count:
        progress = False
        for stratum in WARMUP_STRATA:
            offset = offsets[stratum]
            if offset < len(available[stratum]) and len(selected) < warmup_count:
                selected.append(available[stratum][offset])
                offsets[stratum] += 1
                progress = True
        if not progress:
            raise ValueError(
                "Insufficient unused molecules in the 11-20 and 21-30 strata "
                "for a disjoint warm-up."
            )
    return selected


def serialize_and_discard(
    representations: Sequence[Optional[np.ndarray]],
) -> dict:
    """Serialize representations in memory, retain only scalar diagnostics."""
    arrays = 0
    elements = 0
    serialized_bytes = 0
    nonfinite_values = 0
    digest = hashlib.sha256()
    for representation in representations:
        if representation is None:
            continue
        array = np.asarray(representation, dtype=np.float32)
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payload = buffer.getvalue()
        digest.update(payload)
        arrays += 1
        elements += int(array.size)
        serialized_bytes += len(payload)
        nonfinite_values += int((~np.isfinite(array)).sum())
        del payload, buffer, array
    return {
        "arrays": arrays,
        "elements": elements,
        "serialized_bytes": serialized_bytes,
        "nonfinite_values": nonfinite_values,
        "discard_digest_sha256": digest.hexdigest(),
    }


def _system_metadata() -> dict:
    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        process = psutil.Process()
        metadata.update(
            {
                "physical_cpu_count": psutil.cpu_count(logical=False),
                "total_ram_bytes": int(memory.total),
                "available_ram_bytes_at_start": int(memory.available),
                "process_rss_bytes_at_start": int(process.memory_info().rss),
            }
        )
    except Exception:
        metadata.update(
            {
                "physical_cpu_count": None,
                "total_ram_bytes": None,
                "available_ram_bytes_at_start": None,
                "process_rss_bytes_at_start": None,
            }
        )
    return metadata


def _process_rss_bytes() -> Optional[int]:
    """Return current process RSS when psutil is available."""
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def _updated_ram_metadata(metadata: dict) -> None:
    try:
        import psutil

        memory = psutil.virtual_memory()
        process = psutil.Process()
        metadata["available_ram_bytes_at_end"] = int(memory.available)
        metadata["process_rss_bytes_at_end"] = int(process.memory_info().rss)
    except Exception:
        metadata["available_ram_bytes_at_end"] = None
        metadata["process_rss_bytes_at_end"] = None


def _timed_batch(
    backend: TimingBackend,
    smiles: Sequence[str],
    conformer_seed: int,
    clock: Callable[[], float],
    serializer: Callable[[Sequence[Optional[np.ndarray]]], dict],
    rss_reader: Callable[[], Optional[int]],
) -> dict:
    rss_before_preprocess = rss_reader()
    start = clock()
    prepared = backend.preprocess(smiles, conformer_seed)
    after_preprocess = clock()
    rss_after_preprocess = rss_reader()
    if len(prepared.payloads) != len(smiles) or len(prepared.statuses) != len(smiles):
        raise ValueError("Backend preprocessing output is not aligned to its input batch.")

    inferred = backend.infer(prepared)
    after_inference = clock()
    rss_after_inference = rss_reader()
    if len(inferred.atom_representations) != len(smiles):
        raise ValueError("Backend inference output is not aligned to its input batch.")

    scalar_summary = serializer(inferred.atom_representations)
    after_serialization = clock()
    rss_after_serialization = rss_reader()
    failures = sum(
        status == "failure" or representation is None
        for status, representation in zip(
            prepared.statuses, inferred.atom_representations
        )
    )
    fallbacks = sum(status.startswith("fallback") for status in prepared.statuses)
    observed_rss = [
        value
        for value in (
            rss_before_preprocess,
            rss_after_preprocess,
            rss_after_inference,
            rss_after_serialization,
        )
        if value is not None
    ]
    return {
        "attempted": len(smiles),
        "successes": len(smiles) - failures,
        "failures": failures,
        "fallbacks": fallbacks,
        "preprocess_seconds": after_preprocess - start,
        "inference_seconds": after_inference - after_preprocess,
        "serialization_discard_seconds": after_serialization - after_inference,
        "serialized_bytes": int(scalar_summary.get("serialized_bytes", 0)),
        "serialized_arrays": int(scalar_summary.get("arrays", 0)),
        "nonfinite_values": int(scalar_summary.get("nonfinite_values", 0)),
        "rss_before_preprocess_bytes": rss_before_preprocess,
        "rss_after_preprocess_bytes": rss_after_preprocess,
        "rss_after_inference_bytes": rss_after_inference,
        "rss_after_serialization_bytes": rss_after_serialization,
        "rss_observed_peak_bytes": max(observed_rss, default=None),
    }


def _chunks(items: Sequence[dict], size: int) -> list[list[dict]]:
    return [list(items[index : index + size]) for index in range(0, len(items), size)]


def _interleaved_stratum_batches(
    selected: Sequence[dict], batch_size: int
) -> list[tuple[str, int, list[dict]]]:
    """Round-robin batches in low/high stratum order to limit drift bias."""
    names = [name for name, _, _ in HEAVY_ATOM_STRATA]
    interleave_order: list[str] = []
    left, right = 0, len(names) - 1
    while left <= right:
        interleave_order.append(names[left])
        if left != right:
            interleave_order.append(names[right])
        left += 1
        right -= 1
    by_stratum = {
        name: _chunks(
            [item for item in selected if item["stratum"] == name], batch_size
        )
        for name in names
    }
    schedule: list[tuple[str, int, list[dict]]] = []
    max_batches = max((len(value) for value in by_stratum.values()), default=0)
    for batch_index in range(max_batches):
        for name in interleave_order:
            if batch_index < len(by_stratum[name]):
                schedule.append((name, batch_index, by_stratum[name][batch_index]))
    return schedule


def _aggregate_strata(
    sample_audit: dict,
    batch_rows: Sequence[dict],
) -> tuple[list[dict], dict]:
    strata_rows = []
    total_estimate = 0.0
    total_lower = 0.0
    total_upper = 0.0
    for name, _, _ in HEAVY_ATOM_STRATA:
        rows = [row for row in batch_rows if row["stratum"] == name]
        audit = sample_audit[name]
        sample_count = sum(row["attempted"] for row in rows)
        stage_names = (
            "preprocess_seconds",
            "inference_seconds",
            "serialization_discard_seconds",
        )
        stage_totals = {
            stage: sum(float(row[stage]) for row in rows) for stage in stage_names
        }
        total_seconds = sum(stage_totals.values())
        per_key_batch_times = [
            sum(float(row[stage]) for stage in stage_names) / row["attempted"]
            for row in rows
            if row["attempted"]
        ]
        population_count = audit["population_count"]
        if sample_count:
            mean_per_key = total_seconds / sample_count
            lower_per_key = min(per_key_batch_times)
            upper_per_key = max(per_key_batch_times) * 1.25
        else:
            mean_per_key = lower_per_key = upper_per_key = 0.0
        estimate = population_count * mean_per_key
        lower = population_count * lower_per_key
        upper = population_count * upper_per_key
        total_estimate += estimate
        total_lower += lower
        total_upper += upper
        rss_keys = (
            "rss_after_preprocess_bytes",
            "rss_after_inference_bytes",
            "rss_after_serialization_bytes",
            "rss_observed_peak_bytes",
        )
        rss_peaks = {
            f"{key.removesuffix('_bytes')}_peak_bytes": max(
                (row[key] for row in rows if row.get(key) is not None),
                default=None,
            )
            for key in rss_keys
        }
        strata_rows.append(
            {
                "stratum": name,
                **audit,
                "batch_count": len(rows),
                "attempted": sample_count,
                "successes": sum(row["successes"] for row in rows),
                "failures": sum(row["failures"] for row in rows),
                "fallbacks": sum(row["fallbacks"] for row in rows),
                **stage_totals,
                "serialized_bytes": sum(row["serialized_bytes"] for row in rows),
                "nonfinite_values": sum(row["nonfinite_values"] for row in rows),
                "mean_seconds_per_key": mean_per_key,
                "extrapolated_seconds": estimate,
                "conservative_lower_seconds": lower,
                "conservative_upper_seconds": upper,
                **rss_peaks,
            }
        )
    extrapolation = {
        "point_seconds": total_estimate,
        "conservative_lower_seconds": total_lower,
        "conservative_upper_seconds": total_upper,
        "primary_estimator": "sum_h (N_h / n_h) * sum_i t_hi",
        "primary_estimator_name": "post_stratified_expansion",
        "interval_method": (
            "sum over strata of observed batch minimum to 1.25 times observed "
            "batch maximum per-key time; engineering range, not a confidence interval"
        ),
    }
    return strata_rows, extrapolation


def run_timing_pilot(
    required_smiles: Sequence[str],
    required_inventory_audit: dict,
    backend: TimingBackend,
    batch_size: int,
    sample_per_stratum: int,
    warmup_count: int = 1,
    conformer_seed: int = C1_CONFORMER_SEED,
    sample_seed: int = C1_CONFORMER_SEED,
    atom_counter: Callable[[str], int] = heavy_atom_count,
    clock: Callable[[], float] = time.perf_counter,
    serializer: Callable[[Sequence[Optional[np.ndarray]]], dict] = serialize_and_discard,
    rss_reader: Optional[Callable[[], Optional[int]]] = None,
) -> dict:
    if conformer_seed != C1_CONFORMER_SEED:
        raise ValueError("This pilot is restricted to C1, conformer seed 42.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if warmup_count < 0:
        raise ValueError("warmup_count cannot be negative")

    selected, sample_audit = stratified_sample(
        required_smiles,
        sample_per_stratum,
        sample_seed,
        atom_counter,
    )
    if not selected:
        raise ValueError("No molecular SMILES were available for timing.")
    required_key_count = len(set(required_smiles))
    if required_inventory_audit.get("required_key_count") != required_key_count:
        raise ValueError("Required-key audit count does not match the supplied key set.")
    warmup_items = select_disjoint_warmup(
        required_smiles,
        selected,
        warmup_count,
        sample_seed,
        atom_counter,
    )

    system = _system_metadata()
    rss_reader = rss_reader or _process_rss_bytes
    rss_before_initialization = rss_reader()
    initialize_start = clock()
    backend.initialize()
    initialize_seconds = clock() - initialize_start
    rss_after_initialization = rss_reader()

    actual_warmup_count = len(warmup_items)
    warmup_seconds = 0.0
    warmup_rows: list[dict] = []
    if actual_warmup_count:
        start = clock()
        for batch in _chunks(warmup_items, batch_size):
            warmup_rows.append(
                _timed_batch(
                    backend,
                    [item["smiles"] for item in batch],
                    conformer_seed,
                    clock,
                    serializer,
                    rss_reader,
                )
            )
        warmup_seconds = clock() - start

    batch_rows: list[dict] = []
    batch_schedule = _interleaved_stratum_batches(selected, batch_size)
    for schedule_index, (stratum_name, stratum_batch_index, batch) in enumerate(
        batch_schedule
    ):
        row = _timed_batch(
            backend,
            [item["smiles"] for item in batch],
            conformer_seed,
            clock,
            serializer,
            rss_reader,
        )
        row.update(
            {
                "stratum": stratum_name,
                "batch_index": schedule_index,
                "stratum_batch_index": stratum_batch_index,
            }
        )
        batch_rows.append(row)

    strata_rows, extrapolation = _aggregate_strata(sample_audit, batch_rows)
    stage_totals = {
        stage: sum(float(row[stage]) for row in batch_rows)
        for stage in (
            "preprocess_seconds",
            "inference_seconds",
            "serialization_discard_seconds",
        )
    }
    steady_seconds = sum(stage_totals.values())
    attempts = sum(row["attempted"] for row in batch_rows)
    successes = sum(row["successes"] for row in batch_rows)
    observed_rows = warmup_rows + batch_rows
    rss_stage_peaks = {
        "initialization_before_bytes": rss_before_initialization,
        "initialization_after_bytes": rss_after_initialization,
        "preprocess_peak_bytes": max(
            (
                row["rss_after_preprocess_bytes"]
                for row in observed_rows
                if row.get("rss_after_preprocess_bytes") is not None
            ),
            default=None,
        ),
        "inference_peak_bytes": max(
            (
                row["rss_after_inference_bytes"]
                for row in observed_rows
                if row.get("rss_after_inference_bytes") is not None
            ),
            default=None,
        ),
        "serialization_peak_bytes": max(
            (
                row["rss_after_serialization_bytes"]
                for row in observed_rows
                if row.get("rss_after_serialization_bytes") is not None
            ),
            default=None,
        ),
    }
    rss_observations = [
        value
        for value in rss_stage_peaks.values()
        if value is not None
    ]
    system["process_rss_stage_peaks_bytes"] = rss_stage_peaks
    system["process_rss_observed_peak_bytes"] = max(rss_observations, default=None)
    _updated_ram_metadata(system)
    sampled_smiles = sorted(item["smiles"] for item in selected)
    return {
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "report_kind": "non_scientific_cpu_timing_pilot",
        "protocol_id": "B-C1-TIMING-PILOT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "conformer_label": "C1",
        "conformer_seed": conformer_seed,
        "sample_seed": sample_seed,
        "batch_size": batch_size,
        "warmup": {
            "count": actual_warmup_count,
            "batch_count": len(warmup_rows),
            "seconds": warmup_seconds,
            "excluded_from_steady_state": True,
            "disjoint_from_timed_sample": not bool(
                {item["smiles"] for item in warmup_items}.intersection(sampled_smiles)
            ),
            "stratum_counts": {
                name: sum(item["stratum"] == name for item in warmup_items)
                for name in WARMUP_STRATA
            },
            "keys_sha256": hashlib.sha256(
                "\n".join(sorted(item["smiles"] for item in warmup_items)).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "rss_observed_peak_bytes": max(
                (
                    row["rss_observed_peak_bytes"]
                    for row in warmup_rows
                    if row.get("rss_observed_peak_bytes") is not None
                ),
                default=None,
            ),
        },
        "initialization": {
            "seconds": initialize_seconds,
            "excluded_from_steady_state": True,
            "rss_before_bytes": rss_before_initialization,
            "rss_after_bytes": rss_after_initialization,
            "rss_observed_peak_bytes": max(
                (
                    value
                    for value in (rss_before_initialization, rss_after_initialization)
                    if value is not None
                ),
                default=None,
            ),
        },
        "required_inventory": dict(required_inventory_audit),
        "required_key_count_exact": required_key_count,
        "sample": {
            "count": len(selected),
            "keys_sha256": hashlib.sha256(
                "\n".join(sampled_smiles).encode("utf-8")
            ).hexdigest(),
            "strata": strata_rows,
            "tail_sampling": "census",
            "batch_schedule": {
                "method": "round_robin_low_high_strata",
                "batch_count": len(batch_rows),
                "stratum_order": [row["stratum"] for row in batch_rows],
            },
        },
        "steady_state": {
            "attempted": attempts,
            "successes": successes,
            "failures": sum(row["failures"] for row in batch_rows),
            "fallbacks": sum(row["fallbacks"] for row in batch_rows),
            **stage_totals,
            "total_seconds": steady_seconds,
            "attempted_throughput_keys_per_second": (
                attempts / steady_seconds if steady_seconds else None
            ),
            "successful_throughput_keys_per_second": (
                successes / steady_seconds if steady_seconds else None
            ),
            "serialized_bytes_in_memory": sum(
                row["serialized_bytes"] for row in batch_rows
            ),
            "nonfinite_values": sum(row["nonfinite_values"] for row in batch_rows),
            "rss_stage_peaks_bytes": {
                "preprocess": max(
                    (
                        row["rss_after_preprocess_bytes"]
                        for row in batch_rows
                        if row.get("rss_after_preprocess_bytes") is not None
                    ),
                    default=None,
                ),
                "inference": max(
                    (
                        row["rss_after_inference_bytes"]
                        for row in batch_rows
                        if row.get("rss_after_inference_bytes") is not None
                    ),
                    default=None,
                ),
                "serialization_discard": max(
                    (
                        row["rss_after_serialization_bytes"]
                        for row in batch_rows
                        if row.get("rss_after_serialization_bytes") is not None
                    ),
                    default=None,
                ),
            },
            "rss_observed_peak_bytes": max(
                (
                    row["rss_observed_peak_bytes"]
                    for row in batch_rows
                    if row.get("rss_observed_peak_bytes") is not None
                ),
                default=None,
            ),
        },
        "extrapolation_for_exact_required_key_count": extrapolation,
        "system": system,
        "backend": backend.metadata(),
        "embedding_artifacts_written": False,
    }


def write_reports(report: dict, json_path: str | Path, csv_path: str | Path) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    rows = report["sample"]["strata"]
    fieldnames = list(rows[0]) if rows else ["stratum"]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class UniMolCpuTimingBackend:
    """Lazy adapter around the pinned Uni-Mol CPU representation path."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        batch_size: int,
        threads: int,
        dictionary_path: Optional[str | Path] = None,
        expected_unimol_version: str = PINNED_UNIMOL_VERSION,
        expected_rdkit_version: str = PINNED_RDKIT_VERSION,
        expected_checkpoint_sha256: str = PINNED_CHECKPOINT_SHA256,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.dictionary_path = (
            self.checkpoint_path.parent / REQUIRED_DICTIONARY_BASENAME
            if dictionary_path is None
            else Path(dictionary_path)
        )
        self.batch_size = batch_size
        self.threads = threads
        self.expected_unimol_version = expected_unimol_version
        self.expected_rdkit_version = expected_rdkit_version
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self._initialized = False
        self._metadata: dict = {}
        self._conformer = None
        self._repr = None
        self._trainer = None
        self._dataset_class = None

    def initialize(self) -> None:
        if self._initialized:
            return
        installed_version = importlib.metadata.version("unimol-tools")
        if not _normalized_versions_equal(
            installed_version, self.expected_unimol_version
        ):
            raise RuntimeError(
                f"Expected unimol-tools {self.expected_unimol_version}, got {installed_version}."
            )
        rdkit_version = importlib.metadata.version("rdkit")
        if not _normalized_versions_equal(rdkit_version, self.expected_rdkit_version):
            raise RuntimeError(
                f"Expected RDKit {self.expected_rdkit_version}, got {rdkit_version}."
            )

        checkpoint_path = self.checkpoint_path.expanduser().resolve()
        dictionary_path = self.dictionary_path.expanduser().resolve()
        if checkpoint_path.name != REQUIRED_CHECKPOINT_BASENAME:
            raise RuntimeError(
                "Pinned Uni-Mol checkpoint must be named "
                f"{REQUIRED_CHECKPOINT_BASENAME}."
            )
        if dictionary_path.name != REQUIRED_DICTIONARY_BASENAME:
            raise RuntimeError(
                f"Pinned Uni-Mol dictionary must be named {REQUIRED_DICTIONARY_BASENAME}."
            )
        if checkpoint_path.parent != dictionary_path.parent:
            raise RuntimeError(
                "Pinned checkpoint and dictionary must be in the same weight directory."
            )
        if not checkpoint_path.is_file() or not dictionary_path.is_file():
            raise FileNotFoundError(
                "Pinned Uni-Mol weight directory must contain both "
                f"{REQUIRED_CHECKPOINT_BASENAME} and {REQUIRED_DICTIONARY_BASENAME}."
            )
        checkpoint = file_fingerprint(checkpoint_path)
        if checkpoint["sha256"] != self.expected_checkpoint_sha256:
            raise RuntimeError("Uni-Mol checkpoint SHA-256 does not match the pinned value.")
        dictionary = file_fingerprint(dictionary_path)

        weight_dir = checkpoint_path.parent
        loaded_unimol_modules = [
            name
            for name in sys.modules
            if name == "unimol_tools" or name.startswith("unimol_tools.")
        ]
        if loaded_unimol_modules:
            loaded_weights = sys.modules.get("unimol_tools.weights.weighthub")
            if loaded_weights is None:
                loaded_weights = sys.modules.get("unimol_tools.weights")
            loaded_weight_dir = getattr(loaded_weights, "WEIGHT_DIR", None)
            if (
                loaded_weight_dir is None
                or Path(loaded_weight_dir).expanduser().resolve() != weight_dir
            ):
                raise RuntimeError(
                    "Uni-Mol was imported before the pinned UNIMOL_WEIGHT_DIR "
                    "could be established; start a fresh process."
                )
        os.environ["UNIMOL_WEIGHT_DIR"] = str(weight_dir)

        # Optional heavy imports stay lazy and occur only after the pinned
        # weight directory is visible to unimol_tools.weights.weighthub.
        import rdkit
        import torch
        import unimol_tools
        from unimol_tools import UniMolRepr
        from unimol_tools.data.conformer import ConformerGen
        from unimol_tools.predictor import MolDataset
        from unimol_tools.tasks import Trainer

        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        np.random.seed(C1_CONFORMER_SEED)
        torch.manual_seed(C1_CONFORMER_SEED)

        self._repr = UniMolRepr(
            data_type="molecule",
            batch_size=self.batch_size,
            remove_hs=True,
            model_name="unimolv1",
            model_size="84m",
            use_cuda=False,
            use_ddp=False,
            use_gpu="all",
        )
        conformer_kwargs = {
            "seed": C1_CONFORMER_SEED,
            "max_atoms": 256,
            "data_type": "molecule",
            "method": "rdkit_random",
            "mode": "fast",
            "remove_hs": True,
            "multi_process": False,
        }
        self._conformer = ConformerGen(**conformer_kwargs)
        self._trainer = Trainer(task="repr", **self._repr.params)
        self._dataset_class = MolDataset

        package_source = Path(inspect.getfile(unimol_tools)).resolve()
        distribution = importlib.metadata.distribution("unimol-tools")
        record_text = distribution.read_text("RECORD") or ""
        rdkit_source = Path(inspect.getfile(rdkit)).resolve()
        rdkit_distribution = importlib.metadata.distribution("rdkit")
        rdkit_record_text = rdkit_distribution.read_text("RECORD") or ""
        self._metadata = {
            "name": "UniMolCpuTimingBackend",
            "device": "cpu",
            "threads": self.threads,
            "batch_size": self.batch_size,
            "unimol_tools": {
                "version": installed_version,
                "package_init": file_fingerprint(package_source),
                "distribution_record_sha256": hashlib.sha256(
                    record_text.encode("utf-8")
                ).hexdigest(),
            },
            "rdkit": {
                "version": rdkit_version,
                "package_init": file_fingerprint(rdkit_source),
                "distribution_record_sha256": hashlib.sha256(
                    rdkit_record_text.encode("utf-8")
                ).hexdigest(),
            },
            "checkpoint": checkpoint,
            "dictionary": dictionary,
            "unimol_weight_dir": str(weight_dir),
            "torch": {
                "version": torch.__version__,
                "num_threads": torch.get_num_threads(),
                "num_interop_threads": torch.get_num_interop_threads(),
            },
            "conformer": {
                "method": "rdkit_random",
                "mode": "fast",
                "remove_hs": True,
                "max_atoms": 256,
                "seed": C1_CONFORMER_SEED,
            },
            "rng": {
                "numpy_seed": C1_CONFORMER_SEED,
                "torch_seed": C1_CONFORMER_SEED,
                "rdkit_embed_seed": C1_CONFORMER_SEED,
            },
        }
        self._initialized = True

    def preprocess(self, smiles: Sequence[str], conformer_seed: int) -> PreparedBatch:
        if not self._initialized or self._conformer is None:
            raise RuntimeError("Backend must be initialized before preprocessing.")
        if conformer_seed != C1_CONFORMER_SEED:
            raise ValueError("Backend is pinned to conformer seed C1=42.")
        payloads: list[Any] = []
        statuses: list[str] = []
        for value in smiles:
            try:
                feature = self._conformer.single_process(value)
                coordinates = np.asarray(feature["src_coord"])
                if np.all(coordinates == 0.0):
                    status = "fallback_zero"
                elif np.all(coordinates[:, 2] == 0.0):
                    status = "fallback_2d"
                else:
                    status = "ok"
                payloads.append(feature)
                statuses.append(status)
            except Exception:
                payloads.append(None)
                statuses.append("failure")
        return PreparedBatch(payloads=payloads, statuses=statuses)

    def infer(self, prepared: PreparedBatch) -> InferenceBatch:
        if not self._initialized:
            raise RuntimeError("Backend must be initialized before inference.")
        valid_indices = [
            index for index, payload in enumerate(prepared.payloads) if payload is not None
        ]
        aligned: list[Optional[np.ndarray]] = [None] * len(prepared.payloads)
        if not valid_indices:
            return InferenceBatch(aligned)
        valid_payloads = [prepared.payloads[index] for index in valid_indices]
        try:
            dataset = self._dataset_class(valid_payloads)
            raw = self._trainer.inference(
                self._repr.model,
                dataset,
                return_repr=True,
                return_atomic_reprs=True,
                feature_name=None,
            )
            if not isinstance(raw, dict):
                raise ValueError("Uni-Mol inference did not return a representation mapping.")
            atomic = raw.get("atomic_reprs", raw.get("atom_repr"))
            if not isinstance(atomic, (list, tuple)) or len(atomic) != len(valid_indices):
                raise ValueError("Uni-Mol atomic representations are not batch-aligned.")
            for index, representation in zip(valid_indices, atomic):
                array = np.asarray(representation, dtype=np.float32)
                if array.ndim != 2:
                    raise ValueError(
                        "Uni-Mol atomic representation for batch index "
                        f"{index} has shape {array.shape}, expected two dimensions."
                    )
                aligned[index] = array
        except Exception as exc:
            raise RuntimeError(
                "Uni-Mol inference failed for batch of "
                f"{len(valid_indices)} valid molecules: {type(exc).__name__}: {exc}"
            ) from exc
        return InferenceBatch(aligned)

    def metadata(self) -> dict:
        if not self._initialized:
            raise RuntimeError("Backend metadata is unavailable before initialization.")
        return dict(self._metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument(
        "--metadata-csv", default="data/uspto_reaction_metadata.csv"
    )
    parser.add_argument("--candidate-jsonl", default="outputs/rerank_dataset.jsonl")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dictionary")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--sample-per-stratum", type=int, default=DEFAULT_SAMPLE_PER_STRATUM
    )
    parser.add_argument("--warmup-count", type=int, default=DEFAULT_WARMUP_COUNT)
    parser.add_argument(
        "--output-json",
        default="outputs/revision_timing/c1_cpu_timing_pilot.json",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/revision_timing/c1_cpu_timing_pilot_strata.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    inventory = collect_official_feature_workload(
        source_csv=args.source_csv,
        metadata_csv=args.metadata_csv,
        candidate_jsonl=args.candidate_jsonl,
    )
    backend = UniMolCpuTimingBackend(
        checkpoint_path=args.checkpoint,
        dictionary_path=args.dictionary,
        batch_size=args.batch_size,
        threads=args.threads,
    )
    report = run_timing_pilot(
        required_smiles=inventory.smiles,
        required_inventory_audit=inventory.audit,
        backend=backend,
        batch_size=args.batch_size,
        sample_per_stratum=args.sample_per_stratum,
        warmup_count=args.warmup_count,
    )
    write_reports(report, args.output_json, args.output_csv)
    print(json.dumps({
        "json_report": str(Path(args.output_json).resolve()),
        "csv_report": str(Path(args.output_csv).resolve()),
        "required_key_count": report["required_key_count_exact"],
        "sample_count": report["sample"]["count"],
        "embedding_artifacts_written": False,
    }, indent=2))


if __name__ == "__main__":
    main()
