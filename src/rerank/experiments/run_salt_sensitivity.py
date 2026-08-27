#!/usr/bin/env python
"""Run the prespecified F3 salt-removal sensitivity behind a test lock."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem
from rdkit.Chem import SaltRemover

from rerank.data.build_conformer_sqlite import (
    EMBEDDING_DIM,
    SeededUniMolBackend,
    discover_weight_paths,
    validate_weight_paths,
)
from rerank.features import (
    _atom_set_similarity,
    _cosine,
    _heavy_atom_ratio,
    _morgan_similarity,
    _reaction_distance,
)
from rerank.revision_tuning import (
    MAX_EPOCHS,
    MIN_IMPROVEMENT,
    PATIENCE,
    PROTOCOL_ID,
    SEEDS,
    GridConfig,
    atomic_pickle_dump,
    config_fingerprint,
    file_fingerprint,
    file_sha256,
    load_selection_bundle,
    train_validation_trial,
    transform_selection_cache,
    validate_selection_bundle,
)
from rerank.experiments.run_tuned_revision import (
    FIXED_TUNING_CONFIG,
    _score_test_arm,
    environment_record,
    immutable_json_dump,
    require_clean_evaluation_result_dir,
    resolve_device,
    summarize_paired_metrics,
    validate_selection_freeze,
)


F3_PROTOCOL_ID = "f3-salt-removal-v1"
CONFORMER_SEED = 42
MAX_ATOMS = 256
# Frozen train/validation-only regeneration audit found ten ordinary rows with
# CPU batch-composition drift from 1.07e-5 to 4.70e-5.  No official-test input
# was opened.  Five times 1e-5 is the smallest rounded tolerance covering that
# fixed audit; paired deltas remain anchored to the exact frozen feature row.
CONTROL_TOLERANCE = 5e-5
FAILURE_CAUSES = (
    "SMILES parse",
    "salt-removal-empty",
    "atom-limit",
    "conformer/coordinate",
    "checkpoint/model",
    "cache-lookup failure",
)
_SALT_REMOVER = SaltRemover.SaltRemover()


@lru_cache(maxsize=500_000)
def canonical_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


@lru_cache(maxsize=500_000)
def salt_remove(smiles: str) -> tuple[str | None, str, int, int]:
    """Return cleaned SMILES, status, original fragments and surviving fragments."""
    value = str(smiles)
    molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        return None, "SMILES parse", 0, 0
    original = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    original_fragments = len(Chem.GetMolFrags(molecule))
    if original_fragments == 1:
        # With dontRemoveEverything=True, a single fragment cannot disappear.
        return original, "unchanged", 1, 1
    stripped = _SALT_REMOVER.StripMol(molecule, dontRemoveEverything=True)
    if stripped is None or stripped.GetNumAtoms() == 0:
        return None, "salt-removal-empty", original_fragments, 0
    cleaned = Chem.MolToSmiles(stripped, canonical=True, isomericSmiles=True)
    surviving = len(Chem.GetMolFrags(stripped))
    status = "changed" if cleaned != original else "unchanged"
    return cleaned, status, original_fragments, surviving


def fragment_keys(smiles: str) -> list[str]:
    result: list[str] = []
    for fragment in str(smiles).split("."):
        if not fragment:
            continue
        canonical = canonical_smiles(fragment)
        if canonical is None:
            raise ValueError(f"F3 cannot canonicalize fragment: {fragment!r}")
        result.append(canonical)
    return result


def _candidate_smiles(candidate: dict[str, Any]) -> str:
    return str(candidate.get("smiles", candidate.get("canonical_smiles", "")))


def _all_candidate_strings(sections: Iterable[tuple[str, list[dict]]]) -> list[str]:
    return sorted(
        {
            _candidate_smiles(candidate)
            for _, candidates in sections
            for candidate in candidates
        }
    )


def _salt_map(strings: list[str]) -> tuple[dict[str, str], dict[str, Any]]:
    mapping: dict[str, str] = {}
    statuses: dict[str, str] = {}
    removed_fragments = 0
    for index, value in enumerate(strings, start=1):
        cleaned, status, original_count, surviving_count = salt_remove(value)
        statuses[value] = status
        if cleaned is None:
            raise RuntimeError(f"F3 salt transform failed for {value!r}: {status}")
        mapping[value] = cleaned
        removed_fragments += max(0, original_count - surviving_count)
        if index % 5_000 == 0 or index == len(strings):
            print(f"F3 salt scan: {index}/{len(strings)} unique candidates", flush=True)
    counts = {name: sum(value == name for value in statuses.values()) for name in (
        "changed", "unchanged", "SMILES parse", "salt-removal-empty"
    )}
    return mapping, {
        "unique_candidate_strings": len(strings),
        "status_counts": counts,
        "removed_fragment_count_across_unique_candidates": removed_fragments,
    }


def _embedding_cause(smiles: str, status: str, embedding, error) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return "SMILES parse"
    if molecule.GetNumAtoms() > MAX_ATOMS:
        return "atom-limit"
    if status in {"fallback_zero", "fallback_2d", "failure_preprocess"}:
        return "conformer/coordinate"
    if embedding is None or error:
        return "checkpoint/model"
    return None


def _embed_required(
    required: list[str], backend: SeededUniMolBackend, batch_items: int
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    embeddings: dict[str, np.ndarray] = {}
    audit_rows: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    backend.initialize()
    started = time.perf_counter()
    with open(os.devnull, "w", encoding="utf-8") as sink:
        for start in range(0, len(required), batch_items):
            chunk = required[start : start + batch_items]
            with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                arrays, statuses, errors = backend.encode_batch(chunk)
            for smiles, array, status, error in zip(
                chunk, arrays, statuses, errors, strict=True
            ):
                status_counts[status] = status_counts.get(status, 0) + 1
                cause = _embedding_cause(smiles, status, array, error)
                if cause is not None:
                    audit_rows.append(
                        {
                            "scope": "F3 regenerated affected inventory",
                            "smiles": smiles,
                            "status": status,
                            "failure_cause": cause,
                            "error": error or "",
                        }
                    )
                if array is None:
                    raise RuntimeError(
                        f"F3 embedding generation failed for {smiles!r}: {status}; {error}"
                    )
                embeddings[smiles] = np.asarray(array, dtype=np.float32)
            completed = min(start + len(chunk), len(required))
            elapsed = max(time.perf_counter() - started, 1e-9)
            rate = completed / elapsed
            eta = (len(required) - completed) / max(rate, 1e-9)
            print(
                f"F3 embeddings: {completed}/{len(required)} | "
                f"{rate:.2f}/s | ETA {eta/60:.1f} min",
                flush=True,
            )
    return embeddings, audit_rows, {
        "required_unique_fragments": len(required),
        "status_counts": status_counts,
        "runtime_seconds": time.perf_counter() - started,
        "backend": backend.metadata(),
    }


def _load_or_embed_required(
    required: list[str],
    backend: SeededUniMolBackend,
    batch_items: int,
    cache_path: str | Path | None,
    checkpoint: Path,
    dictionary: Path,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    required_sha = "sha256:" + hashlib.sha256(
        "\n".join(required).encode("utf-8")
    ).hexdigest()
    expected = {
        "record_kind": "f3_affected_atom_embedding_cache",
        "protocol_id": F3_PROTOCOL_ID,
        "conformer_seed": CONFORMER_SEED,
        "required_smiles_sha256": required_sha,
        "checkpoint_sha256": "sha256:" + file_sha256(checkpoint),
        "dictionary_sha256": "sha256:" + file_sha256(dictionary),
    }
    path = Path(cache_path).resolve() if cache_path else None
    if path is not None and path.exists():
        with open(path, "rb") as handle:
            cached = pickle.load(handle)
        if any(cached.get(key) != value for key, value in expected.items()):
            raise ValueError(f"F3 affected embedding cache fingerprint mismatch: {path}")
        embeddings = cached["embeddings"]
        if set(embeddings) != set(required):
            raise ValueError("F3 affected embedding cache key set mismatch.")
        audit = dict(cached["embedding_audit"])
        audit["cache_reused"] = True
        audit["cache"] = file_fingerprint(path)
        return embeddings, list(cached["generated_non_ok"]), audit

    embeddings, generated_non_ok, audit = _embed_required(
        required, backend, batch_items
    )
    audit["cache_reused"] = False
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            **expected,
            "required_unique_fragments": len(required),
            "embeddings": embeddings,
            "generated_non_ok": generated_non_ok,
            "embedding_audit": audit,
        }
        atomic_pickle_dump(record, path)
        audit["cache"] = file_fingerprint(path)
    return embeddings, generated_non_ok, audit


def _embedding_for(smiles: str, embeddings: dict[str, np.ndarray]) -> np.ndarray:
    arrays = []
    for key in fragment_keys(smiles):
        if key not in embeddings:
            raise KeyError(f"F3 cache-lookup failure for {key!r}")
        arrays.append(embeddings[key])
    if not arrays:
        raise ValueError(f"F3 has no surviving fragments for {smiles!r}")
    return np.concatenate(arrays, axis=0)


def _feature_row(
    product: str,
    candidate: str,
    prior: float,
    embeddings: dict[str, np.ndarray],
) -> np.ndarray:
    product_embedding = _embedding_for(product, embeddings)
    candidate_embedding = _embedding_for(candidate, embeddings)
    product_mean = product_embedding.mean(axis=0)
    reaction_vector = product_mean - candidate_embedding.mean(axis=0)
    return np.asarray(
        [
            prior,
            _morgan_similarity(product, candidate),
            _atom_set_similarity(product_embedding, candidate_embedding),
            _reaction_distance(product_embedding, candidate_embedding),
            _cosine(product_mean, reaction_vector),
            float(len(fragment_keys(candidate))),
            _heavy_atom_ratio(product, candidate),
        ],
        dtype=np.float32,
    )


def _inventory_sections(container: dict[str, Any], selection: bool) -> list[tuple[str, list[dict]]]:
    if selection:
        sections = [
            (str(row["product_smiles"]), row["candidates"])
            for row in container["train_products"]
        ]
        sections.extend(
            (str(product), candidates)
            for product, candidates in container["validation_payload"]["eval_pwc"]
        )
        return sections
    return [(str(product), candidates) for product, candidates in container["eval_pwc"]]


def _required_fragments(
    sections: list[tuple[str, list[dict]]], mapping: dict[str, str]
) -> tuple[list[str], set[str]]:
    required: set[str] = set()
    affected_products: set[str] = set()
    for product, candidates in sections:
        for candidate in candidates:
            original = _candidate_smiles(candidate)
            cleaned = mapping[original]
            if cleaned == canonical_smiles(original):
                continue
            affected_products.add(product)
            required.update(fragment_keys(product))
            required.update(fragment_keys(original))
            required.update(fragment_keys(cleaned))
    return sorted(required), affected_products


def _transform_matrix(
    product: str,
    candidates: list[dict],
    matrix: np.ndarray,
    mapping: dict[str, str],
    embeddings: dict[str, np.ndarray],
    frozen_non_ok: dict[str, str],
    audit: dict[str, Any],
) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32).copy()
    if result.shape != (len(candidates), 7):
        raise ValueError("F3 requires candidate-aligned seven-column features.")
    for index, candidate in enumerate(candidates):
        original = _candidate_smiles(candidate)
        cleaned = mapping[original]
        if cleaned == canonical_smiles(original):
            continue
        prior = float(candidate["prior"])
        regenerated = _feature_row(product, original, prior, embeddings)
        difference = np.abs(regenerated - result[index])
        audit["control_pairs_recomputed"] += 1
        audit["maximum_control_absolute_difference"] = max(
            audit["maximum_control_absolute_difference"], float(difference.max())
        )
        if float(difference.max()) > CONTROL_TOLERANCE:
            audit["control_feature_mismatches"] += 1
            implicated = {
                key: frozen_non_ok[key]
                for key in fragment_keys(product) + fragment_keys(original)
                if key in frozen_non_ok
            }
            detail = {
                "product_smiles": product,
                "candidate_smiles": original,
                "maximum_absolute_difference": float(difference.max()),
                "frozen_non_ok_fragments": implicated,
            }
            audit["control_mismatch_details"].append(detail)
            if implicated:
                audit["fallback_attributable_control_mismatches"] += 1
            else:
                audit["unexplained_control_feature_mismatches"] += 1
        regenerated_cleaned = _feature_row(product, cleaned, prior, embeddings)
        # Anchor the intervention to the exact frozen row.  This is identical to
        # the absolute cleaned feature when the control regenerates within the
        # tolerance, and isolates the salt delta for historical fallback rows
        # whose deleted atom cache cannot be reconstructed byte-for-byte.
        transformed = result[index] + (regenerated_cleaned - regenerated)
        transformed[0] = result[index, 0]
        if transformed[0] != result[index, 0]:
            raise RuntimeError("F3 changed the frozen candidate prior.")
        result[index] = transformed
        audit["salt_changed_pair_rows"] += 1
    return result


def _write_fallback_audit(
    output: Path,
    existing_status_csv: str | Path,
    generated_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with open(existing_status_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            status = str(row["status"])
            cause = _embedding_cause(row["smiles"], status, np.empty((1, 1)), row.get("error"))
            rows.append(
                {
                    "scope": "frozen seed-42 non-ok inventory",
                    "smiles": row["smiles"],
                    "status": status,
                    "failure_cause": cause or "conformer/coordinate",
                    "error": row.get("error") or "",
                }
            )
    rows.extend(generated_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scope", "smiles", "status", "failure_cause", "error"),
        )
        writer.writeheader()
        writer.writerows(rows)
    cause_counts = {cause: 0 for cause in FAILURE_CAUSES}
    for row in rows:
        cause_counts[row["failure_cause"]] += 1
    return {
        "artifact": file_fingerprint(output),
        "audited_non_ok_rows": len(rows),
        "frozen_zero_fallback_rows": sum(
            row["scope"] == "frozen seed-42 non-ok inventory"
            and row["status"] == "fallback_zero"
            for row in rows
        ),
        "cause_counts": cause_counts,
    }


def transform_container(
    container: dict[str, Any],
    *,
    selection: bool,
    checkpoint: Path,
    dictionary: Path,
    device: str,
    batch_size: int,
    threads: int,
    batch_items: int,
    embedding_cache: str | Path | None,
    fallback_status_csv: str | Path,
    fallback_audit_csv: Path,
) -> dict[str, Any]:
    sections = _inventory_sections(container, selection)
    strings = _all_candidate_strings(sections)
    mapping, salt_audit = _salt_map(strings)
    required, affected_products = _required_fragments(sections, mapping)
    backend = SeededUniMolBackend(
        CONFORMER_SEED,
        checkpoint,
        dictionary,
        batch_size=batch_size,
        threads=threads,
        device=device,
    )
    embeddings, generated_non_ok, embedding_audit = _load_or_embed_required(
        required,
        backend,
        batch_items,
        embedding_cache,
        checkpoint,
        dictionary,
    )
    with open(fallback_status_csv, encoding="utf-8", newline="") as handle:
        frozen_non_ok = {
            row["smiles"]: row["status"] for row in csv.DictReader(handle)
        }
    feature_audit = {
        "control_pairs_recomputed": 0,
        "control_feature_mismatches": 0,
        "fallback_attributable_control_mismatches": 0,
        "unexplained_control_feature_mismatches": 0,
        "control_mismatch_details": [],
        "maximum_control_absolute_difference": 0.0,
        "control_tolerance": CONTROL_TOLERANCE,
        "control_tolerance_basis": (
            "smallest rounded bound covering the frozen train/validation-only "
            "CPU regeneration audit (observed maximum 4.696846008300781e-05); "
            "official test not loaded"
        ),
        "salt_changed_pair_rows": 0,
        "affected_products": len(affected_products),
    }
    if selection:
        for row in container["train_products"]:
            row["features"] = _transform_matrix(
                str(row["product_smiles"]),
                row["candidates"],
                row["features"],
                mapping,
                embeddings,
                frozen_non_ok,
                feature_audit,
            )
        validation = container["validation_payload"]
        validation["eval_features"] = [
            _transform_matrix(
                str(product), candidates, matrix, mapping, embeddings,
                frozen_non_ok, feature_audit
            )
            for (product, candidates), matrix in zip(
                validation["eval_pwc"], validation["eval_features"], strict=True
            )
        ]
    else:
        container["eval_features"] = [
            _transform_matrix(
                str(product), candidates, matrix, mapping, embeddings,
                frozen_non_ok, feature_audit
            )
            for (product, candidates), matrix in zip(
                container["eval_pwc"], container["eval_features"], strict=True
            )
        ]
    if feature_audit["unexplained_control_feature_mismatches"]:
        raise RuntimeError(
            "F3 regenerated control has mismatches outside the frozen non-ok "
            "inventory: "
            f"{feature_audit['unexplained_control_feature_mismatches']} pairs; "
            "first details="
            + json.dumps(
                [
                    row
                    for row in feature_audit["control_mismatch_details"]
                    if not row["frozen_non_ok_fragments"]
                ][:10],
                sort_keys=True,
            )
        )
    fallback_audit = _write_fallback_audit(
        fallback_audit_csv, fallback_status_csv, generated_non_ok
    )
    del embeddings
    return {
        "salt": salt_audit,
        "embedding": embedding_audit,
        "features": feature_audit,
        "fallback": fallback_audit,
    }


def prepare_selection(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    fallback_path = Path(args.fallback_audit).resolve()
    for target in (output, manifest_path, fallback_path):
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite F3 artifact: {target}")
    started = time.perf_counter()
    primary = validate_selection_freeze(args.primary_freeze)
    if "sha256:" + file_sha256(args.primary_selection) != primary[
        "selection_bundle_sha256"
    ]:
        raise ValueError("Primary selection bundle does not belong to its freeze.")
    checkpoint, dictionary = discover_weight_paths(args.checkpoint, args.dictionary)
    assets = validate_weight_paths(checkpoint, dictionary)
    bundle = load_selection_bundle(args.primary_selection)
    audit = transform_container(
        bundle,
        selection=True,
        checkpoint=checkpoint,
        dictionary=dictionary,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        threads=args.embedding_threads,
        batch_items=args.embedding_batch_items,
        embedding_cache=args.embedding_cache,
        fallback_status_csv=args.fallback_status_csv,
        fallback_audit_csv=fallback_path,
    )
    bundle["comparator"] = "same frozen cap10 model with current fragmentwise handling"
    bundle["single_intended_change"] = "RDKit default SaltRemover molecular input"
    bundle["f3_protocol_id"] = F3_PROTOCOL_ID
    bundle["f3_feature_audit"] = audit
    bundle["f3_settings"] = {
        "salt_definition": "RDKit SaltRemover default",
        "dontRemoveEverything": True,
        "preserve_stereochemistry": True,
        "surviving_fragment_aggregation": "existing fragmentwise concatenation",
        "feature_update": "frozen original plus paired regenerated salt delta",
        "ground_truth_matching_changed": False,
    }
    bundle["representation_provenance"] = {
        **bundle["representation_provenance"],
        "source_protocol_id": F3_PROTOCOL_ID,
        "base_source_protocol_id": bundle["representation_provenance"].get(
            "source_protocol_id"
        ),
        "backend": audit["embedding"]["backend"],
    }
    bundle["input_fingerprints"] = {
        **bundle["input_fingerprints"],
        "primary_selection_bundle": file_fingerprint(args.primary_selection),
        "primary_selection_freeze": file_fingerprint(args.primary_freeze),
        "frozen_seed42_non_ok": file_fingerprint(args.fallback_status_csv),
        "checkpoint": assets["checkpoint"],
        "dictionary": assets["dictionary"],
    }
    validate_selection_bundle(bundle)
    atomic_pickle_dump(bundle, output)
    manifest = {
        "schema_version": 1,
        "record_kind": "f3_salt_train_validation_prepare",
        "protocol_id": F3_PROTOCOL_ID,
        "comparator": "current fragmentwise handling",
        "single_intended_change": "RDKit default SaltRemover molecular input",
        "output": file_fingerprint(output),
        "primary_freeze": file_fingerprint(args.primary_freeze),
        "primary_selection": file_fingerprint(args.primary_selection),
        "fallback_audit": file_fingerprint(fallback_path),
        "audit": audit,
        "runtime_seconds": time.perf_counter() - started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_partition_loaded": False,
    }
    immutable_json_dump(manifest, manifest_path, "F3 prepare manifest")
    return manifest


def _config(primary: dict[str, Any], arm: str) -> GridConfig:
    body = primary[f"selected_{arm}"]["config"]
    config = GridConfig(**body)
    if config_fingerprint(config) != primary[f"selected_{arm}"]["config_fingerprint"]:
        raise ValueError("Frozen primary configuration fingerprint mismatch.")
    return config


def _trial_path(root: Path, arm: str, seed: int) -> Path:
    return root / "validation" / arm / f"seed_{seed}" / "trial.json"


def _validate_trial(
    root: Path, arm: str, seed: int, config: GridConfig, bundle_sha: str
) -> dict[str, Any]:
    path = _trial_path(root, arm, seed)
    trial = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "record_kind": "f3_salt_validation_trial",
        "protocol_id": F3_PROTOCOL_ID,
        "arm": arm,
        "seed": seed,
        "config_fingerprint": config_fingerprint(config),
        "selection_bundle_sha256": bundle_sha,
        "test_partition_loaded": False,
    }
    if any(trial.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid F3 validation trial: {path}")
    for rel_key, hash_key in (
        ("checkpoint_relpath", "checkpoint_sha256"),
        ("normalizer_relpath", "normalizer_sha256"),
    ):
        artifact = (root / trial[rel_key]).resolve()
        if not artifact.is_relative_to(root.resolve()):
            raise ValueError("F3 artifact escapes its output root.")
        if "sha256:" + file_sha256(artifact) != trial[hash_key]:
            raise ValueError(f"F3 artifact checksum mismatch: {artifact}")
    return trial


def fit_validation(args: argparse.Namespace) -> dict[str, Any]:
    primary = validate_selection_freeze(args.primary_freeze)
    bundle = load_selection_bundle(args.selection_bundle)
    if bundle.get("f3_protocol_id") != F3_PROTOCOL_ID:
        raise ValueError("Selection bundle is not F3.")
    config = _config(primary, args.arm)
    transform = primary["selected_prior_transform"]
    cache = transform_selection_cache(bundle, args.arm, transform)
    root = Path(args.output_root).resolve()
    bundle_sha = "sha256:" + file_sha256(args.selection_bundle)
    device = resolve_device(args.device)
    completed: dict[str, Any] = {}
    for seed in SEEDS:
        result_path = _trial_path(root, args.arm, seed)
        if result_path.exists():
            completed[str(seed)] = _validate_trial(
                root, args.arm, seed, config, bundle_sha
            )
            continue
        run_dir = result_path.parent
        checkpoint = run_dir / "best_checkpoint.pt"
        normalizer = run_dir / "normalizer.npz"
        if checkpoint.exists() or normalizer.exists():
            raise FileExistsError(f"Partial F3 trial preserved: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        trained = train_validation_trial(
            cache,
            config,
            seed,
            device,
            checkpoint,
            normalizer,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            min_improvement=MIN_IMPROVEMENT,
        )
        trial = {
            "record_kind": "f3_salt_validation_trial",
            "protocol_id": F3_PROTOCOL_ID,
            "primary_protocol_id": PROTOCOL_ID,
            "arm": args.arm,
            "seed": seed,
            "config": asdict(config),
            "config_fingerprint": config_fingerprint(config),
            "prior_transform": transform,
            "checkpoint_relpath": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": "sha256:" + file_sha256(checkpoint),
            "normalizer_relpath": normalizer.relative_to(root).as_posix(),
            "normalizer_sha256": "sha256:" + file_sha256(normalizer),
            "selection_bundle_sha256": bundle_sha,
            "single_intended_change": "RDKit default SaltRemover molecular input",
            "test_partition_loaded": False,
            "runtime_seconds": time.perf_counter() - started,
            **trained,
        }
        immutable_json_dump(trial, result_path, "F3 validation trial")
        completed[str(seed)] = trial
    return {"arm": args.arm, "completed_seeds": sorted(completed), "device": device}


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    primary = validate_selection_freeze(args.primary_freeze)
    bundle = load_selection_bundle(args.selection_bundle)
    if bundle.get("f3_protocol_id") != F3_PROTOCOL_ID:
        raise ValueError("Selection bundle is not F3.")
    root = Path(args.output_root).resolve()
    bundle_sha = "sha256:" + file_sha256(args.selection_bundle)
    selected: dict[str, Any] = {}
    for arm in ("baseline", "augmented"):
        config = _config(primary, arm)
        selected[arm] = {
            "trials": {
                str(seed): _validate_trial(root, arm, seed, config, bundle_sha)
                for seed in SEEDS
            }
        }
    record = {
        "schema_version": 1,
        "record_kind": "f3_salt_model_freeze",
        "protocol_id": F3_PROTOCOL_ID,
        "primary_selection_fingerprint": primary["selection_fingerprint"],
        "primary_freeze": file_fingerprint(args.primary_freeze),
        "selection_bundle": file_fingerprint(args.selection_bundle),
        "prepare_manifest": file_fingerprint(args.prepare_manifest),
        "retained_primary_train_test_cache_sha256": primary[
            "retained_train_test_cache_sha256"
        ],
        "selected_prior_transform": primary["selected_prior_transform"],
        "selected_baseline": selected["baseline"],
        "selected_augmented": selected["augmented"],
        "single_intended_change": "RDKit default SaltRemover molecular input",
        "config_selection_performed_for_f3": False,
        "seeds": list(SEEDS),
        "test_partition_loaded": False,
        "complete": True,
    }
    record["freeze_fingerprint"] = "sha256:" + hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    immutable_json_dump(record, args.output, "F3 model freeze")
    return record


def _load_test_payload(freeze_record: dict[str, Any], cache_path: str) -> dict:
    expected = str(freeze_record["retained_primary_train_test_cache_sha256"])
    if file_sha256(cache_path) != expected.removeprefix("sha256:"):
        raise ValueError("Primary train/test cache checksum mismatch.")
    with open(cache_path, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("F3 evaluation payload is not official test only.")
    return payload


def evaluate_test(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = require_clean_evaluation_result_dir(args.result_dir)
    started = time.perf_counter()
    freeze_record = json.loads(Path(args.f3_freeze).read_text(encoding="utf-8"))
    if not freeze_record.get("complete") or freeze_record.get("protocol_id") != F3_PROTOCOL_ID:
        raise PermissionError("Official test requires a complete F3 model freeze.")
    original_fingerprint = freeze_record["freeze_fingerprint"]
    payload = _load_test_payload(freeze_record, args.primary_train_test_cache)
    checkpoint, dictionary = discover_weight_paths(args.checkpoint, args.dictionary)
    test_audit = transform_container(
        payload,
        selection=False,
        checkpoint=checkpoint,
        dictionary=dictionary,
        device=args.embedding_device,
        batch_size=args.embedding_batch_size,
        threads=args.embedding_threads,
        batch_items=args.embedding_batch_items,
        embedding_cache=args.embedding_cache,
        fallback_status_csv=args.fallback_status_csv,
        fallback_audit_csv=result_dir / "fallback_audit.csv",
    )
    device = resolve_device(args.device)
    root = Path(args.output_root).resolve()
    salted_baseline = _score_test_arm(
        payload,
        freeze_record["selected_baseline"],
        freeze_record["selected_prior_transform"],
        "baseline",
        root,
        result_dir / "predictions",
        device,
    )
    salted_augmented = _score_test_arm(
        payload,
        freeze_record["selected_augmented"],
        freeze_record["selected_prior_transform"],
        "augmented",
        root,
        result_dir / "predictions",
        device,
    )
    primary_manifest_path = Path(args.primary_test_manifest).resolve()
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    if (
        primary_manifest.get("selection_fingerprint")
        != freeze_record["primary_selection_fingerprint"]
        or not primary_manifest.get("test_partition_loaded_only_after_selection_freeze")
    ):
        raise ValueError("Frozen primary test manifest is incompatible with F3.")
    if json.loads(Path(args.f3_freeze).read_text(encoding="utf-8"))[
        "freeze_fingerprint"
    ] != original_fingerprint:
        raise RuntimeError("F3 freeze changed during test evaluation.")
    record = {
        "schema_version": 1,
        "record_kind": "f3_salt_post_freeze_test",
        "protocol_id": F3_PROTOCOL_ID,
        "f3_freeze": file_fingerprint(args.f3_freeze),
        "primary_current_handling_manifest": file_fingerprint(primary_manifest_path),
        "current_handling_metrics": primary_manifest["per_seed_metrics"],
        "salt_removed_metrics": {
            "baseline": salted_baseline,
            "augmented": salted_augmented,
        },
        "salt_removed_descriptive_summary": summarize_paired_metrics(
            salted_baseline, salted_augmented
        ),
        "test_feature_audit": test_audit,
        "single_intended_change": "RDKit default SaltRemover molecular input",
        "ground_truth_matching_changed": False,
        "test_partition_loaded_only_after_f3_freeze": True,
        "environment": environment_record(device),
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "runtime_seconds": time.perf_counter() - started,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    immutable_json_dump(record, result_dir / "manifest.json", "F3 test manifest")
    return record


def _embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint")
    parser.add_argument("--dictionary")
    parser.add_argument("--embedding-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=8)
    parser.add_argument("--embedding-threads", type=int, default=12)
    parser.add_argument("--embedding-batch-items", type=int, default=256)
    parser.add_argument("--embedding-cache")
    parser.add_argument("--fallback-status-csv", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--primary-freeze", required=True)
    prepare.add_argument("--primary-selection", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--fallback-audit", required=True)
    _embedding_arguments(prepare)
    fit = subparsers.add_parser("fit-validation")
    fit.add_argument("--primary-freeze", required=True)
    fit.add_argument("--selection-bundle", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--arm", choices=("baseline", "augmented"), required=True)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--primary-freeze", required=True)
    freeze_parser.add_argument("--selection-bundle", required=True)
    freeze_parser.add_argument("--prepare-manifest", required=True)
    freeze_parser.add_argument("--output-root", required=True)
    freeze_parser.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate-test")
    evaluate.add_argument("--f3-freeze", required=True)
    evaluate.add_argument("--primary-train-test-cache", required=True)
    evaluate.add_argument("--primary-test-manifest", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    _embedding_arguments(evaluate)
    args = parser.parse_args()
    result = {
        "prepare": prepare_selection,
        "fit-validation": fit_validation,
        "freeze": freeze,
        "evaluate-test": evaluate_test,
    }[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
