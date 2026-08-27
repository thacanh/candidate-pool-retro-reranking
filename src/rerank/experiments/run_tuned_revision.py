#!/usr/bin/env python
"""Sharded, resumable D1--D3 runner for the approved ``cap10-tuned-v1``.

The command is intentionally phased.  Search never accepts a train/test cache;
it accepts only a restricted train+official-validation bundle.  The legacy
test payload is opened only by ``evaluate-test`` after a cryptographically
fingerprinted selection-freeze record has passed validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rerank.evaluate import evaluate_reranking
from rerank.features import FeatureNormalizer
from rerank.model import RankerMLP
from rerank.revision_tuning import (
    AUGMENTED_COLUMNS,
    BASELINE_COLUMNS,
    CAPACITY_CONTROL_ID,
    CAPACITY_PLAN_SCHEMA,
    CAPACITY_TRIAL_SCHEMA,
    CapacitySettings,
    GRID_TIE_EPSILON,
    MAX_EPOCHS,
    MIN_IMPROVEMENT,
    PATIENCE,
    PRIOR_TRANSFORMS,
    PROTOCOL_ID,
    SEEDS,
    TRIAL_SCHEMA,
    assert_d3_capacity_match,
    atomic_json_dump,
    capacity_arm_config,
    capacity_settings_fingerprint,
    config_fingerprint,
    enumerate_d1_grid,
    feature_columns_for_arm,
    file_fingerprint,
    file_sha256,
    load_selection_bundle,
    prepare_selection_bundle,
    select_prior_transform,
    select_shared_config,
    shard_config_indices,
    train_validation_trial,
    transform_feature_matrix,
    transform_selection_cache,
    validate_capacity_settings,
)


FIXED_TUNING_CONFIG = {
    "architecture": "one-hidden-layer RankerMLP",
    "use_batch_norm": False,
    "optimizer": "AdamW",
    "weight_decay": 1e-3,
    "scheduler": "CosineAnnealingLR after epoch 5",
    "batch_size": 256,
    "gradient_clip": 1.0,
    "max_negative_pairs_per_positive": 5,
    "negative_mining": "seeded random",
    "normalization": "per-seed, training-pair-only FeatureNormalizer",
    "max_epochs": MAX_EPOCHS,
    "early_stopping_patience": PATIENCE,
    "minimum_validation_mrr_improvement": MIN_IMPROVEMENT,
    "selection_split": "official valid",
    "selection_metric": "conditional MRR",
    "selection_seeds": list(SEEDS),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def canonical_fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(SEEDS):
        raise argparse.ArgumentTypeError("Seeds must be a unique subset of 42,43,44,45,46.")
    return seeds


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def environment_record(device: str) -> dict:
    packages = {}
    for package in ("torch", "numpy", "pandas", "rdkit"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": device,
        "packages": packages,
    }
    if device == "cuda":
        result.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return result


def trial_path(output_root: Path, arm: str, transform: str, index: int, seed: int) -> Path:
    return (
        output_root
        / "search"
        / arm
        / transform
        / f"config_{index:03d}"
        / f"seed_{seed}"
        / "trial.json"
    )


def _trial_artifacts_valid(result: dict, output_root: Path, expected: dict) -> bool:
    if any(result.get(key) != value for key, value in expected.items()):
        return False
    for path_key, hash_key in (
        ("checkpoint_relpath", "checkpoint_sha256"),
        ("normalizer_relpath", "normalizer_sha256"),
    ):
        path = (output_root / result.get(path_key, "")).resolve()
        if not path.is_relative_to(output_root.resolve()):
            return False
        if not path.is_file() or "sha256:" + file_sha256(path) != result.get(hash_key):
            return False
    return result.get("status") == "completed"


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def require_immutable_output_absent(path: str | Path, label: str) -> Path:
    """Fail closed instead of replacing a scientific decision/output record."""

    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    existing = [candidate for candidate in (target, temporary) if candidate.exists()]
    if existing:
        raise FileExistsError(
            f"Refusing to overwrite existing or partial {label}: "
            + ", ".join(str(candidate.resolve()) for candidate in existing)
        )
    return target


def immutable_json_dump(payload: dict, path: str | Path, label: str) -> None:
    target = require_immutable_output_absent(path, label)
    atomic_json_dump(payload, target)


def require_no_partial_trial_artifacts(
    result_path: str | Path,
    checkpoint_path: str | Path,
    normalizer_path: str | Path,
) -> None:
    """A missing result record never authorizes replacing retained trial state."""

    result = Path(result_path)
    if result.exists():
        return
    partial = [
        path
        for path in (
            result.with_suffix(result.suffix + ".tmp"),
            Path(checkpoint_path),
            Path(normalizer_path),
        )
        if path.exists()
    ]
    if partial:
        raise FileExistsError(
            "Trial result is absent but retained artifacts already exist; preserve "
            "and inspect them instead of overwriting: "
            + ", ".join(str(path.resolve()) for path in partial)
        )


def require_clean_evaluation_result_dir(path: str | Path) -> Path:
    """Evaluation is append-never: partial or completed outputs are immutable."""

    result_dir = Path(path).resolve()
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty evaluation result directory: {result_dir}"
        )
    return result_dir


def _require_prior_freeze(path: str | Path, bundle_sha256: str) -> dict:
    record = _load_json(path)
    if record.get("record_kind") != "prior_transform_freeze":
        raise PermissionError("Augmented search requires a prior-transform freeze record.")
    if record.get("protocol_id") != PROTOCOL_ID:
        raise PermissionError("Prior freeze has the wrong protocol ID.")
    if record.get("selection_bundle_sha256") != bundle_sha256:
        raise PermissionError("Prior freeze belongs to a different selection bundle.")
    expected = dict(record)
    supplied = expected.pop("freeze_fingerprint", None)
    if supplied != canonical_fingerprint(expected):
        raise PermissionError("Prior freeze fingerprint is invalid.")
    return record


def run_search(args) -> None:
    bundle = load_selection_bundle(args.selection_bundle)
    bundle_sha256 = "sha256:" + file_sha256(args.selection_bundle)
    if args.arm == "augmented":
        if not args.prior_freeze:
            raise PermissionError("Augmented search cannot start before --prior-freeze.")
        freeze = _require_prior_freeze(args.prior_freeze, bundle_sha256)
        selected_transform = freeze["selected_prior_transform"]
        if args.prior_transform != selected_transform:
            raise PermissionError(
                f"Augmented search must use frozen transform {selected_transform!r}."
            )

    output_root = Path(args.output_root).resolve()
    selected_indices = shard_config_indices(args.shard_index, args.shard_count)
    grid = enumerate_d1_grid()
    device = resolve_device(args.device)
    selection_cache = transform_selection_cache(
        bundle, args.arm, args.prior_transform
    )
    started = time.perf_counter()
    completed = 0
    resumed = 0
    compact_progress = bool(getattr(args, "compact_progress", False))
    stop_after_epoch = getattr(args, "stop_after_epoch", None)
    stop_margin_seconds = float(getattr(args, "stop_margin_seconds", 600.0))
    if stop_margin_seconds < 0:
        raise ValueError("--stop-margin-seconds cannot be negative.")
    total_trials = len(selected_indices) * len(args.seeds)
    phase = (
        f"{PRIOR_TRANSFORMS.index(args.prior_transform) + 1}/{len(PRIOR_TRANSFORMS)}"
        if args.arm == "baseline"
        else "1/1"
    )
    progress = None
    if compact_progress:
        device_label = device
        if device == "cuda":
            device_label = f"cuda:{torch.cuda.get_device_name(0)}"
        print(
            f"{args.arm} phase {phase} | prior={args.prior_transform} | "
            f"shard={args.shard_index + 1}/{args.shard_count} | "
            f"device={device_label} | trials={total_trials}",
            flush=True,
        )
        progress = tqdm(
            total=total_trials,
            desc=f"{args.arm}/{args.prior_transform}",
            unit="trial",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stdout,
        )
    paused_for_budget = False
    try:
        for config_index in selected_indices:
            config = grid[config_index]
            for seed in args.seeds:
                # A budget stop is checked only between immutable trials.  It
                # never interrupts a checkpoint/normalizer pair mid-write.
                if (
                    stop_after_epoch is not None
                    and time.time() >= float(stop_after_epoch) - stop_margin_seconds
                ):
                    paused_for_budget = True
                    break
                if progress is not None:
                    progress.set_postfix(
                        cfg=f"{config.index:03d}", seed=seed, state="running",
                        refresh=True,
                    )
                result_path = trial_path(
                    output_root, args.arm, args.prior_transform, config.index, seed
                )
                run_dir = result_path.parent
                checkpoint = run_dir / "best_checkpoint.pt"
                normalizer = run_dir / "normalizer.npz"
                expected = {
                    "trial_schema": TRIAL_SCHEMA,
                    "protocol_id": PROTOCOL_ID,
                    "arm": args.arm,
                    "prior_transform": args.prior_transform,
                    "config_index": config.index,
                    "config_fingerprint": config_fingerprint(config),
                    "seed": seed,
                    "selection_bundle_sha256": bundle_sha256,
                }
                if result_path.is_file():
                    previous = _load_json(result_path)
                    if _trial_artifacts_valid(previous, output_root, expected):
                        resumed += 1
                        if progress is not None:
                            progress.set_postfix(
                                cfg=f"{config.index:03d}", seed=seed, state="resumed",
                                refresh=False,
                            )
                            progress.update(1)
                        continue
                    raise RuntimeError(
                        f"Existing trial is incomplete or incompatible; preserve and inspect {result_path}."
                    )

                require_no_partial_trial_artifacts(result_path, checkpoint, normalizer)
                run_dir.mkdir(parents=True, exist_ok=True)
                trial_started = time.perf_counter()
                trained = train_validation_trial(
                    selection_cache,
                    config,
                    seed,
                    device,
                    checkpoint,
                    normalizer,
                    max_epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                    min_improvement=MIN_IMPROVEMENT,
                )
                trial_seconds = time.perf_counter() - trial_started
                result = {
                    **expected,
                    "status": "completed",
                    "comparator": (
                        "validation-tuned prior+2D baseline"
                        if args.arm == "augmented"
                        else "other baseline prior transforms under equal search budget"
                    ),
                    "single_intended_change": (
                        "add three Uni-Mol-derived pair-level scalars"
                        if args.arm == "augmented"
                        else "prior transform"
                    ),
                    "config": asdict(config),
                    "checkpoint_relpath": checkpoint.relative_to(output_root).as_posix(),
                    "normalizer_relpath": normalizer.relative_to(output_root).as_posix(),
                    "max_epochs": MAX_EPOCHS,
                    "patience": PATIENCE,
                    "minimum_improvement": MIN_IMPROVEMENT,
                    "validation_metric": "official-validation conditional MRR",
                    "fixed_training_config": FIXED_TUNING_CONFIG,
                    "representation_provenance": bundle["representation_provenance"],
                    "test_partition_loaded": False,
                    "runtime_seconds": trial_seconds,
                    "created_at_utc": utc_now(),
                    **trained,
                }
                atomic_json_dump(result, result_path)
                completed += 1
                if progress is not None:
                    progress.set_postfix(
                        cfg=f"{config.index:03d}",
                        seed=seed,
                        mrr=f"{trained['best_validation_mrr']:.4f}",
                        epoch=trained["best_epoch"],
                        sec=f"{trial_seconds:.0f}",
                        refresh=False,
                    )
                    progress.update(1)
            if paused_for_budget:
                break
    finally:
        if progress is not None:
            progress.close()

    if paused_for_budget:
        print(
            f"SAFE PAUSE {args.arm}/{args.prior_transform} shard "
            f"{args.shard_index + 1}/{args.shard_count} | "
            f"new={completed} resumed={resumed} | no partial trial created | "
            "rerun the same launcher to resume",
            flush=True,
        )
        raise SystemExit(75)

    manifest = {
        "manifest_kind": "tuning_search_shard",
        "protocol_id": PROTOCOL_ID,
        "comparator": "equal 81-configuration validation-only search budget",
        "single_intended_change": args.arm,
        "arm": args.arm,
        "prior_transform": args.prior_transform,
        "seeds": list(args.seeds),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "config_indices": list(selected_indices),
        "selection_bundle": file_fingerprint(args.selection_bundle),
        "representation_provenance": bundle["representation_provenance"],
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "completed_trials": completed,
        "resumed_trials": resumed,
        "runtime_seconds": time.perf_counter() - started,
        "git_commit": git_commit(),
        "environment": environment_record(device),
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    manifest_path = (
        output_root
        / "shard_manifests"
        / (
            f"{args.arm}_{args.prior_transform}_shard_{args.shard_index:03d}"
            f"_of_{args.shard_count:03d}_seeds_{'-'.join(map(str, args.seeds))}.json"
        )
    )
    atomic_json_dump(manifest, manifest_path)
    if compact_progress:
        print(
            f"COMPLETE {args.arm}/{args.prior_transform} shard "
            f"{args.shard_index + 1}/{args.shard_count} | "
            f"new={completed} resumed={resumed} | "
            f"runtime={manifest['runtime_seconds'] / 60:.1f} min | "
            f"manifest={manifest_path}",
            flush=True,
        )
    else:
        print(json.dumps(manifest, indent=2))


def collect_trials(
    output_root: Path, arm: str, transform: str, bundle_sha256: str
) -> list[dict]:
    base = output_root / "search" / arm / transform
    trials = []
    grid = enumerate_d1_grid()
    for path in sorted(base.glob("config_*/seed_*/trial.json")):
        result = _load_json(path)
        config_index = int(result.get("config_index", -1))
        seed = int(result.get("seed", -1))
        if not 0 <= config_index < len(grid) or seed not in SEEDS:
            raise RuntimeError(f"Trial has an out-of-protocol config or seed: {path}")
        expected = {
            "trial_schema": TRIAL_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "arm": arm,
            "prior_transform": transform,
            "config_index": config_index,
            "config_fingerprint": config_fingerprint(grid[config_index]),
            "seed": seed,
            "selection_bundle_sha256": bundle_sha256,
        }
        if not _trial_artifacts_valid(result, output_root, expected):
            raise RuntimeError(f"Trial result or retained artifacts failed validation: {path}")
        if result.get("config") != asdict(grid[config_index]):
            raise RuntimeError(f"Trial config body disagrees with frozen enumeration: {path}")
        if result.get("test_partition_loaded") is not False:
            raise PermissionError(f"A selection trial accessed the test partition: {path}")
        trials.append(result)
    return trials


def run_select_prior(args) -> None:
    require_immutable_output_absent(args.output, "prior-transform freeze")
    output_root = Path(args.output_root).resolve()
    bundle_sha256 = "sha256:" + file_sha256(args.selection_bundle)
    bundle = load_selection_bundle(args.selection_bundle)
    trials = {
        transform: collect_trials(output_root, "baseline", transform, bundle_sha256)
        for transform in PRIOR_TRANSFORMS
    }
    selection = select_prior_transform(trials)
    record = {
        "record_kind": "prior_transform_freeze",
        "protocol_id": PROTOCOL_ID,
        "comparator": "raw/log/rank baseline searches with equal 81-point budgets",
        "single_intended_change": "prior transform",
        "selection_bundle_sha256": bundle_sha256,
        "representation_provenance": bundle["representation_provenance"],
        "selected_prior_transform": selection["selected_prior_transform"],
        "selected_baseline": selection["selected_baseline"],
        "all_transforms": selection["all_transforms"],
        "tie_epsilon": GRID_TIE_EPSILON,
        "tie_order": list(PRIOR_TRANSFORMS),
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    record["freeze_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "prior-transform freeze")
    print(json.dumps(record, indent=2))


def _trim_selected(selected: dict, output_root: Path) -> dict:
    """Keep selected seed artifacts and scores, but omit per-epoch curves."""

    result = {key: value for key, value in selected.items() if key != "trials"}
    result["trials"] = {}
    for seed, trial in selected["trials"].items():
        checkpoint = (output_root / trial["checkpoint_relpath"]).resolve()
        normalizer = (output_root / trial["normalizer_relpath"]).resolve()
        if not checkpoint.is_relative_to(output_root) or not normalizer.is_relative_to(
            output_root
        ):
            raise RuntimeError("A selected artifact path escapes the tuning output root.")
        if "sha256:" + file_sha256(checkpoint) != trial["checkpoint_sha256"]:
            raise RuntimeError(f"Selected checkpoint failed its fingerprint: {checkpoint}")
        if "sha256:" + file_sha256(normalizer) != trial["normalizer_sha256"]:
            raise RuntimeError(f"Selected normalizer failed its fingerprint: {normalizer}")
        result["trials"][seed] = {
            key: value for key, value in trial.items() if key != "history"
        }
    return result


def run_freeze_selection(args) -> None:
    require_immutable_output_absent(args.output, "model-selection freeze")
    output_root = Path(args.output_root).resolve()
    bundle_sha256 = "sha256:" + file_sha256(args.selection_bundle)
    bundle = load_selection_bundle(args.selection_bundle)
    prior = _require_prior_freeze(args.prior_freeze, bundle_sha256)
    transform = prior["selected_prior_transform"]
    augmented = select_shared_config(
        collect_trials(output_root, "augmented", transform, bundle_sha256)
    )
    record = {
        "record_kind": "model_selection_freeze",
        "protocol_id": PROTOCOL_ID,
        "comparator": "validation-tuned prior+2D baseline",
        "single_intended_change": "add three Uni-Mol-derived pair-level scalars",
        "selection_bundle_sha256": bundle_sha256,
        "representation_provenance": bundle["representation_provenance"],
        "retained_train_test_cache_sha256": bundle["input_fingerprints"][
            "retained_train_test_cache"
        ]["sha256"],
        "prior_freeze_fingerprint": prior["freeze_fingerprint"],
        "selected_prior_transform": transform,
        "selected_baseline": _trim_selected(prior["selected_baseline"], output_root),
        "selected_augmented": _trim_selected(augmented, output_root),
        "d3_capacity_assertion": assert_d3_capacity_match(),
        "seeds": list(SEEDS),
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "minimum_improvement": MIN_IMPROVEMENT,
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "selection_metric": "mean best official-validation conditional MRR across seeds 42--46",
        "selected_models_retrained_on_train_plus_validation": False,
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    record["selection_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "model-selection freeze")
    print(json.dumps(record, indent=2))


def validate_selection_freeze(path: str | Path) -> dict:
    """Fail before any test path is opened unless selection is truly frozen."""

    record = _load_json(path)
    if record.get("record_kind") != "model_selection_freeze":
        raise PermissionError("Test evaluation requires a model-selection freeze record.")
    if record.get("protocol_id") != PROTOCOL_ID or tuple(record.get("seeds", ())) != SEEDS:
        raise PermissionError("Selection freeze does not match cap10-tuned-v1 seeds 42--46.")
    expected = dict(record)
    supplied = expected.pop("selection_fingerprint", None)
    if supplied != canonical_fingerprint(expected):
        raise PermissionError("Selection-freeze fingerprint is invalid.")
    if record.get("test_partition_loaded") is not False:
        raise PermissionError("Selection freeze must attest that test was not loaded.")
    return record


def load_test_cache_after_freeze(
    selection_freeze_path: str | Path, train_test_cache_path: str | Path
) -> tuple[dict, dict]:
    freeze = validate_selection_freeze(selection_freeze_path)
    # Deliberately below the validation call: an invalid freeze cannot cause
    # even an attempted open/unpickle of the test-containing artifact.
    expected_sha256 = freeze.get("retained_train_test_cache_sha256")
    if not expected_sha256 or file_sha256(train_test_cache_path) != expected_sha256:
        raise PermissionError("Official-test cache does not match the frozen selection input.")
    with open(train_test_cache_path, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    required = {"eval_pwc", "eval_ground_truths", "eval_metadata", "eval_features"}
    if not required.issubset(payload):
        raise ValueError("Retained train/test cache lacks its official-test payload.")
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("Post-selection evaluation payload is not exclusively test.")
    return freeze, payload


def run_prepare_capacity(args) -> None:
    """Freeze the otherwise under-specified non-width D3 settings."""

    require_immutable_output_absent(args.output, "D-CAPACITY execution plan")
    bundle = load_selection_bundle(args.selection_bundle)
    bundle_sha256 = "sha256:" + file_sha256(args.selection_bundle)
    prior = _require_prior_freeze(args.prior_freeze, bundle_sha256)
    decision_note = str(args.decision_note).strip()
    if not decision_note:
        raise ValueError("D-CAPACITY requires a non-empty prespecification decision note.")
    settings = CapacitySettings(
        dropout=float(args.dropout),
        learning_rate=float(args.learning_rate),
        margin=float(args.margin),
    )
    validate_capacity_settings(settings)
    record = {
        "record_kind": "capacity_execution_plan",
        "plan_schema": CAPACITY_PLAN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "control_id": CAPACITY_CONTROL_ID,
        "comparator": "seven-input width-32 augmented RankerMLP (289 parameters)",
        "single_intended_change": (
            "increase the four-input baseline hidden width from 32 to 48 "
            "to match the comparator's trainable parameter count"
        ),
        "selection_bundle_sha256": bundle_sha256,
        "representation_provenance": bundle["representation_provenance"],
        "retained_train_test_cache_sha256": bundle["input_fingerprints"][
            "retained_train_test_cache"
        ]["sha256"],
        "prior_freeze_fingerprint": prior["freeze_fingerprint"],
        "prior_transform": prior["selected_prior_transform"],
        "non_width_settings": asdict(settings),
        "non_width_settings_fingerprint": capacity_settings_fingerprint(settings),
        "decision_note": decision_note,
        "capacity_assertion": assert_d3_capacity_match(),
        "paired_seeds": list(SEEDS),
        "search_budget": "one explicitly frozen configuration x five paired seeds per arm",
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    record["capacity_plan_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "D-CAPACITY execution plan")
    print(json.dumps(record, indent=2))


def _require_capacity_plan(path: str | Path, bundle_sha256: str | None = None) -> dict:
    record = _load_json(path)
    if (
        record.get("record_kind") != "capacity_execution_plan"
        or record.get("plan_schema") != CAPACITY_PLAN_SCHEMA
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("control_id") != CAPACITY_CONTROL_ID
    ):
        raise PermissionError("D-CAPACITY requires a valid capacity execution plan.")
    expected = dict(record)
    supplied = expected.pop("capacity_plan_fingerprint", None)
    if supplied != canonical_fingerprint(expected):
        raise PermissionError("D-CAPACITY execution-plan fingerprint is invalid.")
    if bundle_sha256 is not None and record.get("selection_bundle_sha256") != bundle_sha256:
        raise PermissionError("D-CAPACITY plan belongs to another selection bundle.")
    if record.get("test_partition_loaded") is not False:
        raise PermissionError("D-CAPACITY plan must be frozen without test access.")
    settings = CapacitySettings(**record.get("non_width_settings", {}))
    validate_capacity_settings(settings)
    if capacity_settings_fingerprint(settings) != record.get(
        "non_width_settings_fingerprint"
    ):
        raise PermissionError("D-CAPACITY non-width settings fingerprint is invalid.")
    if not str(record.get("decision_note", "")).strip():
        raise PermissionError("D-CAPACITY plan lacks its required decision note.")
    if record.get("capacity_assertion") != assert_d3_capacity_match():
        raise PermissionError("D-CAPACITY plan has stale parameter-count assertions.")
    return record


def capacity_trial_path(output_root: Path, arm: str, seed: int) -> Path:
    return output_root / "capacity" / "search" / arm / f"seed_{seed}" / "trial.json"


def run_capacity(args) -> None:
    bundle = load_selection_bundle(args.selection_bundle)
    bundle_sha256 = "sha256:" + file_sha256(args.selection_bundle)
    plan = _require_capacity_plan(args.capacity_plan, bundle_sha256)
    settings = CapacitySettings(**plan["non_width_settings"])
    config = capacity_arm_config(settings, args.arm)
    capacity = assert_d3_capacity_match()[args.arm]
    if capacity["parameters"] != 289 or capacity["hidden_width"] != config.hidden_width:
        raise AssertionError("D-CAPACITY runtime model differs from its frozen plan.")
    selection_cache = transform_selection_cache(
        bundle, args.arm, plan["prior_transform"]
    )
    output_root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    completed = 0
    resumed = 0
    started = time.perf_counter()
    for seed in args.seeds:
        result_path = capacity_trial_path(output_root, args.arm, seed)
        run_dir = result_path.parent
        checkpoint = run_dir / "best_checkpoint.pt"
        normalizer = run_dir / "normalizer.npz"
        expected = {
            "trial_schema": CAPACITY_TRIAL_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "control_id": CAPACITY_CONTROL_ID,
            "capacity_plan_fingerprint": plan["capacity_plan_fingerprint"],
            "selection_bundle_sha256": bundle_sha256,
            "arm": args.arm,
            "seed": seed,
        }
        if result_path.is_file():
            previous = _load_json(result_path)
            if _trial_artifacts_valid(previous, output_root, expected):
                resumed += 1
                continue
            raise RuntimeError(
                f"Existing D-CAPACITY trial is incompatible; preserve and inspect {result_path}."
            )
        require_no_partial_trial_artifacts(result_path, checkpoint, normalizer)
        run_dir.mkdir(parents=True, exist_ok=True)
        trial_started = time.perf_counter()
        trained = train_validation_trial(
            selection_cache,
            config,
            seed,
            device,
            checkpoint,
            normalizer,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            min_improvement=MIN_IMPROVEMENT,
        )
        if trained["model_parameters"] != 289:
            raise AssertionError("D-CAPACITY trained a model with a non-289 parameter count.")
        result = {
            **expected,
            "status": "completed",
            "comparator": plan["comparator"],
            "single_intended_change": plan["single_intended_change"],
            "prior_transform": plan["prior_transform"],
            "representation_provenance": plan["representation_provenance"],
            "config": asdict(config),
            "non_width_settings_fingerprint": plan["non_width_settings_fingerprint"],
            "checkpoint_relpath": checkpoint.relative_to(output_root).as_posix(),
            "normalizer_relpath": normalizer.relative_to(output_root).as_posix(),
            "fixed_training_config": FIXED_TUNING_CONFIG,
            "validation_metric": "official-validation conditional MRR",
            "test_partition_loaded": False,
            "runtime_seconds": time.perf_counter() - trial_started,
            "created_at_utc": utc_now(),
            **trained,
        }
        atomic_json_dump(result, result_path)
        completed += 1

    manifest = {
        "manifest_kind": "capacity_execution_shard",
        "protocol_id": PROTOCOL_ID,
        "control_id": CAPACITY_CONTROL_ID,
        "comparator": plan["comparator"],
        "single_intended_change": plan["single_intended_change"],
        "capacity_plan": file_fingerprint(args.capacity_plan),
        "capacity_plan_fingerprint": plan["capacity_plan_fingerprint"],
        "selection_bundle": file_fingerprint(args.selection_bundle),
        "representation_provenance": plan["representation_provenance"],
        "arm": args.arm,
        "seeds": list(args.seeds),
        "completed_trials": completed,
        "resumed_trials": resumed,
        "runtime_seconds": time.perf_counter() - started,
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "search_budget": "one explicitly frozen configuration x five paired seeds per arm",
        "environment": environment_record(device),
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    path = (
        output_root
        / "capacity"
        / "shard_manifests"
        / f"{args.arm}_seeds_{'-'.join(map(str, args.seeds))}.json"
    )
    atomic_json_dump(manifest, path)
    print(json.dumps(manifest, indent=2))


def collect_capacity_trials(
    output_root: Path, arm: str, plan: dict, bundle_sha256: str
) -> dict[str, dict]:
    settings = CapacitySettings(**plan["non_width_settings"])
    config = capacity_arm_config(settings, arm)
    trials = {}
    for seed in SEEDS:
        path = capacity_trial_path(output_root, arm, seed)
        if not path.is_file():
            raise RuntimeError(f"Missing D-CAPACITY {arm} seed {seed}: {path}")
        trial = _load_json(path)
        expected = {
            "trial_schema": CAPACITY_TRIAL_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "control_id": CAPACITY_CONTROL_ID,
            "capacity_plan_fingerprint": plan["capacity_plan_fingerprint"],
            "selection_bundle_sha256": bundle_sha256,
            "arm": arm,
            "seed": seed,
        }
        if not _trial_artifacts_valid(trial, output_root, expected):
            raise RuntimeError(f"D-CAPACITY trial failed artifact validation: {path}")
        if trial.get("config") != asdict(config) or trial.get("model_parameters") != 289:
            raise RuntimeError(f"D-CAPACITY trial violates its frozen architecture: {path}")
        if trial.get("test_partition_loaded") is not False:
            raise PermissionError(f"D-CAPACITY trial accessed test before freeze: {path}")
        trials[str(seed)] = {key: value for key, value in trial.items() if key != "history"}
    return trials


def run_freeze_capacity(args) -> None:
    require_immutable_output_absent(args.output, "D-CAPACITY selection freeze")
    output_root = Path(args.output_root).resolve()
    load_selection_bundle(args.selection_bundle)
    bundle_sha256 = "sha256:" + file_sha256(args.selection_bundle)
    plan = _require_capacity_plan(args.capacity_plan, bundle_sha256)
    baseline = collect_capacity_trials(output_root, "baseline", plan, bundle_sha256)
    augmented = collect_capacity_trials(output_root, "augmented", plan, bundle_sha256)
    paired = set(baseline) == set(augmented) == {str(seed) for seed in SEEDS}
    if not paired:
        raise RuntimeError("D-CAPACITY requires paired baseline/augmented seeds 42--46.")
    for seed in map(str, SEEDS):
        pairing_fields = (
            "n_train_pairs",
            "n_validation_reactions",
            "n_validation_candidate_rows",
        )
        if any(baseline[seed][field] != augmented[seed][field] for field in pairing_fields):
            raise RuntimeError(f"D-CAPACITY pairing gate failed for seed {seed}.")
    record = {
        "record_kind": "capacity_selection_freeze",
        "protocol_id": PROTOCOL_ID,
        "control_id": CAPACITY_CONTROL_ID,
        "comparator": plan["comparator"],
        "single_intended_change": plan["single_intended_change"],
        "capacity_plan_fingerprint": plan["capacity_plan_fingerprint"],
        "selection_bundle_sha256": bundle_sha256,
        "representation_provenance": plan["representation_provenance"],
        "retained_train_test_cache_sha256": plan[
            "retained_train_test_cache_sha256"
        ],
        "prior_transform": plan["prior_transform"],
        "non_width_settings": plan["non_width_settings"],
        "non_width_settings_fingerprint": plan["non_width_settings_fingerprint"],
        "capacity_assertion": assert_d3_capacity_match(),
        "paired_seeds": list(SEEDS),
        "pairing_gate": {
            "passed": True,
            "identical_selection_bundle": True,
            "identical_pair_sampling_seeds": True,
            "identical_train_pair_counts": True,
            "identical_validation_reaction_order": True,
            "identical_early_stopping_budget": True,
        },
        "search_budget": "one explicitly frozen configuration x five paired seeds per arm",
        "selected_baseline": {"trials": baseline},
        "selected_augmented": {"trials": augmented},
        "selection_metric": "per-seed best official-validation conditional MRR",
        "selected_models_retrained_on_train_plus_validation": False,
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    record["capacity_freeze_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "D-CAPACITY selection freeze")
    print(json.dumps(record, indent=2))


def validate_capacity_freeze(path: str | Path) -> dict:
    record = _load_json(path)
    if (
        record.get("record_kind") != "capacity_selection_freeze"
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("control_id") != CAPACITY_CONTROL_ID
        or tuple(record.get("paired_seeds", ())) != SEEDS
    ):
        raise PermissionError("Official-test D-CAPACITY evaluation requires a valid freeze.")
    expected = dict(record)
    supplied = expected.pop("capacity_freeze_fingerprint", None)
    if supplied != canonical_fingerprint(expected):
        raise PermissionError("D-CAPACITY selection-freeze fingerprint is invalid.")
    if record.get("test_partition_loaded") is not False:
        raise PermissionError("D-CAPACITY freeze must attest that test was not loaded.")
    if record.get("capacity_assertion") != assert_d3_capacity_match():
        raise PermissionError("D-CAPACITY freeze has stale parameter counts.")
    return record


def load_capacity_test_cache_after_freeze(
    capacity_freeze_path: str | Path, train_test_cache_path: str | Path
) -> tuple[dict, dict]:
    freeze = validate_capacity_freeze(capacity_freeze_path)
    # As for primary evaluation, do not touch the test-bearing pickle until the
    # dedicated D-CAPACITY freeze has been authenticated.
    expected_sha256 = freeze.get("retained_train_test_cache_sha256")
    if not expected_sha256 or file_sha256(train_test_cache_path) != expected_sha256:
        raise PermissionError("D-CAPACITY test cache does not match its frozen input.")
    with open(train_test_cache_path, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    required = {"eval_pwc", "eval_ground_truths", "eval_metadata", "eval_features"}
    if not required.issubset(payload):
        raise ValueError("Retained train/test cache lacks its official-test payload.")
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("D-CAPACITY evaluation payload is not exclusively test.")
    return freeze, payload


def _score_test_arm(
    payload: dict,
    selected: dict,
    transform: str,
    arm: str,
    output_root: Path,
    prediction_dir: Path,
    device: str,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    columns = feature_columns_for_arm(arm)
    input_dim = len(columns)
    metrics = {}
    prediction_targets = {
        seed: prediction_dir / f"{arm}_seed_{seed}.csv" for seed in seeds
    }
    existing_predictions = [
        path for path in prediction_targets.values() if path.exists()
    ]
    if existing_predictions:
        raise FileExistsError(
            "Refusing to overwrite existing reaction-level predictions: "
            + ", ".join(str(path.resolve()) for path in existing_predictions)
        )
    for seed in seeds:
        trial = selected["trials"][str(seed)]
        config = trial["config"]
        checkpoint = (output_root / trial["checkpoint_relpath"]).resolve()
        normalizer_path = (output_root / trial["normalizer_relpath"]).resolve()
        if not checkpoint.is_relative_to(output_root) or not normalizer_path.is_relative_to(
            output_root
        ):
            raise RuntimeError("A selected artifact path escapes the tuning output root.")
        if "sha256:" + file_sha256(checkpoint) != trial["checkpoint_sha256"]:
            raise RuntimeError(f"Selected checkpoint hash changed: {checkpoint}")
        if "sha256:" + file_sha256(normalizer_path) != trial["normalizer_sha256"]:
            raise RuntimeError(f"Selected normalizer hash changed: {normalizer_path}")
        normalizer = FeatureNormalizer.load(str(normalizer_path))
        model = RankerMLP(
            input_dim=input_dim,
            hidden_dims=[int(config["hidden_width"])],
            dropout=float(config["dropout"]),
            use_batch_norm=False,
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        results = []
        with torch.no_grad():
            for (_, candidates), matrix in zip(payload["eval_pwc"], payload["eval_features"]):
                features = transform_feature_matrix(matrix, transform, columns)
                normalized = normalizer.transform(features)
                scores = model.score(
                    torch.as_tensor(normalized, dtype=torch.float32, device=device)
                ).detach().cpu().numpy()
                order = np.argsort(-scores, kind="stable")
                results.append(([candidates[index] for index in order], scores[order]))
        prediction_dir.mkdir(parents=True, exist_ok=True)
        evaluation = evaluate_reranking(
            products_with_candidates=payload["eval_pwc"],
            ground_truths=payload["eval_ground_truths"],
            reranker=None,
            ks=[1, 3, 5, 10],
            output_csv=str(prediction_targets[seed]),
            precomputed_reranked_results=results,
            reaction_metadata=payload["eval_metadata"],
        )
        metrics[str(seed)] = {
            "top1": evaluation.reranked_accuracy[1],
            "top3": evaluation.reranked_accuracy[3],
            "top5": evaluation.reranked_accuracy[5],
            "top10": evaluation.reranked_accuracy[10],
            "mrr": evaluation.reranked_mrr,
            "baseline_top1": evaluation.baseline_accuracy[1],
            "baseline_mrr": evaluation.baseline_mrr,
            "n_test_reactions": evaluation.n_products,
        }
    return metrics


def summarize_paired_metrics(baseline: dict, augmented: dict) -> dict:
    """Compact descriptive report; inferential analysis remains a later task."""

    expected_seeds = {str(seed) for seed in SEEDS}
    if set(baseline) != set(augmented) or set(baseline) != expected_seeds:
        raise ValueError("Metric summaries require paired seeds 42--46.")
    summary = {"baseline": {}, "augmented": {}, "paired_delta": {}}
    for metric in ("top1", "top3", "top5", "top10", "mrr"):
        baseline_values = np.asarray(
            [baseline[str(seed)][metric] for seed in SEEDS], dtype=np.float64
        )
        augmented_values = np.asarray(
            [augmented[str(seed)][metric] for seed in SEEDS], dtype=np.float64
        )
        for label, values in (
            ("baseline", baseline_values),
            ("augmented", augmented_values),
        ):
            summary[label][metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }
        differences = augmented_values - baseline_values
        summary["paired_delta"][metric] = {
            "mean": float(np.mean(differences)),
            "sample_std": float(np.std(differences, ddof=1)),
            "per_seed": {
                str(seed): float(value) for seed, value in zip(SEEDS, differences)
            },
        }
    return summary


def run_evaluate_test(args) -> None:
    result_dir = require_clean_evaluation_result_dir(args.result_dir)
    freeze, payload = load_test_cache_after_freeze(
        args.selection_freeze, args.train_test_cache
    )
    device = resolve_device(args.device)
    output_root = Path(args.output_root).resolve()
    transform = freeze["selected_prior_transform"]
    baseline = _score_test_arm(
        payload,
        freeze["selected_baseline"],
        transform,
        "baseline",
        output_root,
        result_dir / "predictions",
        device,
    )
    augmented = _score_test_arm(
        payload,
        freeze["selected_augmented"],
        transform,
        "augmented",
        output_root,
        result_dir / "predictions",
        device,
    )
    manifest = {
        "manifest_kind": "post_selection_test_evaluation",
        "protocol_id": PROTOCOL_ID,
        "comparator": "validation-tuned prior+2D baseline",
        "single_intended_change": "add three Uni-Mol-derived pair-level scalars",
        "selection_freeze": file_fingerprint(args.selection_freeze),
        "selection_fingerprint": freeze["selection_fingerprint"],
        "representation_provenance": freeze["representation_provenance"],
        "train_test_cache": file_fingerprint(args.train_test_cache),
        "selected_prior_transform": transform,
        "per_seed_metrics": {"baseline": baseline, "augmented": augmented},
        "descriptive_summary": summarize_paired_metrics(baseline, augmented),
        "test_partition_loaded_only_after_selection_freeze": True,
        "environment": environment_record(device),
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "git_commit": git_commit(),
        "created_at_utc": utc_now(),
    }
    immutable_json_dump(
        manifest, result_dir / "manifest.json", "post-selection test manifest"
    )
    print(json.dumps(manifest, indent=2))


def run_evaluate_capacity_test(args) -> None:
    result_dir = require_clean_evaluation_result_dir(args.result_dir)
    freeze, payload = load_capacity_test_cache_after_freeze(
        args.capacity_freeze, args.train_test_cache
    )
    device = resolve_device(args.device)
    output_root = Path(args.output_root).resolve()
    transform = freeze["prior_transform"]
    baseline = _score_test_arm(
        payload,
        freeze["selected_baseline"],
        transform,
        "baseline",
        output_root,
        result_dir / "predictions",
        device,
    )
    augmented = _score_test_arm(
        payload,
        freeze["selected_augmented"],
        transform,
        "augmented",
        output_root,
        result_dir / "predictions",
        device,
    )
    manifest = {
        "manifest_kind": "capacity_post_selection_test_evaluation",
        "protocol_id": PROTOCOL_ID,
        "control_id": CAPACITY_CONTROL_ID,
        "comparator": freeze["comparator"],
        "single_intended_change": freeze["single_intended_change"],
        "capacity_freeze": file_fingerprint(args.capacity_freeze),
        "capacity_freeze_fingerprint": freeze["capacity_freeze_fingerprint"],
        "representation_provenance": freeze["representation_provenance"],
        "train_test_cache": file_fingerprint(args.train_test_cache),
        "prior_transform": transform,
        "non_width_settings": freeze["non_width_settings"],
        "capacity_assertion": freeze["capacity_assertion"],
        "per_seed_metrics": {"baseline": baseline, "augmented": augmented},
        "descriptive_summary": summarize_paired_metrics(baseline, augmented),
        "test_partition_loaded_only_after_capacity_freeze": True,
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "environment": environment_record(device),
        "git_commit": git_commit(),
        "created_at_utc": utc_now(),
    }
    immutable_json_dump(
        manifest, result_dir / "manifest.json", "D-CAPACITY test manifest"
    )
    print(json.dumps(manifest, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a train+valid-only bundle.")
    prepare.add_argument("--train-test-cache", required=True)
    prepare.add_argument("--validation-cache", required=True)
    prepare.add_argument("--conformer-seed", required=True, type=int)
    prepare.add_argument("--output", required=True)

    search = subparsers.add_parser("search", help="Run/resume one D1 search shard.")
    search.add_argument("--selection-bundle", required=True)
    search.add_argument("--output-root", required=True)
    search.add_argument("--arm", required=True, choices=["baseline", "augmented"])
    search.add_argument("--prior-transform", required=True, choices=PRIOR_TRANSFORMS)
    search.add_argument("--prior-freeze")
    search.add_argument("--shard-index", type=int, default=0)
    search.add_argument("--shard-count", type=int, default=1)
    search.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    search.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    search.add_argument(
        "--compact-progress",
        action="store_true",
        help="Show one concise trial bar and suppress the full final manifest dump.",
    )
    search.add_argument(
        "--stop-after-epoch",
        type=float,
        help="Unix epoch deadline; pause safely between completed trials.",
    )
    search.add_argument(
        "--stop-margin-seconds",
        type=float,
        default=600.0,
        help="Do not start another trial inside this deadline safety margin.",
    )

    prior = subparsers.add_parser("select-prior", help="Freeze D2 from baseline only.")
    prior.add_argument("--selection-bundle", required=True)
    prior.add_argument("--output-root", required=True)
    prior.add_argument("--output", required=True)

    freeze = subparsers.add_parser("freeze-selection", help="Freeze D1 arm configs.")
    freeze.add_argument("--selection-bundle", required=True)
    freeze.add_argument("--prior-freeze", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--output", required=True)

    evaluate = subparsers.add_parser(
        "evaluate-test", help="Evaluate frozen checkpoints on official test."
    )
    evaluate.add_argument("--selection-freeze", required=True)
    evaluate.add_argument("--train-test-cache", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    capacity_plan = subparsers.add_parser(
        "prepare-capacity",
        help="Freeze explicitly chosen non-width settings for D-CAPACITY.",
    )
    capacity_plan.add_argument("--selection-bundle", required=True)
    capacity_plan.add_argument("--prior-freeze", required=True)
    capacity_plan.add_argument("--dropout", required=True, type=float)
    capacity_plan.add_argument("--learning-rate", required=True, type=float)
    capacity_plan.add_argument("--margin", required=True, type=float)
    capacity_plan.add_argument("--decision-note", required=True)
    capacity_plan.add_argument("--output", required=True)

    capacity_run = subparsers.add_parser(
        "run-capacity", help="Run/resume one paired-budget D-CAPACITY arm."
    )
    capacity_run.add_argument("--selection-bundle", required=True)
    capacity_run.add_argument("--capacity-plan", required=True)
    capacity_run.add_argument("--output-root", required=True)
    capacity_run.add_argument("--arm", required=True, choices=["baseline", "augmented"])
    capacity_run.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    capacity_run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    capacity_freeze = subparsers.add_parser(
        "freeze-capacity", help="Freeze paired D-CAPACITY validation checkpoints."
    )
    capacity_freeze.add_argument("--selection-bundle", required=True)
    capacity_freeze.add_argument("--capacity-plan", required=True)
    capacity_freeze.add_argument("--output-root", required=True)
    capacity_freeze.add_argument("--output", required=True)

    capacity_evaluate = subparsers.add_parser(
        "evaluate-capacity-test",
        help="Evaluate D-CAPACITY only after its dedicated freeze.",
    )
    capacity_evaluate.add_argument("--capacity-freeze", required=True)
    capacity_evaluate.add_argument("--train-test-cache", required=True)
    capacity_evaluate.add_argument("--output-root", required=True)
    capacity_evaluate.add_argument("--result-dir", required=True)
    capacity_evaluate.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )

    capacity = subparsers.add_parser(
        "check-capacity", help="Assert the prespecified D3 parameter counts."
    )
    capacity.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        bundle = prepare_selection_bundle(
            args.train_test_cache,
            args.validation_cache,
            args.output,
            args.conformer_seed,
        )
        print(
            json.dumps(
                {
                    "output": str(Path(args.output).resolve()),
                    "protocol_id": bundle["protocol_id"],
                    "test_fields_discarded": bundle["audit"]["test_fields_discarded"],
                },
                indent=2,
            )
        )
    elif args.command == "search":
        run_search(args)
    elif args.command == "select-prior":
        run_select_prior(args)
    elif args.command == "freeze-selection":
        run_freeze_selection(args)
    elif args.command == "evaluate-test":
        run_evaluate_test(args)
    elif args.command == "prepare-capacity":
        run_prepare_capacity(args)
    elif args.command == "run-capacity":
        run_capacity(args)
    elif args.command == "freeze-capacity":
        run_freeze_capacity(args)
    elif args.command == "evaluate-capacity-test":
        run_evaluate_capacity_test(args)
    elif args.command == "check-capacity":
        if args.output:
            require_immutable_output_absent(args.output, "D-CAPACITY assertion record")
        record = {
            "protocol_id": PROTOCOL_ID,
            "control": "D3 capacity match",
            "comparator": "seven-input width-32 augmented RankerMLP",
            "single_intended_change": "four-input baseline hidden width 32 to 48",
            "assertion": assert_d3_capacity_match(),
            "fixed_training_config": FIXED_TUNING_CONFIG,
            "git_commit": git_commit(),
            "environment": environment_record("cpu"),
            "created_at_utc": utc_now(),
        }
        if args.output:
            immutable_json_dump(
                record, args.output, "D-CAPACITY assertion record"
            )
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
