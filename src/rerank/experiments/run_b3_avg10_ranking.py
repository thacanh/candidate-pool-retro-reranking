#!/usr/bin/env python
"""Fit and evaluate the prespecified B3 ten-conformer scalar average.

The B3 feature artifacts are already frozen.  This runner reuses the augmented
configuration selected by D1 on the single-conformer representation; B3 gets
no new grid search.  Training and official-validation early stopping finish
and are frozen before this command is allowed to open either official-test
cache or the primary post-test manifest.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from rerank.analysis.analyze_encoder_attribution import clustered_intervals
from rerank.revision_tuning import (
    BASELINE_COLUMNS,
    MAX_EPOCHS,
    MIN_IMPROVEMENT,
    PATIENCE,
    PROTOCOL_ID as PRIMARY_PROTOCOL_ID,
    SEEDS,
    SELECTION_BUNDLE_SCHEMA,
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
    canonical_fingerprint,
    environment_record,
    git_commit,
    immutable_json_dump,
    require_clean_evaluation_result_dir,
    resolve_device,
    validate_selection_freeze,
)


PROTOCOL_ID = "cap10-tuned-b3-avg10-v1"
FEATURE_PROTOCOL_ID = "cap10-conformer-avg10-features-v1"
CONFORMER_SEEDS = tuple(range(42, 52))
COMPARATOR = "single-conformer seed-42 augmented cap10-tuned-v1"
SINGLE_INTENDED_CHANGE = (
    "replace each of the three seed-42 Uni-Mol-derived scalars with its "
    "arithmetic mean across conformer seeds 42--51"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_pickle(path: str | Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _candidate_identity(candidates) -> list[tuple[str, float]]:
    return [
        (
            str(item.get("canonical_smiles", item.get("smiles", ""))),
            float(item["prior"]),
        )
        for item in candidates
    ]


def _assert_baseline_columns(left, right, label: str) -> None:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 7:
        raise RuntimeError(f"{label}: seven-column feature shapes differ.")
    if not np.array_equal(left[:, BASELINE_COLUMNS], right[:, BASELINE_COLUMNS]):
        raise RuntimeError(f"{label}: prior+2D columns differ.")


def assert_selection_pairing(primary: Mapping, b3: Mapping) -> dict:
    if b3.get("representation_provenance", {}).get("kind") != (
        "multi_conformer_scalar_average"
    ):
        raise RuntimeError("B3 selection bundle lacks multi-conformer provenance.")
    primary_train = primary["train_products"]
    b3_train = b3["train_products"]
    if len(primary_train) != len(b3_train):
        raise RuntimeError("Primary/B3 training product counts differ.")
    for index, (left, right) in enumerate(zip(primary_train, b3_train)):
        left_key = str(left.get("product_key", left.get("product_smiles", "")))
        right_key = str(right.get("product_key", right.get("product_smiles", "")))
        if left_key != right_key:
            raise RuntimeError(f"Training product order differs at index {index}.")
        if _candidate_identity(left.get("candidates", ())) != _candidate_identity(
            right.get("candidates", ())
        ):
            raise RuntimeError(f"Training candidate order differs at index {index}.")
        if list(left["positive_indices"]) != list(right["positive_indices"]):
            raise RuntimeError(f"Training positives differ at index {index}.")
        if list(left["negative_indices"]) != list(right["negative_indices"]):
            raise RuntimeError(f"Training negatives differ at index {index}.")
        _assert_baseline_columns(left["features"], right["features"], f"train[{index}]")

    left_valid = primary["validation_payload"]
    right_valid = b3["validation_payload"]
    for key in ("eval_ground_truths", "eval_metadata"):
        if left_valid[key] != right_valid[key]:
            raise RuntimeError(f"Official-validation {key} differs.")
    if len(left_valid["eval_pwc"]) != len(right_valid["eval_pwc"]):
        raise RuntimeError("Official-validation reaction counts differ.")
    for index, ((lp, lc), (rp, rc), lf, rf) in enumerate(
        zip(
            left_valid["eval_pwc"],
            right_valid["eval_pwc"],
            left_valid["eval_features"],
            right_valid["eval_features"],
        )
    ):
        if str(lp) != str(rp) or _candidate_identity(lc) != _candidate_identity(rc):
            raise RuntimeError(f"Validation candidate pairing differs at index {index}.")
        _assert_baseline_columns(lf, rf, f"valid[{index}]")
    return {
        "status": "exact",
        "train_products": len(primary_train),
        "validation_reactions": len(left_valid["eval_pwc"]),
        "baseline_columns_zero_based": list(BASELINE_COLUMNS),
    }


def assert_test_pairing(primary_payload: Mapping, b3_payload: Mapping) -> dict:
    for key in ("eval_ground_truths", "eval_metadata"):
        if primary_payload[key] != b3_payload[key]:
            raise RuntimeError(f"Official-test {key} differs.")
    if len(primary_payload["eval_pwc"]) != len(b3_payload["eval_pwc"]):
        raise RuntimeError("Official-test reaction counts differ.")
    for index, ((lp, lc), (rp, rc), lf, rf) in enumerate(
        zip(
            primary_payload["eval_pwc"],
            b3_payload["eval_pwc"],
            primary_payload["eval_features"],
            b3_payload["eval_features"],
        )
    ):
        if str(lp) != str(rp) or _candidate_identity(lc) != _candidate_identity(rc):
            raise RuntimeError(f"Test candidate pairing differs at index {index}.")
        _assert_baseline_columns(lf, rf, f"test[{index}]")
    return {"status": "exact", "test_reactions": len(primary_payload["eval_pwc"])}


def _immutable_pickle(payload: Mapping, path: str | Path, label: str) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite existing or partial {label}: {target}")
    atomic_pickle_dump(payload, target)


def run_prepare(args) -> None:
    main_blob = _load_pickle(args.train_test_cache)
    validation_blob = _load_pickle(args.validation_cache)
    if main_blob.get("protocol_id") != FEATURE_PROTOCOL_ID:
        raise RuntimeError("B3 train/test cache has the wrong feature protocol.")
    if validation_blob.get("protocol_id") != FEATURE_PROTOCOL_ID:
        raise RuntimeError("B3 validation cache has the wrong feature protocol.")
    if tuple(validation_blob.get("conformer_seeds", ())) != CONFORMER_SEEDS:
        raise RuntimeError("B3 validation cache does not contain seeds 42--51 exactly.")
    main = main_blob.get("payload", main_blob)
    validation = validation_blob.get("payload", validation_blob)
    if main.get("feature_mode") != "3d+prior":
        raise RuntimeError("B3 retained cache is not seven-column 3d+prior.")
    if any(str(row.get("source_split")) != "valid" for row in validation["eval_metadata"]):
        raise PermissionError("B3 selection payload is not exclusively validation.")
    bundle = {
        "selection_bundle_schema": SELECTION_BUNDLE_SCHEMA,
        "protocol_id": PRIMARY_PROTOCOL_ID,
        "experiment_protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "representation_provenance": {
            "kind": "multi_conformer_scalar_average",
            "encoder_control_has_conformer": True,
            "conformer_seed": None,
            "conformer_seeds": list(CONFORMER_SEEDS),
            "aggregation": (
                "arithmetic mean of each pair-level scalar after fragment handling"
            ),
            "atom_embeddings_averaged": False,
            "source_protocol_id": FEATURE_PROTOCOL_ID,
        },
        "seeds": list(SEEDS),
        "feature_mode": "3d+prior",
        "train_split": "train",
        "validation_split": "valid",
        "train_products": main["train_products"],
        "validation_payload": validation,
        "audit": {
            "train": main.get("audit"),
            "validation": validation_blob.get("audit"),
            "test_fields_discarded": sorted(
                key for key in main if key.startswith("eval_")
            ),
        },
        "input_fingerprints": {
            "retained_train_test_cache": file_fingerprint(args.train_test_cache),
            "official_validation_cache": file_fingerprint(args.validation_cache),
        },
    }
    validate_selection_bundle(bundle)
    _immutable_pickle(bundle, args.output, "B3 train+validation selection bundle")
    print(
        json.dumps(
            {
                "status": "prepared",
                "output": str(Path(args.output).resolve()),
                "train_products": len(bundle["train_products"]),
                "validation_reactions": len(validation["eval_pwc"]),
                "test_fields_discarded": bundle["audit"]["test_fields_discarded"],
            },
            indent=2,
        )
    )


def _selected_primary_config(freeze: Mapping) -> GridConfig:
    body = freeze.get("selected_augmented", {}).get("config")
    if not isinstance(body, Mapping):
        raise RuntimeError("Primary freeze lacks its augmented configuration.")
    config = GridConfig(**body)
    expected = freeze["selected_augmented"].get("config_fingerprint")
    if config_fingerprint(config) != expected:
        raise RuntimeError("Primary augmented configuration fingerprint is invalid.")
    return config


def _trial_path(root: Path, seed: int) -> Path:
    return root / "validation" / f"seed_{seed}" / "trial.json"


def _validate_trial(root: Path, seed: int, config: GridConfig, bundle_sha: str) -> dict:
    path = _trial_path(root, seed)
    trial = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "record_kind": "b3_avg10_validation_trial",
        "protocol_id": PROTOCOL_ID,
        "seed": seed,
        "config_fingerprint": config_fingerprint(config),
        "selection_bundle_sha256": bundle_sha,
        "test_partition_loaded": False,
    }
    if any(trial.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Invalid B3 validation trial: {path}")
    for rel_key, hash_key in (
        ("checkpoint_relpath", "checkpoint_sha256"),
        ("normalizer_relpath", "normalizer_sha256"),
    ):
        artifact = (root / trial[rel_key]).resolve()
        if not artifact.is_relative_to(root):
            raise RuntimeError(f"B3 artifact escapes its output root: {artifact}")
        if "sha256:" + file_sha256(artifact) != trial[hash_key]:
            raise RuntimeError(f"B3 artifact checksum failed: {artifact}")
    return trial


def run_fit(args) -> None:
    primary_freeze = validate_selection_freeze(args.primary_selection_freeze)
    primary_bundle = load_selection_bundle(args.primary_selection_bundle)
    if "sha256:" + file_sha256(args.primary_selection_bundle) != primary_freeze.get(
        "selection_bundle_sha256"
    ):
        raise RuntimeError("Primary selection bundle does not belong to its freeze.")
    b3_bundle = load_selection_bundle(args.b3_selection_bundle)
    pairing = assert_selection_pairing(primary_bundle, b3_bundle)
    config = _selected_primary_config(primary_freeze)
    transform = primary_freeze["selected_prior_transform"]
    cache = transform_selection_cache(b3_bundle, "augmented", transform)
    root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    bundle_sha = "sha256:" + file_sha256(args.b3_selection_bundle)
    for seed in SEEDS:
        result_path = _trial_path(root, seed)
        if result_path.exists():
            _validate_trial(root, seed, config, bundle_sha)
            print(f"B3 seed {seed}: retained complete trial", flush=True)
            continue
        run_dir = result_path.parent
        checkpoint = run_dir / "best_checkpoint.pt"
        normalizer = run_dir / "normalizer.npz"
        if checkpoint.exists() or normalizer.exists():
            raise FileExistsError(f"Partial B3 trial preserved for inspection: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"B3 seed {seed}: fitting frozen config {config.index}", flush=True)
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
            "record_kind": "b3_avg10_validation_trial",
            "protocol_id": PROTOCOL_ID,
            "seed": seed,
            "config": asdict(config),
            "config_fingerprint": config_fingerprint(config),
            "prior_transform": transform,
            "checkpoint_relpath": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": "sha256:" + file_sha256(checkpoint),
            "normalizer_relpath": normalizer.relative_to(root).as_posix(),
            "normalizer_sha256": "sha256:" + file_sha256(normalizer),
            "primary_selection_fingerprint": primary_freeze["selection_fingerprint"],
            "selection_bundle_sha256": bundle_sha,
            "representation_provenance": b3_bundle["representation_provenance"],
            "pairing_gate": pairing,
            "comparator": COMPARATOR,
            "single_intended_change": SINGLE_INTENDED_CHANGE,
            "test_partition_loaded": False,
            "runtime_seconds": time.perf_counter() - started,
            **trained,
        }
        immutable_json_dump(trial, result_path, "B3 validation trial")
        print(
            f"B3 seed {seed}: complete, epoch={trained['best_epoch']}, "
            f"validation MRR={trained['best_validation_mrr']:.6f}",
            flush=True,
        )


def run_freeze(args) -> None:
    primary = validate_selection_freeze(args.primary_selection_freeze)
    config = _selected_primary_config(primary)
    bundle = load_selection_bundle(args.b3_selection_bundle)
    root = Path(args.output_root).resolve()
    bundle_sha = "sha256:" + file_sha256(args.b3_selection_bundle)
    trials = {
        str(seed): _validate_trial(root, seed, config, bundle_sha) for seed in SEEDS
    }
    record = {
        "record_kind": "b3_avg10_selection_freeze",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "selection_policy": "reuse D1 augmented config without a B3 grid search",
        "primary_selection_fingerprint": primary["selection_fingerprint"],
        "primary_selected_config": asdict(config),
        "primary_selected_config_fingerprint": config_fingerprint(config),
        "prior_transform": primary["selected_prior_transform"],
        "b3_selection_bundle": file_fingerprint(args.b3_selection_bundle),
        "retained_b3_train_test_cache_sha256": bundle["input_fingerprints"]
        ["retained_train_test_cache"]["sha256"],
        "representation_provenance": bundle["representation_provenance"],
        "selected_b3": {"trials": trials},
        "seeds": list(SEEDS),
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "test_partition_loaded": False,
        "created_at_utc": utc_now(),
    }
    record["freeze_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "B3 model-selection freeze")
    print(json.dumps(record, indent=2))


def _validate_b3_freeze(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = dict(record)
    supplied = expected.pop("freeze_fingerprint", None)
    if (
        record.get("record_kind") != "b3_avg10_selection_freeze"
        or record.get("protocol_id") != PROTOCOL_ID
        or tuple(record.get("seeds", ())) != SEEDS
        or record.get("test_partition_loaded") is not False
        or supplied != canonical_fingerprint(expected)
    ):
        raise PermissionError("Invalid B3 freeze; official test remains locked.")
    return record


def _load_test_payload(path: str | Path, expected_sha: str) -> dict:
    if file_sha256(path) != expected_sha:
        raise PermissionError("Official-test cache fingerprint differs from its freeze.")
    blob = _load_pickle(path)
    payload = blob.get("payload", blob)
    required = {"eval_pwc", "eval_ground_truths", "eval_metadata", "eval_features"}
    if not required.issubset(payload):
        raise RuntimeError("Official-test cache is incomplete.")
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("Evaluation cache is not exclusively official test.")
    return payload


def _metric_summary(b3: Mapping, primary_manifest: Mapping) -> dict:
    primary_aug = primary_manifest["per_seed_metrics"]["augmented"]
    primary_base = primary_manifest["per_seed_metrics"]["baseline"]
    result = {
        "b3_avg10": {},
        "primary_single_conformer": {},
        "primary_2d": {},
        "b3_minus_single_conformer": {},
        "b3_minus_2d": {},
    }
    for metric in ("top1", "top3", "top5", "top10", "mrr"):
        values = {
            "b3": np.asarray([b3[str(seed)][metric] for seed in SEEDS]),
            "single": np.asarray(
                [primary_aug[str(seed)][metric] for seed in SEEDS]
            ),
            "base": np.asarray(
                [primary_base[str(seed)][metric] for seed in SEEDS]
            ),
        }
        for label, source in (
            ("b3_avg10", values["b3"]),
            ("primary_single_conformer", values["single"]),
            ("primary_2d", values["base"]),
        ):
            result[label][metric] = {
                "mean": float(np.mean(source)),
                "sample_std": float(np.std(source, ddof=1)),
            }
        for label, delta in (
            ("b3_minus_single_conformer", values["b3"] - values["single"]),
            ("b3_minus_2d", values["b3"] - values["base"]),
        ):
            result[label][metric] = {
                "mean": float(np.mean(delta)),
                "sample_std": float(np.std(delta, ddof=1)),
                "per_seed": {
                    str(seed): float(value) for seed, value in zip(SEEDS, delta)
                },
            }
    return result


def run_evaluate(args) -> None:
    freeze = _validate_b3_freeze(args.b3_freeze)
    primary_freeze = validate_selection_freeze(args.primary_selection_freeze)
    if freeze["primary_selection_fingerprint"] != primary_freeze["selection_fingerprint"]:
        raise PermissionError("B3 and primary selection freezes do not match.")
    # Official-test artifacts are opened only after both selection freezes pass.
    b3_payload = _load_test_payload(
        args.b3_train_test_cache, freeze["retained_b3_train_test_cache_sha256"]
    )
    primary_payload = _load_test_payload(
        args.primary_train_test_cache,
        primary_freeze["retained_train_test_cache_sha256"],
    )
    pairing = assert_test_pairing(primary_payload, b3_payload)
    result_dir = require_clean_evaluation_result_dir(args.result_dir)
    root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    metrics = _score_test_arm(
        b3_payload,
        freeze["selected_b3"],
        freeze["prior_transform"],
        "augmented",
        root,
        result_dir / "predictions",
        device,
    )
    primary_manifest = json.loads(
        Path(args.primary_test_manifest).read_text(encoding="utf-8")
    )
    if (
        primary_manifest.get("selection_fingerprint")
        != primary_freeze["selection_fingerprint"]
        or primary_manifest.get("test_partition_loaded_only_after_selection_freeze")
        is not True
    ):
        raise PermissionError("Primary post-freeze test manifest is invalid.")
    manifest = {
        "manifest_kind": "b3_avg10_post_freeze_test_evaluation",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "b3_freeze": file_fingerprint(args.b3_freeze),
        "primary_selection_fingerprint": primary_freeze["selection_fingerprint"],
        "primary_test_manifest": file_fingerprint(args.primary_test_manifest),
        "b3_train_test_cache": file_fingerprint(args.b3_train_test_cache),
        "pairing_gate": pairing,
        "per_seed_metrics": metrics,
        "descriptive_summary": _metric_summary(metrics, primary_manifest),
        "representation_provenance": freeze["representation_provenance"],
        "test_partition_loaded_only_after_b3_and_primary_freezes": True,
        "environment": environment_record(device),
        "git_commit": git_commit(),
        "created_at_utc": utc_now(),
    }
    immutable_json_dump(manifest, result_dir / "manifest.json", "B3 test manifest")
    print(json.dumps(manifest["descriptive_summary"], indent=2))


def _prediction_matrix(
    result_root: Path,
    prefix: str,
    manifest_metrics: Mapping,
    reference: pd.DataFrame | None = None,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict]]:
    values = {"top1": [], "mrr": []}
    pairing = None
    files = []
    columns = ["reaction_id", "product_smiles", "source_split", "ground_truth"]
    for seed in SEEDS:
        path = result_root / "predictions" / f"{prefix}_seed_{seed}.csv"
        frame = pd.read_csv(path)
        if len(frame) != 3985 or not set(columns).issubset(frame):
            raise RuntimeError(f"B3 analysis prediction shape/schema failed: {path}")
        current = frame.loc[:, columns].astype(str).reset_index(drop=True)
        if pairing is None:
            pairing = current
        elif not pairing.equals(current):
            raise RuntimeError(f"Prediction pairing differs within {result_root}: {path}")
        if reference is not None and not reference.equals(current):
            raise RuntimeError(f"Prediction pairing differs from primary: {path}")
        top1 = frame["reranked_hit@1"].to_numpy(dtype=np.float64)
        mrr = frame["reranked_rr"].to_numpy(dtype=np.float64)
        for metric, array in (("top1", top1), ("mrr", mrr)):
            expected = float(manifest_metrics[str(seed)][metric])
            if not np.isclose(array.mean(), expected, rtol=0.0, atol=1e-12):
                raise RuntimeError(f"Prediction/manifest mismatch: {path}, {metric}")
            values[metric].append(array)
        files.append(file_fingerprint(path))
    assert pairing is not None
    return (
        {metric: np.stack(rows, axis=0) for metric, rows in values.items()},
        pairing,
        files,
    )


def run_analyze(args) -> None:
    primary_root = Path(args.primary_result_root).resolve()
    b3_root = Path(args.b3_result_root).resolve()
    primary_manifest = json.loads(
        (primary_root / "manifest.json").read_text(encoding="utf-8")
    )
    b3_manifest = json.loads((b3_root / "manifest.json").read_text(encoding="utf-8"))
    if primary_manifest.get("test_partition_loaded_only_after_selection_freeze") is not True:
        raise PermissionError("Primary post-freeze result is invalid.")
    if b3_manifest.get("test_partition_loaded_only_after_b3_and_primary_freezes") is not True:
        raise PermissionError("B3 post-freeze result is invalid.")
    baseline, pairing, baseline_files = _prediction_matrix(
        primary_root,
        "baseline",
        primary_manifest["per_seed_metrics"]["baseline"],
    )
    single, _, single_files = _prediction_matrix(
        primary_root,
        "augmented",
        primary_manifest["per_seed_metrics"]["augmented"],
        pairing,
    )
    b3, _, b3_files = _prediction_matrix(
        b3_root,
        "augmented",
        b3_manifest["per_seed_metrics"],
        pairing,
    )
    clusters = pairing["product_smiles"].to_numpy(dtype=object)
    comparisons = {}
    for label, differences in (
        ("b3_minus_single_conformer", {m: b3[m] - single[m] for m in b3}),
        ("b3_minus_2d", {m: b3[m] - baseline[m] for m in b3}),
    ):
        comparisons[label] = {
            metric: clustered_intervals(
                array,
                clusters,
                n_bootstrap=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            for metric, array in differences.items()
        }
    record = {
        "record_kind": "b3_avg10_clustered_inference",
        "protocol_id": PROTOCOL_ID,
        "comparator": COMPARATOR,
        "single_intended_change": SINGLE_INTENDED_CHANGE,
        "method": "paired canonical-product-clustered bootstrap after seed averaging",
        "comparisons": comparisons,
        "primary_manifest": file_fingerprint(primary_root / "manifest.json"),
        "b3_manifest": file_fingerprint(b3_root / "manifest.json"),
        "prediction_files": {
            "primary_2d": baseline_files,
            "primary_single_conformer": single_files,
            "b3_avg10": b3_files,
        },
        "created_at_utc": utc_now(),
    }
    immutable_json_dump(record, args.output, "B3 clustered-inference record")
    print(json.dumps(record["comparisons"], indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--train-test-cache", required=True)
    prepare.add_argument("--validation-cache", required=True)
    prepare.add_argument("--output", required=True)
    fit = sub.add_parser("fit-validation")
    fit.add_argument("--primary-selection-freeze", required=True)
    fit.add_argument("--primary-selection-bundle", required=True)
    fit.add_argument("--b3-selection-bundle", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--primary-selection-freeze", required=True)
    freeze.add_argument("--b3-selection-bundle", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-test")
    evaluate.add_argument("--primary-selection-freeze", required=True)
    evaluate.add_argument("--b3-freeze", required=True)
    evaluate.add_argument("--primary-train-test-cache", required=True)
    evaluate.add_argument("--b3-train-test-cache", required=True)
    evaluate.add_argument("--primary-test-manifest", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--primary-result-root", required=True)
    analyze.add_argument("--b3-result-root", required=True)
    analyze.add_argument("--output", required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=10_000)
    analyze.add_argument("--bootstrap-seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare":
        run_prepare(args)
    elif args.command == "fit-validation":
        run_fit(args)
    elif args.command == "freeze":
        run_freeze(args)
    elif args.command == "evaluate-test":
        run_evaluate(args)
    else:
        run_analyze(args)


if __name__ == "__main__":
    main()
