#!/usr/bin/env python
"""Matched post-primary ranking for Morgan/GROVER seven-column controls.

The primary D1 freeze supplies the prior transform and augmented MLP
hyperparameters.  A control receives no new grid search.  It is retrained on
its own train features with official-validation early stopping, then frozen
before either official-test cache is opened.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rerank.revision_tuning import (
    BASELINE_COLUMNS,
    MAX_EPOCHS,
    MIN_IMPROVEMENT,
    PATIENCE,
    PROTOCOL_ID,
    SEEDS,
    GridConfig,
    config_fingerprint,
    file_fingerprint,
    file_sha256,
    load_selection_bundle,
    train_validation_trial,
    transform_selection_cache,
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


CONTROL_PROTOCOL_ID = "cap10-tuned-encoder-control-v1"


def _candidate_identity(candidates) -> list[tuple[str, float]]:
    return [
        (str(item.get("canonical_smiles", item.get("smiles", ""))), float(item["prior"]))
        for item in candidates
    ]


def _assert_matrix_baseline_equal(left, right, label: str) -> None:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 7:
        raise RuntimeError(f"{label}: seven-column feature shapes differ.")
    if not np.array_equal(left[:, BASELINE_COLUMNS], right[:, BASELINE_COLUMNS]):
        raise RuntimeError(f"{label}: frozen prior+2D baseline columns differ.")


def assert_selection_baseline_identity(primary: dict, control: dict) -> dict:
    """Require identical train/valid ordering, candidates and four 2D columns."""
    if primary.get("representation_provenance", {}).get("kind") != "indexed_conformer":
        raise RuntimeError("Primary bundle is not an indexed Uni-Mol conformer.")
    if control.get("representation_provenance", {}).get("kind") != "encoder_control_without_conformer":
        raise RuntimeError("Control bundle lacks no-conformer encoder provenance.")
    primary_train = primary["train_products"]
    control_train = control["train_products"]
    if len(primary_train) != len(control_train):
        raise RuntimeError("Primary/control training product counts differ.")
    for index, (left, right) in enumerate(zip(primary_train, control_train)):
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
        _assert_matrix_baseline_equal(left["features"], right["features"], f"train[{index}]")

    left_valid = primary["validation_payload"]
    right_valid = control["validation_payload"]
    for key in ("eval_ground_truths", "eval_metadata"):
        if left_valid[key] != right_valid[key]:
            raise RuntimeError(f"Official-validation {key} differs.")
    if len(left_valid["eval_pwc"]) != len(right_valid["eval_pwc"]):
        raise RuntimeError("Official-validation reaction counts differ.")
    for index, ((lp, lc), (rp, rc), lf, rf) in enumerate(
        zip(
            left_valid["eval_pwc"], right_valid["eval_pwc"],
            left_valid["eval_features"], right_valid["eval_features"],
        )
    ):
        if str(lp) != str(rp) or _candidate_identity(lc) != _candidate_identity(rc):
            raise RuntimeError(f"Validation candidate pairing differs at index {index}.")
        _assert_matrix_baseline_equal(lf, rf, f"valid[{index}]")
    return {
        "train_products": len(primary_train),
        "validation_reactions": len(left_valid["eval_pwc"]),
        "baseline_columns_zero_based": list(BASELINE_COLUMNS),
        "status": "exact",
    }


def assert_test_baseline_identity(primary_payload: dict, control_payload: dict) -> dict:
    for key in ("eval_ground_truths", "eval_metadata"):
        if primary_payload[key] != control_payload[key]:
            raise RuntimeError(f"Official-test {key} differs.")
    if len(primary_payload["eval_pwc"]) != len(control_payload["eval_pwc"]):
        raise RuntimeError("Official-test reaction counts differ.")
    for index, ((lp, lc), (rp, rc), lf, rf) in enumerate(
        zip(
            primary_payload["eval_pwc"], control_payload["eval_pwc"],
            primary_payload["eval_features"], control_payload["eval_features"],
        )
    ):
        if str(lp) != str(rp) or _candidate_identity(lc) != _candidate_identity(rc):
            raise RuntimeError(f"Test candidate pairing differs at index {index}.")
        _assert_matrix_baseline_equal(lf, rf, f"test[{index}]")
    return {"test_reactions": len(primary_payload["eval_pwc"]), "status": "exact"}


def _selected_primary_config(freeze: dict) -> GridConfig:
    body = freeze.get("selected_augmented", {}).get("config")
    if not isinstance(body, dict):
        raise RuntimeError("Primary freeze has no selected augmented configuration.")
    config = GridConfig(**body)
    if config_fingerprint(config) != freeze["selected_augmented"].get("config_fingerprint"):
        raise RuntimeError("Primary selected configuration fingerprint is invalid.")
    return config


def _trial_path(root: Path, seed: int) -> Path:
    return root / "validation" / f"seed_{seed}" / "trial.json"


def run_fit(args) -> None:
    primary_freeze = validate_selection_freeze(args.primary_selection_freeze)
    primary_bundle = load_selection_bundle(args.primary_selection_bundle)
    if "sha256:" + file_sha256(args.primary_selection_bundle) != primary_freeze["selection_bundle_sha256"]:
        raise RuntimeError("Primary selection bundle does not belong to its freeze.")
    control_bundle = load_selection_bundle(args.control_selection_bundle)
    pairing = assert_selection_baseline_identity(primary_bundle, control_bundle)
    config = _selected_primary_config(primary_freeze)
    transform = primary_freeze["selected_prior_transform"]
    cache = transform_selection_cache(control_bundle, "augmented", transform)
    root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    for seed in SEEDS:
        result_path = _trial_path(root, seed)
        if result_path.exists():
            _validate_trial(
                root, seed, config,
                "sha256:" + file_sha256(args.control_selection_bundle),
            )
            continue
        run_dir = result_path.parent
        checkpoint = run_dir / "best_checkpoint.pt"
        normalizer = run_dir / "normalizer.npz"
        if checkpoint.exists() or normalizer.exists():
            raise FileExistsError(
                f"Partial encoder-control trial preserved for inspection: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        trained = train_validation_trial(
            cache, config, seed, device, checkpoint, normalizer,
            max_epochs=MAX_EPOCHS, patience=PATIENCE,
            min_improvement=MIN_IMPROVEMENT,
        )
        trial = {
            "record_kind": "encoder_control_validation_trial",
            "protocol_id": CONTROL_PROTOCOL_ID,
            "primary_protocol_id": PROTOCOL_ID,
            "seed": seed,
            "config": asdict(config),
            "config_fingerprint": config_fingerprint(config),
            "prior_transform": transform,
            "checkpoint_relpath": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": "sha256:" + file_sha256(checkpoint),
            "normalizer_relpath": normalizer.relative_to(root).as_posix(),
            "normalizer_sha256": "sha256:" + file_sha256(normalizer),
            "primary_selection_fingerprint": primary_freeze["selection_fingerprint"],
            "control_selection_bundle_sha256": "sha256:" + file_sha256(args.control_selection_bundle),
            "representation_provenance": control_bundle["representation_provenance"],
            "baseline_pairing_gate": pairing,
            "test_partition_loaded": False,
            "runtime_seconds": time.perf_counter() - started,
            **trained,
        }
        immutable_json_dump(trial, result_path, "encoder-control validation trial")


def _validate_trial(root: Path, seed: int, config: GridConfig, bundle_sha: str) -> dict:
    path = _trial_path(root, seed)
    trial = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "record_kind": "encoder_control_validation_trial",
        "protocol_id": CONTROL_PROTOCOL_ID,
        "seed": seed,
        "config_fingerprint": config_fingerprint(config),
        "control_selection_bundle_sha256": bundle_sha,
        "test_partition_loaded": False,
    }
    if any(trial.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Invalid encoder-control validation trial: {path}")
    for rel_key, hash_key in (
        ("checkpoint_relpath", "checkpoint_sha256"),
        ("normalizer_relpath", "normalizer_sha256"),
    ):
        artifact = (root / trial[rel_key]).resolve()
        if not artifact.is_relative_to(root) or "sha256:" + file_sha256(artifact) != trial[hash_key]:
            raise RuntimeError(f"Encoder-control artifact checksum failed: {artifact}")
    return trial


def run_freeze(args) -> None:
    primary = validate_selection_freeze(args.primary_selection_freeze)
    config = _selected_primary_config(primary)
    bundle = load_selection_bundle(args.control_selection_bundle)
    root = Path(args.output_root).resolve()
    bundle_sha = "sha256:" + file_sha256(args.control_selection_bundle)
    trials = {str(seed): _validate_trial(root, seed, config, bundle_sha) for seed in SEEDS}
    record = {
        "record_kind": "encoder_control_selection_freeze",
        "protocol_id": CONTROL_PROTOCOL_ID,
        "comparator": "frozen cap10-tuned-v1 four-input 2D baseline",
        "single_intended_change": "replace Uni-Mol atom states with the declared encoder control",
        "primary_selection_fingerprint": primary["selection_fingerprint"],
        "primary_selected_config": asdict(config),
        "primary_selected_config_fingerprint": config_fingerprint(config),
        "prior_transform": primary["selected_prior_transform"],
        "control_selection_bundle": file_fingerprint(args.control_selection_bundle),
        "retained_control_train_test_cache_sha256": bundle["input_fingerprints"]["retained_train_test_cache"]["sha256"],
        "representation_provenance": bundle["representation_provenance"],
        "selected_control": {"trials": trials},
        "seeds": list(SEEDS),
        "test_partition_loaded": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["freeze_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "encoder-control selection freeze")


def _validate_control_freeze(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = record.pop("freeze_fingerprint", None)
    if (
        record.get("record_kind") != "encoder_control_selection_freeze"
        or record.get("protocol_id") != CONTROL_PROTOCOL_ID
        or record.get("test_partition_loaded") is not False
        or supplied != canonical_fingerprint(record)
    ):
        raise PermissionError("Invalid encoder-control freeze; official test remains locked.")
    record["freeze_fingerprint"] = supplied
    return record


def _load_test_payload(path: str | Path, expected_sha: str) -> dict:
    if file_sha256(path) != expected_sha:
        raise PermissionError("Official-test cache fingerprint differs from its freeze.")
    with open(path, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("Evaluation cache is not exclusively official test.")
    return payload


def run_evaluate(args) -> None:
    freeze = _validate_control_freeze(args.control_freeze)
    primary = validate_selection_freeze(args.primary_selection_freeze)
    if freeze["primary_selection_fingerprint"] != primary["selection_fingerprint"]:
        raise PermissionError("Control and primary freezes do not match.")
    # Both test-bearing files are opened only after both freezes validate.
    control_payload = _load_test_payload(
        args.control_train_test_cache,
        freeze["retained_control_train_test_cache_sha256"],
    )
    primary_payload = _load_test_payload(
        args.primary_train_test_cache,
        primary["retained_train_test_cache_sha256"],
    )
    pairing = assert_test_baseline_identity(primary_payload, control_payload)
    result_dir = require_clean_evaluation_result_dir(args.result_dir)
    root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    metrics = _score_test_arm(
        control_payload, freeze["selected_control"], freeze["prior_transform"],
        "augmented", root, result_dir / "predictions", device,
    )
    manifest = {
        "manifest_kind": "encoder_control_post_freeze_test_evaluation",
        "protocol_id": CONTROL_PROTOCOL_ID,
        "primary_selection_fingerprint": primary["selection_fingerprint"],
        "control_freeze": file_fingerprint(args.control_freeze),
        "representation_provenance": freeze["representation_provenance"],
        "baseline_pairing_gate": pairing,
        "per_seed_metrics": metrics,
        "primary_2d_predictions_are_the_frozen_comparator": True,
        "test_partition_loaded_only_after_both_freezes": True,
        "environment": environment_record(device),
        "git_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    immutable_json_dump(manifest, result_dir / "manifest.json", "encoder-control test manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-validation")
    fit.add_argument("--primary-selection-freeze", required=True)
    fit.add_argument("--primary-selection-bundle", required=True)
    fit.add_argument("--control-selection-bundle", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--primary-selection-freeze", required=True)
    freeze.add_argument("--control-selection-bundle", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-test")
    evaluate.add_argument("--primary-selection-freeze", required=True)
    evaluate.add_argument("--control-freeze", required=True)
    evaluate.add_argument("--primary-train-test-cache", required=True)
    evaluate.add_argument("--control-train-test-cache", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "fit-validation":
        run_fit(args)
    elif args.command == "freeze":
        run_freeze(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
