"""Run the prespecified F2 Morgan-chirality sensitivity behind a test lock."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

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


F2_PROTOCOL_ID = "f2-morgan-chirality-v1"
MORGAN_COLUMN = 1
MORGAN_SETTINGS = {"radius": 2, "fpSize": 2048}
_GENERATORS: dict[bool, Any] = {}


def _generator(use_chirality: bool):
    if use_chirality not in _GENERATORS:
        _GENERATORS[use_chirality] = GetMorganGenerator(
            radius=2, fpSize=2048, includeChirality=use_chirality
        )
    return _GENERATORS[use_chirality]


@lru_cache(maxsize=500_000)
def _fingerprint(smiles: str, use_chirality: bool):
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        raise ValueError(f"F2 cannot parse a frozen SMILES: {smiles!r}")
    return _generator(use_chirality).GetFingerprint(molecule)


def morgan_similarity(left: str, right: str, use_chirality: bool) -> float:
    return float(
        DataStructs.TanimotoSimilarity(
            _fingerprint(str(left), use_chirality),
            _fingerprint(str(right), use_chirality),
        )
    )


def replace_morgan_column(
    product: str,
    candidates: Iterable[dict[str, Any]],
    matrix: np.ndarray,
    audit: dict[str, Any],
) -> np.ndarray:
    candidates = list(candidates)
    result = np.asarray(matrix, dtype=np.float32).copy()
    if result.ndim != 2 or result.shape[1] != 7 or len(result) != len(candidates):
        raise ValueError("F2 requires candidate-aligned seven-column features.")
    for index, candidate in enumerate(candidates):
        smiles = str(candidate.get("smiles", candidate.get("canonical_smiles", "")))
        false_value = np.float32(morgan_similarity(product, smiles, False))
        true_value = np.float32(morgan_similarity(product, smiles, True))
        difference = abs(float(false_value) - float(result[index, MORGAN_COLUMN]))
        audit["pairs_checked"] += 1
        audit["maximum_false_cache_absolute_difference"] = max(
            audit["maximum_false_cache_absolute_difference"], difference
        )
        if false_value != result[index, MORGAN_COLUMN]:
            audit["false_cache_mismatches"] += 1
        if true_value != false_value:
            audit["chirality_changed_pairs"] += 1
        result[index, MORGAN_COLUMN] = true_value
    return result


def _new_audit() -> dict[str, Any]:
    return {
        "pairs_checked": 0,
        "false_cache_mismatches": 0,
        "maximum_false_cache_absolute_difference": 0.0,
        "chirality_changed_pairs": 0,
    }


def _require_false_equivalence(audit: dict[str, Any]) -> None:
    if audit["false_cache_mismatches"]:
        raise RuntimeError(
            "F2 false-chirality recomputation differs from the frozen primary cache: "
            f"{audit['false_cache_mismatches']} pairs."
        )


def prepare_selection(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite F2 selection bundle: {output}")
    primary_freeze = validate_selection_freeze(args.primary_freeze)
    if "sha256:" + file_sha256(args.primary_selection) != primary_freeze[
        "selection_bundle_sha256"
    ]:
        raise ValueError("Primary selection bundle does not belong to its freeze.")
    bundle = load_selection_bundle(args.primary_selection)
    audit = _new_audit()
    for product in bundle["train_products"]:
        product["features"] = replace_morgan_column(
            str(product["product_smiles"]),
            product["candidates"],
            product["features"],
            audit,
        )
    validation = bundle["validation_payload"]
    validation["eval_features"] = [
        replace_morgan_column(product, candidates, matrix, audit)
        for (product, candidates), matrix in zip(
            validation["eval_pwc"], validation["eval_features"], strict=True
        )
    ]
    _require_false_equivalence(audit)
    bundle["comparator"] = "same frozen cap10 model with Morgan useChirality=False"
    bundle["single_intended_change"] = "Morgan useChirality=False to True"
    bundle["f2_protocol_id"] = F2_PROTOCOL_ID
    bundle["f2_feature_audit"] = audit
    bundle["f2_settings"] = {
        **MORGAN_SETTINGS,
        "control": {"includeChirality": False},
        "sensitivity": {"includeChirality": True},
        "changed_column_zero_based": MORGAN_COLUMN,
    }
    bundle["representation_provenance"] = {
        **bundle["representation_provenance"],
        "source_protocol_id": F2_PROTOCOL_ID,
        "base_source_protocol_id": bundle["representation_provenance"].get(
            "source_protocol_id"
        ),
    }
    bundle["input_fingerprints"] = {
        **bundle["input_fingerprints"],
        "primary_selection_bundle": file_fingerprint(args.primary_selection),
        "primary_selection_freeze": file_fingerprint(args.primary_freeze),
    }
    validate_selection_bundle(bundle)
    atomic_pickle_dump(bundle, output)
    return {
        "protocol_id": F2_PROTOCOL_ID,
        "output": file_fingerprint(output),
        "feature_audit": audit,
        "test_partition_loaded": False,
    }


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
        "record_kind": "f2_chirality_validation_trial",
        "protocol_id": F2_PROTOCOL_ID,
        "arm": arm,
        "seed": seed,
        "config_fingerprint": config_fingerprint(config),
        "selection_bundle_sha256": bundle_sha,
        "test_partition_loaded": False,
    }
    if any(trial.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Invalid F2 validation trial: {path}")
    for rel_key, hash_key in (
        ("checkpoint_relpath", "checkpoint_sha256"),
        ("normalizer_relpath", "normalizer_sha256"),
    ):
        artifact = (root / trial[rel_key]).resolve()
        if not artifact.is_relative_to(root.resolve()):
            raise ValueError("F2 model artifact escapes its output root.")
        if "sha256:" + file_sha256(artifact) != trial[hash_key]:
            raise ValueError(f"F2 model artifact checksum mismatch: {artifact}")
    return trial


def fit_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.arm not in {"baseline", "augmented"}:
        raise ValueError("F2 arm must be baseline or augmented.")
    primary = validate_selection_freeze(args.primary_freeze)
    bundle = load_selection_bundle(args.selection_bundle)
    if bundle.get("f2_protocol_id") != F2_PROTOCOL_ID:
        raise ValueError("Selection bundle is not the frozen F2 sensitivity.")
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
            raise FileExistsError(f"Partial F2 trial preserved for inspection: {run_dir}")
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
            "record_kind": "f2_chirality_validation_trial",
            "protocol_id": F2_PROTOCOL_ID,
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
            "single_intended_change": "Morgan useChirality=False to True",
            "test_partition_loaded": False,
            "runtime_seconds": time.perf_counter() - started,
            **trained,
        }
        immutable_json_dump(trial, result_path, "F2 chirality validation trial")
        completed[str(seed)] = trial
    return {"arm": args.arm, "completed_seeds": sorted(completed), "device": device}


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    primary = validate_selection_freeze(args.primary_freeze)
    bundle = load_selection_bundle(args.selection_bundle)
    if bundle.get("f2_protocol_id") != F2_PROTOCOL_ID:
        raise ValueError("Selection bundle is not F2.")
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
        "record_kind": "f2_chirality_model_freeze",
        "protocol_id": F2_PROTOCOL_ID,
        "primary_selection_fingerprint": primary["selection_fingerprint"],
        "primary_freeze": file_fingerprint(args.primary_freeze),
        "selection_bundle": file_fingerprint(args.selection_bundle),
        "retained_primary_train_test_cache_sha256": primary[
            "retained_train_test_cache_sha256"
        ],
        "selected_prior_transform": primary["selected_prior_transform"],
        "selected_baseline": selected["baseline"],
        "selected_augmented": selected["augmented"],
        "single_intended_change": "Morgan useChirality=False to True",
        "config_selection_performed_for_f2": False,
        "seeds": list(SEEDS),
        "test_partition_loaded": False,
        "complete": True,
    }
    record["freeze_fingerprint"] = "sha256:" + __import__("hashlib").sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    immutable_json_dump(record, args.output, "F2 chirality model freeze")
    return record


def _load_frozen_test_payload(freeze_record: dict[str, Any], cache_path: str) -> dict:
    expected = str(freeze_record["retained_primary_train_test_cache_sha256"])
    expected = expected.removeprefix("sha256:")
    if file_sha256(cache_path) != expected:
        raise ValueError("Primary train/test cache checksum mismatch.")
    with open(cache_path, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("F2 evaluation payload is not official test only.")
    return payload


def _primary_prediction_fingerprints(manifest_path: Path) -> dict[str, Any]:
    prediction_dir = manifest_path.parent / "predictions"
    return {
        arm: {
            str(seed): file_fingerprint(prediction_dir / f"{arm}_seed_{seed}.csv")
            for seed in SEEDS
        }
        for arm in ("baseline", "augmented")
    }


def evaluate_test(args: argparse.Namespace) -> dict[str, Any]:
    result_dir = require_clean_evaluation_result_dir(args.result_dir)
    freeze_record = json.loads(Path(args.f2_freeze).read_text(encoding="utf-8"))
    if not freeze_record.get("complete") or freeze_record.get("protocol_id") != F2_PROTOCOL_ID:
        raise PermissionError("Official test requires a complete F2 model freeze.")
    original_fingerprint = freeze_record["freeze_fingerprint"]
    payload = _load_frozen_test_payload(freeze_record, args.primary_train_test_cache)
    audit = _new_audit()
    payload["eval_features"] = [
        replace_morgan_column(product, candidates, matrix, audit)
        for (product, candidates), matrix in zip(
            payload["eval_pwc"], payload["eval_features"], strict=True
        )
    ]
    _require_false_equivalence(audit)
    device = resolve_device(args.device)
    root = Path(args.output_root).resolve()
    true_baseline = _score_test_arm(
        payload,
        freeze_record["selected_baseline"],
        freeze_record["selected_prior_transform"],
        "baseline",
        root,
        result_dir / "predictions",
        device,
    )
    true_augmented = _score_test_arm(
        payload,
        freeze_record["selected_augmented"],
        freeze_record["selected_prior_transform"],
        "augmented",
        root,
        result_dir / "predictions",
        device,
    )
    primary_manifest_path = Path(args.primary_test_manifest)
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    if (
        primary_manifest.get("selection_fingerprint")
        != freeze_record["primary_selection_fingerprint"]
        or not primary_manifest.get("test_partition_loaded_only_after_selection_freeze")
    ):
        raise ValueError("Frozen primary false-chirality test manifest is incompatible.")
    if json.loads(Path(args.f2_freeze).read_text(encoding="utf-8"))[
        "freeze_fingerprint"
    ] != original_fingerprint:
        raise RuntimeError("F2 freeze changed during test evaluation.")
    record = {
        "schema_version": 1,
        "record_kind": "f2_chirality_post_freeze_test",
        "protocol_id": F2_PROTOCOL_ID,
        "f2_freeze": file_fingerprint(args.f2_freeze),
        "primary_false_chirality_manifest": file_fingerprint(primary_manifest_path),
        "primary_false_chirality_predictions": _primary_prediction_fingerprints(
            primary_manifest_path
        ),
        "false_chirality_metrics": primary_manifest["per_seed_metrics"],
        "true_chirality_metrics": {
            "baseline": true_baseline,
            "augmented": true_augmented,
        },
        "true_chirality_descriptive_summary": summarize_paired_metrics(
            true_baseline, true_augmented
        ),
        "test_feature_audit": audit,
        "single_intended_change": "Morgan useChirality=False to True",
        "test_partition_loaded_only_after_f2_freeze": True,
        "environment": environment_record(device),
        "fixed_training_config": FIXED_TUNING_CONFIG,
    }
    immutable_json_dump(record, result_dir / "manifest.json", "F2 test manifest")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--primary-freeze", required=True)
    prepare.add_argument("--primary-selection", required=True)
    prepare.add_argument("--output", required=True)
    fit = subparsers.add_parser("fit-validation")
    fit.add_argument("--primary-freeze", required=True)
    fit.add_argument("--selection-bundle", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--arm", choices=("baseline", "augmented"), required=True)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--primary-freeze", required=True)
    freeze_parser.add_argument("--selection-bundle", required=True)
    freeze_parser.add_argument("--output-root", required=True)
    freeze_parser.add_argument("--output", required=True)
    evaluate = subparsers.add_parser("evaluate-test")
    evaluate.add_argument("--f2-freeze", required=True)
    evaluate.add_argument("--primary-train-test-cache", required=True)
    evaluate.add_argument("--primary-test-manifest", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
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
