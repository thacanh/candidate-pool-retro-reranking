#!/usr/bin/env python
"""G1 extension of frozen cap10-tuned-v1 arms to paired seeds 42--61."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rerank.revision_tuning import (
    MAX_EPOCHS,
    MIN_IMPROVEMENT,
    PATIENCE,
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


G1_PROTOCOL_ID = "cap10-tuned-20seed-v1"
G1_SEEDS = tuple(range(42, 62))


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds).issubset(G1_SEEDS):
        raise argparse.ArgumentTypeError("Seeds must be a unique subset of 42--61.")
    return seeds


def _config(freeze: dict, arm: str) -> GridConfig:
    body = freeze[f"selected_{arm}"]["config"]
    config = GridConfig(**body)
    if config_fingerprint(config) != freeze[f"selected_{arm}"]["config_fingerprint"]:
        raise RuntimeError(f"Frozen {arm} configuration fingerprint failed.")
    return config


def _trial_path(root: Path, arm: str, seed: int) -> Path:
    return root / "validation" / arm / f"seed_{seed}" / "trial.json"


def _validate_trial(root: Path, arm: str, seed: int, config: GridConfig, bundle_sha: str, primary_fp: str) -> dict:
    path = _trial_path(root, arm, seed)
    trial = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "record_kind": "g1_validation_trial", "protocol_id": G1_PROTOCOL_ID,
        "arm": arm, "seed": seed, "config_fingerprint": config_fingerprint(config),
        "selection_bundle_sha256": bundle_sha,
        "primary_selection_fingerprint": primary_fp, "test_partition_loaded": False,
    }
    if any(trial.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"G1 validation trial is incompatible: {path}")
    for rel_key, hash_key in (("checkpoint_relpath", "checkpoint_sha256"), ("normalizer_relpath", "normalizer_sha256")):
        artifact = (root / trial[rel_key]).resolve()
        if not artifact.is_relative_to(root) or trial[hash_key] != "sha256:" + file_sha256(artifact):
            raise RuntimeError(f"G1 artifact checksum failed: {artifact}")
    return trial


def run_fit(args) -> None:
    freeze = validate_selection_freeze(args.primary_selection_freeze)
    bundle = load_selection_bundle(args.selection_bundle)
    bundle_sha = "sha256:" + file_sha256(args.selection_bundle)
    if bundle_sha != freeze["selection_bundle_sha256"]:
        raise RuntimeError("G1 selection bundle does not belong to the primary freeze.")
    config = _config(freeze, args.arm)
    cache = transform_selection_cache(bundle, args.arm, freeze["selected_prior_transform"])
    root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    for seed in args.seeds:
        path = _trial_path(root, args.arm, seed)
        if path.exists():
            _validate_trial(root, args.arm, seed, config, bundle_sha, freeze["selection_fingerprint"])
            continue
        run_dir = path.parent
        checkpoint = run_dir / "best_checkpoint.pt"
        normalizer = run_dir / "normalizer.npz"
        if checkpoint.exists() or normalizer.exists():
            raise FileExistsError(f"Partial G1 trial preserved for inspection: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        trained = train_validation_trial(
            cache, config, seed, device, checkpoint, normalizer,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, min_improvement=MIN_IMPROVEMENT,
            allowed_seeds=G1_SEEDS,
        )
        trial = {
            "record_kind": "g1_validation_trial", "protocol_id": G1_PROTOCOL_ID,
            "comparator": "same frozen arm configuration across paired seeds 42--61",
            "single_intended_change": "training RNG seed",
            "arm": args.arm, "seed": seed, "config": asdict(config),
            "config_fingerprint": config_fingerprint(config),
            "selection_bundle_sha256": bundle_sha,
            "primary_selection_fingerprint": freeze["selection_fingerprint"],
            "prior_transform": freeze["selected_prior_transform"],
            "checkpoint_relpath": checkpoint.relative_to(root).as_posix(),
            "checkpoint_sha256": "sha256:" + file_sha256(checkpoint),
            "normalizer_relpath": normalizer.relative_to(root).as_posix(),
            "normalizer_sha256": "sha256:" + file_sha256(normalizer),
            "test_partition_loaded": False,
            "runtime_seconds": time.perf_counter() - started,
            **trained,
        }
        immutable_json_dump(trial, path, "G1 validation trial")


def run_freeze(args) -> None:
    primary = validate_selection_freeze(args.primary_selection_freeze)
    bundle = load_selection_bundle(args.selection_bundle)
    bundle_sha = "sha256:" + file_sha256(args.selection_bundle)
    if bundle_sha != primary["selection_bundle_sha256"]:
        raise RuntimeError("G1 selection bundle does not belong to primary freeze.")
    root = Path(args.output_root).resolve()
    selected = {}
    for arm in ("baseline", "augmented"):
        config = _config(primary, arm)
        selected[arm] = {
            "config": asdict(config),
            "config_fingerprint": config_fingerprint(config),
            "trials": {
                str(seed): _validate_trial(root, arm, seed, config, bundle_sha, primary["selection_fingerprint"])
                for seed in G1_SEEDS
            },
        }
    record = {
        "record_kind": "g1_20seed_freeze", "protocol_id": G1_PROTOCOL_ID,
        "comparator": "frozen validation-tuned baseline and augmented arms",
        "single_intended_change": "extend paired training seeds from 42--46 to 42--61",
        "primary_selection_fingerprint": primary["selection_fingerprint"],
        "selection_bundle_sha256": bundle_sha,
        "retained_train_test_cache_sha256": bundle["input_fingerprints"]["retained_train_test_cache"]["sha256"],
        "prior_transform": primary["selected_prior_transform"],
        "selected_baseline": selected["baseline"],
        "selected_augmented": selected["augmented"],
        "seeds": list(G1_SEEDS), "test_partition_loaded": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["freeze_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "G1 20-seed freeze")


def validate_freeze(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = record.pop("freeze_fingerprint", None)
    if (
        record.get("record_kind") != "g1_20seed_freeze"
        or record.get("protocol_id") != G1_PROTOCOL_ID
        or tuple(record.get("seeds", ())) != G1_SEEDS
        or record.get("test_partition_loaded") is not False
        or supplied != canonical_fingerprint(record)
    ):
        raise PermissionError("Invalid G1 freeze; official test remains locked.")
    record["freeze_fingerprint"] = supplied
    return record


def run_evaluate(args) -> None:
    freeze = validate_freeze(args.g1_freeze)
    if file_sha256(args.train_test_cache) != freeze["retained_train_test_cache_sha256"]:
        raise PermissionError("G1 official-test cache differs from its freeze.")
    with open(args.train_test_cache, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("G1 evaluation payload is not exclusively official test.")
    result = require_clean_evaluation_result_dir(args.result_dir)
    root = Path(args.output_root).resolve()
    device = resolve_device(args.device)
    predictions = result / "predictions"
    baseline = _score_test_arm(
        payload, freeze["selected_baseline"], freeze["prior_transform"], "baseline",
        root, predictions, device, seeds=G1_SEEDS,
    )
    augmented = _score_test_arm(
        payload, freeze["selected_augmented"], freeze["prior_transform"], "augmented",
        root, predictions, device, seeds=G1_SEEDS,
    )
    manifest = {
        "manifest_kind": "g1_20seed_post_freeze_test_evaluation",
        "protocol_id": G1_PROTOCOL_ID, "g1_freeze": file_fingerprint(args.g1_freeze),
        "per_seed_metrics": {"baseline": baseline, "augmented": augmented},
        "test_partition_loaded_only_after_20seed_freeze": True,
        "fixed_training_config": FIXED_TUNING_CONFIG,
        "environment": environment_record(device), "git_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    immutable_json_dump(manifest, result / "manifest.json", "G1 test manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit-validation")
    fit.add_argument("--primary-selection-freeze", required=True)
    fit.add_argument("--selection-bundle", required=True)
    fit.add_argument("--output-root", required=True)
    fit.add_argument("--arm", choices=("baseline", "augmented"), required=True)
    fit.add_argument("--seeds", type=parse_seeds, default=G1_SEEDS)
    fit.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--primary-selection-freeze", required=True)
    freeze.add_argument("--selection-bundle", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-test")
    evaluate.add_argument("--g1-freeze", required=True)
    evaluate.add_argument("--train-test-cache", required=True)
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
