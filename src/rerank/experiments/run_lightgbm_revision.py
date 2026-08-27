#!/usr/bin/env python
"""Phased deterministic D4 LambdaMART control for cap10-tuned-v1.

Search consumes only train plus official validation.  Official test is opened
only after both 27-configuration arm searches have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rerank.evaluate import evaluate_reranking
from rerank.revision_tuning import (
    AUGMENTED_COLUMNS,
    BASELINE_COLUMNS,
    PROTOCOL_ID,
    file_fingerprint,
    file_sha256,
    load_selection_bundle,
    transform_feature_matrix,
)
from rerank.study_data import canonicalize_reactant_set
from rerank.experiments.run_tuned_revision import (
    _require_prior_freeze,
    canonical_fingerprint,
    environment_record,
    git_commit,
    immutable_json_dump,
    require_clean_evaluation_result_dir,
)


D4_PROTOCOL_ID = "cap10-lightgbm-v1"
MAX_TREES = 2000
EARLY_STOPPING_ROUNDS = 50
TIE_EPSILON = 1e-12


@dataclass(frozen=True)
class LightGBMConfig:
    index: int
    num_leaves: int
    min_child_samples: int
    learning_rate: float


def enumerate_grid() -> tuple[LightGBMConfig, ...]:
    result = []
    for leaves in (7, 15, 31):
        for minimum in (10, 20, 50):
            for rate in (0.03, 0.1, 0.2):
                result.append(LightGBMConfig(len(result), leaves, minimum, rate))
    if len(result) != 27:
        raise AssertionError("D4 grid must contain exactly 27 configurations.")
    return tuple(result)


def config_fingerprint(config: LightGBMConfig) -> str:
    return "sha256:" + hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _labels_from_indices(n: int, indices) -> np.ndarray:
    labels = np.zeros(n, dtype=np.int8)
    labels[np.asarray(list(indices), dtype=np.int64)] = 1
    return labels


def _eval_labels(candidates, ground_truth: str) -> np.ndarray:
    truth = canonicalize_reactant_set(str(ground_truth))
    labels = np.asarray(
        [canonicalize_reactant_set(str(item["smiles"])) == truth for item in candidates],
        dtype=np.int8,
    )
    if labels.sum() < 1:
        raise RuntimeError("A covered validation/test query has no positive candidate.")
    return labels


def build_query_arrays(bundle: dict, arm: str, transform: str) -> dict:
    columns = BASELINE_COLUMNS if arm == "baseline" else AUGMENTED_COLUMNS
    train_features = []
    train_labels = []
    train_groups = []
    for product in bundle["train_products"]:
        features = transform_feature_matrix(product["features"], transform, columns)
        labels = np.asarray(
            product.get("labels", _labels_from_indices(len(features), product["positive_indices"])),
            dtype=np.int8,
        )
        if len(features) != len(labels) or labels.sum() < 1 or labels.sum() == len(labels):
            raise RuntimeError("Training query lacks a valid positive/negative relevance set.")
        train_features.append(features)
        train_labels.append(labels)
        train_groups.append(len(labels))

    valid = bundle["validation_payload"]
    valid_features = []
    valid_labels = []
    valid_groups = []
    for (_, candidates), truth, features in zip(
        valid["eval_pwc"], valid["eval_ground_truths"], valid["eval_features"]
    ):
        transformed = transform_feature_matrix(features, transform, columns)
        labels = _eval_labels(candidates, truth)
        if len(transformed) != len(labels):
            raise RuntimeError("Validation candidates and features are misaligned.")
        valid_features.append(transformed)
        valid_labels.append(labels)
        valid_groups.append(len(labels))

    return {
        "train_x": np.concatenate(train_features).astype(np.float32, copy=False),
        "train_y": np.concatenate(train_labels).astype(np.int8, copy=False),
        "train_groups": np.asarray(train_groups, dtype=np.int32),
        "valid_x": np.concatenate(valid_features).astype(np.float32, copy=False),
        "valid_y": np.concatenate(valid_labels).astype(np.int8, copy=False),
        "valid_groups": np.asarray(valid_groups, dtype=np.int32),
    }


def conditional_mrr(scores, labels, groups) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    groups = np.asarray(groups, dtype=np.int64)
    if int(groups.sum()) != len(scores) or len(scores) != len(labels):
        raise ValueError("Scores, labels and query groups are misaligned.")
    reciprocals = []
    start = 0
    for size in groups:
        stop = start + int(size)
        order = np.argsort(-scores[start:stop], kind="stable")
        positives = np.flatnonzero(labels[start:stop][order] > 0)
        reciprocals.append(0.0 if len(positives) == 0 else 1.0 / (int(positives[0]) + 1))
        start = stop
    return float(np.mean(reciprocals))


def group_fingerprint(arrays: dict) -> str:
    digest = hashlib.sha256()
    for name in ("train_groups", "train_y", "valid_groups", "valid_y"):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(value.dtype.str.encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return "sha256:" + digest.hexdigest()


def _trial_path(root: Path, arm: str, index: int) -> Path:
    return root / "search" / arm / f"config_{index:02d}" / "trial.json"


def _model_path(root: Path, arm: str, index: int) -> Path:
    return _trial_path(root, arm, index).with_name("model.txt")


def _train_one(arrays: dict, config: LightGBMConfig, model_path: Path) -> dict:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("D4 requires the pinned lightgbm==4.6.0 environment.") from exc

    train = lgb.Dataset(
        arrays["train_x"], label=arrays["train_y"],
        group=arrays["train_groups"], free_raw_data=False,
    )
    valid = lgb.Dataset(
        arrays["valid_x"], label=arrays["valid_y"],
        group=arrays["valid_groups"], reference=train, free_raw_data=False,
    )
    labels = arrays["valid_y"]
    groups = arrays["valid_groups"]

    def metric(predictions, _dataset):
        return "conditional_mrr", conditional_mrr(predictions, labels, groups), True

    params = {
        "objective": "lambdarank",
        "metric": "None",
        "label_gain": [0, 1],
        "deterministic": True,
        "force_col_wise": True,
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "num_threads": 1,
        "verbosity": -1,
        "seed": 2026,
        "data_random_seed": 2026,
        "feature_fraction_seed": 2026,
        "bagging_seed": 2026,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "learning_rate": config.learning_rate,
    }
    booster = lgb.train(
        params,
        train,
        num_boost_round=MAX_TREES,
        valid_sets=[valid],
        valid_names=["official_valid"],
        feval=metric,
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False
            )
        ],
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = model_path.with_suffix(".txt.tmp")
    booster.save_model(str(temporary), num_iteration=booster.best_iteration)
    os.replace(temporary, model_path)
    return {
        "best_iteration": int(booster.best_iteration),
        "best_validation_mrr": float(booster.best_score["official_valid"]["conditional_mrr"]),
    }


def run_search(args) -> None:
    bundle = load_selection_bundle(args.selection_bundle)
    bundle_sha = "sha256:" + file_sha256(args.selection_bundle)
    prior = _require_prior_freeze(args.prior_freeze, bundle_sha)
    transform = prior["selected_prior_transform"]
    arrays = build_query_arrays(bundle, args.arm, transform)
    pairing = group_fingerprint(arrays)
    root = Path(args.output_root).resolve()
    for config in enumerate_grid():
        if config.index % args.shard_count != args.shard_index:
            continue
        result_path = _trial_path(root, args.arm, config.index)
        model_path = _model_path(root, args.arm, config.index)
        if result_path.exists():
            trial = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                trial.get("record_kind") != "lightgbm_validation_trial"
                or trial.get("protocol_id") != D4_PROTOCOL_ID
                or trial.get("arm") != args.arm
                or trial.get("config_fingerprint") != config_fingerprint(config)
                or trial.get("selection_bundle_sha256") != bundle_sha
                or trial.get("prior_freeze_fingerprint") != prior["freeze_fingerprint"]
                or trial.get("prior_transform") != transform
                or trial.get("query_group_fingerprint") != pairing
                or not model_path.is_file()
                or trial.get("model_sha256") != "sha256:" + file_sha256(model_path)
            ):
                raise RuntimeError(f"Existing D4 trial failed validation: {result_path}")
            continue
        if model_path.exists():
            raise FileExistsError(f"Partial D4 model preserved for inspection: {model_path}")
        started = time.perf_counter()
        trained = _train_one(arrays, config, model_path)
        record = {
            "record_kind": "lightgbm_validation_trial",
            "protocol_id": D4_PROTOCOL_ID,
            "primary_protocol_id": PROTOCOL_ID,
            "arm": args.arm,
            "config": asdict(config),
            "config_fingerprint": config_fingerprint(config),
            "selection_bundle_sha256": bundle_sha,
            "prior_freeze_fingerprint": prior["freeze_fingerprint"],
            "prior_transform": transform,
            "query_group_fingerprint": pairing,
            "model_relpath": model_path.relative_to(root).as_posix(),
            "model_sha256": "sha256:" + file_sha256(model_path),
            "test_partition_loaded": False,
            "runtime_seconds": time.perf_counter() - started,
            **trained,
        }
        immutable_json_dump(record, result_path, "D4 validation trial")


def _collect(root: Path, arm: str, bundle_sha: str, prior_fingerprint: str) -> dict:
    trials = []
    for config in enumerate_grid():
        path = _trial_path(root, arm, config.index)
        trial = json.loads(path.read_text(encoding="utf-8"))
        model = root / trial["model_relpath"]
        if (
            trial.get("record_kind") != "lightgbm_validation_trial"
            or trial.get("protocol_id") != D4_PROTOCOL_ID
            or trial.get("arm") != arm
            or trial.get("config_fingerprint") != config_fingerprint(config)
            or trial.get("selection_bundle_sha256") != bundle_sha
            or trial.get("prior_freeze_fingerprint") != prior_fingerprint
            or trial.get("model_sha256") != "sha256:" + file_sha256(model)
            or trial.get("test_partition_loaded") is not False
        ):
            raise RuntimeError(f"Invalid D4 trial: {path}")
        trials.append(trial)
    best = trials[0]
    for trial in trials[1:]:
        if trial["best_validation_mrr"] > best["best_validation_mrr"] + TIE_EPSILON:
            best = trial
    return {"selected": best, "all_validation_mrr": [t["best_validation_mrr"] for t in trials]}


def run_freeze(args) -> None:
    bundle = load_selection_bundle(args.selection_bundle)
    bundle_sha = "sha256:" + file_sha256(args.selection_bundle)
    prior = _require_prior_freeze(args.prior_freeze, bundle_sha)
    root = Path(args.output_root).resolve()
    baseline = _collect(root, "baseline", bundle_sha, prior["freeze_fingerprint"])
    augmented = _collect(root, "augmented", bundle_sha, prior["freeze_fingerprint"])
    if baseline["selected"]["query_group_fingerprint"] != augmented["selected"]["query_group_fingerprint"]:
        raise RuntimeError("D4 arms do not have identical query groups and labels.")
    record = {
        "record_kind": "lightgbm_selection_freeze",
        "protocol_id": D4_PROTOCOL_ID,
        "comparator": "four-input deterministic LambdaMART",
        "single_intended_change": "add three Uni-Mol-derived scalar features",
        "selection_bundle_sha256": bundle_sha,
        "retained_train_test_cache_sha256": bundle["input_fingerprints"]["retained_train_test_cache"]["sha256"],
        "prior_transform": prior["selected_prior_transform"],
        "prior_freeze_fingerprint": prior["freeze_fingerprint"],
        "query_group_fingerprint": baseline["selected"]["query_group_fingerprint"],
        "selected_baseline": baseline["selected"],
        "selected_augmented": augmented["selected"],
        "grid": [asdict(item) for item in enumerate_grid()],
        "max_trees": MAX_TREES,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "test_partition_loaded": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    record["freeze_fingerprint"] = canonical_fingerprint(record)
    immutable_json_dump(record, args.output, "D4 selection freeze")


def validate_freeze(path: str | Path) -> dict:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    supplied = record.pop("freeze_fingerprint", None)
    if (
        record.get("record_kind") != "lightgbm_selection_freeze"
        or record.get("protocol_id") != D4_PROTOCOL_ID
        or record.get("test_partition_loaded") is not False
        or supplied != canonical_fingerprint(record)
    ):
        raise PermissionError("Invalid D4 freeze; official test remains locked.")
    record["freeze_fingerprint"] = supplied
    return record


def _score_payload(payload: dict, trial: dict, arm: str, transform: str, root: Path, output: Path):
    import lightgbm as lgb

    model_path = (root / trial["model_relpath"]).resolve()
    if not model_path.is_relative_to(root) or trial["model_sha256"] != "sha256:" + file_sha256(model_path):
        raise RuntimeError("Frozen D4 model checksum failed.")
    booster = lgb.Booster(model_file=str(model_path))
    columns = BASELINE_COLUMNS if arm == "baseline" else AUGMENTED_COLUMNS
    reranked = []
    for (_, candidates), features in zip(payload["eval_pwc"], payload["eval_features"]):
        matrix = transform_feature_matrix(features, transform, columns)
        scores = booster.predict(matrix, num_iteration=booster.best_iteration)
        order = np.argsort(-scores, kind="stable")
        reranked.append(([candidates[i] for i in order], np.asarray(scores)[order]))
    evaluation = evaluate_reranking(
        products_with_candidates=payload["eval_pwc"],
        ground_truths=payload["eval_ground_truths"],
        reranker=None,
        ks=[1, 3, 5, 10],
        output_csv=str(output),
        precomputed_reranked_results=reranked,
        reaction_metadata=payload["eval_metadata"],
    )
    return {
        "top1": evaluation.reranked_accuracy[1], "top3": evaluation.reranked_accuracy[3],
        "top5": evaluation.reranked_accuracy[5], "top10": evaluation.reranked_accuracy[10],
        "mrr": evaluation.reranked_mrr, "n_test_reactions": evaluation.n_products,
    }


def run_evaluate(args) -> None:
    freeze = validate_freeze(args.selection_freeze)
    if file_sha256(args.train_test_cache) != freeze["retained_train_test_cache_sha256"]:
        raise PermissionError("D4 official-test cache does not match the freeze.")
    with open(args.train_test_cache, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload", blob)
    if any(str(row.get("source_split")) != "test" for row in payload["eval_metadata"]):
        raise PermissionError("D4 evaluation payload is not official test only.")
    result = require_clean_evaluation_result_dir(args.result_dir)
    predictions = result / "predictions"
    predictions.mkdir(parents=True)
    root = Path(args.output_root).resolve()
    baseline = _score_payload(
        payload, freeze["selected_baseline"], "baseline", freeze["prior_transform"],
        root, predictions / "baseline.csv",
    )
    augmented = _score_payload(
        payload, freeze["selected_augmented"], "augmented", freeze["prior_transform"],
        root, predictions / "augmented.csv",
    )
    manifest = {
        "manifest_kind": "lightgbm_post_freeze_test_evaluation",
        "protocol_id": D4_PROTOCOL_ID,
        "selection_freeze": file_fingerprint(args.selection_freeze),
        "per_arm_metrics": {"baseline": baseline, "augmented": augmented},
        "test_partition_loaded_only_after_freeze": True,
        "environment": environment_record("cpu"),
        "git_commit": git_commit(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    immutable_json_dump(manifest, result / "manifest.json", "D4 test manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    search = sub.add_parser("search")
    search.add_argument("--selection-bundle", required=True)
    search.add_argument("--prior-freeze", required=True)
    search.add_argument("--output-root", required=True)
    search.add_argument("--arm", choices=("baseline", "augmented"), required=True)
    search.add_argument("--shard-index", type=int, default=0)
    search.add_argument("--shard-count", type=int, default=1)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--selection-bundle", required=True)
    freeze.add_argument("--prior-freeze", required=True)
    freeze.add_argument("--output-root", required=True)
    freeze.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate-test")
    evaluate.add_argument("--selection-freeze", required=True)
    evaluate.add_argument("--train-test-cache", required=True)
    evaluate.add_argument("--output-root", required=True)
    evaluate.add_argument("--result-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "search":
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ValueError("Require 0 <= shard-index < shard-count.")
        run_search(args)
    elif args.command == "freeze":
        run_freeze(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
