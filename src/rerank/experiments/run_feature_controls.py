#!/usr/bin/env python
"""Run deterministic capacity controls and Uni-Mol feature ablations.

All variants are derived from the same frozen seven-column official-split
feature cache.  This avoids recomputing embeddings and guarantees identical
candidate identities, priors, reaction partitions, and candidate ordering.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rerank.evaluate import evaluate_reranking
from rerank.features import fit_normalizer_from_dataset
from rerank.model import RankerMLP
from rerank.study_data import file_fingerprint, make_pairwise_dataset
from rerank.trainer import RankerTrainer, TrainerConfig
from rerank.experiments.run_controlled_study import FROZEN_CONFIG, _sample_std, _score_precomputed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("feature_controls")


BASE_FEATURE_NAMES = [
    "prior_or_log_prob",
    "morgan_similarity",
    "atom_set_similarity",
    "reaction_distance",
    "cosine_reaction_vec",
    "n_fragments",
    "heavy_atom_ratio",
]

VARIANTS = {
    "prior_2d": {
        "columns": [0, 1, 5, 6],
        "feature_names": [
            "prior_or_log_prob",
            "morgan_similarity",
            "n_fragments",
            "heavy_atom_ratio",
        ],
        "description": "Prior plus the three conventional 2D/size descriptors.",
    },
    "unimol_atom": {
        "columns": [0, 1, 2, 5, 6],
        "feature_names": [
            "prior_or_log_prob",
            "morgan_similarity",
            "atom_set_similarity",
            "n_fragments",
            "heavy_atom_ratio",
        ],
        "description": "Prior+2D plus symmetric Uni-Mol atom-set similarity only.",
    },
    "unimol_distance": {
        "columns": [0, 1, 3, 5, 6],
        "feature_names": [
            "prior_or_log_prob",
            "morgan_similarity",
            "reaction_distance",
            "n_fragments",
            "heavy_atom_ratio",
        ],
        "description": "Prior+2D plus distance between mean Uni-Mol embeddings only.",
    },
    "unimol_cosine": {
        "columns": [0, 1, 4, 5, 6],
        "feature_names": [
            "prior_or_log_prob",
            "morgan_similarity",
            "cosine_reaction_vec",
            "n_fragments",
            "heavy_atom_ratio",
        ],
        "description": "Prior+2D plus product/reaction-vector cosine only.",
    },
    "unimol_all": {
        "columns": [0, 1, 2, 3, 4, 5, 6],
        "feature_names": BASE_FEATURE_NAMES,
        "description": "Prior+2D plus all three Uni-Mol-derived features.",
    },
    "permuted_unimol": {
        "columns": [0, 1, 2, 3, 4, 5, 6],
        "feature_names": [
            "prior_or_log_prob",
            "morgan_similarity",
            "permuted_atom_set_similarity",
            "permuted_reaction_distance",
            "permuted_cosine_reaction_vec",
            "n_fragments",
            "heavy_atom_ratio",
        ],
        "description": (
            "Capacity-matched negative control: each Uni-Mol column is independently "
            "permuted across candidate rows within the train and test splits."
        ),
    },
}


def _parse_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items).difference(VARIANTS))
    if not items or unknown:
        raise argparse.ArgumentTypeError(
            f"Variants must be selected from {sorted(VARIANTS)}; unknown={unknown}"
        )
    return items


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique list.")
    return seeds


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _seed_before_model_construction(seed: int) -> None:
    """Seed every RNG before constructing the model and the trainer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _select_columns(matrix: np.ndarray, columns: list[int]) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != len(BASE_FEATURE_NAMES):
        raise ValueError(f"Expected a seven-column feature matrix, got {array.shape}.")
    return array[:, columns].astype(np.float32, copy=True)


def _permute_unimol_columns(
    matrices: list[np.ndarray], rng: np.random.Generator
) -> list[np.ndarray]:
    """Independently permute the three Uni-Mol columns across candidate rows."""
    lengths = [len(matrix) for matrix in matrices]
    if not lengths or sum(lengths) == 0:
        return [np.asarray(matrix, dtype=np.float32).copy() for matrix in matrices]
    joined = np.concatenate(
        [np.asarray(matrix, dtype=np.float32) for matrix in matrices], axis=0
    ).copy()
    for column in (2, 3, 4):
        joined[:, column] = joined[rng.permutation(len(joined)), column]
    result = []
    start = 0
    for length in lengths:
        result.append(joined[start : start + length].copy())
        start += length
    return result


def transform_feature_cache(
    payload: dict, variant: str, control_seed: int = 20260803
) -> dict:
    """Return a transformed cache view without mutating the frozen base cache."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    config = VARIANTS[variant]
    columns = config["columns"]
    train_raw = [product["features"] for product in payload["train_products"]]
    eval_raw = payload["eval_features"]

    if variant == "permuted_unimol":
        train_matrices = _permute_unimol_columns(
            train_raw, np.random.default_rng(control_seed)
        )
        eval_matrices = _permute_unimol_columns(
            eval_raw, np.random.default_rng(control_seed + 1)
        )
    else:
        train_matrices = [_select_columns(matrix, columns) for matrix in train_raw]
        eval_matrices = [_select_columns(matrix, columns) for matrix in eval_raw]

    train_products = []
    for product, features in zip(payload["train_products"], train_matrices):
        transformed = dict(product)
        transformed["features"] = features
        train_products.append(transformed)

    transformed_payload = dict(payload)
    transformed_payload["feature_mode"] = variant
    transformed_payload["train_products"] = train_products
    transformed_payload["eval_features"] = eval_matrices
    transformed_payload["audit"] = dict(payload["audit"])
    transformed_payload["audit"]["feature_mode"] = variant
    return transformed_payload


def _load_base_cache(path: Path) -> dict:
    logger.info("Loading frozen seven-column feature cache: %s", path)
    with open(path, "rb") as handle:
        blob = pickle.load(handle)
    payload = blob.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("Base cache does not contain a payload dictionary.")
    sample = payload["train_products"][0]["features"]
    if np.asarray(sample).shape[1] != len(BASE_FEATURE_NAMES):
        raise ValueError("Base cache is not the expected seven-column Uni-Mol cache.")
    return payload


def _parameter_count(input_dim: int) -> int:
    model = RankerMLP(
        input_dim=input_dim,
        hidden_dims=FROZEN_CONFIG["hidden_dims"],
        dropout=FROZEN_CONFIG["dropout"],
        use_batch_norm=FROZEN_CONFIG["use_batch_norm"],
    )
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _run_variant(
    variant: str,
    feature_cache: dict,
    output_root: Path,
    seeds: list[int],
    device: str,
    control_seed: int,
    base_cache_path: Path,
) -> None:
    config = VARIANTS[variant]
    output_dir = output_root / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    metrics_path = output_dir / "per_seed_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as handle:
            all_metrics = json.load(handle)
    else:
        all_metrics = {}
    seed_timings: dict[str, float] = {}
    input_dim = len(config["feature_names"])

    for seed in seeds:
        seed_key = str(seed)
        evaluation_path = output_dir / f"eval_seed{seed}.csv"
        if seed_key in all_metrics and evaluation_path.exists():
            logger.info("Skipping completed variant=%s seed=%d", variant, seed)
            continue
        logger.info("Starting variant=%s seed=%d", variant, seed)
        seed_start = time.perf_counter()
        train_dataset = make_pairwise_dataset(
            feature_cache,
            seed=seed,
            max_neg_per_pos=FROZEN_CONFIG["max_neg_per_pos"],
            negative_mining=FROZEN_CONFIG["negative_mining"],
        )
        normalizer = fit_normalizer_from_dataset(train_dataset)
        normalizer.save(str(output_dir / f"normalizer_seed{seed}.npz"))
        train_dataset.apply_normalizer(normalizer)

        _seed_before_model_construction(seed)
        model = RankerMLP(
            input_dim=input_dim,
            hidden_dims=FROZEN_CONFIG["hidden_dims"],
            dropout=FROZEN_CONFIG["dropout"],
            use_batch_norm=FROZEN_CONFIG["use_batch_norm"],
        )
        checkpoint = output_dir / f"ranker_seed{seed}_best.pt"
        trainer = RankerTrainer(
            model,
            TrainerConfig(
                learning_rate=FROZEN_CONFIG["learning_rate"],
                weight_decay=FROZEN_CONFIG["weight_decay"],
                margin=FROZEN_CONFIG["margin"],
                n_epochs=FROZEN_CONFIG["epochs"],
                batch_size=FROZEN_CONFIG["batch_size"],
                val_fraction=FROZEN_CONFIG["validation_fraction"],
                checkpoint_path=str(checkpoint),
                seed=seed,
                device=device,
            ),
        )
        trainer.fit(train_dataset)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        model.to("cpu").eval()
        precomputed = _score_precomputed(
            feature_cache["eval_pwc"],
            feature_cache["eval_features"],
            normalizer,
            model,
        )
        evaluation = evaluate_reranking(
            products_with_candidates=feature_cache["eval_pwc"],
            ground_truths=feature_cache["eval_ground_truths"],
            reranker=None,
            ks=[1, 3, 5, 10],
            output_csv=str(evaluation_path),
            precomputed_reranked_results=precomputed,
            reaction_metadata=feature_cache["eval_metadata"],
        )
        all_metrics[seed_key] = {
            "top1": evaluation.reranked_accuracy[1],
            "top3": evaluation.reranked_accuracy[3],
            "top5": evaluation.reranked_accuracy[5],
            "top10": evaluation.reranked_accuracy[10],
            "mrr": evaluation.reranked_mrr,
            "baseline_top1": evaluation.baseline_accuracy[1],
            "baseline_top3": evaluation.baseline_accuracy[3],
            "baseline_top5": evaluation.baseline_accuracy[5],
            "baseline_top10": evaluation.baseline_accuracy[10],
            "baseline_mrr": evaluation.baseline_mrr,
            "n_train_pairs": len(train_dataset),
            "n_eval_reactions": evaluation.n_products,
        }
        seed_timings[seed_key] = time.perf_counter() - seed_start
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(all_metrics, handle, indent=2)

    metric_names = ["top1", "top3", "top5", "top10", "mrr"]
    summary = {
        "timestamp": started_at,
        "variant": variant,
        "seeds": json.dumps(seeds),
        "input_dim": input_dim,
        "parameter_count": _parameter_count(input_dim),
    }
    for metric in metric_names:
        values = [all_metrics[str(seed)][metric] for seed in seeds]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = _sample_std(values)
        summary[f"baseline_{metric}"] = all_metrics[str(seeds[0])][
            f"baseline_{metric}"
        ]
    pd.DataFrame([summary]).to_csv(output_dir / "experiment_summary.csv", index=False)

    manifest = {
        "study": "capacity_control_and_unimol_feature_ablation",
        "timestamp": started_at,
        "git_commit": _git_commit(),
        "variant": variant,
        "description": config["description"],
        "feature_names": config["feature_names"],
        "input_dim": input_dim,
        "parameter_count": summary["parameter_count"],
        "seeds": seeds,
        "model_rng_seeded_before_construction": True,
        "control_permutation_seed": control_seed if variant == "permuted_unimol" else None,
        "frozen_config": FROZEN_CONFIG,
        "base_feature_cache": file_fingerprint(base_cache_path),
        "feature_cache_audit": feature_cache["audit"],
        "per_seed_metrics": all_metrics,
        "seed_runtime_seconds": seed_timings,
        "torch_num_threads": torch.get_num_threads(),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Completed variant=%s", variant)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-feature-cache",
        default="outputs/study_cache/official_3d_prior_schema1.pkl",
    )
    parser.add_argument(
        "--variants",
        type=_parse_csv,
        default=_parse_csv(",".join(VARIANTS)),
    )
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=_parse_seeds("42,43,44,45,46")
    )
    parser.add_argument("--control-seed", type=int, default=20260803)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default="outputs/revision_controls")
    args = parser.parse_args()

    if args.torch_threads < 1:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    base_path = Path(args.base_feature_cache)
    base_payload = _load_base_cache(base_path)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        transformed = transform_feature_cache(
            base_payload, variant, control_seed=args.control_seed
        )
        _run_variant(
            variant,
            transformed,
            output_root,
            args.seeds,
            args.device,
            args.control_seed,
            base_path,
        )


if __name__ == "__main__":
    main()
