#!/usr/bin/env python
"""Run the frozen, official-split 2D versus 2D+3D reranking study.

Only ``feature_mode`` differs between study arms.  The MLP, optimizer, BPR
loss, candidate pools, official split, pair sampling, and evaluation protocol
are fixed in ``FROZEN_CONFIG`` and recorded in every manifest.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rerank.cached_encoder import CachedUniMolEncoder, SqliteCachedUniMolEncoder
from rerank.evaluate import evaluate_reranking
from rerank.features import FEATURE_NAMES_MAP, FeatureExtractor, fit_normalizer_from_dataset
from rerank.model import RankerMLP
from rerank.study_data import (
    STUDY_CACHE_SCHEMA,
    build_official_feature_cache,
    file_fingerprint,
    load_candidate_pools,
    load_reactions,
    make_pairwise_dataset,
)
from rerank.trainer import RankerTrainer, TrainerConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("controlled_study")


FROZEN_CONFIG = {
    "architecture": "RankerMLP",
    "hidden_dims": [32],
    "dropout": 0.1,
    "use_batch_norm": False,
    "loss": "BPR",
    "margin": 0.1,
    "optimizer": "AdamW",
    "learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "epochs": 50,
    "batch_size": 256,
    "max_neg_per_pos": 5,
    "negative_mining": "random",
    "train_split": "train",
    "eval_split": "test",
    "exclude_cross_split_train_products": True,
    "validation_fraction": 0.0,
}


class _NoOpEncoder:
    """Guard object: a 2D arm must never request UniMol embeddings."""

    def __getattr__(self, name):
        raise RuntimeError(f"The 2D study arm unexpectedly requested encoder.{name}.")


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("Seeds must be a non-empty unique list.")
    return seeds


def _seed_before_model_construction(seed: int) -> None:
    """Seed every RNG before model initialization as well as training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _input_fingerprints(args) -> dict:
    repair_path = f"{args.atom_cache}.repair.sqlite"
    return {
        "source_csv": file_fingerprint(args.source_csv),
        "metadata_csv": file_fingerprint(args.metadata_csv),
        "candidate_jsonl": file_fingerprint(args.candidate_jsonl),
        # Hashing the 18 GB atom cache on every invocation is disproportionate.
        "atom_embedding_cache": (
            file_fingerprint(args.atom_cache, include_sha256=False)
            if args.feature_mode == "3d+prior"
            else None
        ),
        "atom_embedding_repair_cache": (
            file_fingerprint(repair_path, include_sha256=False)
            if args.feature_mode == "3d+prior" and os.path.exists(repair_path)
            else None
        ),
    }


def _load_or_build_feature_cache(args, extractor: FeatureExtractor) -> tuple[dict, dict]:
    cache_dir = Path(args.feature_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (
        f"official_{args.feature_mode.replace('+', '_')}_schema{STUDY_CACHE_SCHEMA}.pkl"
    )
    fingerprints = _input_fingerprints(args)

    if cache_path.exists() and not args.force_rebuild_feature_cache:
        logger.info("Loading official-split feature cache: %s", cache_path)
        with open(cache_path, "rb") as handle:
            blob = pickle.load(handle)
        if blob.get("schema_version") != STUDY_CACHE_SCHEMA:
            raise RuntimeError("Feature-cache schema mismatch; rebuild the cache.")
        if blob.get("feature_mode") != args.feature_mode:
            raise RuntimeError("Feature-cache representation mismatch.")
        if blob.get("input_fingerprints") != fingerprints:
            raise RuntimeError(
                "Feature-cache inputs changed. Re-run with --force-rebuild-feature-cache."
            )
        return blob["payload"], fingerprints

    reactions = load_reactions(args.source_csv, args.metadata_csv)
    pools, candidate_audit = load_candidate_pools(args.candidate_jsonl)
    payload = build_official_feature_cache(
        reactions=reactions,
        pools=pools,
        feature_extractor=extractor,
        feature_mode=args.feature_mode,
        train_split=FROZEN_CONFIG["train_split"],
        eval_split=FROZEN_CONFIG["eval_split"],
        exclude_cross_split_train_products=FROZEN_CONFIG[
            "exclude_cross_split_train_products"
        ],
    )
    payload["candidate_audit"] = candidate_audit
    blob = {
        "schema_version": STUDY_CACHE_SCHEMA,
        "feature_mode": args.feature_mode,
        "input_fingerprints": fingerprints,
        "payload": payload,
    }
    logger.info("Saving official-split feature cache: %s", cache_path)
    with open(cache_path, "wb") as handle:
        pickle.dump(blob, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return payload, fingerprints


def _score_precomputed(
    eval_pwc: list,
    eval_features: list[np.ndarray],
    normalizer,
    model: RankerMLP,
):
    results = []
    for (_, candidates), raw_features in zip(eval_pwc, eval_features):
        normalized = normalizer.transform(np.asarray(raw_features, dtype=np.float32))
        scores = model.score_numpy(normalized)
        # Stable descending order keeps the prior order for exact score ties.
        order = np.argsort(-scores, kind="stable")
        results.append(
            ([candidates[index] for index in order], scores[order])
        )
    return results


def _sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-jsonl", default="outputs/rerank_dataset.jsonl")
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument(
        "--metadata-csv", default="data/uspto_reaction_metadata.csv"
    )
    parser.add_argument("--atom-cache", default="outputs/atom_embeddings.pkl")
    parser.add_argument(
        "--feature-mode",
        required=True,
        choices=["2d+prior", "3d+prior"],
    )
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("42,43,44,45,46"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-cache-dir", default="outputs/study_cache")
    parser.add_argument("--force-rebuild-feature-cache", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    if args.feature_mode == "3d+prior":
        encoder_class = (
            SqliteCachedUniMolEncoder
            if str(args.atom_cache).lower().endswith((".sqlite", ".sqlite3", ".db"))
            else CachedUniMolEncoder
        )
        encoder = encoder_class(
            args.atom_cache,
            fallback_device=args.device,
            log_misses=False,
            strict=False,
        )
    else:
        encoder = _NoOpEncoder()
    extractor = FeatureExtractor(encoder, feature_mode=args.feature_mode)

    feature_cache, input_fingerprints = _load_or_build_feature_cache(args, extractor)
    if feature_cache.get("schema_version") != STUDY_CACHE_SCHEMA:
        raise RuntimeError("Unsupported study feature-cache payload.")

    all_metrics: dict[str, dict] = {}
    seed_timings: dict[str, float] = {}
    for seed in args.seeds:
        logger.info("Starting frozen study arm=%s seed=%d", args.feature_mode, seed)
        seed_start = time.perf_counter()
        train_dataset = make_pairwise_dataset(
            feature_cache,
            seed=seed,
            max_neg_per_pos=FROZEN_CONFIG["max_neg_per_pos"],
            negative_mining=FROZEN_CONFIG["negative_mining"],
        )
        normalizer = fit_normalizer_from_dataset(train_dataset)
        normalizer_path = output_dir / f"normalizer_seed{seed}.npz"
        normalizer.save(str(normalizer_path))
        train_dataset.apply_normalizer(normalizer)

        _seed_before_model_construction(seed)
        model = RankerMLP(
            input_dim=len(FEATURE_NAMES_MAP[args.feature_mode]),
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
                device=args.device,
            ),
        )
        trainer.fit(train_dataset)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
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
            output_csv=str(output_dir / f"eval_seed{seed}.csv"),
            precomputed_reranked_results=precomputed,
            reaction_metadata=feature_cache["eval_metadata"],
        )
        metrics = {
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
        all_metrics[str(seed)] = metrics
        seed_timings[str(seed)] = time.perf_counter() - seed_start
        with open(output_dir / "per_seed_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(all_metrics, handle, indent=2)

    metric_names = ["top1", "top3", "top5", "top10", "mrr"]
    summary = {
        "timestamp": started_at,
        "feature_mode": args.feature_mode,
        "seeds": json.dumps(args.seeds),
    }
    for metric in metric_names:
        values = [all_metrics[str(seed)][metric] for seed in args.seeds]
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = _sample_std(values)
        summary[f"baseline_{metric}"] = all_metrics[str(args.seeds[0])][
            f"baseline_{metric}"
        ]
    pd.DataFrame([summary]).to_csv(output_dir / "experiment_summary.csv", index=False)

    feature_stats = {
        "feature_names": FEATURE_NAMES_MAP[args.feature_mode],
        "per_seed": {
            str(seed): {
                "normalizer_path": str((output_dir / f"normalizer_seed{seed}.npz").resolve())
            }
            for seed in args.seeds
        },
    }
    with open(output_dir / "feature_stats.json", "w", encoding="utf-8") as handle:
        json.dump(feature_stats, handle, indent=2)

    manifest = {
        "study": "controlled_2d_vs_2d_plus_3d",
        "timestamp": started_at,
        "git_commit": _git_commit(),
        "feature_mode": args.feature_mode,
        "seeds": args.seeds,
        "device": args.device,
        "model_rng_seeded_before_construction": True,
        "frozen_config": FROZEN_CONFIG,
        "input_fingerprints": input_fingerprints,
        "feature_cache_audit": feature_cache["audit"],
        "candidate_audit": feature_cache.get("candidate_audit"),
        "per_seed_metrics": all_metrics,
        "seed_runtime_seconds": seed_timings,
        "encoder_coverage": (
            encoder.get_coverage_metrics()
            if args.feature_mode == "3d+prior"
            else None
        ),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Completed controlled study arm %s", args.feature_mode)


if __name__ == "__main__":
    main()
