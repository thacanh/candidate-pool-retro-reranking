#!/usr/bin/env python
"""Paired post-freeze inference for the D5 rxn-ebm FF-EBM comparison."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rerank.analysis.analyze_controlled_study import (
    clustered_bootstrap,
    paired_cluster_permutation_test,
)
from rerank.external_ebm import PUBLISHED_EBM_SEEDS, PROTOCOL_ID
from rerank.revision_tuning import file_fingerprint
from rerank.study_data import canonicalize_smiles


EXPECTED_COVERED_REACTIONS = 3_985
EXPECTED_OFFICIAL_TEST_REACTIONS = 5_004
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
PERMUTATION_REPLICATES = 100_000
PERMUTATION_SEED = 2027


def _verify_record(record: Mapping[str, Any], path: Path) -> None:
    actual = file_fingerprint(path)
    if (actual["size_bytes"], actual["sha256"]) != (
        int(record["size_bytes"]),
        record["sha256"],
    ):
        raise RuntimeError(f"Frozen fingerprint differs after relocation: {path}")


def _load_reference(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("reaction_id").reset_index(drop=True)
    required = {
        "reaction_id",
        "source_split",
        "candidate_count",
        "product_smiles",
        "baseline_hit@1",
        "baseline_rr",
        "baseline_rank",
        "reranked_hit@1",
        "reranked_rr",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Primary reference lacks columns: {sorted(missing)}")
    if len(frame) != EXPECTED_COVERED_REACTIONS or frame["reaction_id"].duplicated().any():
        raise ValueError("Primary reference does not contain 3,985 unique covered reactions.")
    if not (frame["source_split"].astype(str) == "test").all():
        raise PermissionError("Primary reference contains a non-test reaction.")
    return frame


def _load_external(path: Path, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("reaction_id").reset_index(drop=True)
    required = {
        "reaction_id",
        "seed",
        "candidate_count",
        "prior_true_rank",
        "external_true_rank",
        "external_top1_candidate",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"External prediction lacks columns: {sorted(missing)}")
    if len(frame) != EXPECTED_COVERED_REACTIONS or frame["reaction_id"].duplicated().any():
        raise ValueError(f"External seed {seed} does not contain exact covered reactions.")
    if not (frame["seed"].astype(int) == seed).all():
        raise ValueError(f"External prediction has the wrong seed: {path}")
    return frame


def align_predictions(reference: pd.DataFrame, external: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "reaction_id",
        "candidate_count",
        "prior_true_rank",
        "external_true_rank",
        "external_top1_candidate",
    ]
    aligned = reference.merge(
        external.loc[:, columns],
        on="reaction_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_primary", "_external"),
    )
    if len(aligned) != len(reference):
        raise ValueError("D5 and primary predictions do not pair exactly.")
    if not np.array_equal(
        aligned["candidate_count_primary"].to_numpy(dtype=int),
        aligned["candidate_count_external"].to_numpy(dtype=int),
    ):
        raise ValueError("D5 changed candidate counts.")
    if not np.array_equal(
        aligned["baseline_rank"].to_numpy(dtype=int),
        aligned["prior_true_rank"].to_numpy(dtype=int),
    ):
        raise ValueError("D5 candidate-prior ranks differ from the frozen primary reference.")
    return aligned


def build_paired_matrices(
    aligned_by_seed: Mapping[int, pd.DataFrame],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    top1_rows: list[np.ndarray] = []
    mrr_rows: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    for seed in PUBLISHED_EBM_SEEDS:
        frame = aligned_by_seed[seed]
        prior_rank = frame["prior_true_rank"].to_numpy(dtype=int)
        external_rank = frame["external_true_rank"].to_numpy(dtype=int)
        candidate_count = frame["candidate_count_external"].to_numpy(dtype=int)
        if np.any(prior_rank < 1) or np.any(prior_rank > candidate_count):
            raise ValueError(f"Invalid prior rank for D5 seed {seed}.")
        if np.any(external_rank < 1) or np.any(external_rank > candidate_count):
            raise ValueError(f"Invalid external rank for D5 seed {seed}.")
        prior_top1 = (prior_rank == 1).astype(np.int8)
        external_top1 = (external_rank == 1).astype(np.int8)
        top1_delta = external_top1 - prior_top1
        mrr_delta = 1.0 / external_rank - 1.0 / prior_rank
        rank_delta = prior_rank - external_rank
        top1_rows.append(top1_delta)
        mrr_rows.append(mrr_delta)
        summaries.append(
            {
                "seed": seed,
                "n_reactions": len(frame),
                "candidate_prior_top1": float(prior_top1.mean()),
                "external_top1": float(external_top1.mean()),
                "top1_delta_external_minus_prior": float(top1_delta.mean()),
                "candidate_prior_mrr": float(np.mean(1.0 / prior_rank)),
                "external_mrr": float(np.mean(1.0 / external_rank)),
                "mrr_delta_external_minus_prior": float(mrr_delta.mean()),
                "top1_promoted": int(np.sum(top1_delta == 1)),
                "top1_degraded": int(np.sum(top1_delta == -1)),
                "top1_unchanged": int(np.sum(top1_delta == 0)),
                "rank_improved": int(np.sum(rank_delta > 0)),
                "rank_degraded": int(np.sum(rank_delta < 0)),
                "rank_unchanged": int(np.sum(rank_delta == 0)),
            }
        )
    return {
        "top1": np.stack(top1_rows),
        "mrr": np.stack(mrr_rows),
    }, pd.DataFrame(summaries)


def _aggregate_metric(
    differences: np.ndarray,
    clusters: np.ndarray,
    candidate_prior: float,
    external_mean: float,
    bootstrap_rng: np.random.Generator,
    permutation_rng: np.random.Generator,
) -> dict[str, Any]:
    point, low, high = clustered_bootstrap(
        differences,
        clusters,
        BOOTSTRAP_REPLICATES,
        bootstrap_rng,
    )
    per_seed = differences.mean(axis=1)
    coverage = EXPECTED_COVERED_REACTIONS / EXPECTED_OFFICIAL_TEST_REACTIONS
    return {
        "candidate_prior_conditional": candidate_prior,
        "external_conditional_mean": external_mean,
        "delta_external_minus_prior": point,
        "ci95": [low, high],
        "paired_cluster_sign_flip_p": paired_cluster_permutation_test(
            differences,
            clusters,
            PERMUTATION_REPLICATES,
            permutation_rng,
        ),
        "per_seed_deltas": {
            str(seed): float(value)
            for seed, value in zip(PUBLISHED_EBM_SEEDS, per_seed)
        },
        "positive_seed_count": int(np.sum(per_seed > 0)),
        "negative_seed_count": int(np.sum(per_seed < 0)),
        "zero_seed_count": int(np.sum(per_seed == 0)),
        "candidate_prior_end_to_end": candidate_prior * coverage,
        "external_end_to_end_mean": external_mean * coverage,
        "end_to_end_delta_external_minus_prior": point * coverage,
    }


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite D5 analysis CSV: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite D5 analysis JSON: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite D5 analysis: {output_dir}")

    d5_manifest_path = Path(args.d5_manifest).resolve()
    primary_manifest_path = Path(args.primary_manifest).resolve()
    d5_manifest = json.loads(d5_manifest_path.read_text(encoding="utf-8"))
    primary_manifest = json.loads(primary_manifest_path.read_text(encoding="utf-8"))
    if (
        d5_manifest.get("protocol_id") != PROTOCOL_ID
        or d5_manifest.get("test_partition_loaded_only_after_freeze") is not True
    ):
        raise PermissionError("D5 analysis requires a valid post-freeze test manifest.")
    if (
        primary_manifest.get("protocol_id") != "cap10-tuned-v1"
        or primary_manifest.get("test_partition_loaded_only_after_selection_freeze")
        is not True
    ):
        raise PermissionError("D5 analysis requires the frozen primary test manifest.")
    d5_cache = d5_manifest["train_test_cache"]
    primary_cache = primary_manifest["train_test_cache"]
    if (d5_cache["size_bytes"], d5_cache["sha256"]) != (
        primary_cache["size_bytes"],
        primary_cache["sha256"],
    ):
        raise ValueError("D5 and primary results belong to different test caches.")

    primary_prediction = Path(args.primary_prediction).resolve()
    reference = _load_reference(primary_prediction)
    primary_seed_metrics = primary_manifest["per_seed_metrics"]["baseline"]["42"]
    if not np.isclose(
        reference["reranked_hit@1"].mean(),
        primary_seed_metrics["top1"],
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        reference["reranked_rr"].mean(),
        primary_seed_metrics["mrr"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Primary prediction is not the frozen seed-42 baseline file.")

    d5_root = d5_manifest_path.parent
    aligned: dict[int, pd.DataFrame] = {}
    input_predictions: dict[str, dict[str, Any]] = {}
    for seed in PUBLISHED_EBM_SEEDS:
        path = d5_root / f"predictions_seed_{seed}.csv"
        _verify_record(d5_manifest["prediction_files"][str(seed)], path)
        frame = _load_external(path, seed)
        aligned[seed] = align_predictions(reference, frame)
        input_predictions[str(seed)] = file_fingerprint(path)

    matrices, per_seed = build_paired_matrices(aligned)
    for seed in PUBLISHED_EBM_SEEDS:
        row = per_seed.loc[per_seed["seed"] == seed].iloc[0]
        expected = d5_manifest["external_per_seed_metrics"][str(seed)]
        if not np.isclose(row["external_top1"], expected["top1"], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"D5 Top-1 prediction/manifest mismatch for seed {seed}.")
        if not np.isclose(row["external_mrr"], expected["mrr"], rtol=0.0, atol=1e-12):
            raise RuntimeError(f"D5 MRR prediction/manifest mismatch for seed {seed}.")

    clusters = np.asarray(
        [canonicalize_smiles(value) or str(value) for value in reference["product_smiles"]],
        dtype=object,
    )
    bootstrap_rng = np.random.default_rng(BOOTSTRAP_SEED)
    permutation_rng = np.random.default_rng(PERMUTATION_SEED)
    aggregate = {
        "top1": _aggregate_metric(
            matrices["top1"],
            clusters,
            float(d5_manifest["baseline_prior_metrics"]["top1"]),
            float(d5_manifest["external_mean_metrics"]["top1"]),
            bootstrap_rng,
            permutation_rng,
        ),
        "mrr": _aggregate_metric(
            matrices["mrr"],
            clusters,
            float(d5_manifest["baseline_prior_metrics"]["mrr"]),
            float(d5_manifest["external_mean_metrics"]["mrr"]),
            bootstrap_rng,
            permutation_rng,
        ),
    }
    shift_columns = [
        "top1_promoted",
        "top1_degraded",
        "top1_unchanged",
        "rank_improved",
        "rank_degraded",
        "rank_unchanged",
    ]
    rank_shift_totals = {
        column: int(per_seed[column].sum()) for column in shift_columns
    }
    rank_shift_totals["top1_net_promotions"] = (
        rank_shift_totals["top1_promoted"] - rank_shift_totals["top1_degraded"]
    )
    rank_shift_totals["rank_net_improvements"] = (
        rank_shift_totals["rank_improved"] - rank_shift_totals["rank_degraded"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = output_dir / "per_seed_rank_shifts.csv"
    _atomic_csv(per_seed, per_seed_path)
    result = {
        "schema_version": 1,
        "record_kind": "external_ebm_paired_clustered_inference",
        "protocol_id": PROTOCOL_ID,
        "comparator": "frozen cap-10 candidate-prior ordering",
        "single_intended_change": "published rxn-ebm feedforward energy reranker",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_seeds": len(PUBLISHED_EBM_SEEDS),
        "seeds": list(PUBLISHED_EBM_SEEDS),
        "n_covered_reactions": EXPECTED_COVERED_REACTIONS,
        "n_official_test_reactions": EXPECTED_OFFICIAL_TEST_REACTIONS,
        "coverage": EXPECTED_COVERED_REACTIONS / EXPECTED_OFFICIAL_TEST_REACTIONS,
        "n_product_clusters": int(len(np.unique(clusters))),
        "aggregate": aggregate,
        "rank_shift_totals_across_seed_reaction_pairs": rank_shift_totals,
        "methods": {
            "bootstrap": {
                "method": "canonical-product-clustered paired bootstrap retaining all seed rows",
                "replicates": BOOTSTRAP_REPLICATES,
                "rng_seed": BOOTSTRAP_SEED,
            },
            "permutation": {
                "method": "two-sided sign flip of seed-averaged reaction differences by canonical product cluster",
                "replicates": PERMUTATION_REPLICATES,
                "rng_seed": PERMUTATION_SEED,
            },
            "end_to_end": "assign zero credit to all uncovered official-test reactions",
        },
        "inputs": {
            "d5_manifest": file_fingerprint(d5_manifest_path),
            "primary_manifest": file_fingerprint(primary_manifest_path),
            "primary_prediction": file_fingerprint(primary_prediction),
            "d5_predictions": input_predictions,
        },
        "per_seed_rank_shifts": file_fingerprint(per_seed_path),
        "test_partition_used_for_training_or_selection": False,
    }
    _atomic_json(result, output_dir / "paired_inference.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d5-manifest", required=True)
    parser.add_argument("--primary-manifest", required=True)
    parser.add_argument("--primary-prediction", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    result = analyze(build_parser().parse_args())
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    print(json.dumps(result["rank_shift_totals_across_seed_reaction_pairs"], indent=2))


if __name__ == "__main__":
    main()
