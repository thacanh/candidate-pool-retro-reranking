#!/usr/bin/env python
"""Analyze the frozen F3 current-handling versus salt-removal predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rerank.analysis.analyze_controlled_study import (
    clustered_bootstrap,
    paired_cluster_permutation_test,
)
from rerank.revision_tuning import file_fingerprint
from rerank.study_data import canonicalize_smiles
from rerank.experiments.run_salt_sensitivity import F3_PROTOCOL_ID, SEEDS


EXPECTED_TEST_REACTIONS = 5_004
EXPECTED_COVERED_REACTIONS = 3_985
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
PERMUTATION_REPLICATES = 100_000
PERMUTATION_SEED = 2027


def _load_prediction(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path).sort_values("reaction_id").reset_index(drop=True)
    required = {
        "reaction_id",
        "product_smiles",
        "ground_truth",
        "reranked_top1",
        "reranked_hit@1",
        "reranked_rr",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if len(frame) != EXPECTED_COVERED_REACTIONS:
        raise ValueError(
            f"{path} has {len(frame)} rows; expected {EXPECTED_COVERED_REACTIONS}."
        )
    if frame["reaction_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate reaction IDs.")
    return frame


def align_predictions(current: pd.DataFrame, salt: pd.DataFrame) -> pd.DataFrame:
    aligned = current.merge(
        salt,
        on="reaction_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_current", "_salt"),
    )
    if len(aligned) != EXPECTED_COVERED_REACTIONS:
        raise ValueError("Current/salt predictions do not pair exactly.")
    for column in ("product_smiles", "ground_truth"):
        if not aligned[f"{column}_current"].equals(aligned[f"{column}_salt"]):
            raise ValueError(f"Current/salt predictions disagree on {column}.")
    return aligned


def paired_statistics(
    aligned_by_seed: dict[int, pd.DataFrame], arm: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    top1_differences: list[np.ndarray] = []
    mrr_differences: list[np.ndarray] = []
    for seed in SEEDS:
        frame = aligned_by_seed[seed]
        current_top1 = frame["reranked_hit@1_current"].to_numpy(dtype=int)
        salt_top1 = frame["reranked_hit@1_salt"].to_numpy(dtype=int)
        current_rr = frame["reranked_rr_current"].to_numpy(dtype=float)
        salt_rr = frame["reranked_rr_salt"].to_numpy(dtype=float)
        top1_delta = salt_top1 - current_top1
        mrr_delta = salt_rr - current_rr
        top1_differences.append(top1_delta)
        mrr_differences.append(mrr_delta)
        per_seed.append(
            {
                "arm": arm,
                "seed": seed,
                "n_reactions": len(frame),
                "top1_current": float(current_top1.mean()),
                "top1_salt_removed": float(salt_top1.mean()),
                "top1_delta_salt_minus_current": float(top1_delta.mean()),
                "mrr_current": float(current_rr.mean()),
                "mrr_salt_removed": float(salt_rr.mean()),
                "mrr_delta_salt_minus_current": float(mrr_delta.mean()),
                "top1_promoted": int(np.sum(top1_delta == 1)),
                "top1_lost": int(np.sum(top1_delta == -1)),
                "top1_unchanged": int(np.sum(top1_delta == 0)),
                "top1_prediction_changed": int(
                    np.sum(
                        frame["reranked_top1_current"].astype(str).to_numpy()
                        != frame["reranked_top1_salt"].astype(str).to_numpy()
                    )
                ),
            }
        )

    top1_matrix = np.stack(top1_differences)
    mrr_matrix = np.stack(mrr_differences)
    first = aligned_by_seed[SEEDS[0]]
    clusters = np.asarray(
        [
            canonicalize_smiles(value) or str(value)
            for value in first["product_smiles_current"]
        ],
        dtype=object,
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    top1_point, top1_low, top1_high = clustered_bootstrap(
        top1_matrix, clusters, BOOTSTRAP_REPLICATES, rng
    )
    mrr_point, mrr_low, mrr_high = clustered_bootstrap(
        mrr_matrix, clusters, BOOTSTRAP_REPLICATES, rng
    )
    permutation_rng = np.random.default_rng(PERMUTATION_SEED)
    aggregate = {
        "arm": arm,
        "n_seeds": len(SEEDS),
        "n_reactions": len(first),
        "n_product_clusters": int(len(np.unique(clusters))),
        "top1_delta_salt_minus_current": top1_point,
        "top1_ci95": [top1_low, top1_high],
        "top1_paired_cluster_sign_flip_p": paired_cluster_permutation_test(
            top1_matrix, clusters, PERMUTATION_REPLICATES, permutation_rng
        ),
        "mrr_delta_salt_minus_current": mrr_point,
        "mrr_ci95": [mrr_low, mrr_high],
        "mrr_paired_cluster_sign_flip_p": paired_cluster_permutation_test(
            mrr_matrix, clusters, PERMUTATION_REPLICATES, permutation_rng
        ),
        "bootstrap": {
            "method": "canonical-product-clustered paired bootstrap",
            "replicates": BOOTSTRAP_REPLICATES,
            "rng_seed": BOOTSTRAP_SEED,
        },
        "permutation": {
            "method": "two-sided sign flip of seed-averaged reaction differences by canonical product cluster",
            "replicates": PERMUTATION_REPLICATES,
            "rng_seed": PERMUTATION_SEED,
        },
    }
    return per_seed, aggregate


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite F3 analysis: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_manifest_path = Path(args.f3_test_manifest).resolve()
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    if (
        test_manifest.get("protocol_id") != F3_PROTOCOL_ID
        or not test_manifest.get("test_partition_loaded_only_after_f3_freeze")
    ):
        raise PermissionError("F3 analysis requires a valid post-freeze test manifest.")

    current_dir = Path(args.current_prediction_dir).resolve()
    salt_dir = Path(args.salt_prediction_dir).resolve()
    per_seed_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "augmented"):
        aligned_by_seed: dict[int, pd.DataFrame] = {}
        inputs[arm] = {}
        for seed in SEEDS:
            current_path = current_dir / f"{arm}_seed_{seed}.csv"
            salt_path = salt_dir / f"{arm}_seed_{seed}.csv"
            aligned_by_seed[seed] = align_predictions(
                _load_prediction(current_path), _load_prediction(salt_path)
            )
            inputs[arm][str(seed)] = {
                "current": file_fingerprint(current_path),
                "salt_removed": file_fingerprint(salt_path),
            }
        seed_rows, aggregate[arm] = paired_statistics(aligned_by_seed, arm)
        per_seed_rows.extend(seed_rows)

    per_seed_path = output_dir / "per_seed_paired_metrics.csv"
    aggregate_path = output_dir / "paired_statistics.json"
    pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "record_kind": "f3_salt_analysis",
        "protocol_id": F3_PROTOCOL_ID,
        "f3_test_manifest": file_fingerprint(test_manifest_path),
        "prediction_inputs": inputs,
        "outputs": {
            "per_seed_paired_metrics": file_fingerprint(per_seed_path),
            "paired_statistics": file_fingerprint(aggregate_path),
        },
        "denominators": {
            "official_test_reactions": EXPECTED_TEST_REACTIONS,
            "covered_reactions": EXPECTED_COVERED_REACTIONS,
            "uncovered_reactions": EXPECTED_TEST_REACTIONS
            - EXPECTED_COVERED_REACTIONS,
        },
        "single_intended_change": "RDKit default SaltRemover molecular input",
        "ground_truth_matching_changed": False,
        "test_partition_used_for_selection": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f3-test-manifest", required=True)
    parser.add_argument("--current-prediction-dir", required=True)
    parser.add_argument("--salt-prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(analyze(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
