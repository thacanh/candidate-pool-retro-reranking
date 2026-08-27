#!/usr/bin/env python
"""Analyze the frozen F2 Morgan-chirality sensitivity predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from rerank.analysis.analyze_controlled_study import (
    clustered_bootstrap,
    paired_cluster_permutation_test,
)
from rerank.revision_tuning import file_fingerprint
from rerank.study_data import canonicalize_smiles
from rerank.experiments.run_chirality_sensitivity import F2_PROTOCOL_ID, SEEDS


EXPECTED_TEST_REACTIONS = 5_004
EXPECTED_COVERED_REACTIONS = 3_985
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2026
PERMUTATION_REPLICATES = 100_000
PERMUTATION_SEED = 2027


def canonical_fragment_set(smiles: str, *, preserve_stereochemistry: bool) -> tuple[str, ...] | None:
    """Return a fragment-order-invariant identity, or None for invalid input."""
    fragments: list[str] = []
    for fragment in str(smiles).split("."):
        molecule = Chem.MolFromSmiles(fragment)
        if molecule is None:
            return None
        fragments.append(
            Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=preserve_stereochemistry,
            )
        )
    return tuple(sorted(fragments))


def is_stereo_only_reference_difference(prediction: str, reference: str) -> bool:
    predicted_stereo = canonical_fragment_set(
        prediction, preserve_stereochemistry=True
    )
    reference_stereo = canonical_fragment_set(reference, preserve_stereochemistry=True)
    predicted_flat = canonical_fragment_set(
        prediction, preserve_stereochemistry=False
    )
    reference_flat = canonical_fragment_set(reference, preserve_stereochemistry=False)
    return bool(
        predicted_stereo is not None
        and reference_stereo is not None
        and predicted_flat is not None
        and reference_flat is not None
        and predicted_stereo != reference_stereo
        and predicted_flat == reference_flat
    )


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


def _align(false_frame: pd.DataFrame, true_frame: pd.DataFrame) -> pd.DataFrame:
    aligned = false_frame.merge(
        true_frame,
        on="reaction_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_false", "_true"),
    )
    if len(aligned) != EXPECTED_COVERED_REACTIONS:
        raise ValueError("False/true chirality predictions do not pair exactly.")
    for column in ("product_smiles", "ground_truth"):
        if not aligned[f"{column}_false"].equals(aligned[f"{column}_true"]):
            raise ValueError(f"False/true predictions disagree on {column}.")
    return aligned


def _paired_statistics(
    aligned_by_seed: dict[int, pd.DataFrame], arm: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_seed: list[dict[str, Any]] = []
    top1_differences: list[np.ndarray] = []
    mrr_differences: list[np.ndarray] = []
    for seed in SEEDS:
        frame = aligned_by_seed[seed]
        false_top1 = frame["reranked_hit@1_false"].to_numpy(dtype=int)
        true_top1 = frame["reranked_hit@1_true"].to_numpy(dtype=int)
        false_rr = frame["reranked_rr_false"].to_numpy(dtype=float)
        true_rr = frame["reranked_rr_true"].to_numpy(dtype=float)
        top1_delta = true_top1 - false_top1
        rr_delta = true_rr - false_rr
        top1_differences.append(top1_delta)
        mrr_differences.append(rr_delta)
        per_seed.append(
            {
                "arm": arm,
                "seed": seed,
                "n_reactions": len(frame),
                "top1_false": float(false_top1.mean()),
                "top1_true": float(true_top1.mean()),
                "top1_delta_true_minus_false": float(top1_delta.mean()),
                "mrr_false": float(false_rr.mean()),
                "mrr_true": float(true_rr.mean()),
                "mrr_delta_true_minus_false": float(rr_delta.mean()),
                "top1_promoted": int(np.sum(top1_delta == 1)),
                "top1_lost": int(np.sum(top1_delta == -1)),
                "top1_unchanged": int(np.sum(top1_delta == 0)),
            }
        )

    top1_matrix = np.stack(top1_differences)
    mrr_matrix = np.stack(mrr_differences)
    first = aligned_by_seed[SEEDS[0]]
    clusters = np.asarray(
        [
            canonicalize_smiles(value) or str(value)
            for value in first["product_smiles_false"]
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
        "top1_delta_true_minus_false": top1_point,
        "top1_ci95": [top1_low, top1_high],
        "top1_paired_cluster_sign_flip_p": paired_cluster_permutation_test(
            top1_matrix, clusters, PERMUTATION_REPLICATES, permutation_rng
        ),
        "mrr_delta_true_minus_false": mrr_point,
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
        raise FileExistsError(f"Refusing to overwrite F2 analysis: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_manifest_path = Path(args.f2_test_manifest).resolve()
    test_manifest = json.loads(test_manifest_path.read_text(encoding="utf-8"))
    if (
        test_manifest.get("protocol_id") != F2_PROTOCOL_ID
        or not test_manifest.get("test_partition_loaded_only_after_f2_freeze")
    ):
        raise PermissionError("F2 analysis requires a valid post-freeze test manifest.")

    false_dir = Path(args.false_prediction_dir).resolve()
    true_dir = Path(args.true_prediction_dir).resolve()
    per_seed_rows: list[dict[str, Any]] = []
    stereo_rows: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {}
    inputs: dict[str, dict[str, dict[str, Any]]] = {}
    for arm in ("baseline", "augmented"):
        aligned_by_seed: dict[int, pd.DataFrame] = {}
        inputs[arm] = {}
        for seed in SEEDS:
            false_path = false_dir / f"{arm}_seed_{seed}.csv"
            true_path = true_dir / f"{arm}_seed_{seed}.csv"
            false_frame = _load_prediction(false_path)
            true_frame = _load_prediction(true_path)
            aligned_by_seed[seed] = _align(false_frame, true_frame)
            inputs[arm][str(seed)] = {
                "false": file_fingerprint(false_path),
                "true": file_fingerprint(true_path),
            }
            for setting, frame in (("false", false_frame), ("true", true_frame)):
                count = sum(
                    is_stereo_only_reference_difference(prediction, reference)
                    for prediction, reference in zip(
                        frame["reranked_top1"], frame["ground_truth"], strict=True
                    )
                )
                stereo_rows.append(
                    {
                        "arm": arm,
                        "morgan_use_chirality": setting == "true",
                        "seed": seed,
                        "official_test_reactions": EXPECTED_TEST_REACTIONS,
                        "covered_reactions": EXPECTED_COVERED_REACTIONS,
                        "uncovered_reactions": EXPECTED_TEST_REACTIONS
                        - EXPECTED_COVERED_REACTIONS,
                        "stereo_only_reference_differences": count,
                    }
                )
        seed_rows, aggregate[arm] = _paired_statistics(aligned_by_seed, arm)
        per_seed_rows.extend(seed_rows)

    per_seed_path = output_dir / "per_seed_paired_metrics.csv"
    stereo_path = output_dir / "stereo_only_counts.csv"
    aggregate_path = output_dir / "paired_statistics.json"
    pd.DataFrame(per_seed_rows).to_csv(per_seed_path, index=False)
    pd.DataFrame(stereo_rows).to_csv(stereo_path, index=False)
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "record_kind": "f2_chirality_analysis",
        "protocol_id": F2_PROTOCOL_ID,
        "f2_test_manifest": file_fingerprint(test_manifest_path),
        "prediction_inputs": inputs,
        "outputs": {
            "per_seed_paired_metrics": file_fingerprint(per_seed_path),
            "stereo_only_counts": file_fingerprint(stereo_path),
            "paired_statistics": file_fingerprint(aggregate_path),
        },
        "denominators": {
            "official_test_reactions": EXPECTED_TEST_REACTIONS,
            "covered_reactions": EXPECTED_COVERED_REACTIONS,
            "uncovered_reactions": EXPECTED_TEST_REACTIONS
            - EXPECTED_COVERED_REACTIONS,
        },
        "stereo_only_definition": (
            "stereochemistry-stripped canonical reactant fragment sets match, "
            "but stereochemistry-preserving sets do not"
        ),
        "single_intended_change": "Morgan useChirality=False to True",
        "test_partition_used_for_selection": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f2-test-manifest", required=True)
    parser.add_argument("--false-prediction-dir", required=True)
    parser.add_argument("--true-prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    print(json.dumps(analyze(parser.parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
