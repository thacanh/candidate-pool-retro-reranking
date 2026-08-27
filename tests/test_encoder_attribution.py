import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rerank.analysis.analyze_encoder_attribution import (
    clustered_intervals,
    load_prediction_set,
)


PAIRING = {
    "reaction_id": [1, 2, 3, 4],
    "source_split": ["test"] * 4,
    "reaction_class": [1, 1, 2, 2],
    "candidate_count": [2] * 4,
    "coverage_rank": [1] * 4,
    "product_smiles": ["A", "B", "C", "D"],
    "ground_truth": ["a", "b", "c", "d"],
    "baseline_hit@1": [0, 0, 1, 1],
    "baseline_hit@3": [1] * 4,
    "baseline_hit@5": [1] * 4,
    "baseline_hit@10": [1] * 4,
    "baseline_rr": [0.5, 0.5, 1.0, 1.0],
    "baseline_rank": [2, 2, 1, 1],
    "baseline_top1": ["x", "y", "c", "d"],
    "baseline_candidates_json": ["[]"] * 4,
}


def _write_result(root: Path, seeds=(42, 43), mismatch=False) -> None:
    predictions = root / "predictions"
    predictions.mkdir(parents=True)
    per_seed = {}
    for seed in seeds:
        frame = pd.DataFrame(PAIRING)
        if mismatch and seed == seeds[-1]:
            frame.loc[1, "product_smiles"] = "CHANGED"
        frame["reranked_hit@1"] = [0, 1, 1, 1]
        frame["reranked_rr"] = [0.5, 1.0, 1.0, 1.0]
        frame.to_csv(predictions / f"augmented_seed_{seed}.csv", index=False)
        per_seed[str(seed)] = {"top1": 0.75, "mrr": 0.875}
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "test_partition_loaded_only_after_both_freezes": True,
                "primary_2d_predictions_are_the_frozen_comparator": True,
                "per_seed_metrics": per_seed,
            }
        ),
        encoding="utf-8",
    )


def test_clustered_interval_is_deterministic_and_uses_seed_mean():
    differences = np.asarray(
        [[0.0, 0.1, 0.0, 0.1], [0.2, 0.1, 0.0, 0.1]], dtype=float
    )
    clusters = np.asarray(["A", "B", "C", "D"], dtype=object)
    first = clustered_intervals(differences, clusters, n_bootstrap=500, seed=17)
    second = clustered_intervals(differences, clusters, n_bootstrap=500, seed=17)
    assert first == second
    assert first["effect"] == pytest.approx(float(differences.mean()))
    assert first["ci90_low"] <= first["effect"] <= first["ci90_high"]
    assert first["ci95_low"] <= first["ci90_low"]
    assert first["ci95_high"] >= first["ci90_high"]


def test_clustered_interval_retains_multirow_product_clusters():
    differences = np.asarray([[0.0, 0.2, 0.4], [0.2, 0.0, 0.4]], dtype=float)
    clusters = np.asarray(["A", "A", "B"], dtype=object)
    result = clustered_intervals(differences, clusters, n_bootstrap=250, seed=9)
    assert result["effect"] == pytest.approx(float(differences.mean()))
    assert result["n_product_clusters"] == 2
    assert result["n_reactions"] == 3


def test_prediction_loader_checks_manifest_and_exact_pairing(tmp_path):
    root = tmp_path / "control"
    _write_result(root)
    loaded = load_prediction_set(
        "control",
        root,
        "augmented",
        "encoder_control",
        seeds=(42, 43),
        expected_reactions=4,
    )
    assert loaded.values["top1"].shape == (2, 4)
    assert loaded.values["mrr"].mean() == pytest.approx(0.875)
    assert len(loaded.files) == 3


def test_prediction_loader_rejects_pairing_mismatch(tmp_path):
    root = tmp_path / "control"
    _write_result(root, mismatch=True)
    with pytest.raises(ValueError, match="Pairing mismatch"):
        load_prediction_set(
            "control",
            root,
            "augmented",
            "encoder_control",
            seeds=(42, 43),
            expected_reactions=4,
        )
