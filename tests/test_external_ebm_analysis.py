from __future__ import annotations

import numpy as np
import pandas as pd

from rerank.analysis.analyze_external_ebm import align_predictions, build_paired_matrices
from rerank.external_ebm import PUBLISHED_EBM_SEEDS


def _reference() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [1, 2, 3],
            "candidate_count": [3, 3, 3],
            "baseline_rank": [1, 2, 3],
            "product_smiles": ["CC", "CO", "CN"],
        }
    )


def _external(seed: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [1, 2, 3],
            "candidate_count": [3, 3, 3],
            "prior_true_rank": [1, 2, 3],
            "external_true_rank": [2, 1, 3],
            "external_top1_candidate": ["A", "B", "C"],
            "seed": [seed, seed, seed],
        }
    )


def test_external_analysis_pairs_and_counts_rank_changes():
    aligned = {
        seed: align_predictions(_reference(), _external(seed))
        for seed in PUBLISHED_EBM_SEEDS
    }
    matrices, summary = build_paired_matrices(aligned)
    assert matrices["top1"].shape == (3, 3)
    assert matrices["mrr"].shape == (3, 3)
    assert np.array_equal(matrices["top1"][0], [-1, 1, 0])
    assert (summary["top1_promoted"] == 1).all()
    assert (summary["top1_degraded"] == 1).all()
    assert (summary["rank_improved"] == 1).all()
    assert (summary["rank_degraded"] == 1).all()
    assert (summary["rank_unchanged"] == 1).all()


def test_external_analysis_rejects_prior_rank_change():
    external = _external(PUBLISHED_EBM_SEEDS[0])
    external.loc[0, "prior_true_rank"] = 2
    try:
        align_predictions(_reference(), external)
    except ValueError as error:
        assert "candidate-prior ranks differ" in str(error)
    else:
        raise AssertionError("Changed candidate-prior rank was not rejected.")


def test_external_analysis_rejects_candidate_count_change():
    external = _external(PUBLISHED_EBM_SEEDS[0])
    external.loc[0, "candidate_count"] = 2
    try:
        align_predictions(_reference(), external)
    except ValueError as error:
        assert "candidate counts" in str(error)
    else:
        raise AssertionError("Changed candidate count was not rejected.")
