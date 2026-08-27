from __future__ import annotations

import numpy as np

from rerank.revision_tuning import SEEDS, validate_selection_bundle
from rerank.experiments.run_b3_avg10_ranking import _metric_summary, assert_test_pairing


def _candidates():
    return [
        {"smiles": "CC.O", "canonical_smiles": "CC.O", "prior": 0.8},
        {"smiles": "C.CO", "canonical_smiles": "C.CO", "prior": 0.2},
    ]


def _payload():
    matrix = np.arange(14, dtype=np.float32).reshape(2, 7)
    return {
        "eval_pwc": [("CCO", _candidates())],
        "eval_ground_truths": ["CC.O"],
        "eval_metadata": [{"source_split": "test", "reaction_id": "1"}],
        "eval_features": [matrix],
    }


def test_multi_conformer_selection_provenance_is_explicit():
    matrix = np.arange(14, dtype=np.float32).reshape(2, 7)
    bundle = {
        "selection_bundle_schema": 1,
        "protocol_id": "cap10-tuned-v1",
        "seeds": list(SEEDS),
        "representation_provenance": {
            "kind": "multi_conformer_scalar_average",
            "encoder_control_has_conformer": True,
            "conformer_seed": None,
            "conformer_seeds": list(range(42, 52)),
            "aggregation": (
                "arithmetic mean of each pair-level scalar after fragment handling"
            ),
            "atom_embeddings_averaged": False,
        },
        "train_products": [
            {
                "features": matrix,
                "candidates": _candidates(),
                "positive_indices": [0],
                "negative_indices": [1],
            }
        ],
        "validation_payload": {
            "eval_pwc": [("CCO", _candidates())],
            "eval_ground_truths": ["CC.O"],
            "eval_metadata": [{"source_split": "valid"}],
            "eval_features": [matrix],
        },
    }
    validate_selection_bundle(bundle)


def test_b3_pairing_rejects_changed_baseline_columns():
    primary = _payload()
    b3 = _payload()
    assert assert_test_pairing(primary, b3)["status"] == "exact"
    b3["eval_features"][0][0, 1] += 1
    try:
        assert_test_pairing(primary, b3)
    except RuntimeError as error:
        assert "prior+2D columns differ" in str(error)
    else:
        raise AssertionError("Changed B3 baseline column was not rejected.")


def test_metric_summary_reports_b3_deltas():
    b3 = {
        str(seed): {metric: 0.8 for metric in ("top1", "top3", "top5", "top10", "mrr")}
        for seed in SEEDS
    }
    primary = {
        "per_seed_metrics": {
            "augmented": {
                str(seed): {
                    metric: 0.7
                    for metric in ("top1", "top3", "top5", "top10", "mrr")
                }
                for seed in SEEDS
            },
            "baseline": {
                str(seed): {
                    metric: 0.6
                    for metric in ("top1", "top3", "top5", "top10", "mrr")
                }
                for seed in SEEDS
            },
        }
    }
    summary = _metric_summary(b3, primary)
    assert np.isclose(summary["b3_minus_single_conformer"]["top1"]["mean"], 0.1)
    assert np.isclose(summary["b3_minus_2d"]["mrr"]["mean"], 0.2)
