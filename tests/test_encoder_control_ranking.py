import copy

import numpy as np
import pytest

from rerank.experiments.run_encoder_control_ranking import assert_selection_baseline_identity


def _bundle(kind):
    features = np.asarray(
        [[0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.1, 7, 8, 9, 10, 11, 12]],
        dtype=np.float32,
    )
    candidates = [
        {"smiles": "C", "canonical_smiles": "C", "prior": 0.9},
        {"smiles": "CC", "canonical_smiles": "CC", "prior": 0.1},
    ]
    return {
        "representation_provenance": {"kind": kind},
        "train_products": [{
            "product_key": "CO", "candidates": candidates,
            "positive_indices": [0], "negative_indices": [1],
            "features": features.copy(),
        }],
        "validation_payload": {
            "eval_pwc": [("CO", candidates)],
            "eval_ground_truths": ["C"],
            "eval_metadata": [{"reaction_id": 1, "source_split": "valid"}],
            "eval_features": [features.copy()],
        },
    }


def test_encoder_control_may_change_only_three_representation_columns():
    primary = _bundle("indexed_conformer")
    control = _bundle("encoder_control_without_conformer")
    control["train_products"][0]["features"][:, 2:5] += 100
    control["validation_payload"]["eval_features"][0][:, 2:5] -= 100
    result = assert_selection_baseline_identity(primary, control)
    assert result["status"] == "exact"

    broken = copy.deepcopy(control)
    broken["train_products"][0]["features"][0, 1] += 1
    with pytest.raises(RuntimeError, match="baseline columns differ"):
        assert_selection_baseline_identity(primary, broken)
