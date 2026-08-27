import numpy as np
import pytest
import rerank.experiments.run_lightgbm_revision as d4

from rerank.experiments.run_lightgbm_revision import (
    build_query_arrays,
    conditional_mrr,
    enumerate_grid,
    group_fingerprint,
)


def _bundle():
    matrix = np.asarray(
        [[0.8, 1, 2, 3, 4, 5, 6], [0.2, 7, 8, 9, 10, 11, 12]],
        dtype=np.float32,
    )
    candidates = [{"smiles": "C", "prior": 0.8}, {"smiles": "CC", "prior": 0.2}]
    return {
        "train_products": [{
            "features": matrix, "positive_indices": [0], "negative_indices": [1]
        }],
        "validation_payload": {
            "eval_pwc": [("CO", candidates)],
            "eval_ground_truths": ["C"],
            "eval_features": [matrix],
        },
    }


def test_d4_grid_order_and_equal_query_groups():
    grid = enumerate_grid()
    assert len(grid) == 27
    assert (grid[0].num_leaves, grid[0].min_child_samples, grid[0].learning_rate) == (7, 10, 0.03)
    assert (grid[-1].num_leaves, grid[-1].min_child_samples, grid[-1].learning_rate) == (31, 50, 0.2)
    baseline = build_query_arrays(_bundle(), "baseline", "raw")
    augmented = build_query_arrays(_bundle(), "augmented", "raw")
    assert baseline["train_x"].shape[1] == 4
    assert augmented["train_x"].shape[1] == 7
    assert group_fingerprint(baseline) == group_fingerprint(augmented)


def test_reaction_level_mrr_uses_stable_within_query_ranks():
    scores = np.asarray([0.1, 0.9, 0.5, 0.5])
    labels = np.asarray([1, 0, 0, 1])
    assert conditional_mrr(scores, labels, [2, 2]) == pytest.approx((0.5 + 0.5) / 2)
    with pytest.raises(ValueError, match="misaligned"):
        conditional_mrr(scores, labels, [3])


def test_pinned_lightgbm_api_trains_and_saves_tiny_model(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm", reason="D4 training requires optional lightgbm==4.6.0")
    arrays = build_query_arrays(_bundle(), "baseline", "raw")
    monkeypatch.setattr(d4, "MAX_TREES", 5)
    monkeypatch.setattr(d4, "EARLY_STOPPING_ROUNDS", 2)
    model = tmp_path / "model.txt"
    result = d4._train_one(arrays, enumerate_grid()[0], model)
    assert model.is_file()
    assert result["best_iteration"] >= 1
    assert 0.0 <= result["best_validation_mrr"] <= 1.0
