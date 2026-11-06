from __future__ import annotations

import numpy as np
import pytest

from rerank.experiments.cross_inference_features import clipping_audit, training_arm_features
from rerank.experiments.run_cross_inference import _checked_diagonal_rank, _score_lists


def test_all_four_schema_cells_and_arms() -> None:
    historical = np.arange(14, dtype=np.float32).reshape(2, 7)
    aiz = np.column_stack((historical, np.ones(2), np.zeros(2))).astype(np.float32)
    for arm, historical_dim, aiz_dim in (("baseline", 4, 6), ("augmented", 7, 9)):
        assert training_arm_features(
            historical,
            training_pool="historical_cap10",
            evaluation_pool="historical_cap10",
            arm=arm,
        ).shape == (2, historical_dim)
        assert training_arm_features(
            aiz, training_pool="historical_cap10", evaluation_pool="aizynthfinder_only", arm=arm
        ).shape == (2, historical_dim)
        assert training_arm_features(
            historical,
            training_pool="aizynthfinder_only",
            evaluation_pool="historical_cap10",
            arm=arm,
        ).shape == (2, aiz_dim)
        assert training_arm_features(
            aiz, training_pool="aizynthfinder_only", evaluation_pool="aizynthfinder_only", arm=arm
        ).shape == (2, aiz_dim)


def test_source_indicator_and_shape_gates() -> None:
    bad = np.zeros((2, 9), dtype=np.float32)
    with pytest.raises(ValueError, match="source indicators"):
        training_arm_features(
            bad,
            training_pool="historical_cap10",
            evaluation_pool="aizynthfinder_only",
            arm="baseline",
        )
    with pytest.raises(ValueError, match="finite"):
        training_arm_features(
            np.full((2, 7), np.nan, dtype=np.float32),
            training_pool="aizynthfinder_only",
            evaluation_pool="historical_cap10",
            arm="augmented",
        )


def test_clipping_audit_is_read_only_and_flags_prior_boundary() -> None:
    raw = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]], dtype=np.float32)
    before = raw.copy()
    audit = clipping_audit(raw, np.zeros(2), np.ones(2), 5.0)
    assert audit["prior_lower_fraction"] == pytest.approx(1 / 3)
    assert audit["prior_upper_fraction"] == pytest.approx(1 / 3)
    assert audit["uninterpretable_prior_clipping_gt_10pct"] is True
    np.testing.assert_array_equal(raw, before)


def test_diagonal_tie_exception_is_exact_input_only() -> None:
    same = np.ones((2, 4), dtype=np.float32)
    mask = np.array([False, True])
    local_scores = np.array([0.2, 0.2])
    frozen_scores = np.array([0.2000004, 0.2])
    rank, audit = _checked_diagonal_rank(2, 1, same, mask, local_scores, frozen_scores)
    assert rank == 1 and audit["indistinguishable_candidate_count"] == 2
    with pytest.raises(ValueError, match="diagonal reproduction"):
        _checked_diagonal_rank(2, 1, same, mask, local_scores, np.array([0.201, 0.2]))
    different = same.copy()
    different[1, 1] += 1
    with pytest.raises(ValueError, match="diagonal reproduction"):
        _checked_diagonal_rank(2, 1, different, mask, local_scores, frozen_scores)
    assert _checked_diagonal_rank(1, 1, different, mask, local_scores, None) == (1, None)

    mixed = np.vstack((np.array([[-1, 0, 0, 0]], dtype=np.float32), same))
    rank, audit = _checked_diagonal_rank(
        3,
        2,
        mixed,
        np.array([False, False, True]),
        np.array([1.0, 0.2, 0.2]),
        np.array([1.0, 0.2000004, 0.2]),
    )
    assert rank == 2 and audit["indistinguishable_candidate_count"] == 2


def test_scoring_keeps_reaction_batch_boundaries() -> None:
    class IdentityNormalizer:
        def transform(self, matrix):
            return matrix

    class BatchSensitiveModel:
        def score(self, matrix):
            return matrix[:, 0] + float(len(matrix))

    matrices = [np.zeros((2, 1), dtype=np.float32), np.zeros((3, 1), dtype=np.float32)]
    scores = _score_lists(BatchSensitiveModel(), IdentityNormalizer(), matrices, "cpu")
    np.testing.assert_array_equal(scores[0], np.full(2, 2, dtype=np.float32))
    np.testing.assert_array_equal(scores[1], np.full(3, 3, dtype=np.float32))
