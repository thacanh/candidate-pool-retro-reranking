import numpy as np
import pytest

from rerank.experiments.run_projected_embedding_probe import (
    PREDICTION_METRIC_COLUMNS,
    collect_existing_prediction_metrics,
)
from rerank.projected_probe import (
    BASE_DIM, HEAD_INPUT_DIM, POOLED_DIM, RAW_INPUT_DIM, ProjectedRanker,
    expected_parameter_count, fit_pair_normalizer, make_pair_indices,
    materialize_rows, selected_base_features,
)


def _product(offset=0.0):
    return {
        "candidates": [{"smiles": f"C{i}"} for i in range(8)],
        "positive_indices": [0],
        "negative_indices": list(range(1, 8)),
        "base_features": np.arange(32, dtype=np.float32).reshape(8, 4) + offset,
        "product_embedding": np.full(POOLED_DIM, 2 + offset, dtype=np.float32),
        "reactant_embeddings": np.arange(8 * POOLED_DIM, dtype=np.float32).reshape(8, POOLED_DIM),
    }


def test_c2_dimensions_and_parameter_count():
    model = ProjectedRanker.build(hidden_width=128, dropout=0.1)
    assert RAW_INPUT_DIM == 1028
    assert HEAD_INPUT_DIM == 36
    assert sum(parameter.numel() for parameter in model.parameters()) == 21281
    assert expected_parameter_count(128) == 21281
    values = np.zeros((3, RAW_INPUT_DIM), dtype=np.float32)
    assert tuple(model.score(__import__("torch").from_numpy(values)).shape) == (3,)


def test_base_columns_are_exactly_frozen_view():
    matrix = np.arange(21, dtype=np.float32).reshape(3, 7)
    observed = selected_base_features(matrix)
    np.testing.assert_array_equal(observed, matrix[:, [0, 1, 5, 6]])
    assert observed.shape == (3, BASE_DIM)


def test_pair_indices_match_seeded_cap5_sampling():
    products = [_product()]
    p1, positive1, negative1 = make_pair_indices(products, 42)
    p2, positive2, negative2 = make_pair_indices(products, 42)
    assert len(p1) == 5
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(positive1, positive2)
    np.testing.assert_array_equal(negative1, negative2)
    assert len(set(negative1.tolist())) == 5


def test_materialization_and_train_pair_only_normalizer():
    product = _product()
    product_indices = np.asarray([0, 0], dtype=np.int32)
    candidate_indices = np.asarray([0, 1], dtype=np.int16)
    rows = materialize_rows([product], product_indices, candidate_indices)
    assert rows.shape == (2, RAW_INPUT_DIM)
    np.testing.assert_array_equal(rows[:, :4], product["base_features"][:2])
    np.testing.assert_array_equal(rows[0, 4:516], product["product_embedding"])
    normalizer = fit_pair_normalizer(rows, rows + 10)
    transformed = normalizer.transform(np.vstack([rows, rows + 10]))
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=2e-5)
    np.testing.assert_allclose(transformed.std(axis=0), 1, atol=2e-5)


def test_wrong_feature_shape_fails_closed():
    with pytest.raises(ValueError):
        selected_base_features(np.zeros((2, 6), dtype=np.float32))


def _write_prediction_csv(path, reaction_ids):
    import csv

    fields = ["reaction_id", "source_split", *PREDICTION_METRIC_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, reaction_id in enumerate(reaction_ids):
            writer.writerow({
                "reaction_id": reaction_id, "source_split": "test",
                "baseline_hit@1": index, "reranked_hit@1": 1,
                "baseline_hit@3": 1, "reranked_hit@3": 1,
                "baseline_hit@5": 1, "reranked_hit@5": 1,
                "baseline_hit@10": 1, "reranked_hit@10": 1,
                "baseline_rr": 0.5 + 0.5 * index,
                "reranked_rr": 1.0,
            })


def test_complete_prediction_csvs_recover_metrics_without_rescoring(tmp_path):
    for seed in (42, 43, 44, 45, 46):
        _write_prediction_csv(tmp_path / f"projected_seed_{seed}.csv", ["10", "11"])
    metrics, records = collect_existing_prediction_metrics(tmp_path)
    assert set(metrics) == {"42", "43", "44", "45", "46"}
    assert metrics["42"]["top1"] == 1.0
    assert metrics["42"]["baseline_top1"] == 0.5
    assert metrics["42"]["mrr"] == 1.0
    assert set(records) == {"42", "43", "44", "45", "46"}


def test_prediction_recovery_refuses_cross_seed_identity_mismatch(tmp_path):
    for seed in (42, 43, 44, 45, 46):
        reaction_ids = ["10", "11"] if seed != 46 else ["10", "12"]
        _write_prediction_csv(tmp_path / f"projected_seed_{seed}.csv", reaction_ids)
    with pytest.raises(PermissionError, match="identities/order"):
        collect_existing_prediction_metrics(tmp_path)
