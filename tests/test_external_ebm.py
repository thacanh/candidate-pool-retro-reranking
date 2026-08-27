import numpy as np

from rerank.external_ebm import (
    CANDIDATE_WIDTH,
    QueryRow,
    adapter_audit,
    evaluation_query_rows,
    ranking_metrics,
    training_query_rows,
)


def _candidate(smiles, canonical=None):
    return {
        "smiles": smiles,
        "canonical_smiles": canonical or smiles,
        "prior": 1.0,
    }


def test_training_adapter_expands_multiple_positives_without_false_negatives():
    products = [
        {
            "product_key": "CCO",
            "product_smiles": "CCO",
            "candidates": [_candidate("C.C"), _candidate("CC"), _candidate("CO")],
            "positive_indices": [0, 2],
            "negative_indices": [1],
        }
    ]
    rows = training_query_rows(products)
    assert len(rows) == 2
    assert rows[0].candidate_smiles == ("C.C", "CC")
    assert rows[1].candidate_smiles == ("CO", "CC")
    assert all(row.true_index == 0 for row in rows)
    assert "CO" not in rows[0].candidate_smiles
    assert "C.C" not in rows[1].candidate_smiles


def test_evaluation_adapter_preserves_order_and_canonical_truth_rank():
    payload = {
        "eval_pwc": [
            (
                "CCO",
                [
                    {"smiles": "CC", "prior": 0.8},
                    {"smiles": "O.C", "prior": 0.2},
                ],
            )
        ],
        "eval_ground_truths": ["C.O"],
        "eval_metadata": [
            {"reaction_id": 7, "source_split": "valid", "coverage_rank": 2}
        ],
    }
    rows = evaluation_query_rows(payload)
    assert rows[0].candidate_smiles == ("CC", "O.C")
    assert rows[0].true_index == 1
    assert rows[0].reaction_id == 7


def test_ranking_metrics_use_stable_energy_ties_and_ignore_padding():
    energies = np.full((2, CANDIDATE_WIDTH), np.inf)
    energies[0, :3] = [0.0, 0.0, 1.0]
    energies[1, :2] = [2.0, 1.0]
    result = ranking_metrics(energies, [1, 1], [3, 2])
    assert result["true_ranks"].tolist() == [2, 1]
    assert result["top1"] == 0.5
    assert result["mrr"] == 0.75


def test_adapter_audit_records_extra_multi_positive_rows():
    rows = [
        QueryRow("CCO", ("CC",), ("CC",), 0, None, "CCO"),
        QueryRow("CCO", ("CO",), ("CO",), 0, None, "CCO"),
    ]
    valid = [QueryRow("CC", ("C",), ("C",), 0, 1, "CC")]
    audit = adapter_audit(rows, valid)
    assert audit["train_rows"] == 2
    assert audit["train_unique_products"] == 1
    assert audit["train_multi_positive_extra_rows"] == 1
