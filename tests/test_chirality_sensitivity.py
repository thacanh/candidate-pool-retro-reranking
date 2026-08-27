import numpy as np
import pickle
import pytest

from rerank.experiments.run_chirality_sensitivity import (
    MORGAN_COLUMN,
    _load_frozen_test_payload,
    _new_audit,
    morgan_similarity,
    replace_morgan_column,
)


def test_chirality_sensitivity_changes_only_morgan_column() -> None:
    product = "F[C@H](Cl)Br"
    candidates = [
        {"smiles": "F[C@@H](Cl)Br", "prior": 0.9},
        {"smiles": "CCO", "prior": 0.1},
    ]
    matrix = np.asarray(
        [
            [0.9, morgan_similarity(product, candidates[0]["smiles"], False), 1, 2, 3, 4, 5],
            [0.1, morgan_similarity(product, candidates[1]["smiles"], False), 6, 7, 8, 9, 10],
        ],
        dtype=np.float32,
    )
    audit = _new_audit()

    transformed = replace_morgan_column(product, candidates, matrix, audit)

    np.testing.assert_array_equal(
        transformed[:, [0, 2, 3, 4, 5, 6]], matrix[:, [0, 2, 3, 4, 5, 6]]
    )
    assert transformed[0, MORGAN_COLUMN] != matrix[0, MORGAN_COLUMN]
    assert audit == {
        "pairs_checked": 2,
        "false_cache_mismatches": 0,
        "maximum_false_cache_absolute_difference": 0.0,
        "chirality_changed_pairs": 1,
    }


def test_chirality_sensitivity_fails_closed_on_false_cache_mismatch() -> None:
    matrix = np.zeros((1, 7), dtype=np.float32)
    matrix[0, MORGAN_COLUMN] = 0.123
    audit = _new_audit()

    replace_morgan_column("CCO", [{"smiles": "CCO"}], matrix, audit)

    assert audit["false_cache_mismatches"] == 1
    assert audit["maximum_false_cache_absolute_difference"] == pytest.approx(0.877)


@pytest.mark.parametrize("prefix", ["", "sha256:"])
def test_frozen_cache_hash_accepts_primary_manifest_format(tmp_path, prefix) -> None:
    cache = tmp_path / "cache.pkl"
    with cache.open("wb") as handle:
        pickle.dump({"payload": {"eval_metadata": [{"source_split": "test"}]}}, handle)
    import hashlib

    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    payload = _load_frozen_test_payload(
        {"retained_primary_train_test_cache_sha256": prefix + digest}, str(cache)
    )

    assert payload["eval_metadata"][0]["source_split"] == "test"
