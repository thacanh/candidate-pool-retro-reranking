import numpy as np

from rerank.experiments.run_salt_sensitivity import (
    CONTROL_TOLERANCE,
    FAILURE_CAUSES,
    _embedding_cause,
    _feature_row,
    _transform_matrix,
    canonical_smiles,
    fragment_keys,
    salt_remove,
)


def test_salt_removal_preserves_stereo_and_surviving_fragments() -> None:
    cleaned, status, original_count, surviving_count = salt_remove(
        "Cl.F[C@H](Br)I.CCO"
    )

    assert status == "changed"
    assert original_count == 3
    assert surviving_count == 2
    assert "[C@H]" in cleaned
    assert "CCO" in cleaned
    assert "Cl" not in cleaned


def test_dont_remove_everything_retains_single_salt() -> None:
    cleaned, status, original_count, surviving_count = salt_remove("Cl")

    assert cleaned == "Cl"
    assert status == "unchanged"
    assert (original_count, surviving_count) == (1, 1)


def test_fallback_categories_are_closed_and_zero_is_coordinate() -> None:
    cause = _embedding_cause("Br", "fallback_zero", np.zeros((1, 512)), None)

    assert cause == "conformer/coordinate"
    assert set(FAILURE_CAUSES) == {
        "SMILES parse",
        "salt-removal-empty",
        "atom-limit",
        "conformer/coordinate",
        "checkpoint/model",
        "cache-lookup failure",
    }
    assert CONTROL_TOLERANCE == 5e-5


def test_transform_changes_only_salt_affected_row_and_preserves_prior() -> None:
    product = "CCO"
    salted = "Cl.F[C@H](Br)I.CCO"
    cleaned, status, *_ = salt_remove(salted)
    assert status == "changed"
    assert cleaned is not None

    keys = set(fragment_keys(product) + fragment_keys(salted) + fragment_keys(cleaned))
    embeddings = {
        key: np.full((max(1, len(key) % 4 + 1), 512), index + 1, dtype=np.float32)
        for index, key in enumerate(sorted(keys))
    }
    candidates = [
        {"smiles": salted, "prior": 0.75},
        {"smiles": "CCN", "prior": 0.25},
    ]
    embeddings[canonical_smiles("CCN")] = np.full((3, 512), 9, dtype=np.float32)
    original = np.vstack(
        [
            _feature_row(product, salted, 0.75, embeddings),
            _feature_row(product, "CCN", 0.25, embeddings),
        ]
    )
    audit = {
        "control_pairs_recomputed": 0,
        "control_feature_mismatches": 0,
        "maximum_control_absolute_difference": 0.0,
        "control_tolerance": 1e-5,
        "salt_changed_pair_rows": 0,
        "affected_products": 1,
    }

    transformed = _transform_matrix(
        product,
        candidates,
        original,
        {salted: cleaned, "CCN": canonical_smiles("CCN")},
        embeddings,
        {},
        audit,
    )

    assert transformed[0, 0] == original[0, 0] == 0.75
    assert not np.array_equal(transformed[0, 1:], original[0, 1:])
    np.testing.assert_array_equal(transformed[1], original[1])
    assert audit["control_pairs_recomputed"] == 1
    assert audit["control_feature_mismatches"] == 0


def test_known_fallback_control_drift_is_audited_and_delta_anchored() -> None:
    product = "CCO"
    salted = "Cl.CCO"
    cleaned, status, *_ = salt_remove(salted)
    assert status == "changed"
    keys = set(fragment_keys(product) + fragment_keys(salted) + fragment_keys(cleaned))
    embeddings = {
        key: np.full((2, 512), index + 1, dtype=np.float32)
        for index, key in enumerate(sorted(keys))
    }
    regenerated = _feature_row(product, salted, 0.5, embeddings)
    frozen = regenerated.copy()
    frozen[2] += 0.1
    audit = {
        "control_pairs_recomputed": 0,
        "control_feature_mismatches": 0,
        "fallback_attributable_control_mismatches": 0,
        "unexplained_control_feature_mismatches": 0,
        "control_mismatch_details": [],
        "maximum_control_absolute_difference": 0.0,
        "control_tolerance": 1e-5,
        "salt_changed_pair_rows": 0,
        "affected_products": 1,
    }

    transformed = _transform_matrix(
        product,
        [{"smiles": salted, "prior": 0.5}],
        frozen[None, :],
        {salted: cleaned},
        embeddings,
        {"Cl": "fallback_zero"},
        audit,
    )

    regenerated_cleaned = _feature_row(product, cleaned, 0.5, embeddings)
    np.testing.assert_allclose(
        transformed[0], frozen + (regenerated_cleaned - regenerated), atol=1e-7
    )
    assert audit["control_feature_mismatches"] == 1
    assert audit["fallback_attributable_control_mismatches"] == 1
    assert audit["unexplained_control_feature_mismatches"] == 0
