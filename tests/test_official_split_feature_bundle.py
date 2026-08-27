from collections import Counter

import numpy as np
import pytest

from rerank.study_data import (
    ReactionRecord,
    attach_post_selection_test_payload,
    build_official_feature_cache,
    build_official_split_feature_bundle,
    canonicalize_reactant_set,
    canonicalize_smiles,
    selection_feature_cache_view,
)


class CountingExtractor:
    def __init__(self):
        self.calls = []

    def extract_features_batch(self, product_smiles, candidates, priors, ranks):
        self.calls.append(product_smiles)
        return np.asarray(
            [[prior, rank, len(candidate)] for candidate, prior, rank in zip(candidates, priors, ranks)],
            dtype=np.float32,
        )


def _reaction(reaction_id, source_split, product, ground_truth):
    return ReactionRecord(
        reaction_id=reaction_id,
        source_split=source_split,
        product_smiles=product,
        product_key=canonicalize_smiles(product),
        ground_truth=ground_truth,
        ground_truth_key=canonicalize_reactant_set(ground_truth),
    )


def _candidate(smiles, prior):
    return {
        "smiles": smiles,
        "prior": prior,
        "canonical_smiles": canonicalize_reactant_set(smiles),
    }


def _synthetic_inputs():
    reactions = [
        _reaction(0, "train", "CCO", "CC.O"),
        # This train reaction overlaps the validation split and must be excluded.
        _reaction(1, "train", "CNO", "CN.O"),
        _reaction(2, "valid", "CNO", "CN.O"),
        _reaction(3, "valid", "CNO", "C.NO"),
        _reaction(4, "valid", "COC", "CO.C"),
        _reaction(5, "test", "CCl", "C.Cl"),
        _reaction(6, "test", "CCl", "C.Cl"),
        _reaction(7, "test", "CBr", "C.Br"),
    ]
    pools = {
        canonicalize_smiles("CCO"): [
            _candidate("CC.O", 0.8),
            _candidate("C.CO", 0.2),
        ],
        canonicalize_smiles("CNO"): [
            _candidate("CN.O", 0.7),
            _candidate("C.NO", 0.3),
        ],
        canonicalize_smiles("CCl"): [
            _candidate("C.Cl", 0.9),
            _candidate("ClC", 0.1),
        ],
    }
    return reactions, pools


def test_default_bundle_extracts_train_and_validation_only():
    reactions, pools = _synthetic_inputs()
    extractor = CountingExtractor()

    bundle = build_official_split_feature_bundle(
        reactions,
        pools,
        extractor,
        feature_mode="2d+prior",
    )

    validation = bundle["validation_payload"]
    assert [item["reaction_id"] for item in validation["eval_metadata"]] == [2, 3]
    assert {item["source_split"] for item in validation["eval_metadata"]} == {"valid"}

    # One shared train extraction followed by validation only. In particular,
    # the covered test product CCl must not reach the extractor.
    assert Counter(extractor.calls) == Counter({"CCO": 1, "CNO": 1})
    assert "CCl" not in extractor.calls
    assert "test_split" not in bundle
    assert "post_selection_test" not in bundle
    assert "evaluation_payloads" not in bundle
    assert bundle["audit"]["train"]["train_overlap_reactions_excluded"] == 1
    assert [item["product_smiles"] for item in bundle["train_products"]] == ["CCO"]

    assert bundle["audit"]["validation"] == {
        "source_split": "valid",
        "reactions_total": 3,
        "reactions_covered": 2,
        "reactions_uncovered": 1,
        "unique_products_total": 2,
        "unique_products_covered": 1,
        "unique_products_with_uncovered_reactions": 1,
        "unique_products_fully_uncovered": 1,
    }


def test_selection_view_is_validation_only_and_test_guard_fails_closed():
    reactions, pools = _synthetic_inputs()
    bundle = build_official_split_feature_bundle(
        reactions,
        pools,
        CountingExtractor(),
        feature_mode="2d+prior",
    )

    selection = selection_feature_cache_view(bundle)
    assert set(selection) == {
        "schema_version",
        "feature_mode",
        "train_products",
        "eval_pwc",
        "eval_ground_truths",
        "eval_metadata",
        "eval_features",
        "audit",
    }
    assert selection["audit"]["eval_split"] == "valid"
    assert [item["reaction_id"] for item in selection["eval_metadata"]] == [2, 3]
    assert {item["source_split"] for item in selection["eval_metadata"]} == {"valid"}
    assert "validation_payload" not in selection
    with pytest.raises(PermissionError, match="restricted to the official validation split"):
        selection_feature_cache_view(bundle, selection_split="test")


def test_post_selection_test_extraction_requires_freeze_evidence():
    reactions, pools = _synthetic_inputs()
    extractor = CountingExtractor()
    bundle = build_official_split_feature_bundle(
        reactions,
        pools,
        extractor,
        feature_mode="2d+prior",
    )
    calls_before_attach = list(extractor.calls)

    with pytest.raises(PermissionError, match="sha256"):
        attach_post_selection_test_payload(
            bundle,
            reactions,
            pools,
            extractor,
            frozen_selection_record={},
        )
    assert extractor.calls == calls_before_attach
    assert "CCl" not in extractor.calls

    with pytest.raises(PermissionError, match="sha256"):
        attach_post_selection_test_payload(
            bundle,
            reactions,
            pools,
            extractor,
            frozen_selection_record={
                "selected_config_fingerprint": "sha256:not-a-real-digest",
            },
        )
    assert extractor.calls == calls_before_attach

    frozen = attach_post_selection_test_payload(
        bundle,
        reactions,
        pools,
        extractor,
        frozen_selection_record={
            "selected_config_fingerprint": "sha256:" + "a" * 64,
        },
    )
    assert "post_selection_test" not in bundle
    test_payload = frozen["post_selection_test"]["payload"]
    assert [item["reaction_id"] for item in test_payload["eval_metadata"]] == [5, 6]
    assert {item["source_split"] for item in test_payload["eval_metadata"]} == {"test"}
    assert Counter(extractor.calls) == Counter({"CCO": 1, "CNO": 1, "CCl": 1})
    assert frozen["audit"]["post_selection_test"] == {
        "source_split": "test",
        "reactions_total": 3,
        "reactions_covered": 2,
        "reactions_uncovered": 1,
        "unique_products_total": 2,
        "unique_products_covered": 1,
        "unique_products_with_uncovered_reactions": 1,
        "unique_products_fully_uncovered": 1,
    }
    with pytest.raises(PermissionError, match="restricted to the official validation split"):
        selection_feature_cache_view(frozen, selection_split="test")


def test_legacy_builder_keeps_exact_top_level_and_audit_schema():
    reactions, pools = _synthetic_inputs()
    cache = build_official_feature_cache(
        reactions,
        pools,
        CountingExtractor(),
        feature_mode="2d+prior",
        eval_split="test",
    )

    assert set(cache) == {
        "schema_version",
        "feature_mode",
        "train_products",
        "eval_pwc",
        "eval_ground_truths",
        "eval_metadata",
        "eval_features",
        "audit",
    }
    assert set(cache["audit"]) == {
        "schema_version",
        "feature_mode",
        "train_split",
        "eval_split",
        "exclude_cross_split_train_products",
        "train_products",
        "train_overlap_reactions_excluded",
        "train_products_uncovered",
        "train_products_without_negative",
        "eval_reactions_total",
        "eval_reactions_covered",
        "eval_reactions_uncovered",
        "eval_unique_products_covered",
    }
    assert cache["audit"]["eval_split"] == "test"
    assert [item["reaction_id"] for item in cache["eval_metadata"]] == [5, 6]
