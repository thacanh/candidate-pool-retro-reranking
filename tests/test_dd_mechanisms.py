import json

import numpy as np
import pytest

from rerank.analysis.analyze_dd_mechanisms import (
    _between_group_statistic,
    _contrast_bootstrap,
    _expanded_effects,
    _holm_adjust,
    _list_descriptors,
    _normalized_entropy,
    _normalized_historical_top10_pool,
    _order_concordance,
    _safe_spearman,
    parse_args,
)


def test_between_group_statistic_is_zero_only_without_mean_heterogeneity():
    labels = np.asarray(["a", "a", "b", "b"])
    assert _between_group_statistic(np.asarray([0.0, 2.0, 0.0, 2.0]), labels) == 0.0
    assert _between_group_statistic(np.asarray([0.0, 0.0, 2.0, 2.0]), labels) > 0.0


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    adjusted = _holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_primary_bootstrap_contrast_preserves_sign():
    labels = np.asarray(["4-6"] * 4 + ["2-3"] * 4)
    values = np.asarray([1.0] * 4 + [0.0] * 4)
    low, high = _contrast_bootstrap(values, labels, "4-6", ("2-3",), 100, 7)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_pool_descriptors_handle_identity_order_entropy_and_similarity():
    historical = [
        {"identity": "CC", "prior": 1.0},
        {"identity": "CO", "prior": 0.5},
        {"identity": "CN", "prior": 0.0},
    ]
    current = [historical[1], historical[0], historical[2]]
    result = _list_descriptors(current, historical, "CC")
    assert result["candidate_count"] == 3
    assert result["jaccard_vs_historical"] == pytest.approx(1.0)
    assert result["shared_candidate_count"] == 3
    assert result["kendall_order_vs_historical"] == pytest.approx(1.0 / 3.0)
    assert 0.0 < result["normalized_stored_prior_mass_entropy"] < 1.0
    assert 0.0 <= result["mean_pairwise_morgan_distance"] <= 1.0
    assert 0.0 <= result["max_nonreference_similarity_to_truth"] <= 1.0


def test_undefined_descriptors_are_not_imputed():
    assert _normalized_entropy([1.0]) is None
    assert _normalized_entropy([0.0, 0.0]) is None
    assert _order_concordance(["CC"], ["CC"]) is None
    n, rho = _safe_spearman([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])
    assert n == 3
    assert rho is None


def test_m2_default_uses_frozen_cap10_common_coverage(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_dd_mechanisms.py",
            "m2",
            "--output-dir", "out",
            "--anchor-prediction-root", "anchor",
            "--k1-manifest", "a", "--k1-manifest", "b", "--k1-manifest", "c",
            "--pool-jsonl", "a", "--pool-jsonl", "b", "--pool-jsonl", "c",
            "--historical-pool-jsonl", "historical",
            "--l1-diagnostics", "l1",
            "--d5-root", "d5",
        ],
    )
    assert parse_args().expected_common_count == 3814


def test_expanded_effects_only_reads_requested_covered_rows():
    covered = {"covered": True}
    for seed in range(42, 62):
        covered[f"baseline_rank_seed_{seed}"] = 2
        covered[f"augmented_rank_seed_{seed}"] = 1
    records = {
        1: covered,
        2: {"covered": False, "candidate_count": 0, "prior_rank": 0},
    }
    result = _expanded_effects(records, {1})
    assert set(result) == {1}
    assert result[1]["mean_20seed_delta_top1"] == pytest.approx(1.0)
    assert result[1]["mean_20seed_delta_mrr"] == pytest.approx(0.5)


def test_expanded_effects_rejects_requested_uncovered_row():
    with pytest.raises(ValueError, match="uncovered reaction"):
        _expanded_effects({2: {"covered": False}}, {2})


def test_historical_pool_normalizes_duplicates_and_stable_prior_order(tmp_path):
    path = tmp_path / "historical.jsonl"
    rows = [
        {"product": "CCO", "reactant": "CC.O", "prior": 0.2},
        {"product": "CCN", "reactant": "CN.C", "prior": 0.9},
        {"product": "CCO", "reactant": "O.CC", "prior": 0.8},
        {"product": "CCO", "reactant": "CO.C", "prior": 0.8},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = _normalized_historical_top10_pool(path, {"CCO"})["CCO"]
    assert [row["identity"] for row in result] == ["CC.O", "C.CO"]
    assert [row["prior"] for row in result] == pytest.approx([0.8, 0.8])
