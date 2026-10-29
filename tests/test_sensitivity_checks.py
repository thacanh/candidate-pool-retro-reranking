from __future__ import annotations

import math
import json
import pickle
from pathlib import Path
import pytest

import numpy as np

from rerank.analysis.analyze_sensitivity_checks import (
    _n1_prior_audit,
    _assert_n1_numeric_equivalence,
    corrected_headroom_rows,
    crossed_transfer_intervals,
    matched_list_cohorts,
    n1_sensitivity_rows,
)


SEEDS = tuple(range(42, 62))


def _historical() -> dict:
    return {
        "per_seed_metrics": {
            "baseline": {str(seed): {"top1": 0.75} for seed in SEEDS},
            "augmented": {str(seed): {"top1": 0.76} for seed in SEEDS},
        }
    }


def _expanded(pool: str, baseline: float, augmented: float) -> dict:
    return {
        "pool_name": pool,
        "test_reactions_covered": 4000,
        "test_reactions_total": 5004,
        "per_seed_metrics": {
            "baseline": {str(seed): {"within_pool": {"top1": baseline}} for seed in SEEDS},
            "augmented": {str(seed): {"within_pool": {"top1": augmented}} for seed in SEEDS},
        },
    }


def test_corrected_headroom_uses_prior_2d_denominator() -> None:
    rows = corrected_headroom_rows(
        _historical(),
        {
            "aizynthfinder_only": _expanded("aizynthfinder_only", 0.80, 0.81),
            "localretro_only": _expanded("localretro_only", 0.60, 0.59),
            "merged": _expanded("merged", 0.50, 0.50),
        },
    )
    by_pool = {row["pool"]: row for row in rows}
    assert math.isclose(by_pool["historical_cap10"]["corrected_capture_rate"], 0.04)
    assert math.isclose(by_pool["aizynthfinder_only"]["corrected_capture_rate"], 0.05)
    assert math.isclose(by_pool["localretro_only"]["corrected_capture_rate"], -0.025)
    assert math.isclose(by_pool["merged"]["corrected_capture_rate"], 0.0)


def test_crossed_transfer_intervals_are_reproducible_and_pool_specific() -> None:
    products = 12
    negative = np.full((20, products), -0.1, dtype=np.float64)
    mixed = np.vstack(
        [
            np.full((10, products), -0.1, dtype=np.float64),
            np.full((10, products), 0.1, dtype=np.float64),
        ]
    )
    matrices = {
        "aizynthfinder_only": {"top1": negative, "mrr": negative / 2},
        "localretro_only": {"top1": mixed, "mrr": mixed / 2},
        "merged": {"top1": negative * 2, "mrr": negative},
    }
    first = crossed_transfer_intervals(matrices, draws=500, rng_seed=2070, batch_size=50)
    second = crossed_transfer_intervals(matrices, draws=500, rng_seed=2070, batch_size=50)
    assert first == second
    assert math.isclose(first[("aizynthfinder_only", "top1")][0], -0.1)
    assert first[("aizynthfinder_only", "top1")][2] < 0
    assert first[("localretro_only", "top1")][1] < 0
    assert first[("localretro_only", "top1")][2] > 0


def test_matched_list_cohort_gate_reproduces_frozen_partition() -> None:
    rows = []
    for reaction_id in range(3814):
        matched = reaction_id < 1745
        product = f"product-{reaction_id}"
        rows.append(
            {
                "reaction_id": str(reaction_id),
                "canonical_product": product,
                "pool": "historical_cap10",
                "candidate_count": "1" if reaction_id == 0 else "10",
                "jaccard_vs_historical": "1",
                "shared_candidate_count": "1" if reaction_id == 0 else "10",
                "kendall_order_vs_historical": "",
            }
        )
        rows.append(
            {
                "reaction_id": str(reaction_id),
                "canonical_product": product,
                "pool": "aizynthfinder_only",
                "candidate_count": ("1" if reaction_id == 0 else "10" if matched else "9"),
                "jaccard_vs_historical": "1" if matched else "0.9",
                "shared_candidate_count": ("1" if reaction_id == 0 else "10" if matched else "9"),
                "kendall_order_vs_historical": "" if reaction_id == 0 else "1",
            }
        )
    cohorts, products = matched_list_cohorts(rows)
    assert len(cohorts["matched_candidate_identity_and_order"]) == 1745
    assert len(cohorts["changed_candidate_list"]) == 2069
    assert products[0] == "product-0"


def test_n1_sensitivity_reports_joint_pipeline_contrast() -> None:
    reaction_ids = [1, 2, 3, 4]
    products = {item: f"product-{item}" for item in reaction_ids}
    historical_top1 = np.zeros((20, 4), dtype=np.float64)
    aizynth_top1 = np.full((20, 4), -0.1, dtype=np.float64)
    historical = {"top1": historical_top1, "mrr": historical_top1 / 2}
    aizynth = {"top1": aizynth_top1, "mrr": aizynth_top1 / 2}
    cohorts = {"matched_candidate_identity_and_order": [1, 2], "changed_candidate_list": [3, 4]}
    effects, contrasts = n1_sensitivity_rows(
        historical,
        aizynth,
        reaction_ids,
        products,
        cohorts,
        bootstrap_samples=100,
        sign_flip_samples=200,
        rng_seed=2080,
    )
    assert len(effects) == 8
    assert len(contrasts) == 4
    assert all(
        math.isclose(row["aizynth_minus_historical_increment"], -0.1)
        for row in contrasts
        if row["metric"] == "top1"
    )
    assert all(
        "matching candidate identities and order" in row["interpretation_scope"]
        for row in contrasts
        if row["cohort"] == "matched_candidate_identity_and_order"
    )
    assert all(
        "including candidate-list changes" in row["interpretation_scope"]
        for row in contrasts
        if row["cohort"] == "changed_candidate_list"
    )


def test_n1_prior_audit_canonicalizes_products_and_uses_first_ten(tmp_path: Path) -> None:
    historical_path = tmp_path / "historical.pkl"
    pool_path = tmp_path / "pool.jsonl"
    payload = {
        "payload": {
            "eval_metadata": [{"reaction_id": 7}],
            "eval_pwc": [("CCO", [{"prior": 0.1}] * 10)],
        }
    }
    historical_path.write_bytes(pickle.dumps(payload))
    pool_path.write_text(
        "\n".join(
            json.dumps({"product": "OCC", "candidate_rank": rank, "prior": 1.0 - rank / 20})
            for rank in range(1, 12)
        )
        + "\n",
        encoding="utf-8",
    )
    audit = _n1_prior_audit(historical_path, pool_path, [7], {7: "CCO"})
    assert audit["audited_reactions"] == 1
    assert audit["exact_stored_prior_list_matches"] == 0


def test_n1_v2_allows_only_interpretation_label_change(tmp_path: Path) -> None:
    reference = tmp_path / "v1.csv"
    reference.write_text(
        "cohort,effect,interpretation_scope\nmatched,0.125,old label\n", encoding="utf-8"
    )
    _assert_n1_numeric_equivalence(
        [{"cohort": "matched", "effect": 0.125, "interpretation_scope": "new label"}], reference
    )
    with pytest.raises(ValueError, match="numeric or cohort value differs"):
        _assert_n1_numeric_equivalence(
            [{"cohort": "matched", "effect": 0.126, "interpretation_scope": "new label"}], reference
        )
