import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rerank.analysis.analyze_round_jk import (
    _average_ranks,
    _candidate_bin,
    _distribution_summary,
    _rank_bin,
    _spearman_with_prior,
)
from rerank.experiments.run_round_jk import (
    EXPECTED_CONFIGS,
    PROTOCOL_ID,
    _assert_approval,
    _assert_isolated_output,
    _parameter_count,
)
from rerank.experiments.run_ws_e_ranking import _truncate_feature_lookup


def test_round_jk_parameter_counts_reject_289_claim():
    assert _parameter_count(6) == 1025
    assert _parameter_count(9) == 1409
    assert 289 not in {_parameter_count(6), _parameter_count(9)}
    assert EXPECTED_CONFIGS["baseline"]["hidden_width"] == 128
    assert EXPECTED_CONFIGS["augmented"]["hidden_width"] == 128


def test_truncation_preserves_frozen_prefix_without_resorting():
    digests = np.arange(15 * 32, dtype=np.uint8).reshape(15, 32)
    features = np.arange(15 * 9, dtype=np.float32).reshape(15, 9)
    truncated = _truncate_feature_lookup({7: (digests, features)}, 10)
    np.testing.assert_array_equal(truncated[7][0], digests[:10])
    np.testing.assert_array_equal(truncated[7][1], features[:10])
    assert not np.shares_memory(truncated[7][0], digests)
    assert not np.shares_memory(truncated[7][1], features)


def test_round_jk_bins_are_exhaustive_for_cap10():
    assert [_candidate_bin(value) for value in range(1, 11)] == [
        "1", "2-3", "2-3", "4-6", "4-6", "4-6", "7-9", "7-9", "7-9", "10"
    ]


def test_l1_spearman_uses_average_ranks_and_excludes_undefined_lists():
    np.testing.assert_allclose(_average_ranks(np.asarray([2.0, 1.0, 1.0])), [3.0, 1.5, 1.5])
    assert _spearman_with_prior(np.asarray([3.0, 2.0, 1.0])) == pytest.approx(1.0)
    assert _spearman_with_prior(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(-1.0)
    assert _spearman_with_prior(np.asarray([1.0])) is None
    assert _spearman_with_prior(np.asarray([2.0, 2.0])) is None


def test_l1_distribution_summary_is_population_based_and_null_safe():
    result = _distribution_summary([1.0, 2.0, 3.0])
    assert result["n"] == 3
    assert result["mean"] == pytest.approx(2.0)
    assert result["std"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert result["median"] == pytest.approx(2.0)
    assert _distribution_summary([]) == {
        "n": 0,
        "mean": None,
        "std": None,
        "min": None,
        "q25": None,
        "median": None,
        "q75": None,
        "max": None,
    }
    assert [_rank_bin(value) for value in range(1, 11)] == [
        "1", "2", "3", "4-5", "4-5", "6-10", "6-10", "6-10", "6-10", "6-10"
    ]


def test_round_jk_output_namespace_isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    allowed = tmp_path / "outputs" / "digital_discovery_round_jk" / "pool"
    assert _assert_isolated_output(allowed) == allowed.resolve()
    with pytest.raises(PermissionError):
        _assert_isolated_output(tmp_path / "outputs" / "revision_analysis" / "bad")
    with pytest.raises(PermissionError):
        _assert_isolated_output(tmp_path / "paper" / "overleaf" / "bad")


def test_pending_approval_fails_closed(tmp_path):
    plan = tmp_path / "analysis_plan.md"
    plan.write_text("draft", encoding="utf-8")
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "status": "pending",
                "width128_d1_resolution": "pending",
                "j4_bin_assignment": "pending",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        _assert_approval(approval, plan)


def test_approved_record_is_bound_to_plan_bytes(tmp_path):
    plan = tmp_path / "analysis_plan.md"
    plan.write_text("approved plan", encoding="utf-8")
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "status": "approved",
                "width128_d1_resolution": "approved",
                "j4_bin_assignment": "median_across_seeds_worse_bin_on_half_tie",
                "supervisor": "test",
                "approval_date": "2026-08-26",
                "analysis_plan_sha256": digest,
                "analysis_plan_commit": "abc123",
                "platform_lock": {"requirements.txt": {"sha256": "abc"}},
            }
        ),
        encoding="utf-8",
    )
    record = _assert_approval(approval, plan)
    assert record["status"] == "approved"
    plan.write_text("changed after approval", encoding="utf-8")
    with pytest.raises(PermissionError):
        _assert_approval(approval, plan)
