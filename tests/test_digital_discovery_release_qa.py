"""Release-facing consistency gates for the Digital Discovery manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "digital_discovery"

if not PAPER.is_dir():
    pytest.skip(
        "manuscript sources are intentionally excluded from the public code release",
        allow_module_level=True,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _release_artifact(legacy: Path, public: Path) -> Path:
    """Use the neutral public alias when running from the compact release."""
    return public if public.is_file() else legacy


def test_single_repository_url_and_neutral_release_paths() -> None:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    si = (PAPER / "supporting_information.tex").read_text(encoding="utf-8")
    metadata = (PAPER / "release_metadata.tex").read_text(encoding="utf-8")

    assert "\\input{release_metadata}" in main
    assert "\\input{release_metadata}" in si
    assert metadata.count("\\newcommand{\\repositoryurl}") == 1
    assert "jcheminform_revision" not in main
    assert "jcheminform_revision" not in si
    assert "No individual descriptor showed a substantial monotonic association" in si


def test_release_alias_hashes_match_current_source_bytes() -> None:
    ledger = json.loads((PAPER / "release_artifact_map.json").read_text(encoding="utf-8"))
    for row in ledger["artifacts"]:
        source = ROOT / row["source_path"]
        if not source.is_file():
            source = ROOT / row["public_path"]
        payload = source.read_bytes()
        assert len(payload) == row["size_bytes"]
        assert hashlib.sha256(payload).hexdigest() == row["sha256"]
        assert "jcheminform_revision" not in row["public_path"]


def test_complete_m2_grid_is_reported() -> None:
    si = (PAPER / "supporting_information.tex").read_text(encoding="utf-8")
    rows = _rows(
        _release_artifact(
            ROOT / "outputs/digital_discovery_round_jk/reanalysis/m2_pool_shift/m2_transfer_loss_associations.csv",
            ROOT / "outputs/transfer_analysis/m2_pool_shift/m2_transfer_loss_associations.csv",
        )
    )
    assert len(rows) == 36
    pool_names = {
        "aizynthfinder_only": "AiZynthFinder-only",
        "localretro_only": "LocalRetro-only",
        "merged": "Merged union",
    }
    feature_names = {
        "candidate_count_shift_vs_historical": "Candidate-count shift",
        "jaccard_vs_historical": "Jaccard overlap",
        "shared_candidate_count": "Shared-candidate count",
        "kendall_order_vs_historical": "Shared-order Kendall",
        "mean_pairwise_morgan_distance_shift_vs_historical": "Morgan-distance shift",
        "max_nonreference_similarity_shift_vs_historical": "Max non-ref. similarity shift",
    }
    paired: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    for row in rows:
        paired.setdefault((row["pool"], row["feature"]), {})[row["metric"]] = row
    assert len(paired) == 18

    def rho(value: str) -> str:
        number = float(value)
        return f"+{number:.4f}" if number >= 0 else f"$-{abs(number):.4f}$"

    for (pool, feature), endpoints in paired.items():
        top1 = endpoints["transfer_delta_top1"]
        mrr = endpoints["transfer_delta_mrr"]
        expected = (
            f'{pool_names[pool]} & {feature_names[feature]} & {top1["n_reactions"]} & '
            f'{rho(top1["spearman_rho"])} & {mrr["n_reactions"]} & '
            f'{rho(mrr["spearman_rho"])}'
        )
        assert expected in si


def test_k2_all_groups_and_m1_secondary_results_are_reported() -> None:
    si = (PAPER / "supporting_information.tex").read_text(encoding="utf-8")
    k2 = _rows(
        _release_artifact(
            ROOT / "outputs/digital_discovery_round_jk/reanalysis/k2_full_reporting_v2/k2_group_effects.csv",
            ROOT / "outputs/transfer_analysis/k2_full_reporting_v2/k2_group_effects.csv",
        )
    )
    assert {(row["group"], row["metric"]) for row in k2} == {
        (group, metric)
        for group in (
            "a_identical_set_and_order",
            "b_identical_set_different_order",
            "c_different_set",
        )
        for metric in ("top1", "mrr")
    }
    for row in k2:
        assert f'{float(row["effect"]):+.6f}' in si

    omnibus = _rows(
        _release_artifact(
            ROOT / "outputs/digital_discovery_round_jk/reanalysis/m1_heterogeneity/m1_omnibus.csv",
            ROOT / "outputs/transfer_analysis/m1_heterogeneity/m1_omnibus.csv",
        )
    )
    contrasts = _rows(
        _release_artifact(
            ROOT / "outputs/digital_discovery_round_jk/reanalysis/m1_heterogeneity/m1_pairwise_contrasts.csv",
            ROOT / "outputs/transfer_analysis/m1_heterogeneity/m1_pairwise_contrasts.csv",
        )
    )
    for row in omnibus:
        assert f'{float(row["between_bin_statistic"]):.6f}' in si
        assert f'{float(row["permutation_p_two_sided"]):.6f}' in si
    for row in contrasts:
        assert f'{float(row["effect"]):+.6f}' in si
        assert f'{float(row["holm_p_within_metric"]):.6f}' in si


def test_chemistry_and_roundtrip_sensitivity_rows_are_present() -> None:
    si = (PAPER / "supporting_information.tex").read_text(encoding="utf-8")
    rows = [
        row
        for row in _rows(
            _release_artifact(
                ROOT / "outputs/jcheminform_revision/numerical_freeze_v2/paired_effects.csv",
                ROOT / "outputs/historical_anchor/numerical_freeze_v2/paired_effects.csv",
            )
        )
        if row["analysis_id"] in {"F1", "F2", "F3"}
    ]
    assert len(rows) == 11
    for row in rows:
        number = float(row["effect"])
        rendered = f"+{number:.6f}" if number >= 0 else f"$-{abs(number):.6f}$"
        assert rendered in si
