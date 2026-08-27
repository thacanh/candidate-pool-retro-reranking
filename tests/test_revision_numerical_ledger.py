from pathlib import Path

import pytest

from rerank.data.build_revision_numerical_ledger import (
    APPROVED_B1_STRATA,
    LEDGER_PROTOCOL_ID,
    build_numerical_ledger,
    file_record,
    validate_file_record,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_builds_paper_facing_ledger_from_frozen_outputs(tmp_path):
    if not (REPO_ROOT / "outputs/revision_analysis/table2_main_results.csv").is_file():
        pytest.skip("full private numerical sources are not included in the compact public release")
    output = tmp_path / "freeze"
    manifest = build_numerical_ledger(REPO_ROOT, output)

    assert manifest["protocol_id"] == LEDGER_PROTOCOL_ID
    assert manifest["paper_gate"]["numerical_sources_frozen"] is True
    assert manifest["paper_gate"]["latex_updated"] is False
    assert manifest["row_counts"]["b1_conformer_stability_summary"] == 12
    assert manifest["row_counts"]["paired_effects"] > 50
    assert len(manifest["validation_checks"]) >= 10

    b1_text = (output / "b1_conformer_stability_summary.csv").read_text(encoding="utf-8")
    assert "rigid_0_4" not in b1_text
    for stratum in APPROVED_B1_STRATA:
        assert stratum in b1_text

    effects = (output / "paired_effects.csv").read_text(encoding="utf-8")
    assert "revised_primary_headline" in effects
    assert "external_reranker_negative_result" in effects
    assert "legacy_sensitivity" in effects


def test_refuses_to_overwrite_even_an_empty_output_directory(tmp_path):
    output = tmp_path / "freeze"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_numerical_ledger(REPO_ROOT, output)


def test_frozen_file_record_detects_tampering(tmp_path):
    path = tmp_path / "source.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")
    record = file_record(path, tmp_path)
    validate_file_record(path, record)

    path.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        validate_file_record(path, record)
