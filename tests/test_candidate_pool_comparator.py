import json
import struct
from pathlib import Path

import pytest

from rerank.analysis.compare_candidate_pools import (
    compare_candidate_pools,
    float32_ulp_distance,
    load_normalized_pools,
    main,
    validate_manifest_schema,
)


def write_jsonl(path: Path, records) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def record(product, reactant, prior, label=0):
    return {
        "product": product,
        "reactant": reactant,
        "prior": prior,
        "label": label,
    }


def valid_manifest():
    return {
        "protocol_id": "A-CAP10-REPRO",
        "comparator": "legacy-cap10-fixed50-v1",
        "single_intended_change": "recorded-environment regeneration",
        "input_fingerprints": {
            "source_csv": {
                "path": "data/uspto_smiles.csv",
                "size_bytes": 12,
                "sha256": "a" * 64,
            }
        },
        "settings": {"candidate_cap": 10},
        "environment": {"python": "3.11"},
        "failures": {"count": 0},
        "counts": {"candidate_records": 2},
        "output": {"path": "regenerated.jsonl"},
        "runtime": {"duration_seconds": 1.0},
    }


def test_fragment_equivalence_label_ignored_and_duplicate_winner(tmp_path):
    reference = write_jsonl(
        tmp_path / "reference.jsonl",
        [
            record("C[C@H](O)F", "CC.O", 0.2, label=0),
            record("C[C@H](O)F", "O.CC", 0.8, label=1),
            record("C[C@H](O)F", "N.Cl", 0.4, label=0),
        ],
    )
    regenerated = write_jsonl(
        tmp_path / "regenerated.jsonl",
        [
            record("C[C@H](O)F", "O.CC", 0.2, label=999),
            record("C[C@H](O)F", "CC.O", 0.8, label=-1),
            record("C[C@H](O)F", "Cl.N", 0.4, label=-1),
        ],
    )
    summary, discrepancies = compare_candidate_pools(reference, regenerated)
    assert summary["comparison_passed"] is True
    assert discrepancies == []
    loaded = load_normalized_pools(reference)
    winner = loaded.pools[loaded.product_order[0]].ordered_candidates()[0]
    assert winner.prior == 0.8
    assert winner.winner_line == 2
    assert loaded.audit["canonical_duplicate_records"] == 1


def test_equal_prior_duplicate_keeps_first_occurrence(tmp_path):
    path = write_jsonl(
        tmp_path / "pool.jsonl",
        [
            record("CCO", "O.CC", 0.7),
            record("CCO", "CC.O", 0.7),
        ],
    )
    loaded = load_normalized_pools(path)
    candidate = loaded.pools[loaded.product_order[0]].ordered_candidates()[0]
    assert candidate.raw_reactant == "O.CC"
    assert candidate.first_seen_line == 1
    assert candidate.winner_line == 1


def test_duplicate_count_mismatch_fails_even_when_normalized_pool_matches(tmp_path):
    reference = write_jsonl(
        tmp_path / "reference.jsonl",
        [record("CCO", "CC.O", 0.7)],
    )
    regenerated = write_jsonl(
        tmp_path / "regenerated.jsonl",
        [
            record("CCO", "CC.O", 0.7),
            record("CCO", "O.CC", 0.7),
        ],
    )
    summary, discrepancies = compare_candidate_pools(reference, regenerated)
    assert summary["comparison_passed"] is False
    assert summary["checks"]["candidate_identity_and_order_exact"] is True
    assert summary["checks"]["audit_counts_exact"] is False
    assert summary["checks"]["audit_count_differences"] == {
        "candidate_records": {"reference": 1, "regenerated": 2},
        "canonical_duplicate_records": {"reference": 0, "regenerated": 1},
        "physical_lines": {"reference": 1, "regenerated": 2},
    }
    assert any(
        issue["type"] == "raw_audit_counts_mismatch"
        for discrepancy in discrepancies
        for issue in discrepancy["issues"]
    )


@pytest.mark.parametrize(
    ("reference_records", "regenerated_records", "issue_type"),
    [
        (
            [record("CCO", "CC.O", 0.8), record("CNO", "CN.O", 0.7)],
            [record("CCO", "CC.O", 0.8)],
            "missing_product",
        ),
        (
            [record("CCO", "CC.O", 0.8)],
            [record("CCO", "CC.O", 0.8), record("CNO", "CN.O", 0.7)],
            "extra_product",
        ),
    ],
)
def test_missing_and_extra_products_fail(
    tmp_path,
    reference_records,
    regenerated_records,
    issue_type,
):
    reference = write_jsonl(tmp_path / "reference.jsonl", reference_records)
    regenerated = write_jsonl(tmp_path / "regenerated.jsonl", regenerated_records)
    summary, discrepancies = compare_candidate_pools(reference, regenerated)
    assert summary["comparison_passed"] is False
    assert any(
        issue["type"] == issue_type
        for discrepancy in discrepancies
        for issue in discrepancy["issues"]
    )


def test_stable_equal_prior_order_must_match(tmp_path):
    reference = write_jsonl(
        tmp_path / "reference.jsonl",
        [record("CCO", "CC.O", 0.5), record("CCO", "CN.O", 0.5)],
    )
    regenerated = write_jsonl(
        tmp_path / "regenerated.jsonl",
        [record("CCO", "CN.O", 0.5), record("CCO", "CC.O", 0.5)],
    )
    summary, discrepancies = compare_candidate_pools(reference, regenerated)
    assert summary["comparison_passed"] is False
    assert summary["checks"]["candidate_identity_and_order_exact"] is False
    assert any(
        issue["type"] == "candidate_identity_or_order_mismatch"
        for discrepancy in discrepancies
        for issue in discrepancy["issues"]
    )


def test_prior_tolerance_boundary(tmp_path):
    reference = write_jsonl(
        tmp_path / "reference.jsonl",
        [record("CCO", "CC.O", 0.5)],
    )
    within = write_jsonl(
        tmp_path / "within.jsonl",
        [record("CCO", "CC.O", 0.5000004)],
    )
    outside = write_jsonl(
        tmp_path / "outside.jsonl",
        [record("CCO", "CC.O", 0.500001)],
    )
    assert compare_candidate_pools(reference, within)[0]["comparison_passed"] is True
    failed, discrepancies = compare_candidate_pools(reference, outside)
    assert failed["comparison_passed"] is False
    assert failed["checks"]["prior_tolerance_passed"] is False
    assert any(
        issue["type"] == "prior_tolerance_exceeded"
        for discrepancy in discrepancies
        for issue in discrepancy["issues"]
    )


def test_exact_prior_fraction_and_float32_ulp_metrics(tmp_path):
    half_bits = struct.unpack(">I", struct.pack(">f", 0.5))[0]
    next_float32_after_half = struct.unpack(
        ">f", struct.pack(">I", half_bits + 1)
    )[0]
    reference = write_jsonl(
        tmp_path / "reference.jsonl",
        [
            record("CCO", "CC.O", 0.75),
            record("CCO", "CN.O", 0.5),
        ],
    )
    regenerated = write_jsonl(
        tmp_path / "regenerated.jsonl",
        [
            record("CCO", "CC.O", 0.75),
            record("CCO", "CN.O", next_float32_after_half),
        ],
    )
    summary, discrepancies = compare_candidate_pools(reference, regenerated)
    assert summary["comparison_passed"] is True
    assert discrepancies == []
    assert summary["checks"]["aligned_prior_count"] == 2
    assert summary["checks"]["exact_float_prior_count"] == 1
    assert summary["checks"]["exact_float_prior_fraction"] == 0.5
    assert summary["checks"]["max_float32_ulp_distance"] == 1
    assert summary["checks"]["float32_ulp_unavailable_count"] == 0
    assert float32_ulp_distance(float("nan"), 0.5) is None
    assert float32_ulp_distance(float("inf"), 0.5) is None


def test_nonfinite_prior_fails_closed(tmp_path):
    reference = write_jsonl(
        tmp_path / "reference.jsonl",
        [record("CCO", "CC.O", 0.5)],
    )
    regenerated = write_jsonl(
        tmp_path / "regenerated.jsonl",
        [record("CCO", "CC.O", float("nan"))],
    )
    summary, discrepancies = compare_candidate_pools(reference, regenerated)
    assert summary["comparison_passed"] is False
    assert summary["regenerated"]["fatal_error_count"] == 1
    assert any(
        issue["type"] == "regenerated_fatal_load_errors"
        for discrepancy in discrepancies
        for issue in discrepancy["issues"]
    )


def test_manifest_schema_accepts_complete_and_reports_missing():
    assert validate_manifest_schema(valid_manifest()) == []
    incomplete = valid_manifest()
    del incomplete["single_intended_change"]
    incomplete["input_fingerprints"]["source_csv"]["sha256"] = "bad"
    errors = validate_manifest_schema(incomplete)
    assert "missing required field: single_intended_change" in errors
    assert any("sha256" in error for error in errors)


def test_cli_emits_reports_and_returns_nonzero_on_failure(tmp_path):
    reference = write_jsonl(
        tmp_path / "reference.jsonl",
        [record("CCO", "CC.O", 0.5)],
    )
    regenerated = write_jsonl(
        tmp_path / "regenerated.jsonl",
        [record("CCO", "CN.O", 0.5)],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(valid_manifest()), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    discrepancies_path = tmp_path / "discrepancies.jsonl"
    exit_code = main(
        [
            "--reference",
            str(reference),
            "--regenerated",
            str(regenerated),
            "--manifest-json",
            str(manifest_path),
            "--summary-json",
            str(summary_path),
            "--discrepancies-jsonl",
            str(discrepancies_path),
        ]
    )
    assert exit_code == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["passed"] is False
    assert summary["manifest_validation"]["valid"] is True
    discrepancies = [
        json.loads(line)
        for line in discrepancies_path.read_text(encoding="utf-8").splitlines()
    ]
    assert discrepancies


def test_cli_requires_valid_manifest_even_when_pools_match(tmp_path):
    records = [record("CCO", "CC.O", 0.5)]
    reference = write_jsonl(tmp_path / "reference.jsonl", records)
    regenerated = write_jsonl(tmp_path / "regenerated.jsonl", records)
    manifest = valid_manifest()
    del manifest["environment"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    summary_path = tmp_path / "summary.json"
    discrepancies_path = tmp_path / "discrepancies.jsonl"
    exit_code = main(
        [
            "--reference",
            str(reference),
            "--regenerated",
            str(regenerated),
            "--manifest-json",
            str(manifest_path),
            "--summary-json",
            str(summary_path),
            "--discrepancies-jsonl",
            str(discrepancies_path),
        ]
    )
    assert exit_code == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["comparison_passed"] is True
    assert summary["manifest_validation"]["valid"] is False
    assert summary["passed"] is False
