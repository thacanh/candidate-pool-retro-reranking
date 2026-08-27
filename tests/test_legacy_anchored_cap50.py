import json
from pathlib import Path

import pytest

from rerank.data.build_legacy_anchored_cap50 import build_legacy_anchored_pool


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path, *, omit_anchor: bool = False) -> tuple[Path, Path, Path]:
    legacy = tmp_path / "legacy.jsonl"
    regenerated = tmp_path / "regenerated.jsonl"
    failed = tmp_path / "failed.json"
    _write_jsonl(
        legacy,
        [
            {"product": "CCO", "reactant": "C.O", "prior": 0.9},
            {"product": "CCO", "reactant": "CC.O", "prior": 0.8},
            {"product": "CCN", "reactant": "C.N", "prior": 0.7},
        ],
    )
    regenerated_records = [
        {"product": "CCO", "reactant": "CC.O", "prior": 0.8},
        {"product": "CCO", "reactant": "O.C", "prior": 0.9},
        {"product": "CCO", "reactant": "CCC", "prior": 0.6},
        {"product": "CCO", "reactant": "N", "prior": 0.5},
        {"product": "CCN", "reactant": "C.N", "prior": 0.7},
        {"product": "CCN", "reactant": "CN", "prior": 0.4},
    ]
    if omit_anchor:
        regenerated_records = [
            record
            for record in regenerated_records
            if not (record["product"] == "CCN" and record["reactant"] == "C.N")
        ]
    _write_jsonl(regenerated, regenerated_records)
    failed.write_text(
        json.dumps({"passed": False, "discrepancy_product_count": 2}),
        encoding="utf-8",
    )
    return legacy, regenerated, failed


def test_builds_immutable_anchor_then_clean_extensions(tmp_path: Path) -> None:
    legacy, regenerated, failed = _inputs(tmp_path)
    output = tmp_path / "anchored"
    result = build_legacy_anchored_pool(
        legacy_cap10=legacy,
        regenerated_cap50=regenerated,
        failed_reproduction_summary=failed,
        output_root=output,
        cap=4,
    )

    records = [
        json.loads(line)
        for line in (output / "candidates_cap50_legacy_anchored.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    ethanol = [record for record in records if record["product"] == "CCO"]
    assert [record["reactant"] for record in ethanol] == ["C.O", "CC.O", "CCC", "N"]
    assert [record["prior"] for record in ethanol[:2]] == [0.9, 0.8]
    assert [record["candidate_source"] for record in ethanol] == [
        "legacy_cap10_anchor",
        "legacy_cap10_anchor",
        "clean_cap50_extension",
        "clean_cap50_extension",
    ]
    assert result["legacy_anchor_candidates"] == 3
    assert result["extension_candidates"] == 3
    assert json.loads((output / "ANCHOR_RELEASE_GATE.json").read_text())["passed"]


def test_missing_legacy_candidate_fails_closed(tmp_path: Path) -> None:
    legacy, regenerated, failed = _inputs(tmp_path, omit_anchor=True)
    output = tmp_path / "anchored"
    with pytest.raises(ValueError, match="legacy anchor candidate is absent"):
        build_legacy_anchored_pool(
            legacy_cap10=legacy,
            regenerated_cap50=regenerated,
            failed_reproduction_summary=failed,
            output_root=output,
        )
    assert not output.exists()


def test_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    legacy, regenerated, failed = _inputs(tmp_path)
    output = tmp_path / "anchored"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        build_legacy_anchored_pool(
            legacy_cap10=legacy,
            regenerated_cap50=regenerated,
            failed_reproduction_summary=failed,
            output_root=output,
        )


def test_requires_a_recorded_failed_reproduction_gate(tmp_path: Path) -> None:
    legacy, regenerated, failed = _inputs(tmp_path)
    failed.write_text(json.dumps({"passed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a recorded failed gate"):
        build_legacy_anchored_pool(
            legacy_cap10=legacy,
            regenerated_cap50=regenerated,
            failed_reproduction_summary=failed,
            output_root=tmp_path / "anchored",
        )


def test_reports_cross_boundary_prior_inversion_without_moving_anchor(tmp_path: Path) -> None:
    legacy, regenerated, failed = _inputs(tmp_path)
    records = [json.loads(line) for line in regenerated.read_text().splitlines()]
    for record in records:
        if record["product"] == "CCO" and record["reactant"] == "CCC":
            record["prior"] = 0.85
    _write_jsonl(regenerated, records)
    output = tmp_path / "anchored"
    build_legacy_anchored_pool(
        legacy_cap10=legacy,
        regenerated_cap50=regenerated,
        failed_reproduction_summary=failed,
        output_root=output,
        cap=4,
    )
    validation = json.loads((output / "anchor_validation.json").read_text())
    assert validation["counts"]["products_with_cross_boundary_raw_prior_inversion"] == 1
    output_records = [
        json.loads(line)
        for line in (output / "candidates_cap50_legacy_anchored.jsonl")
        .read_text()
        .splitlines()
    ]
    ethanol = [record for record in output_records if record["product"] == "CCO"]
    assert [record["reactant"] for record in ethanol[:2]] == ["C.O", "CC.O"]
