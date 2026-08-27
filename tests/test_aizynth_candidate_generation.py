import json
from pathlib import Path

import pytest

from rerank.data.generate_aizynth_candidate_pools import (
    exact_string_deduplicate,
    flat_candidate_records,
    merge_chunks,
    validate_chunk,
    write_chunk,
)


def _record(source_index: int, product: str, outcomes: list[tuple[str, float]]):
    return {
        "source_row_index": source_index,
        "product": product,
        "status": "ok" if outcomes else "empty",
        "actions_returned": len(outcomes),
        "actions_visited": len(outcomes),
        "errors": [],
        "raw_outcomes": [
            {
                "reactant": reactant,
                "prior": prior,
                "action_index": index,
                "outcome_index": 0,
                "raw_outcome_index": index,
            }
            for index, (reactant, prior) in enumerate(outcomes)
        ],
    }


def test_deduplication_happens_after_each_raw_prefix():
    raw = [
        {"reactant": "A", "prior": 0.9, "raw_outcome_index": 0, "action_index": 0},
        {"reactant": "A", "prior": 0.8, "raw_outcome_index": 1, "action_index": 1},
        {"reactant": "B", "prior": 0.7, "raw_outcome_index": 2, "action_index": 2},
        {"reactant": "C", "prior": 0.6, "raw_outcome_index": 3, "action_index": 3},
    ]
    assert [row["reactant"] for row in exact_string_deduplicate(raw, 2)] == ["A"]
    assert [row["reactant"] for row in exact_string_deduplicate(raw, 4)] == ["A", "B", "C"]


def test_higher_prior_duplicate_replaces_value_without_moving_order():
    raw = [
        {"reactant": "A", "prior": 0.2, "raw_outcome_index": 0, "action_index": 0},
        {"reactant": "B", "prior": 0.3, "raw_outcome_index": 1, "action_index": 1},
        {"reactant": "A", "prior": 0.8, "raw_outcome_index": 2, "action_index": 2},
    ]
    result = exact_string_deduplicate(raw, 3)
    assert [row["reactant"] for row in result] == ["A", "B"]
    assert result[0]["prior"] == pytest.approx(0.8)
    assert result[0]["raw_outcome_index"] == 2


def test_flat_records_keep_source_identity_and_prefix():
    record = _record(17, "P", [(f"R{i}", 1.0 / (i + 1)) for i in range(12)])
    cap10 = flat_candidate_records(record, 10)
    cap50 = flat_candidate_records(record, 50)
    assert len(cap10) == 10
    assert len(cap50) == 12
    assert cap50[:10] == cap10
    assert all(row["source_row_index"] == 17 and row["product"] == "P" for row in cap50)


def test_chunk_validation_detects_tampering(tmp_path: Path):
    tasks = [(0, "P0"), (1, "P1")]
    path = tmp_path / "chunk.jsonl"
    write_chunk(path, [_record(0, "P0", [("A", 1.0)]), _record(1, "P1", [])])
    assert len(validate_chunk(path, tasks)) == 2
    rows = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(rows[1])
    tampered["product"] = "WRONG"
    rows[1] = json.dumps(tampered)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="product mismatch"):
        validate_chunk(path, tasks)


def test_merge_is_source_ordered_and_builds_both_caps(tmp_path: Path):
    products = [(0, "P0"), (1, "P1"), (2, "P2")]
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    write_chunk(
        chunks / "chunk_000000_000002.jsonl",
        [
            _record(0, "P0", [(f"A{i}", 1 - i / 100) for i in range(12)]),
            _record(1, "P1", [("B", 0.5), ("B", 0.4)]),
        ],
    )
    write_chunk(
        chunks / "chunk_000002_000003.jsonl",
        [_record(2, "P2", [])],
    )
    result = merge_chunks(products, chunks, 2, tmp_path / "out")
    assert result["raw_product_records"] == 3
    assert result["cap10_candidate_records"] == 11
    assert result["cap50_candidate_records"] == 13
    assert result["empty_products"] == 1

    cap10 = [
        json.loads(line)
        for line in (tmp_path / "out" / "candidates_cap10.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["product"] for row in cap10] == ["P0"] * 10 + ["P1"]
