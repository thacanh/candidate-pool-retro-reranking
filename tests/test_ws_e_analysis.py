import json
from pathlib import Path

import pytest

from rerank.analysis.analyze_ws_e_results import analyze_pool, classify_top1
from rerank.study_data import file_fingerprint
from rerank.ws_e_streaming import RANKING_PROTOCOL_ID


def test_ws_e_analysis_reproduces_paired_rank_metrics(tmp_path: Path) -> None:
    ranks = tmp_path / "reaction_ranks.jsonl"
    records = [
        {"reaction_id": 0, "covered": True},
        {"reaction_id": 1, "covered": True},
        {"reaction_id": 2, "covered": False},
        {"reaction_id": 3, "covered": True},
    ]
    for record in records:
        for seed in range(42, 47):
            if not record["covered"]:
                continue
            baseline = {0: 2, 1: 1, 3: 2}[record["reaction_id"]]
            augmented = {0: 1, 1: 2, 3: 1}[record["reaction_id"]]
            record[f"baseline_rank_seed_{seed}"] = baseline
            record[f"augmented_rank_seed_{seed}"] = augmented
    ranks.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    per_seed = {"baseline": {}, "augmented": {}}
    for seed in range(42, 47):
        per_seed["baseline"][str(seed)] = {
            "within_pool": {"top1": 1 / 3, "mrr": 2 / 3},
            "end_to_end": {"top1": 1 / 4, "mrr": 1 / 2},
        }
        per_seed["augmented"][str(seed)] = {
            "within_pool": {"top1": 2 / 3, "mrr": 5 / 6},
            "end_to_end": {"top1": 1 / 2, "mrr": 5 / 8},
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "protocol_id": RANKING_PROTOCOL_ID,
                "pool_name": "aizynthfinder_only",
                "test_partition_loaded_only_after_model_freeze": True,
                "test_reactions_total": 4,
                "test_reactions_covered": 3,
                "coverage": 0.75,
                "predictions": file_fingerprint(ranks),
                "per_seed_metrics": per_seed,
            }
        ),
        encoding="utf-8",
    )

    rows, statistics = analyze_pool(
        manifest,
        {0: "P1", 1: "P1", 2: "P2", 3: "P3"},
        n_bootstrap=100,
        n_permutations=100,
        rng_seed=2026,
    )

    assert len(rows) == 4
    assert statistics["scopes"]["within_pool"]["top1"]["delta"] == pytest.approx(1 / 3)
    assert statistics["scopes"]["within_pool"]["mrr"]["delta"] == pytest.approx(1 / 6)
    assert statistics["scopes"]["end_to_end"]["top1"]["delta"] == pytest.approx(1 / 4)
    assert statistics["scopes"]["within_pool"]["top1"]["n_product_clusters"] == 2
    assert classify_top1(statistics) in {
        "positive_effect_supported",
        "no_clear_difference",
    }
