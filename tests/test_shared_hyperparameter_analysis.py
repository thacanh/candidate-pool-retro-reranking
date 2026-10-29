from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from rerank.experiments import run_shared_hyperparameter_sensitivity as shared


def test_alt_matrix_zero_rank_has_zero_rr_without_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "reaction_ranks.jsonl"
    rows = []
    for reaction_id, covered, rank in ((1, True, 1), (2, False, 0), (3, True, 4)):
        row = {"reaction_id": reaction_id, "covered": covered}
        for arm in ("baseline", "augmented"):
            for seed in shared.SEEDS:
                row[f"{arm}_rank_seed_{seed}"] = rank
        rows.append(row)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    matrices, _ = shared._load_alt_matrix(path)
    mrr = matrices["mrr"]["baseline"]
    assert np.isfinite(mrr).all()
    assert mrr[:, 0].tolist() == [1.0] * len(shared.SEEDS)
    assert mrr[:, 1].tolist() == [0.0] * len(shared.SEEDS)
    assert mrr[:, 2].tolist() == [0.25] * len(shared.SEEDS)


def test_pool_specific_clusters_do_not_require_historical_coverage() -> None:
    full_test_cluster = {10: "product-a", 11: "product-b", 12: "product-c"}
    historical_covered = {10: 0, 11: 1}
    alternative_covered = np.asarray([10, 12])

    clusters = np.asarray([full_test_cluster[int(item)] for item in alternative_covered])
    assert clusters.tolist() == ["product-a", "product-c"]
    assert 12 not in historical_covered
