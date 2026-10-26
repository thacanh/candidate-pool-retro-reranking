from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rerank.experiments import run_shared_hyperparameter_sensitivity as shared
from rerank.ws_e_streaming import fingerprint


def _json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _approval_root(tmp_path: Path) -> Path:
    for relative in (
        "docs/analysis_plan.md",
        "tests/test_shared_hyperparameter_sensitivity.py",
        "src/rerank/experiments/run_ws_e_ranking.py",
        "src/rerank/tuning.py",
        "src/rerank/analysis/analyze_predictions.py",
        "src/rerank/study_data.py",
        "environment.yml",
        "requirements.txt",
        "pyproject.toml",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    return tmp_path


def test_symmetric_strata_are_frozen_and_distinct() -> None:
    assert shared.SEEDS == (42, 43, 44, 45, 46)
    assert shared.STRATA["baseline_selected"] == {
        "dropout": 0.0,
        "hidden_width": 128,
        "index": 62,
        "learning_rate": 0.001,
        "margin": 0.3,
    }
    assert shared.STRATA["augmented_selected"] == {
        "dropout": 0.1,
        "hidden_width": 128,
        "index": 70,
        "learning_rate": 0.001,
        "margin": 0.1,
    }
    assert set(shared.MATCHING_ARM.values()) == {"baseline", "augmented"}


def test_holm_adjustment_matches_step_down_definition() -> None:
    values = np.asarray([0.01, 0.04, 0.03, 0.002])
    adjusted = shared.holm_adjust(values)
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert np.all(adjusted >= values)


def test_immutable_json_is_resumable_but_refuses_change(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    shared.write_immutable_json(path, {"value": 1})
    shared.write_immutable_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        shared.write_immutable_json(path, {"value": 2})


def test_approval_binds_plan_runner_test_and_environment(tmp_path: Path) -> None:
    root = _approval_root(tmp_path)
    record = shared.create_approval(root, "a" * 40)
    assert record["protocol_id"] == shared.PROTOCOL_ID
    assert shared.validate_approval(root)["approval_fingerprint"] == record["approval_fingerprint"]

    (root / "docs/analysis_plan.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        shared.validate_approval(root)


def test_selection_subset_keeps_only_prespecified_seeds(tmp_path: Path) -> None:
    root = _approval_root(tmp_path)
    approval = shared.create_approval(root, "b" * 40)
    source_path = shared._source_selection(root, "merged")
    source = {
        "protocol_id": "dd-expanded-pool-cap10-d1-v1",
        "pool_name": "merged",
        "candidate_cap": 10,
        "seeds": list(range(42, 62)),
        "train_pairs": {str(seed): {"path": f"pair-{seed}", "sha256": "x"} for seed in range(42, 62)},
        "train_audit": {str(seed): {"pairs": 1} for seed in range(42, 62)},
        "test_partition_loaded": False,
        "test_ground_truth_loaded": False,
    }
    _json(source_path, source)
    subset = shared._selection_subset(root, "merged", approval)
    assert subset["protocol_id"] == shared.PROTOCOL_ID
    assert subset["seeds"] == list(shared.SEEDS)
    assert set(subset["train_pairs"]) == {str(seed) for seed in shared.SEEDS}
    assert set(subset["train_audit"]) == {str(seed) for seed in shared.SEEDS}
    assert subset["source_selection_manifest"] == fingerprint(source_path)


def test_imported_trial_keeps_artifacts_read_only_and_verified(tmp_path: Path) -> None:
    root = _approval_root(tmp_path)
    approval = shared.create_approval(root, "c" * 40)
    checkpoint = tmp_path / "source/checkpoint.pt"
    normalizer = tmp_path / "source/normalizer.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    normalizer.write_bytes(b"normalizer")
    source_path = tmp_path / "source/trial.json"
    _json(
        source_path,
        {
            "pool_name": "aizynthfinder_only",
            "arm": "baseline",
            "seed": 42,
            "config": shared.STRATA["baseline_selected"],
            "checkpoint": fingerprint(checkpoint),
            "normalizer": fingerprint(normalizer),
            "test_partition_loaded": False,
        },
    )
    imported = shared._imported_trial(
        source_path,
        "aizynthfinder_only",
        "baseline_selected",
        "baseline",
        42,
        approval,
    )
    assert imported["imported_existing_trial"] is True
    assert imported["protocol_id"] == shared.PROTOCOL_ID
    assert imported["checkpoint"]["path"] == str(checkpoint.resolve())

    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="byte size changed"):
        shared._imported_trial(
            source_path,
            "aizynthfinder_only",
            "baseline_selected",
            "baseline",
            42,
            approval,
        )
