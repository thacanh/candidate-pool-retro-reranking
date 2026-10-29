"""Synthetic integration gates for the N3 release; never use scientific outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workflows/transfer"))
from run_prior_comparison import run_stage, PROTOCOL
from check_prior_inputs import local_record
from run_prior_selection import _pool_inputs
import run_prior_selection as runner
from rerank.experiments import select_prior_transform as selection_module
from rerank.experiments import run_pool_ranking as ranking
from rerank.experiments import run_prior_sensitivity as prior
from rerank.experiments.protocol_gate import fingerprint
from rerank.pool_streaming import (atomic_json, atomic_npz, build_pool_index,
    digest_array, POOL_PROTOCOL_ID, RANKING_PROTOCOL_ID, FULL_FEATURE_NAMES,
    BASELINE_COLUMNS, AUGMENTED_COLUMNS)


def test_pool_filename_and_relocation_fail_closed(tmp_path):
    args = NS(pool_root=tmp_path, index_root=tmp_path, products_jsonl="p",
              source_csv="s", metadata_csv="m")
    assert Path(_pool_inputs(args, "merged")["pool_jsonl"]).name == "merged_canonical_union.jsonl"
    assert Path(_pool_inputs(args, "localretro_only")["pool_jsonl"]).name == "localretro_only.jsonl"
    original = tmp_path / "old" / "a.txt"
    original.parent.mkdir()
    original.write_text("source")
    with pytest.raises(ValueError, match="relocated"):
        local_record(fingerprint(original), tmp_path / "new")


def test_completed_stage_resume_verifies_archive_and_current_outputs(tmp_path):
    result, archives, log = tmp_path / "result", tmp_path / "archives", tmp_path / "log.txt"
    result.mkdir()
    atomic_json(result / "manifest.json", {"status": "complete", "protocol_id": PROTOCOL,
        "comparator": "synthetic", "single_intended_change": "test fixture"})
    approval = tmp_path / "approval.json"
    atomic_json(approval, {"status": "approved", "approved_protocol_ids": [PROTOCOL]})
    command = [sys.executable, "-c", "raise AssertionError('must not rerun')"]
    for _ in range(2):
        run_stage("N3_S1", result, archives, approval, log, "manifest.json", command)
    with (result / "manifest.json").open("a") as f:
        f.write(" ")
    with pytest.raises(ValueError, match="changed"):
        run_stage("N3_S1", result, archives, approval, log, "manifest.json", command)


def test_partial_stage_is_preserved_not_overwritten(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    (result / "checkpoint.pt").write_bytes(b"partial")
    with pytest.raises(RuntimeError, match="partial"):
        run_stage("N3_S2", result, tmp_path / "archives", "unused", tmp_path / "log",
                  "choice_freeze.json", [sys.executable, "-c", "pass"])
    assert (result / "checkpoint.pt").read_bytes() == b"partial"


@pytest.mark.parametrize("transform", ["log", "rank"])
def test_transformed_features_prepare_fit_freeze_test_and_archive(tmp_path, monkeypatch, transform):
    # Real implementation all the way through; tiny synthetic chemistry, 2 epochs.
    # Production epochs, seeds, data and approval are never modified by this test.
    monkeypatch.setattr(prior, "require_approval", lambda *_: {})
    monkeypatch.setattr(ranking, "MAX_EPOCHS", 2)
    monkeypatch.setattr(ranking, "PATIENCE", 1)
    products, pool = tmp_path / "products.jsonl", tmp_path / "pool.jsonl"
    values = [("CC", ["C.C", "C.N"]), ("CO", ["C.O", "C.N"]),
              ("CN", ["C.N", "C.C"])]
    products.write_text("".join(json.dumps({"product_rank": i, "product": p,
        "canonical_product": p}) + "\n" for i, (p, _) in enumerate(values)), encoding="utf-8")
    records = [{"protocol_id": POOL_PROTOCOL_ID, "product": p, "reactant": c,
        "candidate_rank": j + 1, "prior": 0.8 if j == 0 else 0.2,
        "source_aizynthfinder": 1, "source_localretro": 0}
        for p, candidates in values for j, c in enumerate(candidates)]
    pool.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    index, index_manifest = tmp_path / "index.npz", tmp_path / "index.json"
    build_pool_index(pool, products, index, index_manifest)
    raw = tmp_path / "raw"
    npz = raw / "shards/shard_000_of_001.npz"
    features = np.arange(54, dtype=np.float32).reshape(6, 9) / 100
    features[:, 0] = [0.8, 0.2] * 3
    atomic_npz(npz, product_rank=np.repeat(np.arange(3, dtype=np.int32), 2),
               candidate_sha256=digest_array(r["reactant"] for r in records), features=features)
    shard = npz.with_suffix(".json")
    atomic_json(shard, {"pool_name": "synthetic", "shard_index": 0,
        "product_rank_start": 0, "product_rank_stop": 3, "output": fingerprint(npz)})
    raw_freeze = raw / "manifest.json"
    atomic_json(raw_freeze, {"protocol_id": RANKING_PROTOCOL_ID, "complete": True,
        "pool_name": "synthetic", "shard_count": 1, "product_count": 3, "candidate_rows": 6,
        "shard_manifests": [fingerprint(shard)], "pool_index_manifest": fingerprint(index_manifest),
        "feature_names": list(FULL_FEATURE_NAMES), "baseline_columns": list(BASELINE_COLUMNS),
        "augmented_columns": list(AUGMENTED_COLUMNS), "single_intended_arm_change": "synthetic control"})
    transformed = tmp_path / "transformed"
    approval = tmp_path / "approval.json"
    atomic_json(approval, {"status": "approved", "approved_protocol_ids": [PROTOCOL]})
    transform_args = NS(approval_record=approval, analysis_plan="unused", pool_name="synthetic",
        raw_feature_freeze=raw_freeze, output_root=transformed, shard_index=0, transform=transform)
    prior.build_shard(transform_args)
    prior.freeze(transform_args)
    source, metadata = tmp_path / "source.csv", tmp_path / "metadata.csv"
    source.write_text("reactants_smiles,products_smiles,set\nC.C,CC,train\nC.O,CO,valid\nC.N,CN,test\n")
    metadata.write_text("reaction_id,reaction_class,source_split\n0,1,train\n1,2,valid\n2,3,test\n")
    selection, models = tmp_path / "selection", tmp_path / "models"
    inputs = dict(feature_freeze=transformed / "manifest.json", pool_index_npz=index,
        pool_index_manifest=index_manifest, products_jsonl=products, pool_jsonl=pool,
        source_csv=source, metadata_csv=metadata)
    prepared = ranking.prepare_selection(NS(**inputs, output_root=selection,
        candidate_cap=10, seeds=(42,), output_protocol_id=PROTOCOL))
    assert prepared["prior_transform"] == transform and not prepared["test_ground_truth_loaded"]
    config = dict(hidden_width=8, dropout=0.0, learning_rate=0.001, margin=0.1)
    primary = tmp_path / "primary.json"
    atomic_json(primary, {"protocol_id": "cap10-tuned-v1",
        "selected_baseline": {"config": config}, "selected_augmented": {"config": config}})
    for arm in ("baseline", "augmented"):
        trial = ranking.fit_validation(NS(selection_root=selection, primary_freeze=primary,
            output_root=models, arm=arm, seed=42, device="cpu"))
        assert not trial["test_partition_loaded"]
    model_freeze = models / "freeze.json"
    frozen = ranking.freeze(NS(selection_root=selection, primary_freeze=primary,
        output_root=models, output=model_freeze))
    assert frozen["selected_prior_transform"] == transform
    result = ranking.evaluate_test(NS(**inputs, model_freeze=model_freeze, selection_root=selection,
        primary_freeze=primary, output_root=models, result_dir=tmp_path / "test", device="cpu"))
    assert result["test_reactions_covered"] == 1
    assert result["test_partition_loaded_only_after_model_freeze"]
    assert set(result["per_seed_metrics"]) == {"baseline", "augmented"}


def test_s2_full_choice_freeze_raw_log_rank_and_s3_raw_branch(tmp_path, monkeypatch):
    monkeypatch.setattr(selection_module, "require_approval", lambda *_: {})
    monkeypatch.setattr(runner, "require_approval", lambda *_: {})
    approval = tmp_path / "approval.json"
    atomic_json(approval, {"status": "approved", "approved_protocol_ids": [PROTOCOL]})
    s2, k1 = tmp_path / "s2", tmp_path / "k1"
    for pool, winner in zip(selection_module.POOLS, ("raw", "log", "rank")):
        raw_trials = {}
        for transform in selection_module.TRANSFORMS:
            base = k1 / pool if transform == "raw" else s2 / pool / transform
            for seed in selection_module.SEEDS:
                trial_dir = base / f"models/validation/baseline/seed_{seed}"
                trial_dir.mkdir(parents=True)
                for name in ("checkpoint.pt", "normalizer.npz"):
                    (trial_dir / name).write_bytes(b"synthetic-not-loadable-model")
                trial = dict(pool_name=pool, arm="baseline", seed=seed,
                    test_partition_loaded=False, best_validation_mrr=0.9 if transform == winner else 0.5,
                    checkpoint=fingerprint(trial_dir / "checkpoint.pt"),
                    normalizer=fingerprint(trial_dir / "normalizer.npz"))
                atomic_json(trial_dir / "trial.json", trial)
                if transform == "raw":
                    raw_trials[str(seed)] = fingerprint(trial_dir / "trial.json")
            if transform != "raw":
                for folder in ("features", "selection"):
                    atomic_json(base / folder / "manifest.json", dict(prior_transform=transform,
                        pool_name=pool, test_ground_truth_loaded=False))
        atomic_json(k1 / pool / "models/freeze.json", dict(complete=True,
            pool_name=pool, selected_prior_transform="raw", trials={"baseline": raw_trials}))
    chosen = selection_module.select(NS(approval_record=approval, analysis_plan="unused",
        k1_root=k1, n3_root=s2, output_root=s2))
    assert [chosen["choices"][p]["selected"] for p in selection_module.POOLS] == ["raw", "log", "rank"]
    assert not chosen["test_partition_loaded"]
    # Independent S3 fixture: every pool selects raw, so no training is allowed.
    raw_choice = tmp_path / "raw_choice"
    chosen["choices"] = {p: {"selected": "raw"} for p in selection_module.POOLS}
    for name in ("manifest.json", "choice_freeze.json"):
        atomic_json(raw_choice / name, chosen)
    monkeypatch.setattr(runner, "_prepare", lambda *a: pytest.fail("raw must not refit"))
    monkeypatch.setattr(runner, "common_cohort", lambda *_: ([2, 3], np.array(["CN", "CCO"])))
    monkeypatch.setattr(runner, "historical_g1_ranks", lambda *_: (np.ones((20, 2)), np.ones((20, 2))))
    for pool in selection_module.POOLS:
        path = k1 / pool / "test_results/reaction_ranks.jsonl"
        path.parent.mkdir()
        rows = [dict(reaction_id=i, covered=True,
            **{f"{arm}_rank_seed_{seed}": 1 for arm in ("baseline", "augmented")
               for seed in range(42, 62)}) for i in (2, 3)]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    result = runner.s3(NS(approval_record=approval, analysis_plan="unused", s2_root=raw_choice,
        s3_root=tmp_path / "s3", k1_root=k1, workers=1, common_reactions_csv="unused",
        historical_prediction_root="unused", bootstrap_samples=10, rng_seed=2180))
    assert result["status"] == "complete" and len(result["raw_selected_no_refit"]) == 3
    run_stage("N3_S3", tmp_path / "s3", tmp_path / "archives", approval,
              tmp_path / "log", "transfer_effects.csv", ["must-not-run"])
