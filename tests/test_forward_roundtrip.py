from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import rerank.experiments.run_forward_roundtrip as f1
from chemformer_jobs import migrate_checkpoint as migration


def _prediction_frame(*, augmented: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "reaction_id": [10, 11],
            "source_split": ["test", "test"],
            "product_smiles": ["CO", "CCO"],
            "ground_truth": ["C", "CC"],
            "baseline_top1": ["C", "CC"],
            "baseline_hit@1": [1, 1],
            "reranked_top1": ["N", "CC" if augmented else "C"],
            "reranked_hit@1": [0, 1 if augmented else 0],
        }
    )


def _prepare_synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    for seed in (42, 43):
        _prediction_frame(augmented=False).to_csv(
            predictions / f"baseline_seed_{seed}.csv", index=False
        )
        _prediction_frame(augmented=True).to_csv(
            predictions / f"augmented_seed_{seed}.csv", index=False
        )
    g1_manifest = tmp_path / "g1_manifest.json"
    g1_manifest.write_text(
        json.dumps(
            {
                "manifest_kind": "g1_20seed_post_freeze_test_evaluation",
                "test_partition_loaded_only_after_20seed_freeze": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        f1,
        "verify_assets",
        lambda *_args, **_kwargs: {"verified": True, "task": "forward_prediction"},
    )
    prepared = tmp_path / "prepared"
    f1.prepare_inputs(
        prediction_dir=predictions,
        g1_manifest_path=g1_manifest,
        output_dir=prepared,
        asset_ledger_path=tmp_path / "unused.json",
        assets=f1.AssetPaths(tmp_path / "c", tmp_path / "v", tmp_path / "r"),
        seeds=(42, 43),
        expected_reactions=2,
    )
    return prepared


def test_canonical_product_preserves_stereochemistry_and_rejects_invalid() -> None:
    assert f1.canonical_product("C[C@@H](O)F") == "C[C@@H](O)F"
    assert f1.canonical_product("not-smiles") is None


def test_prepare_and_evaluate_frozen_roundtrip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = _prepare_synthetic(tmp_path, monkeypatch)
    usage = pd.read_csv(prepared / "frozen_top1_usages.csv")
    assert len(usage) == 10
    assert set(usage["system"]) == set(f1.SYSTEMS)

    inference = pd.read_csv(prepared / "inference_map.csv")
    prepare_manifest_path = prepared / "prepare_manifest.json"
    prepare_manifest = json.loads(prepare_manifest_path.read_text(encoding="utf-8"))
    for fingerprint in prepare_manifest["outputs"].values():
        fingerprint["path"] = "/relocated/bundle/" + Path(fingerprint["path"]).name
    prepare_manifest_path.write_text(
        json.dumps(prepare_manifest), encoding="utf-8"
    )
    output_rows = []
    for row in inference.itertuples(index=False):
        correct = row.precursor_smiles in {"N", "CC"}
        output_rows.append(
            {
                "ground_truth": row.product_smiles,
                "sampled_smiles_1": row.product_smiles if correct else "C",
                "sampled_smiles_2": row.product_smiles,
                "sampled_smiles_3": "invalid",
                "sampled_smiles_4": "C",
                "sampled_smiles_5": "C",
            }
        )
    model_output = tmp_path / "chemformer_output.tsv"
    pd.DataFrame(output_rows).to_csv(model_output, sep="\t", index=False)
    results = tmp_path / "results"
    manifest = f1.evaluate_output(
        prepare_dir=prepared,
        chemformer_output_path=model_output,
        result_dir=results,
        seeds=(42, 43),
        expected_reactions=2,
        bootstrap_replicates=50,
        permutation_replicates=100,
    )
    assert manifest["protocol_id"] == f1.F1_PROTOCOL_ID
    table = pd.read_csv(results / "system_metrics.csv").set_index("system")
    assert table.loc[f1.SYSTEM_AUGMENTED, "roundtrip_beam1_mean"] > table.loc[
        f1.SYSTEM_2D, "roundtrip_beam1_mean"
    ]
    assert (table["roundtrip_beam5_mean"] == 1.0).all()
    paired = json.loads((results / "paired_statistics.json").read_text())
    assert paired["roundtrip_beam1"]["augmented_minus_2d"] > 0


def test_prepare_rejects_unpaired_frozen_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    _prediction_frame(augmented=False).to_csv(
        predictions / "baseline_seed_42.csv", index=False
    )
    changed = _prediction_frame(augmented=True)
    changed.loc[0, "product_smiles"] = "CCC"
    changed.to_csv(predictions / "augmented_seed_42.csv", index=False)
    manifest = tmp_path / "g1.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_kind": "g1_20seed_post_freeze_test_evaluation",
                "test_partition_loaded_only_after_20seed_freeze": True,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(f1, "verify_assets", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="disagree on product_smiles"):
        f1.prepare_inputs(
            prediction_dir=predictions,
            g1_manifest_path=manifest,
            output_dir=tmp_path / "prepared",
            asset_ledger_path=tmp_path / "unused.json",
            assets=f1.AssetPaths(tmp_path / "c", tmp_path / "v", tmp_path / "r"),
            seeds=(42,),
            expected_reactions=2,
        )


def test_checkpoint_migration_changes_only_vocabulary_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.ckpt"
    output = tmp_path / "migrated.ckpt"
    manifest = tmp_path / "migration.json"
    torch.save(
        {
            "pytorch-lightning_version": "1.7.1",
            "epoch": 99,
            "global_step": 7800,
            "hyper_parameters": {"vocab_size": 523, "other": "unchanged"},
            "state_dict": {"weight": torch.arange(12, dtype=torch.float32).reshape(3, 4)},
        },
        source,
    )
    monkeypatch.setattr(migration, "torch", torch, raising=False)
    monkeypatch.setattr(migration, "EXPECTED_SOURCE_SIZE", source.stat().st_size)
    monkeypatch.setattr(migration, "EXPECTED_SOURCE_SHA256", migration.sha256(source))
    record = migration.migrate(source, output, manifest)
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    assert loaded["hyper_parameters"] == {
        "vocabulary_size": 523,
        "other": "unchanged",
    }
    assert torch.equal(
        loaded["state_dict"]["weight"],
        torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )
    metadata = record["checkpoint_metadata"]
    assert metadata["state_dict_logical_sha256_before"] == metadata[
        "state_dict_logical_sha256_after"
    ]
