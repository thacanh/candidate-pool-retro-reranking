from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import rerank.experiments.run_ws_e_ranking as ws_e_ranking

from rerank.ws_e_streaming import (
    AUGMENTED_COLUMNS,
    BASELINE_COLUMNS,
    FULL_FEATURE_NAMES,
    POOL_PROTOCOL_ID,
    arm_view,
    build_pool_index,
    combine_embeddings,
    core_scalar_row,
    core_scalar_rows,
    full_feature_matrix,
    identity_digest,
    raw_identity_digest,
    load_pool_index,
    atomic_json,
    atomic_npz,
    fingerprint,
    FEATURE_PROTOCOL_ID,
    read_product_records,
)
from rerank.experiments.run_ws_e_ranking import evaluate_test, fit_validation, join_shard, prepare_selection


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_pool_index_supports_missing_products_and_random_access(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    pool = tmp_path / "pool.jsonl"
    index = tmp_path / "index.npz"
    manifest = tmp_path / "index.json"
    _write_jsonl(
        products,
        [
            {"product_rank": 0, "product": "CC", "canonical_product": "CC"},
            {"product_rank": 1, "product": "CO", "canonical_product": "CO"},
            {"product_rank": 2, "product": "CN", "canonical_product": "CN"},
        ],
    )
    _write_jsonl(
        pool,
        [
            {
                "protocol_id": POOL_PROTOCOL_ID,
                "product": "CC",
                "reactant": "C.C",
                "candidate_rank": 1,
                "prior": 1.0,
                "source_aizynthfinder": 1,
                "source_localretro": 0,
            },
            {
                "protocol_id": POOL_PROTOCOL_ID,
                "product": "CN",
                "reactant": "C.N",
                "candidate_rank": 1,
                "prior": 1.0,
                "source_aizynthfinder": 0,
                "source_localretro": 1,
            },
        ],
    )
    result = build_pool_index(pool, products, index, manifest)
    assert result["candidate_count"] == 2
    assert result["products_without_candidates"] == 1
    loaded = load_pool_index(index, manifest, products, pool)
    assert read_product_records(pool, loaded, 1) == []
    assert read_product_records(pool, loaded, 2)[0]["reactant"] == "C.N"


def test_source_indicators_are_matched_and_only_unimol_columns_differ() -> None:
    records = [
        {
            "prior": 0.75,
            "source_aizynthfinder": 1,
            "source_localretro": 0,
        },
        {
            "prior": 0.50,
            "source_aizynthfinder": 1,
            "source_localretro": 1,
        },
    ]
    core = np.asarray(
        [
            [0.1, 0.2, 3.0, 0.4, 2.0, 1.1],
            [0.5, 0.6, 7.0, 0.8, 1.0, 0.9],
        ],
        dtype=np.float32,
    )
    full = full_feature_matrix(records, core)
    baseline = arm_view(full, "baseline")
    augmented = arm_view(full, "augmented")
    assert full.shape == (2, len(FULL_FEATURE_NAMES))
    assert baseline.shape[1] == 6
    assert augmented.shape[1] == 9
    np.testing.assert_array_equal(baseline, full[:, BASELINE_COLUMNS])
    np.testing.assert_array_equal(augmented, full[:, AUGMENTED_COLUMNS])
    np.testing.assert_array_equal(baseline[:, -2:], full[:, -2:])
    assert set(AUGMENTED_COLUMNS).difference(BASELINE_COLUMNS) == {2, 3, 4}


def test_core_scalar_row_and_missing_fragment_fallback_are_finite() -> None:
    product = np.ones((2, 512), dtype=np.float32)
    fragment = np.full((3, 512), 0.5, dtype=np.float32)
    combined = combine_embeddings(("C", "N"), {"C": fragment, "N": None})
    assert combined.shape == (3, 512)
    row = core_scalar_row("CC", "C.N", product, combined)
    grouped = core_scalar_rows("CC", ["C.N"], product, [combined])
    assert row.shape == (6,)
    np.testing.assert_allclose(grouped[0], row, rtol=0.0, atol=1e-6)
    assert np.isfinite(row).all()
    assert row[4] == 2.0
    assert identity_digest("C.N") == identity_digest("C.N")
    assert identity_digest("C.N") == identity_digest("N.C")
    assert raw_identity_digest("C.N") != raw_identity_digest("N.C")


def test_official_test_fails_closed_before_any_label_input(tmp_path: Path) -> None:
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"protocol_id": "wrong", "complete": False}))
    args = Namespace(model_freeze=str(freeze))
    with pytest.raises(PermissionError, match="complete immutable"):
        evaluate_test(args)


def test_pool_join_reuses_merged_scalars_by_candidate_identity(tmp_path: Path) -> None:
    products = tmp_path / "products.jsonl"
    merged = tmp_path / "merged.jsonl"
    subset = tmp_path / "subset.jsonl"
    merged_index = tmp_path / "merged_index.npz"
    merged_index_manifest = tmp_path / "merged_index.json"
    subset_index = tmp_path / "subset_index.npz"
    subset_index_manifest = tmp_path / "subset_index.json"
    _write_jsonl(
        products,
        [{"product_rank": 0, "product": "CC", "canonical_product": "CC"}],
    )
    base = {
        "protocol_id": POOL_PROTOCOL_ID,
        "product": "CC",
        "source_aizynthfinder": 1,
        "source_localretro": 0,
    }
    records = [
        {**base, "reactant": "C.C", "candidate_rank": 1, "prior": 1.0},
        {**base, "reactant": "C.N", "candidate_rank": 2, "prior": 0.5},
    ]
    _write_jsonl(merged, records)
    _write_jsonl(
        subset,
        [{**records[1], "reactant": "N.C", "candidate_rank": 1, "prior": 1.0}],
    )
    build_pool_index(merged, products, merged_index, merged_index_manifest)
    build_pool_index(subset, products, subset_index, subset_index_manifest)

    scalar_npz = tmp_path / "scalar.npz"
    atomic_npz(
        scalar_npz,
        product_rank=np.asarray([0, 0], dtype=np.int32),
        candidate_sha256=np.frombuffer(
            raw_identity_digest("C.C") + raw_identity_digest("C.N"), dtype=np.uint8
        ).reshape(2, 32),
        core_features=np.asarray(
            [[0.1, 0.2, 0.3, 0.4, 2.0, 1.0], [0.5, 0.6, 0.7, 0.8, 2.0, 1.1]],
            dtype=np.float32,
        ),
    )
    scalar_manifest = tmp_path / "scalar.json"
    atomic_json(
        scalar_manifest,
        {
            "protocol_id": FEATURE_PROTOCOL_ID,
            "product_rank_start": 0,
            "product_rank_stop": 1,
            "output": {
                **fingerprint(scalar_npz),
                "path": "/workspace/retired-run/scalars/shards/scalar.npz",
            },
        },
    )
    scalar_freeze = tmp_path / "scalar_freeze.json"
    atomic_json(
        scalar_freeze,
        {
            "protocol_id": FEATURE_PROTOCOL_ID,
            "complete": True,
            "shard_count": 1,
            "shard_manifests": [
                {
                    **fingerprint(scalar_manifest),
                    "path": "/workspace/retired-run/scalars/shards/scalar.json",
                }
            ],
        },
    )
    output = tmp_path / "joined.npz"
    output_manifest = tmp_path / "joined.json"
    join_shard(
        Namespace(
            scalar_freeze=str(scalar_freeze),
            shard_count=1,
            shard_index=0,
            pool_index_npz=str(subset_index),
            pool_index_manifest=str(subset_index_manifest),
            merged_jsonl=str(merged),
            merged_index_npz=str(merged_index),
            merged_index_manifest=str(merged_index_manifest),
            products_jsonl=str(products),
            pool_jsonl=str(subset),
            output=str(output),
            manifest=str(output_manifest),
            pool_name="subset",
        )
    )
    joined = np.load(output, allow_pickle=False)
    assert joined["features"].shape == (1, 9)
    np.testing.assert_allclose(
        joined["features"][0, 1:7],
        np.asarray([0.5, 0.6, 0.7, 0.8, 2.0, 1.1], dtype=np.float32),
    )
    assert joined["features"][0, 0] == 1.0


def test_selection_builds_seed_pairs_and_does_not_load_test_ground_truth(
    tmp_path: Path,
) -> None:
    products = tmp_path / "products.jsonl"
    pool = tmp_path / "pool.jsonl"
    index = tmp_path / "index.npz"
    index_manifest = tmp_path / "index.json"
    _write_jsonl(
        products,
        [
            {"product_rank": 0, "product": "CC", "canonical_product": "CC"},
            {"product_rank": 1, "product": "CO", "canonical_product": "CO"},
            {"product_rank": 2, "product": "CN", "canonical_product": "CN"},
        ],
    )
    records = []
    candidates = (("CC", ("C.C", "C.N")), ("CO", ("C.O", "C.N")), ("CN", ("C.N", "C.C")))
    for product, values in candidates:
        for rank, reactant in enumerate(values, start=1):
            records.append(
                {
                    "protocol_id": POOL_PROTOCOL_ID,
                    "product": product,
                    "reactant": reactant,
                    "candidate_rank": rank,
                    "prior": 1.0 if rank == 1 else 0.0,
                    "source_aizynthfinder": 1,
                    "source_localretro": 0,
                }
            )
    _write_jsonl(pool, records)
    build_pool_index(pool, products, index, index_manifest)

    feature_npz = tmp_path / "features.npz"
    feature_rows = np.arange(6 * 9, dtype=np.float32).reshape(6, 9) / 100.0
    atomic_npz(
        feature_npz,
        product_rank=np.repeat(np.arange(3, dtype=np.int32), 2),
        candidate_sha256=np.frombuffer(
            b"".join(identity_digest(record["reactant"]) for record in records),
            dtype=np.uint8,
        ).reshape(6, 32),
        features=feature_rows,
    )
    feature_manifest = tmp_path / "feature_shard.json"
    atomic_json(
        feature_manifest,
        {
            "protocol_id": "ws-e-three-pool-frozen-ranker-v1",
            "pool_name": "synthetic",
            "output": fingerprint(feature_npz),
        },
    )
    feature_freeze = tmp_path / "feature_freeze.json"
    atomic_json(
        feature_freeze,
        {
            "protocol_id": "ws-e-three-pool-frozen-ranker-v1",
            "pool_name": "synthetic",
            "complete": True,
            "shard_manifests": [fingerprint(feature_manifest)],
            "pool_index_manifest": fingerprint(index_manifest),
        },
    )
    source = tmp_path / "source.csv"
    source.write_text(
        "reactants_smiles,products_smiles,set\n"
        "C.C,CC,train\n"
        "C.O,CO,valid\n"
        # Invalid test reference would fail if selection canonicalized it.
        "definitely_not_smiles,CN,test\n",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "reaction_id,reaction_class,source_split\n"
        "0,1,train\n1,2,valid\n2,3,test\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "selection"
    result = prepare_selection(
        Namespace(
            feature_freeze=str(feature_freeze),
            pool_index_npz=str(index),
            pool_index_manifest=str(index_manifest),
            products_jsonl=str(products),
            pool_jsonl=str(pool),
            source_csv=str(source),
            metadata_csv=str(metadata),
            output_root=str(output_root),
        )
    )
    assert result["test_partition_loaded"] is False
    assert result["test_ground_truth_loaded"] is False
    assert result["selection_load_audit"]["test_ground_truth_loaded"] is False
    assert result["validation_audit"]["reactions_covered"] == 1
    for seed in range(42, 47):
        pairs = np.load(output_root / f"train_pairs_seed_{seed}.npz")
        assert pairs["positive"].shape == (1, 9)
        assert pairs["negative"].shape == (1, 9)


def test_validation_fit_uses_frozen_config_without_test(tmp_path: Path, monkeypatch) -> None:
    selection_root = tmp_path / "selection"
    selection_root.mkdir()
    pairs = selection_root / "train_pairs_seed_42.npz"
    valid = selection_root / "official_valid.npz"
    positive = np.asarray([[1.0] * 9, [0.8] * 9], dtype=np.float32)
    negative = np.asarray([[0.0] * 9, [0.2] * 9], dtype=np.float32)
    atomic_npz(pairs, positive=positive, negative=negative)
    valid_features = np.asarray([[1.0] * 9, [0.0] * 9], dtype=np.float32)
    atomic_npz(
        valid,
        features=valid_features,
        offsets=np.asarray([0, 2], dtype=np.int64),
        match_mask=np.asarray([True, False]),
        reaction_id=np.asarray([1], dtype=np.int64),
        product_rank=np.asarray([0], dtype=np.int32),
    )
    atomic_json(
        selection_root / "manifest.json",
        {
            "pool_name": "synthetic",
            "train_pairs": {"42": fingerprint(pairs)},
            "validation": fingerprint(valid),
        },
    )
    config = {
        "index": 62,
        "hidden_width": 8,
        "dropout": 0.0,
        "learning_rate": 0.001,
        "margin": 0.1,
    }
    primary = tmp_path / "primary.json"
    atomic_json(
        primary,
        {
            "protocol_id": "cap10-tuned-v1",
            "selected_baseline": {"config": config},
            "selected_augmented": {"config": {**config, "index": 70}},
        },
    )
    monkeypatch.setattr(ws_e_ranking, "MAX_EPOCHS", 2)
    monkeypatch.setattr(ws_e_ranking, "PATIENCE", 1)
    result = fit_validation(
        Namespace(
            seed=42,
            arm="baseline",
            selection_root=str(selection_root),
            primary_freeze=str(primary),
            output_root=str(tmp_path / "models"),
            device="cpu",
        )
    )
    assert result["config"] == config
    assert result["test_partition_loaded"] is False
    assert Path(result["checkpoint"]["path"]).is_file()
    assert result["epochs_completed"] <= 2
