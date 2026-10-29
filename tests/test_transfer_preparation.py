"""Synthetic gates for isolated N3/N4 preparation without official-test data."""

from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from rerank.experiments.protocol_gate import fingerprint
from rerank.experiments import build_pool_morgan_features as n4
from rerank.pool_streaming import (
    POOL_PROTOCOL_ID, RANKING_PROTOCOL_ID, atomic_json, atomic_npz,
    build_pool_index, digest_array,
)

from rerank.experiments.run_prior_sensitivity import transformed_shard
from rerank.experiments.select_prior_transform import choose_transform
from rerank.experiments.run_pool_ranking import _selection_arm_view


def test_n3_prior_transform_is_per_list_and_never_reorders() -> None:
    features = np.zeros((5, 9), dtype=np.float32)
    features[:, 0] = [0.8, 0.3, 0.3, 0.6, 0.2]
    features[:, 1:] = np.arange(40, dtype=np.float32).reshape(5, 8)
    ranks = np.asarray([0, 0, 0, 1, 1])
    result = transformed_shard(features, ranks, "rank")
    np.testing.assert_allclose(result[:, 0], [1.0, 0.25, 0.25, 1.0, 0.0])
    np.testing.assert_array_equal(result[:, 1:], features[:, 1:])
    assert result is not features


def test_n3_choice_tie_priority() -> None:
    assert choose_transform({"raw": 0.8, "log": 0.8 + 5e-13, "rank": 0.7}) == "raw"
    assert choose_transform({"raw": 0.7, "log": 0.8, "rank": 0.8}) == "log"
    assert choose_transform({"raw": 0.7, "log": 0.71, "rank": 0.8}) == "rank"
    with pytest.raises(ValueError):
        choose_transform({"raw": 0.8, "rank": 0.7})


def test_morgan_seven_column_selection_view_is_manifest_driven() -> None:
    features = np.arange(14, dtype=np.float32).reshape(2, 7)
    selection = {"baseline_columns": [0, 1, 5, 6],
                 "augmented_columns": list(range(7))}
    np.testing.assert_array_equal(_selection_arm_view(features, selection, "baseline"),
                                  features[:, [0, 1, 5, 6]])
    np.testing.assert_array_equal(_selection_arm_view(features, selection, "augmented"),
                                  features)
    with pytest.raises(ValueError):
        _selection_arm_view(features, {**selection, "baseline_columns": [0, 7]}, "baseline")


def test_n4_synthetic_shard_keeps_candidate_order_and_zero_candidate_product(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(n4, "require_approval", lambda *_: {})

    class Encoder:
        def initialize(self):
            pass

        def metadata(self):
            return {"name": "synthetic-encoder"}

    monkeypatch.setattr(n4, "MorganAtomEncoder", Encoder)
    monkeypatch.setattr(n4, "compute_query_control_features",
                        lambda _encoder, _product, candidates, _batch: np.asarray(
                            [[float(index), 0.5, 0.25] for index in range(len(candidates))],
                            dtype=np.float32))
    products = tmp_path / "products.jsonl"
    products.write_text(
        '{"product_rank":0,"product":"CCO","canonical_product":"CCO"}\n'
        '{"product_rank":1,"product":"CCN","canonical_product":"CCN"}\n',
        encoding="utf-8",
    )
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(
        '{"protocol_id":"%s","product":"CCO","candidate_rank":%d,'
        '"reactant":"%s","source_aizynthfinder":1,"source_localretro":0}\n'
        % (POOL_PROTOCOL_ID, rank, candidate)
        for rank, candidate in enumerate(("C.C", "CO", "O.C"), 1)
    ), encoding="utf-8")
    index_npz = tmp_path / "index.npz"
    index_manifest = tmp_path / "index.json"
    build_pool_index(pool, products, index_npz, index_manifest)
    source_root = tmp_path / "raw"
    source_npz = source_root / "shards" / "shard_000_of_001.npz"
    source_json = source_npz.with_suffix(".json")
    source = np.arange(27, dtype=np.float32).reshape(3, 9)
    atomic_npz(source_npz, product_rank=np.zeros(3, dtype=np.int32),
               candidate_sha256=digest_array(("C.C", "CO", "O.C")),
               features=source)
    atomic_json(source_json, {"product_rank_start": 0, "product_rank_stop": 2,
                              "output": fingerprint(source_npz)})
    source_freeze = source_root / "manifest.json"
    atomic_json(source_freeze, {
        "protocol_id": RANKING_PROTOCOL_ID, "complete": True, "pool_name": "synthetic",
        "shard_count": 1, "shard_manifests": [fingerprint(source_json)],
        "pool_index_manifest": fingerprint(index_manifest),
    })
    result = n4.build_shard(SimpleNamespace(
        approval_record="unused", analysis_plan="unused", raw_feature_freeze=str(source_freeze),
        pool_name="synthetic", shard_count=1, shard_index=0,
        pool_index_npz=str(index_npz), pool_index_manifest=str(index_manifest),
        products_jsonl=str(products), pool_jsonl=str(pool),
        output_root=str(tmp_path / "out"), encoder_batch_size=2,
    ))
    assert result["candidate_rows"] == 3
    with np.load(tmp_path / "out/shards/shard_000_of_001.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["candidate_sha256"],
                                      digest_array(("C.C", "CO", "O.C")))
        np.testing.assert_array_equal(arrays["features"][:, [0, 1, 5, 6]],
                                      source[:, [0, 1, 5, 6]])
        np.testing.assert_array_equal(arrays["features"][:, 2:5],
                                      [[0, 0.5, 0.25], [1, 0.5, 0.25],
                                       [2, 0.5, 0.25]])
