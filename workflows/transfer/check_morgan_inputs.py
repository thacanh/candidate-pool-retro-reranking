"""Read-only N4 path, checksum, approval and schema gate before feature work."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from package_inputs import PRIMARY, MORGAN, FEATURES, POOLS, INDEX
from check_prior_inputs import local_record
from run_morgan_comparison import _pool_inputs, PROTOCOL_ID
from rerank.experiments.protocol_gate import require_approval, sha256
from rerank.experiments.run_morgan_comparison import _artifact
from rerank.experiments.run_tuned import validate_selection_freeze
from rerank.tuning import load_selection_bundle
from rerank.pool_streaming import load_pool_index


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check():
    for runner in (
        "run_morgan_comparison.py",
        "run_morgan_comparison.py",
        "build_pool_morgan_features.py",
    ):
        parent = "scripts" if runner == "run_morgan_comparison.py" else "src/rerank/experiments"
        require_approval(
            "docs/dd_remote_approval.json",
            "docs/analysis_plan.md",
            Path(parent) / runner,
            PROTOCOL_ID,
        )
    primary = validate_selection_freeze(Path(PRIMARY) / "freeze/model_selection.json")
    control = read(Path(MORGAN) / "ranking/freeze.json")
    assert control["protocol_id"] == "cap10-tuned-encoder-control-v1"
    assert control["primary_selection_fingerprint"] == primary["selection_fingerprint"]
    assert control["primary_selected_config"] == primary["selected_augmented"]["config"]
    assert control["prior_transform"] == "raw"
    selection = local_record(control["control_selection_bundle"], Path(MORGAN) / "ranking")
    assert (
        load_selection_bundle(selection)["representation_provenance"]["source_protocol_id"]
        == "C-MORGAN-ATOM"
    )
    assert (
        sha256(Path(MORGAN) / "cache/train_test.pkl")
        == control["retained_control_train_test_cache_sha256"]
    )
    for seed in range(42, 47):
        trial = control["selected_control"]["trials"][str(seed)]
        assert trial["seed"] == seed and trial["test_partition_loaded"] is False
        for kind in ("checkpoint", "normalizer"):
            _artifact(Path(MORGAN) / "ranking/artifacts", trial, kind)
        assert (
            Path(MORGAN) / f"ranking/test_results/predictions/augmented_seed_{seed}.csv"
        ).is_file()
    for seed in range(42, 62):
        assert (
            Path(PRIMARY) / f"g1_20seed/test_results/predictions/baseline_seed_{seed}.csv"
        ).is_file()
    args = SimpleNamespace(
        pool_root=POOLS,
        index_root=INDEX,
        products_jsonl=f"{POOLS}/products.jsonl",
        source_csv="data/uspto_smiles.csv",
        metadata_csv="data/uspto_reaction_metadata.csv",
    )
    for pool in ("aizynthfinder_only", "localretro_only", "merged"):
        paths = _pool_inputs(args, pool)
        load_pool_index(
            paths["pool_index_npz"],
            paths["pool_index_manifest"],
            paths["products_jsonl"],
            paths["pool_jsonl"],
        )
        folder = Path(FEATURES) / pool
        raw = read(folder / "manifest.json")
        assert raw["pool_name"] == pool and raw["complete"] and raw["shard_count"] == 64
        local_record(raw["pool_index_manifest"], INDEX)
        stop, count = 0, 0
        for i, record in enumerate(raw["shard_manifests"]):
            path = local_record(record, folder / "shards")
            shard = read(path)
            assert shard["shard_index"] == i and shard["product_rank_start"] == stop
            stop = shard["product_rank_stop"]
            with np.load(local_record(shard["output"], path.parent), allow_pickle=False) as arrays:
                n = len(arrays["product_rank"])
                assert arrays["features"].shape == (n, 9) and np.isfinite(arrays["features"]).all()
                count += n
        assert stop == raw["product_count"] and count == raw["candidate_rows"]
        print(f"N4 PREFLIGHT: {pool}, 64 shards verified", flush=True)
    return dict(
        status="pass",
        feature_shards=192,
        historical_morgan_models=5,
        scientific_training_performed=False,
        official_test_cache_unpickled=False,
    )


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
