"""Read-only N3 input/schema/relocation gate, before any long-running stage."""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import numpy as np

from package_inputs import PRIMARY, FEATURES, K1, POOLS, INDEX
from run_prior_selection import _pool_inputs
from rerank.experiments.protocol_gate import require_approval, sha256
from rerank.experiments.select_prior_transform import POOLS as POOL_NAMES, PROTOCOL_ID
from rerank.pool_streaming import load_pool_index
from rerank.analysis.transfer_statistics import common_cohort, historical_g1_ranks, k1_ranks
from package_inputs import COMMON_COHORT


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def local_record(record, directory):
    # Deliberately never fall back to the original workstation's absolute path.
    path = Path(directory) / PureWindowsPath(record["path"]).name
    if (not path.is_file() or path.stat().st_size != int(record["size_bytes"])
            or sha256(path) != record["sha256"].removeprefix("sha256:")):
        raise ValueError(f"Missing or changed relocated artifact: {path}")
    return path


def check(root=Path(".")):
    root = Path(root).resolve()
    approval = root / "docs/dd_remote_approval.json"
    plan = root / "docs/analysis_plan.md"
    for runner in ("run_prior_diagnostics.py", "run_prior_sensitivity.py",
                   "select_prior_transform.py", "run_prior_selection.py"):
        folder = "scripts" if runner == "run_prior_selection.py" else "src/rerank/experiments"
        require_approval(approval, plan, root / folder / runner, PROTOCOL_ID)
    args = SimpleNamespace(pool_root=root / POOLS, index_root=root / INDEX,
                           products_jsonl=root / POOLS / "products.jsonl",
                           source_csv=root / "data/uspto_smiles.csv",
                           metadata_csv=root / "data/uspto_reaction_metadata.csv")
    checks = {}
    for pool in POOL_NAMES:
        inputs = _pool_inputs(args, pool)
        load_pool_index(inputs["pool_index_npz"], inputs["pool_index_manifest"],
                        inputs["products_jsonl"], inputs["pool_jsonl"])
        folder = root / FEATURES / pool
        feature = read(folder / "manifest.json")
        if (not feature["complete"] or feature["pool_name"] != pool
                or feature["shard_count"] != 64):
            raise ValueError(f"Wrong feature freeze: {pool}")
        rows, stop = 0, 0
        for i, record in enumerate(feature["shard_manifests"]):
            shard_path = local_record(record, folder / "shards")
            shard = read(shard_path)
            if shard["shard_index"] != i or shard["product_rank_start"] != stop:
                raise ValueError(f"Shard identity/coverage mismatch: {shard_path}")
            stop = shard["product_rank_stop"]
            with np.load(local_record(shard["output"], shard_path.parent)) as arrays:
                n = len(arrays["product_rank"])
                if (arrays["features"].shape != (n, 9)
                        or arrays["candidate_sha256"].shape != (n, 32)
                        or not np.isfinite(arrays["features"]).all()):
                    raise ValueError(f"Invalid shard array: {shard_path}")
                rows += n
        if rows != feature["candidate_rows"] or stop != feature["product_count"]:
            raise ValueError(f"Incomplete feature coverage: {pool}")
        models = root / K1 / pool / "models"
        freeze = read(models / "freeze.json")
        if not freeze["complete"] or freeze["selected_prior_transform"] != "raw":
            raise ValueError(f"Wrong frozen raw comparator: {pool}")
        selection_dir = root / K1 / pool / "selection"
        selection = read(local_record(freeze["selection_manifest"], selection_dir))
        for record in selection["train_pairs"].values():
            local_record(record, selection_dir)
        local_record(selection["validation"], selection_dir)
        for arm in ("baseline", "augmented"):
            for seed in range(42, 62):
                directory = models / "validation" / arm / f"seed_{seed}"
                trial = read(local_record(freeze["trials"][arm][str(seed)], directory))
                if (trial["pool_name"] != pool or trial["arm"] != arm
                        or trial["seed"] != seed or trial["test_partition_loaded"]):
                    raise ValueError(f"Trial identity mismatch: {directory}")
                for key in ("checkpoint", "normalizer"):
                    local_record(trial[key], directory)
        test = read(root / K1 / pool / "test_results/manifest.json")
        local_record(test["feature_freeze"], folder)
        local_record(test["model_freeze"], models)
        local_record(test["predictions"], root / K1 / pool / "test_results")
        checks[pool] = {"shards": 64, "candidate_rows": rows, "models": 40,
                        "pool_file": Path(inputs["pool_jsonl"]).name}
        print(f"N3 PREFLIGHT: {pool} inputs and 40 frozen models verified", flush=True)
    historical = root / PRIMARY / "g1_20seed"
    freeze = read(historical / "freeze.json")
    cache = root / "outputs/jcheminform_revision/conformers/seed_42/features/official_3d_prior_schema1.pkl"
    if sha256(cache) != freeze["retained_train_test_cache_sha256"]:
        raise ValueError("Historical cache differs")
    for arm in ("baseline", "augmented"):
        for seed in range(42, 62):
            trial = freeze[f"selected_{arm}"]["trials"][str(seed)]
            for key in ("checkpoint", "normalizer"):
                if sha256(historical / trial[f"{key}_relpath"]) != trial[f"{key}_sha256"].removeprefix("sha256:"):
                    raise ValueError(f"Historical {key} differs: {arm}/{seed}")
            if not (historical / f"test_results/predictions/{arm}_seed_{seed}.csv").is_file():
                raise FileNotFoundError(f"Historical predictions: {arm}/{seed}")
    ids, _ = common_cohort(root / COMMON_COHORT)
    historical_g1_ranks(historical / "test_results/predictions", ids)
    for pool in POOL_NAMES:
        k1_ranks(root / K1 / pool / "test_results/reaction_ranks.jsonl", ids)
    return {"status": "pass", "read_only": True, "training_performed": False,
            "common_reactions": len(ids),
            "pool_checks": checks, "historical_models": 40}


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
