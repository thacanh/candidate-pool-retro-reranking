"""One isolated N4 instance: Morgan features, 20-seed fits, freeze, test, ZIP-ready.

The code checks all three pool-feature freezes and the historical 20-seed
Morgan model freeze before opening any official-test references.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rerank.analysis.transfer_statistics import common_cohort, k1_ranks, transfer_summary, write_csv
from rerank.experiments.protocol_gate import fingerprint, require_approval, sha256
from rerank.experiments.run_morgan_comparison import evaluate as evaluate_historical
from rerank.experiments.run_morgan_comparison import freeze as freeze_historical
from rerank.experiments.build_pool_morgan_features import freeze as freeze_features
from rerank.experiments.run_pool_ranking import (
    evaluate_test,
    freeze as freeze_models,
    prepare_selection,
)
from rerank.pool_streaming import atomic_json


PROTOCOL_ID = "dd-n4-morgan-atom-transfer-v1"
POOLS = ("aizynthfinder_only", "localretro_only", "merged")
SEEDS = tuple(range(42, 62))


def _read(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pool_inputs(args: argparse.Namespace, pool: str) -> dict:
    return {
        "pool_jsonl": str(
            Path(args.pool_root)
            / ("merged_canonical_union.jsonl" if pool == "merged" else f"{pool}.jsonl")
        ),
        "products_jsonl": args.products_jsonl,
        "pool_index_npz": str(Path(args.index_root) / f"{pool}_index.npz"),
        "pool_index_manifest": str(Path(args.index_root) / f"{pool}_index.json"),
        "source_csv": args.source_csv,
        "metadata_csv": args.metadata_csv,
    }


def _task(spec: tuple) -> tuple[str, int]:
    label, command, log_path = spec
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False, env=os.environ.copy()
        )
    return label, result.returncode


def _run_many(tasks: list[tuple], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_task, spec): spec for spec in tasks}
        for future in as_completed(futures):
            spec = futures[future]
            label, code = future.result()
            print(f"N4 {label} rc={code}", flush=True)
            if code:
                raise RuntimeError(f"N4 {label} failed; inspect {spec[2]}")


def _feature_tasks(args: argparse.Namespace, root: Path) -> list[tuple]:
    tasks = []
    for pool in POOLS:
        raw = Path(args.feature_root) / pool / "manifest.json"
        source = _read(raw)
        if source.get("pool_name") != pool or not source.get("complete"):
            raise ValueError("N4 source WS-E feature freeze differs.")
        paths = _pool_inputs(args, pool)
        count = int(source["shard_count"])
        # Keep each Python worker alive across eight shards so the 0.3--0.9 GB
        # source pool/index is checksummed once per worker, not once per shard.
        for first in range(0, count, 8):
            group = tuple(range(first, min(first + 8, count)))
            command = [
                sys.executable,
                "-m",
                "rerank.experiments.build_pool_morgan_features",
                "build-range",
                "--approval-record",
                args.approval_record,
                "--analysis-plan",
                args.analysis_plan,
                "--pool-name",
                pool,
                "--pool-index-manifest",
                paths["pool_index_manifest"],
                "--pool-index-npz",
                paths["pool_index_npz"],
                "--pool-jsonl",
                paths["pool_jsonl"],
                "--products-jsonl",
                paths["products_jsonl"],
                "--raw-feature-freeze",
                str(raw),
                "--shard-count",
                str(count),
                "--shard-indices",
                ",".join(map(str, group)),
                "--output-root",
                str(root / pool / "features"),
            ]
            log = root / "logs" / f"features_{pool}_{first:03d}-{group[-1]:03d}.log"
            tasks.append((f"features/{pool}/{first}-{group[-1]}", command, log))
    return tasks


def _fit_tasks(args: argparse.Namespace, root: Path) -> list[tuple]:
    tasks = []
    for pool in POOLS:
        selection = root / pool / "selection"
        for arm in ("baseline", "augmented"):
            for seed in SEEDS:
                command = [
                    sys.executable,
                    "-m",
                    "rerank.experiments.run_pool_ranking",
                    "fit-validation",
                    "--selection-root",
                    str(selection),
                    "--primary-freeze",
                    args.primary_freeze,
                    "--output-root",
                    str(root / pool / "models"),
                    "--arm",
                    arm,
                    "--seed",
                    str(seed),
                    "--device",
                    args.device,
                ]
                log = root / "logs" / f"fit_{pool}_{arm}_{seed}.log"
                tasks.append((f"fit/{pool}/{arm}/{seed}", command, log))
    for seed in range(47, 62):
        command = [
            sys.executable,
            "-m",
            "rerank.experiments.run_morgan_comparison",
            "fit-one",
            "--approval-record",
            args.approval_record,
            "--analysis-plan",
            args.analysis_plan,
            "--output-root",
            str(root / "historical" / "models"),
            "--primary-freeze",
            args.primary_freeze,
            "--existing-control-freeze",
            args.morgan_control_freeze,
            "--control-selection-bundle",
            args.morgan_selection_bundle,
            "--seed",
            str(seed),
            "--device",
            args.device,
        ]
        log = root / "logs" / f"fit_historical_augmented_{seed}.log"
        tasks.append((f"fit/historical/augmented/{seed}", command, log))
    return tasks


def run(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    root = Path(args.output_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("N4 output exists; refusing partial overwrite.")
    primary = _read(args.primary_freeze)
    control = _read(args.morgan_control_freeze)
    if (
        primary.get("protocol_id") != "cap10-tuned-v1"
        or control.get("primary_selected_config") != primary["selected_augmented"]["config"]
    ):
        raise ValueError("N4 augmented configuration differs from the historical Morgan control.")
    _run_many(_feature_tasks(args, root), args.feature_workers)
    for pool in POOLS:
        paths = _pool_inputs(args, pool)
        source = _read(Path(args.feature_root) / pool / "manifest.json")
        feature_root = root / pool / "features"
        freeze_features(
            SimpleNamespace(
                approval_record=args.approval_record,
                analysis_plan=args.analysis_plan,
                output_root=str(feature_root),
                pool_name=pool,
                pool_index_manifest=paths["pool_index_manifest"],
                shard_count=int(source["shard_count"]),
            )
        )
        prepare_selection(
            SimpleNamespace(
                feature_freeze=str(feature_root / "manifest.json"),
                output_root=str(root / pool / "selection"),
                candidate_cap=10,
                seeds=SEEDS,
                output_protocol_id=PROTOCOL_ID,
                **paths,
            )
        )
    _run_many(_fit_tasks(args, root), args.fit_workers)
    historical_models = root / "historical" / "models"
    freeze_historical(
        SimpleNamespace(
            approval_record=args.approval_record,
            analysis_plan=args.analysis_plan,
            primary_freeze=args.primary_freeze,
            existing_control_freeze=args.morgan_control_freeze,
            existing_control_root=args.morgan_control_root,
            control_selection_bundle=args.morgan_selection_bundle,
            g1_freeze=args.g1_freeze,
            output_root=str(historical_models),
        )
    )
    for pool in POOLS:
        item = root / pool
        freeze_models(
            SimpleNamespace(
                selection_root=str(item / "selection"),
                primary_freeze=args.primary_freeze,
                output_root=str(item / "models"),
                output=str(item / "models" / "freeze.json"),
            )
        )
    return finish(args, root, started)


def finish(args, root, started):
    """Evaluation only; all feature/model freezes must already exist and validate."""
    historical_models = root / 'historical' / 'models'
    # The complete validation-only freeze set exists before any official test read.
    evaluate_historical(
        SimpleNamespace(
            approval_record=args.approval_record,
            analysis_plan=args.analysis_plan,
            output_root=str(historical_models),
            existing_control_freeze=args.morgan_control_freeze,
            control_cache=args.morgan_test_cache,
            historical_baseline_predictions=args.historical_prediction_root,
            existing_control_predictions=args.morgan_old_predictions,
            result_dir=str(root / "historical" / "test_results"),
            device=args.device,
            reproduction_audit=getattr(args, 'reproduction_audit', None),
        )
    )
    for pool in POOLS:
        print(f'N4 evaluating frozen pool: {pool}', flush=True)
        item = root / pool
        evaluate_test(
            SimpleNamespace(
                model_freeze=str(item / "models" / "freeze.json"),
                feature_freeze=str(item / "features" / "manifest.json"),
                selection_root=str(item / "selection"),
                primary_freeze=args.primary_freeze,
                output_root=str(item / "models"),
                result_dir=str(item / "test_results"),
                device=args.device,
                **_pool_inputs(args, pool),
            )
        )
    ids, clusters = common_cohort(args.common_reactions_csv)
    print('N4 paired transfer statistics', flush=True)
    with np.load(
        root / "historical" / "test_results" / "reaction_ranks.npz", allow_pickle=False
    ) as arrays:
        rank_by_id = {int(rid): index for index, rid in enumerate(arrays["reaction_ids"])}
        if not set(ids).issubset(rank_by_id):
            raise ValueError("Historical Morgan test ranks miss the 3,814 common cohort.")
        positions = [rank_by_id[rid] for rid in ids]
        historical = (
            arrays["baseline_ranks"][:, positions].copy(),
            arrays["augmented_ranks"][:, positions].copy(),
        )
    expanded = {
        pool: k1_ranks(root / pool / "test_results" / "reaction_ranks.jsonl", ids) for pool in POOLS
    }
    effect_path = root / "transfer_effects.csv"
    write_csv(
        effect_path,
        transfer_summary(
            historical, expanded, clusters, draws=args.bootstrap_samples, seed=args.rng_seed
        ),
    )
    result = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "comparator": "historical and three expanded cap-10 pools with atom-centred Morgan features",
        "single_intended_change": "candidate pool only; same seven Morgan-control features and matched frozen ranking procedure",
        "common_reactions": len(ids),
        "seeds": list(SEEDS),
        "feature_freezes": {
            pool: fingerprint(root / pool / "features" / "manifest.json") for pool in POOLS
        },
        "model_freezes": {
            pool: fingerprint(root / pool / "models" / "freeze.json") for pool in POOLS
        },
        "historical_model_freeze": fingerprint(historical_models / "freeze.json"),
        "historical_test": fingerprint(root / "historical" / "test_results" / "manifest.json"),
        "expanded_tests": {
            pool: fingerprint(root / pool / "test_results" / "manifest.json") for pool in POOLS
        },
        "transfer_effects": fingerprint(effect_path),
        "test_partition_loaded_only_after_all_four_model_freezes": True,
        "approval": fingerprint(args.approval_record),
        "runtime_seconds": time.perf_counter() - started,
        "runtime_scope": "evaluation/statistics continuation"
        if getattr(args, 'reproduction_audit', None)
        else "full pipeline",
    }
    atomic_json(root / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--products-jsonl", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--primary-freeze", required=True)
    parser.add_argument("--g1-freeze", required=True)
    parser.add_argument("--historical-prediction-root", required=True)
    parser.add_argument("--morgan-control-freeze", required=True)
    parser.add_argument("--morgan-control-root", required=True)
    parser.add_argument("--morgan-selection-bundle", required=True)
    parser.add_argument("--morgan-test-cache", required=True)
    parser.add_argument("--morgan-old-predictions", required=True)
    parser.add_argument("--common-reactions-csv", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--fit-workers", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=2230)
    args = parser.parse_args()
    if min(args.feature_workers, args.fit_workers, args.bootstrap_samples) < 1:
        parser.error("worker/bootstrap counts must be positive")
    result = run(args)
    print(json.dumps({"status": result["status"], "protocol_id": PROTOCOL_ID}, sort_keys=True))


if __name__ == "__main__":
    main()
