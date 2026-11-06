"""Isolated N3 S2/S3 remote stages; S1 has a separate frozen diagnostic runner.

S2 creates/archives validation-only selection before S3 may touch official test.
S4's 81-grid extension is intentionally not part of this launcher.
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

from rerank.analysis.transfer_statistics import (
    common_cohort,
    historical_g1_ranks,
    k1_ranks,
    transfer_summary,
    write_csv,
)
from rerank.experiments.protocol_gate import fingerprint, require_approval, sha256
from rerank.experiments.run_prior_sensitivity import build_shard, freeze as freeze_features
from rerank.experiments.select_prior_transform import POOLS, PROTOCOL_ID, select
from rerank.experiments.run_pool_ranking import (
    evaluate_test,
    freeze as freeze_models,
    prepare_selection,
)
from rerank.pool_streaming import atomic_json


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


def _prepare(
    args: argparse.Namespace, feature: Path, selection: Path, pool: str, seeds: tuple[int, ...]
) -> dict:
    return prepare_selection(
        SimpleNamespace(
            feature_freeze=str(feature),
            **_pool_inputs(args, pool),
            output_root=str(selection),
            candidate_cap=10,
            seeds=seeds,
            output_protocol_id=PROTOCOL_ID,
        )
    )


def _fit_task(spec: tuple) -> tuple[str, int]:
    selection, model_root, primary, pool, transform, arm, seed, device, log_root = spec
    log = Path(log_root) / f"{pool}_{transform}_{arm}_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "rerank.experiments.run_pool_ranking",
        "fit-validation",
        "--selection-root",
        str(selection),
        "--primary-freeze",
        str(primary),
        "--output-root",
        str(model_root),
        "--arm",
        arm,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    with log.open("x", encoding="utf-8") as handle:
        result = subprocess.run(
            command, stdout=handle, stderr=subprocess.STDOUT, check=False, env=os.environ.copy()
        )
    return str(log), result.returncode


def _fit_many(args: argparse.Namespace, tasks: list[tuple], max_workers: int) -> None:
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fit_task, spec): spec for spec in tasks}
        for future in as_completed(futures):
            spec = futures[future]
            log, code = future.result()
            print(f"N3 {spec[3]}/{spec[4]}/{spec[5]}/{spec[6]} rc={code} log={log}", flush=True)
            if code != 0:
                raise RuntimeError(f"N3 validation fit failed; inspect {log}")


def prepare_s2(args: argparse.Namespace) -> list[tuple]:
    """Prepare the exact six validation inputs; separable for release QA without fitting."""
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    root = Path(args.s2_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "N3-S2 output exists; resume requires an explicit audit, not overwrite."
        )
    tasks = []
    for pool in POOLS:
        source = Path(args.feature_root) / pool / "manifest.json"
        raw = _read(source)
        if raw.get("pool_name") != pool or not raw.get("complete"):
            raise ValueError("N3 raw feature freeze differs.")
        for transform in ("log", "rank"):
            print(
                f"N3-S2 preparing {pool}/{transform}: feature shards and validation inputs",
                flush=True,
            )
            item = root / pool / transform
            feature = item / "features"
            for shard_index in range(int(raw["shard_count"])):
                build_shard(
                    SimpleNamespace(
                        approval_record=args.approval_record,
                        analysis_plan=args.analysis_plan,
                        pool_name=pool,
                        transform=transform,
                        raw_feature_freeze=str(source),
                        output_root=str(feature),
                        shard_index=shard_index,
                    )
                )
            freeze_features(
                SimpleNamespace(
                    approval_record=args.approval_record,
                    analysis_plan=args.analysis_plan,
                    pool_name=pool,
                    transform=transform,
                    raw_feature_freeze=str(source),
                    output_root=str(feature),
                )
            )
            selection = item / "selection"
            _prepare(args, feature / "manifest.json", selection, pool, tuple(range(42, 47)))
            print(f"N3-S2 prepared {pool}/{transform}", flush=True)
            for seed in range(42, 47):
                tasks.append(
                    (
                        selection,
                        item / "models",
                        args.primary_freeze,
                        pool,
                        transform,
                        "baseline",
                        seed,
                        args.device,
                        root / "logs",
                    )
                )
    return tasks


def s2(args: argparse.Namespace) -> dict:
    tasks = prepare_s2(args)
    root = Path(args.s2_root).resolve()
    _fit_many(args, tasks, args.workers)
    result = select(
        SimpleNamespace(
            approval_record=args.approval_record,
            analysis_plan=args.analysis_plan,
            k1_root=args.k1_root,
            n3_root=str(root),
            output_root=str(root),
        )
    )
    print("N3-S2 validation choice frozen; official test has not been loaded.", flush=True)
    return result


def s3(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    require_approval(args.approval_record, args.analysis_plan, __file__, PROTOCOL_ID)
    s2_root = Path(args.s2_root).resolve()
    choice_path = s2_root / "choice_freeze.json"
    choice = _read(choice_path)
    if (
        choice.get("status") != "complete"
        or choice.get("protocol_id") != PROTOCOL_ID
        or choice.get("test_partition_loaded") is not False
        or sha256(choice_path) != sha256(s2_root / "manifest.json")
    ):
        raise PermissionError("N3 S2 selection was not frozen before S3 test access.")
    root = Path(args.s3_root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("N3-S3 output exists; refusing partial overwrite.")
    tasks = []
    selected = {pool: choice["choices"][pool]["selected"] for pool in POOLS}
    for pool, transform in selected.items():
        if transform == "raw":
            continue
        print(f"N3-S3 preparing {pool}/{transform}: 20 paired seeds", flush=True)
        feature = s2_root / pool / transform / "features" / "manifest.json"
        selection = root / pool / "selection"
        _prepare(args, feature, selection, pool, tuple(range(42, 62)))
        for arm in ("baseline", "augmented"):
            for seed in range(42, 62):
                tasks.append(
                    (
                        selection,
                        root / pool / "models",
                        args.primary_freeze,
                        pool,
                        transform,
                        arm,
                        seed,
                        args.device,
                        root / "logs",
                    )
                )
    _fit_many(args, tasks, args.workers)
    # Lock every non-raw pool model before loading any official-test reference.
    for pool, transform in selected.items():
        if transform == "raw":
            continue
        item = root / pool
        freeze_models(
            SimpleNamespace(
                selection_root=str(item / "selection"),
                primary_freeze=args.primary_freeze,
                output_root=str(item / "models"),
                output=str(item / "models" / "freeze.json"),
            )
        )
    result_paths = {}
    for pool, transform in selected.items():
        if transform == "raw":
            result_paths[pool] = Path(args.k1_root) / pool / "test_results" / "reaction_ranks.jsonl"
            continue
        item = root / pool
        feature = s2_root / pool / transform / "features" / "manifest.json"
        selection = item / "selection"
        model_root = item / "models"
        model_freeze = model_root / "freeze.json"
        actual = _read(model_freeze)
        if actual.get("selected_prior_transform") != transform:
            raise PermissionError("N3 model freeze did not retain validation-selected transform.")
        test = item / "test_results"
        evaluate_test(
            SimpleNamespace(
                model_freeze=str(model_freeze),
                feature_freeze=str(feature),
                selection_root=str(selection),
                primary_freeze=args.primary_freeze,
                output_root=str(model_root),
                result_dir=str(test),
                device=args.device,
                **_pool_inputs(args, pool),
            )
        )
        result_paths[pool] = test / "reaction_ranks.jsonl"
    ids, clusters = common_cohort(args.common_reactions_csv)
    historical = historical_g1_ranks(args.historical_prediction_root, ids)
    expanded = {pool: k1_ranks(path, ids) for pool, path in result_paths.items()}
    rows = transfer_summary(
        historical, expanded, clusters, draws=args.bootstrap_samples, seed=args.rng_seed
    )
    effects_path = root / "transfer_effects.csv"
    write_csv(effects_path, rows)
    result = {
        "schema_version": 1,
        "status": "complete",
        "stage": "N3_S3",
        "protocol_id": PROTOCOL_ID,
        "comparator": "pool-specific validation-selected prior versus historical frozen G1",
        "single_intended_change": "refit both arms only for pools selecting nonraw prior",
        "choices": selected,
        "raw_selected_no_refit": [p for p, t in selected.items() if t == "raw"],
        "common_reactions": len(ids),
        "seeds": list(range(42, 62)),
        "s2_choice_freeze": fingerprint(choice_path),
        "result_predictions": {pool: fingerprint(path) for pool, path in result_paths.items()},
        "transfer_effects": fingerprint(effects_path),
        "test_partition_loaded_only_after_choice_and_model_freezes": True,
        "approval": fingerprint(args.approval_record),
        "runtime_seconds": time.perf_counter() - started,
    }
    atomic_json(root / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("s2", "s3"))
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--k1-root", required=True)
    parser.add_argument("--feature-root", required=True)
    parser.add_argument("--pool-root", required=True)
    parser.add_argument("--index-root", required=True)
    parser.add_argument("--products-jsonl", required=True)
    parser.add_argument("--source-csv", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--primary-freeze", required=True)
    parser.add_argument("--historical-prediction-root")
    parser.add_argument("--common-reactions-csv")
    parser.add_argument("--s2-root", required=True)
    parser.add_argument("--s3-root")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--rng-seed", type=int, default=2180)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.stage == "s3" and not all(
        (args.s3_root, args.historical_prediction_root, args.common_reactions_csv)
    ):
        parser.error(
            "S3 requires --s3-root, --historical-prediction-root and --common-reactions-csv"
        )
    result = s2(args) if args.stage == "s2" else s3(args)
    print(json.dumps({"status": result["status"], "stage": args.stage}, sort_keys=True))


if __name__ == "__main__":
    main()
