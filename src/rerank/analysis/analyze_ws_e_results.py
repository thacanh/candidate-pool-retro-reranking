"""Summarize the frozen WS-E three-pool official-test comparison.

The script consumes only immutable post-freeze reaction-rank artifacts.  It
reports five-seed means plus canonical-product-clustered paired uncertainty for
the augmented-minus-baseline effect, separately within each covered pool and
end to end over all 5,004 official-test reactions.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from rerank.analysis.analyze_controlled_study import clustered_bootstrap, paired_cluster_permutation_test
from rerank.study_data import file_fingerprint, load_reactions
from rerank.ws_e_streaming import RANKING_PROTOCOL_ID, atomic_json


SEEDS = tuple(range(42, 47))
POOLS = ("aizynthfinder_only", "localretro_only", "merged")
SCOPES = ("within_pool", "end_to_end")
METRICS = ("top1", "mrr")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rank_metric(rank: int, metric: str) -> float:
    if rank <= 0:
        return 0.0
    if metric == "top1":
        return float(rank == 1)
    if metric == "mrr":
        return 1.0 / float(rank)
    raise ValueError(f"Unsupported metric: {metric}")


def analyze_pool(
    manifest_path: Path,
    product_by_reaction: dict[int, str],
    n_bootstrap: int,
    n_permutations: int,
    rng_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol_id") != RANKING_PROTOCOL_ID:
        raise ValueError("Unexpected WS-E ranking protocol.")
    if not manifest.get("test_partition_loaded_only_after_model_freeze"):
        raise PermissionError("WS-E test results were not created behind the model-freeze gate.")

    pool = str(manifest["pool_name"])
    recorded_predictions = Path(manifest["predictions"]["path"])
    predictions_path = recorded_predictions
    if not predictions_path.is_file():
        predictions_path = manifest_path.parent / recorded_predictions.name
    prediction_fingerprint = file_fingerprint(predictions_path)
    if prediction_fingerprint["sha256"] != manifest["predictions"]["sha256"]:
        raise ValueError("WS-E reaction-rank checksum mismatch.")

    records = _load_jsonl(predictions_path)
    if len(records) != int(manifest["test_reactions_total"]):
        raise ValueError("WS-E reaction-rank count mismatch.")
    reaction_ids = [int(record["reaction_id"]) for record in records]
    if len(reaction_ids) != len(set(reaction_ids)):
        raise ValueError("WS-E reaction ranks contain duplicate reaction IDs.")
    try:
        all_clusters = np.asarray(
            [product_by_reaction[reaction_id] for reaction_id in reaction_ids], dtype=object
        )
    except KeyError as exc:
        raise ValueError(f"Unknown official-test reaction ID: {exc.args[0]}") from exc

    summary_rows: list[dict[str, Any]] = []
    statistics: dict[str, Any] = {
        "pool_name": pool,
        "coverage": float(manifest["coverage"]),
        "covered": int(manifest["test_reactions_covered"]),
        "total": int(manifest["test_reactions_total"]),
        "prediction_artifact": prediction_fingerprint,
        "scopes": {},
    }
    for scope_index, scope in enumerate(SCOPES):
        mask = np.asarray(
            [bool(record["covered"]) for record in records], dtype=bool
        )
        if scope == "end_to_end":
            mask[:] = True
        clusters = all_clusters[mask]
        scope_result: dict[str, Any] = {}
        for metric_index, metric in enumerate(METRICS):
            baseline = np.asarray(
                [
                    [
                        _rank_metric(int(record.get(f"baseline_rank_seed_{seed}", 0)), metric)
                        for record in records
                        if scope == "end_to_end" or bool(record["covered"])
                    ]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            augmented = np.asarray(
                [
                    [
                        _rank_metric(int(record.get(f"augmented_rank_seed_{seed}", 0)), metric)
                        for record in records
                        if scope == "end_to_end" or bool(record["covered"])
                    ]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            differences = augmented - baseline
            rng = np.random.default_rng(rng_seed + 100 * scope_index + 10 * metric_index)
            point, low, high = clustered_bootstrap(
                differences, clusters, n_bootstrap, rng
            )
            permutation_rng = np.random.default_rng(
                rng_seed + 10_000 + 100 * scope_index + 10 * metric_index
            )
            p_value = paired_cluster_permutation_test(
                differences, clusters, n_permutations, permutation_rng
            )
            baseline_mean = float(baseline.mean())
            augmented_mean = float(augmented.mean())
            expected_baseline = float(
                np.mean(
                    [
                        manifest["per_seed_metrics"]["baseline"][str(seed)][scope][metric]
                        for seed in SEEDS
                    ]
                )
            )
            expected_augmented = float(
                np.mean(
                    [
                        manifest["per_seed_metrics"]["augmented"][str(seed)][scope][metric]
                        for seed in SEEDS
                    ]
                )
            )
            if not np.isclose(baseline_mean, expected_baseline, atol=1e-12):
                raise ValueError("Baseline summary does not reproduce the frozen manifest.")
            if not np.isclose(augmented_mean, expected_augmented, atol=1e-12):
                raise ValueError("Augmented summary does not reproduce the frozen manifest.")

            scope_result[metric] = {
                "baseline_mean": baseline_mean,
                "augmented_mean": augmented_mean,
                "delta": point,
                "ci95": [low, high],
                "paired_cluster_sign_flip_p": p_value,
                "n_reactions": int(mask.sum()),
                "n_product_clusters": int(len(np.unique(clusters))),
            }
            summary_rows.append(
                {
                    "pool": pool,
                    "scope": scope,
                    "metric": metric,
                    "baseline_mean": baseline_mean,
                    "augmented_mean": augmented_mean,
                    "delta": point,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_cluster_sign_flip_p": p_value,
                    "n_reactions": int(mask.sum()),
                    "n_product_clusters": int(len(np.unique(clusters))),
                    "coverage": float(manifest["coverage"]),
                }
            )
        statistics["scopes"][scope] = scope_result
    return summary_rows, statistics


def classify_top1(statistics: dict[str, Any]) -> str:
    low, high = statistics["scopes"]["within_pool"]["top1"]["ci95"]
    if low > 0:
        return "positive_effect_supported"
    if high < 0:
        return "negative_effect_supported"
    return "no_clear_difference"


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite WS-E analysis: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    test_reactions = {
        record.reaction_id: record.product_key
        for record in load_reactions(args.source_csv, args.metadata_csv)
        if record.source_split == "test"
    }
    if len(test_reactions) != 5004:
        raise ValueError("Expected exactly 5,004 official-test reactions.")

    rows: list[dict[str, Any]] = []
    pool_statistics: dict[str, Any] = {}
    input_manifests: dict[str, Any] = {}
    for pool, manifest_value in zip(POOLS, args.result_manifest, strict=True):
        manifest_path = Path(manifest_value)
        pool_rows, statistics = analyze_pool(
            manifest_path,
            test_reactions,
            args.bootstrap_samples,
            args.permutation_samples,
            args.rng_seed,
        )
        if statistics["pool_name"] != pool:
            raise ValueError("WS-E result manifests are not in the prescribed pool order.")
        rows.extend(pool_rows)
        statistics["top1_assessment"] = classify_top1(statistics)
        pool_statistics[pool] = statistics
        input_manifests[pool] = file_fingerprint(manifest_path)

    summary_csv = output_dir / "ws_e_three_pool_summary.csv"
    with summary_csv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    statistics_json = output_dir / "ws_e_paired_statistics.json"
    atomic_json(
        statistics_json,
        {
            "protocol_id": RANKING_PROTOCOL_ID,
            "method": "canonical-product-clustered paired bootstrap and sign flip",
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            "rng_seed": args.rng_seed,
            "pools": pool_statistics,
        },
    )
    analysis_manifest = {
        "schema_version": 1,
        "record_kind": "ws_e_three_pool_analysis",
        "protocol_id": RANKING_PROTOCOL_ID,
        "inputs": input_manifests,
        "source_csv": file_fingerprint(args.source_csv),
        "metadata_csv": file_fingerprint(args.metadata_csv),
        "outputs": {
            "summary_csv": file_fingerprint(summary_csv),
            "paired_statistics": file_fingerprint(statistics_json),
        },
        "test_partition_loaded_from_frozen_results_only": True,
        "pool_assessments": {
            pool: pool_statistics[pool]["top1_assessment"] for pool in POOLS
        },
    }
    atomic_json(output_dir / "manifest.json", analysis_manifest)
    return analysis_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-manifest", action="append", required=True)
    parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=100_000)
    parser.add_argument("--rng-seed", type=int, default=2026)
    args = parser.parse_args()
    if len(args.result_manifest) != len(POOLS):
        parser.error("Pass exactly three result manifests in AiZ, LocalRetro, merged order.")
    print(json.dumps(run_analysis(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
