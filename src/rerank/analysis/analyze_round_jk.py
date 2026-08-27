"""Frozen-result J1-J4, K1-K2 and L1 analyses for the independent DD paper.

The module consumes immutable JoC artifacts by checksum and writes only to the
Digital Discovery J/K namespace.  It does not train, generate candidates, or
modify the JoC numerical freeze.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from rerank.analysis.analyze_revision_predictions import (
    crossed_product_seed_bootstrap,
    product_cluster_bootstrap,
)
from rerank.experiments.run_round_jk import (
    OUTPUT_NAMESPACE,
    PROTOCOL_ID as K1_PROTOCOL_ID,
    SEEDS,
    _approval_provenance,
    _assert_isolated_output,
)
from rerank.study_data import canonicalize_reactant_set, canonicalize_smiles, file_fingerprint
from rerank.study_data import load_reactions
from rerank.ws_e_streaming import atomic_json


PROTOCOL_ID = "dd-round-jk-reanalysis-v1"
K2_REPORTING_PROTOCOL_ID = "dd-posthoc-k2-full-reporting-v2"
J12_SYNC_PROTOCOL_ID = "dd-round-jk-j1-j2-filtered-v2"
L1_PROTOCOL_ID = "dd-rxn-ebm-score-diagnostic-v1"
D5_PROTOCOL_ID = "D-EXTERNAL-RXN-EBM-FF-CAP10-v1"
D5_SEEDS = (0, 20210423, 77777777)
POOL_ORDER = ("historical_cap10", "aizynthfinder_only", "localretro_only", "merged")
METRICS = ("top1", "top3", "top5", "top10", "mrr")
FINAL_K1_COVERAGE = {
    "aizynthfinder_only": 4012,
    "localretro_only": 4462,
    "merged": 4584,
}
FINAL_COMMON_COVERED = 3814


def _j12_sync_approval(approval_path: str | Path, plan_path: str | Path) -> dict:
    raw = _load_json(approval_path)
    if (
        raw.get("final_inference_status") != "approved_frozen_artifacts_only"
        or raw.get("j12_sync_protocol_id") != J12_SYNC_PROTOCOL_ID
        or not raw.get("j12_sync_approval_date")
        or not raw.get("j12_sync_approval_quote")
    ):
        raise PermissionError("Filtered-v2 J1/J2 synchronization lacks approval.")
    result = _approval_provenance(approval_path, plan_path)
    result.update(
        {
            "j12_sync_protocol_id": raw["j12_sync_protocol_id"],
            "j12_sync_approval_date": raw["j12_sync_approval_date"],
            "j12_sync_approval_quote": raw["j12_sync_approval_quote"],
        }
    )
    return result


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metric(rank: int, metric: str) -> float:
    if rank <= 0:
        return 0.0
    if metric == "mrr":
        return 1.0 / rank
    return float(rank <= int(metric.removeprefix("top")))


def _metrics(ranks: Iterable[int]) -> dict[str, float]:
    values = [int(rank) for rank in ranks]
    if not values:
        raise ValueError("Metric denominator is empty.")
    return {metric: float(np.mean([_metric(rank, metric) for rank in values])) for metric in METRICS}


def _prediction_path(manifest_path: Path, manifest: Mapping) -> Path:
    recorded = Path(str(manifest["predictions"]["path"]))
    path = recorded if recorded.is_file() else manifest_path.parent / recorded.name
    actual = file_fingerprint(path)
    if actual["sha256"] != manifest["predictions"]["sha256"]:
        raise ValueError(f"Frozen prediction checksum mismatch: {path}")
    return path


def _anchor_records(path: str | Path) -> dict[int, dict]:
    rows = _read_csv(path)
    result = {}
    for row in rows:
        reaction_id = int(row["reaction_id"])
        if reaction_id in result:
            raise ValueError("Anchor prediction contains duplicate reaction IDs.")
        result[reaction_id] = {
            "reaction_id": reaction_id,
            "covered": True,
            "prior_rank": int(row["baseline_rank"]),
            "candidate_count": int(row["candidate_count"]),
            "product_smiles": row["product_smiles"],
        }
    if len(result) != 3985:
        raise ValueError(f"Expected 3,985 anchor-covered reactions, found {len(result)}.")
    return result


def _expanded_records(manifest_path: str | Path) -> tuple[dict, dict[int, dict]]:
    path = Path(manifest_path)
    manifest = _load_json(path)
    rows = _read_jsonl(_prediction_path(path, manifest))
    records = {int(row["reaction_id"]): row for row in rows}
    if len(records) != len(rows) or len(records) != int(manifest["test_reactions_total"]):
        raise ValueError("Expanded-pool predictions have invalid reaction coverage.")
    return manifest, records


def _anchor_gain(prediction_root: str | Path) -> float:
    effects = []
    root = Path(prediction_root)
    for seed in SEEDS:
        baseline = {int(row["reaction_id"]): row for row in _read_csv(root / f"baseline_seed_{seed}.csv")}
        augmented = {int(row["reaction_id"]): row for row in _read_csv(root / f"augmented_seed_{seed}.csv")}
        if baseline.keys() != augmented.keys():
            raise ValueError("Anchor paired prediction IDs differ.")
        effects.extend(
            float(int(augmented[key]["reranked_hit@1"]))
            - float(int(baseline[key]["reranked_hit@1"]))
            for key in baseline
        )
    return float(np.mean(effects))


def j1_j2(args: argparse.Namespace) -> dict:
    approval = _j12_sync_approval(args.approval_record, args.analysis_plan)
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite J1/J2 output: {output}")
    anchor = _anchor_records(args.anchor_baseline_csv)
    pool_records: dict[str, dict[int, dict]] = {"historical_cap10": anchor}
    manifests: dict[str, dict] = {}
    for pool, path in zip(POOL_ORDER[1:], args.expanded_manifest, strict=True):
        manifest, records = _expanded_records(path)
        seeds = tuple(sorted(int(seed) for seed in manifest.get("per_seed_metrics", {}).get("baseline", {})))
        augmented_seeds = tuple(
            sorted(int(seed) for seed in manifest.get("per_seed_metrics", {}).get("augmented", {}))
        )
        covered = sum(bool(row.get("covered")) for row in records.values())
        if (
            manifest.get("protocol_id") != K1_PROTOCOL_ID
            or manifest.get("pool_name") != pool
            or manifest.get("source_pool_protocol_id")
            != "ws-e-localretro-three-pools-filtered-v2"
            or int(manifest.get("candidate_cap", 0)) != 10
            or manifest.get("candidate_ordering")
            != "first rows of the frozen pool order; no re-sorting"
            or int(manifest.get("test_reactions_total", 0)) != 5004
            or int(manifest.get("test_reactions_covered", 0)) != FINAL_K1_COVERAGE[pool]
            or covered != FINAL_K1_COVERAGE[pool]
            or seeds != SEEDS
            or augmented_seeds != SEEDS
            or manifest.get("test_partition_loaded_only_after_model_freeze") is not True
        ):
            raise ValueError(f"{pool} is not the approved final filtered-v2 K1 result.")
        manifests[pool] = manifest
        pool_records[pool] = records

    covered_sets = {
        pool: {key for key, row in rows.items() if bool(row.get("covered", True))}
        for pool, rows in pool_records.items()
    }
    common = set.intersection(*(covered_sets[pool] for pool in POOL_ORDER))
    if len(common) != FINAL_COMMON_COVERED:
        raise ValueError(
            f"Expected {FINAL_COMMON_COVERED} final common covered reactions, found {len(common)}."
        )

    gains = {"historical_cap10": _anchor_gain(args.anchor_prediction_root)}
    for pool in POOL_ORDER[1:]:
        manifest = manifests[pool]
        values = []
        for seed in sorted(manifest["per_seed_metrics"]["baseline"], key=int):
            values.append(
                float(manifest["per_seed_metrics"]["augmented"][seed]["within_pool"]["top1"])
                - float(manifest["per_seed_metrics"]["baseline"][seed]["within_pool"]["top1"])
            )
        gains[pool] = float(np.mean(values))

    metric_rows = []
    headroom_rows = []
    for pool in POOL_ORDER:
        records = pool_records[pool]
        own_ids = sorted(covered_sets[pool])
        own_metrics = _metrics(int(records[key]["prior_rank"]) for key in own_ids)
        common_metrics = _metrics(int(records[key]["prior_rank"]) for key in sorted(common))
        for scope, values, denominator in (
            ("own_covered", own_metrics, len(own_ids)),
            ("common_four_pool_covered", common_metrics, len(common)),
        ):
            metric_rows.append(
                {
                    "pool": pool,
                    "scope": scope,
                    "n_reactions": denominator,
                    **values,
                }
            )
        headroom = 1.0 - own_metrics["top1"]
        capture = None if headroom == 0.0 else gains[pool] / headroom
        headroom_rows.append(
            {
                "pool": pool,
                "coverage_count": len(own_ids),
                "coverage_total": 5004,
                "coverage": len(own_ids) / 5004,
                "prior_top1_within_pool": own_metrics["top1"],
                "headroom": headroom,
                "observed_augmented_minus_baseline_top1": gains[pool],
                "capture_rate": capture,
                "common_covered_count": len(common),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "j1_prior_only.csv", metric_rows)
    _write_csv(output / "j2_headroom_capture.csv", headroom_rows)
    result = {
        "schema_version": 1,
        "record_kind": "dd_round_j1_j2",
        "protocol_id": J12_SYNC_PROTOCOL_ID,
        "comparator": "four frozen candidate pools",
        "single_intended_change": (
            "align J1/J2 reporting to the final frozen filtered-v2 K1 cap-10 pools"
        ),
        "common_covered_count": len(common),
        "supersedes_for_reporting": (
            "dd-round-jk-reanalysis-v1 J1/J2 with 3,939 common reactions; old output preserved"
        ),
        "inputs": {
            "anchor_baseline_csv": file_fingerprint(args.anchor_baseline_csv),
            "anchor_prediction_root": str(Path(args.anchor_prediction_root).resolve()),
            "expanded_manifests": [file_fingerprint(path) for path in args.expanded_manifest],
        },
        "outputs": {
            "j1": file_fingerprint(output / "j1_prior_only.csv"),
            "j2": file_fingerprint(output / "j2_headroom_capture.csv"),
        },
        "round_jk_approval": approval,
        "training_performed": False,
        "retuning_performed": False,
        "candidate_generation_performed": False,
        "test_partition_loaded_from_frozen_predictions_only": True,
    }
    atomic_json(output / "manifest.json", result)
    return result


def _load_anchor_matrices(prediction_root: str | Path):
    root = Path(prediction_root)
    baseline_rows = []
    augmented_rows = []
    reference = None
    for seed in SEEDS:
        baseline = _read_csv(root / f"baseline_seed_{seed}.csv")
        augmented = _read_csv(root / f"augmented_seed_{seed}.csv")
        left = {int(row["reaction_id"]): row for row in baseline}
        right = {int(row["reaction_id"]): row for row in augmented}
        if left.keys() != right.keys():
            raise ValueError("Anchor paired prediction IDs differ.")
        ids = sorted(left)
        if reference is None:
            reference = [left[key] for key in ids]
        elif [int(row["reaction_id"]) for row in reference] != ids:
            raise ValueError("Anchor reaction order differs across seeds.")
        baseline_rows.append([left[key] for key in ids])
        augmented_rows.append([right[key] for key in ids])
    assert reference is not None
    return reference, baseline_rows, augmented_rows


def _candidate_bin(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 6:
        return "4-6"
    if count <= 9:
        return "7-9"
    if count == 10:
        return "10"
    raise ValueError(f"Historical cap-10 row has candidate_count={count}.")


def _rank_bin(rank: int) -> str:
    if rank == 1:
        return "1"
    if rank == 2:
        return "2"
    if rank == 3:
        return "3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    raise ValueError(f"Historical cap-10 baseline rank outside 1..10: {rank}")


def j3_j4(args: argparse.Namespace) -> dict:
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite J3/J4 output: {output}")
    reference, baseline, augmented = _load_anchor_matrices(args.anchor_prediction_root)
    if len(reference) != 3985:
        raise ValueError("J3/J4 require exactly 3,985 anchor-covered reactions.")
    clusters = np.asarray(
        [canonicalize_smiles(row["product_smiles"]) for row in reference], dtype=object
    )
    if any(value is None for value in clusters):
        raise ValueError("J3/J4 found a non-canonicalizable product.")
    candidate_bins = np.asarray([_candidate_bin(int(row["candidate_count"])) for row in reference])
    top1 = np.asarray(
        [
            [float(int(a["reranked_rank"]) == 1) - float(int(b["reranked_rank"]) == 1) for b, a in zip(bs, aug)]
            for bs, aug in zip(baseline, augmented)
        ],
        dtype=np.float64,
    )
    mrr = np.asarray(
        [
            [1.0 / int(a["reranked_rank"]) - 1.0 / int(b["reranked_rank"]) for b, a in zip(bs, aug)]
            for bs, aug in zip(baseline, augmented)
        ],
        dtype=np.float64,
    )
    singleton = candidate_bins == "1"
    if not np.all(top1[:, singleton] == 0.0) or not np.all(mrr[:, singleton] == 0.0):
        raise RuntimeError("J3 singleton-bin sanity gate failed: non-zero paired effect.")

    j3_rows = []
    for index, label in enumerate(("1", "2-3", "4-6", "7-9", "10")):
        mask = candidate_bins == label
        for metric_index, (metric, values) in enumerate((("top1", top1), ("mrr", mrr))):
            result = product_cluster_bootstrap(
                values[:, mask], clusters[mask], args.bootstrap_samples, args.rng_seed + index * 10 + metric_index
            )
            j3_rows.append(
                {
                    "candidate_count_bin": label,
                    "metric": metric,
                    "n_reactions": int(mask.sum()),
                    "n_product_clusters": int(len(np.unique(clusters[mask]))),
                    **result,
                }
            )

    baseline_ranks = np.asarray(
        [[int(row["reranked_rank"]) for row in rows] for rows in baseline], dtype=np.int32
    )
    augmented_ranks = np.asarray(
        [[int(row["reranked_rank"]) for row in rows] for rows in augmented], dtype=np.int32
    )
    # The approved rule rounds half-rank medians upward to the worse rank.
    assigned_ranks = np.ceil(np.median(baseline_ranks, axis=0)).astype(np.int32)
    rank_bins = np.asarray([_rank_bin(int(rank)) for rank in assigned_ranks])
    mean_change = (augmented_ranks - baseline_ranks).mean(axis=0)
    j4_rows = []
    for label in ("1", "2", "3", "4-5", "6-10"):
        mask = rank_bins == label
        changes = mean_change[mask]
        j4_rows.append(
            {
                "baseline_reference_rank_bin": label,
                "n_reactions": int(mask.sum()),
                "promoted": int(np.sum(changes < 0)),
                "degraded": int(np.sum(changes > 0)),
                "unchanged": int(np.sum(changes == 0)),
                "mean_augmented_minus_baseline_rank_change": float(changes.mean()),
                "harm_rate": float(np.mean(changes > 0)) if label == "1" else None,
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "j3_candidate_count_strata.csv", j3_rows)
    _write_csv(output / "j4_baseline_rank_strata.csv", j4_rows)
    result = {
        "schema_version": 1,
        "record_kind": "dd_round_j3_j4",
        "protocol_id": PROTOCOL_ID,
        "comparator": "frozen 20-seed historical cap-10 paired predictions",
        "single_intended_change": "prespecified diagnostic stratum",
        "bootstrap_samples": args.bootstrap_samples,
        "rng_seed": args.rng_seed,
        "j3_singleton_gate": "pass",
        "j4_bin_assignment": "ceil of median prior+2D reference rank across seeds",
        "input_prediction_root": str(Path(args.anchor_prediction_root).resolve()),
        "outputs": {
            "j3": file_fingerprint(output / "j3_candidate_count_strata.csv"),
            "j4": file_fingerprint(output / "j4_baseline_rank_strata.csv"),
        },
        "round_jk_approval": approval,
        "test_partition_loaded_from_frozen_predictions_only": True,
    }
    atomic_json(output / "manifest.json", result)
    return result


def _regenerated_top10(path: str | Path, products: set[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            product = canonicalize_smiles(str(row["product"]))
            if product not in products or len(result[product]) >= 10:
                continue
            candidate = canonicalize_reactant_set(str(row["reactant"]))
            if candidate is None:
                raise ValueError("K2 regenerated candidate is not canonicalizable.")
            result[product].append(candidate)
    return dict(result)


def k2(args: argparse.Namespace) -> dict:
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite K2 output: {output}")
    reference, baseline, augmented = _load_anchor_matrices(args.anchor_prediction_root)
    products = {canonicalize_smiles(row["product_smiles"]) for row in reference}
    if None in products:
        raise ValueError("K2 found a non-canonicalizable anchor product.")
    regenerated = _regenerated_top10(args.regenerated_cap50_jsonl, products)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(reference):
        product = canonicalize_smiles(row["product_smiles"])
        historical_raw = json.loads(row["baseline_candidates_json"])
        historical = [canonicalize_reactant_set(value) for value in historical_raw]
        if any(value is None for value in historical):
            raise ValueError("K2 historical candidate is not canonicalizable.")
        current = regenerated.get(product, [])
        if historical == current:
            group = "a_identical_set_and_order"
        elif len(historical) == len(current) and set(historical) == set(current):
            group = "b_identical_set_different_order"
        else:
            group = "c_different_set"
        groups[group].append(index)

    group_labels = (
        "a_identical_set_and_order",
        "b_identical_set_different_order",
        "c_different_set",
    )
    if len(groups["a_identical_set_and_order"]) == 0:
        raise ValueError("K2 group (a) is empty.")
    rows = []
    for group_index, label in enumerate(group_labels):
        selected = np.asarray(groups[label], dtype=np.int64)
        clusters = np.asarray(
            [canonicalize_smiles(reference[index]["product_smiles"]) for index in selected],
            dtype=object,
        )
        top1 = np.asarray(
            [
                [
                    float(int(augmented[s][i]["reranked_rank"]) == 1)
                    - float(int(baseline[s][i]["reranked_rank"]) == 1)
                    for i in selected
                ]
                for s in range(len(SEEDS))
            ],
            dtype=np.float64,
        )
        mrr = np.asarray(
            [
                [
                    1.0 / int(augmented[s][i]["reranked_rank"])
                    - 1.0 / int(baseline[s][i]["reranked_rank"])
                    for i in selected
                ]
                for s in range(len(SEEDS))
            ],
            dtype=np.float64,
        )
        for metric_index, (metric, values) in enumerate((("top1", top1), ("mrr", mrr))):
            row = {
                "group": label,
                "metric": metric,
                "n_reactions": len(selected),
                "effect": float(values.mean()),
                "ci95_low": None,
                "ci95_high": None,
                "analysis_role": "supplementary_descriptive",
            }
            if label == "a_identical_set_and_order":
                row.update(
                    product_cluster_bootstrap(
                        values,
                        clusters,
                        args.bootstrap_samples,
                        args.rng_seed + group_index * 2 + metric_index,
                    )
                )
                row["analysis_role"] = "prespecified_inferential"
            rows.append(row)
    group_rows = [
        {"group": label, "n_reactions": len(groups[label])}
        for label in group_labels
    ]
    if sum(row["n_reactions"] for row in group_rows) != 3985:
        raise RuntimeError("K2 groups do not partition the 3,985 anchor reactions.")
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "k2_group_sizes.csv", group_rows)
    _write_csv(output / "k2_group_effects.csv", rows)
    result = {
        "schema_version": 1,
        "record_kind": "dd_round_k2",
        "protocol_id": K2_REPORTING_PROTOCOL_ID,
        "comparator": "historical cap-10 versus clean regenerated cap-50 truncated to ten",
        "single_intended_change": "complete supplementary reporting across the three frozen K2 identity/order groups",
        "regenerated_source": file_fingerprint(args.regenerated_cap50_jsonl),
        "legacy_anchored_cap50_forbidden": True,
        "group_sizes": {row["group"]: row["n_reactions"] for row in group_rows},
        "outputs": {
            "groups": file_fingerprint(output / "k2_group_sizes.csv"),
            "group_effects": file_fingerprint(output / "k2_group_effects.csv"),
        },
        "inferential_scope": "Only group A retains the prespecified product-cluster interval; groups B and C are descriptive point estimates.",
        "round_jk_approval": approval,
        "test_partition_loaded_from_frozen_predictions_only": True,
    }
    atomic_json(output / "manifest.json", result)
    return result


def k1(args: argparse.Namespace) -> dict:
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite K1 analysis: {output}")
    product_by_reaction = {
        reaction.reaction_id: reaction.product_key
        for reaction in load_reactions(args.source_csv, args.metadata_csv)
        if reaction.source_split == "test"
    }
    if len(product_by_reaction) != 5004:
        raise ValueError("K1 expects exactly 5,004 official-test reactions.")
    rows = []
    inputs = {}
    for pool, manifest_value in zip(POOL_ORDER[1:], args.result_manifest, strict=True):
        manifest_path = Path(manifest_value)
        manifest = _load_json(manifest_path)
        if (
            manifest.get("protocol_id") != K1_PROTOCOL_ID
            or manifest.get("pool_name") != pool
            or int(manifest.get("candidate_cap", 0)) != 10
        ):
            raise ValueError(f"K1 result manifest is invalid for {pool}.")
        if not manifest.get("test_partition_loaded_only_after_model_freeze"):
            raise PermissionError("K1 result bypassed the model-freeze test gate.")
        predictions = _read_jsonl(_prediction_path(manifest_path, manifest))
        covered = [row for row in predictions if bool(row["covered"])]
        clusters = np.asarray(
            [product_by_reaction[int(row["reaction_id"])] for row in covered], dtype=object
        )
        for metric_index, metric in enumerate(("top1", "mrr")):
            differences = np.asarray(
                [
                    [
                        _metric(int(row[f"augmented_rank_seed_{seed}"]), metric)
                        - _metric(int(row[f"baseline_rank_seed_{seed}"]), metric)
                        for row in covered
                    ]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            product_only = product_cluster_bootstrap(
                differences,
                clusters,
                args.bootstrap_samples,
                args.rng_seed + metric_index,
            )
            seed_marginal = crossed_product_seed_bootstrap(
                differences,
                clusters,
                args.bootstrap_samples,
                args.rng_seed + 100 + metric_index,
            )
            rows.append(
                {
                    "pool": pool,
                    "metric": metric,
                    "covered": len(covered),
                    "total": len(predictions),
                    "coverage_at_10": len(covered) / len(predictions),
                    "effect": product_only["effect"],
                    "product_cluster_ci95_low": product_only["ci95_low"],
                    "product_cluster_ci95_high": product_only["ci95_high"],
                    "seed_marginal_ci95_low": seed_marginal["ci95_low"],
                    "seed_marginal_ci95_high": seed_marginal["ci95_high"],
                    "n_product_clusters": len(np.unique(clusters)),
                    "n_seeds": len(SEEDS),
                }
            )
        inputs[pool] = file_fingerprint(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "k1_truncated_pool_effects.csv", rows)
    result = {
        "schema_version": 1,
        "record_kind": "dd_round_k1_analysis",
        "protocol_id": K1_PROTOCOL_ID,
        "comparator": "frozen D1 baseline and augmented arms at cap 10",
        "single_intended_change": "expanded-pool candidate composition at fixed list cap",
        "seeds": list(SEEDS),
        "bootstrap_samples": args.bootstrap_samples,
        "rng_seed": args.rng_seed,
        "inputs": inputs,
        "output": file_fingerprint(output / "k1_truncated_pool_effects.csv"),
        "round_jk_approval": approval,
        "test_partition_loaded_from_post_freeze_predictions_only": True,
    }
    atomic_json(output / "manifest.json", result)
    return result


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return ascending average ranks, using exact equality for ties."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Rank input must be one-dimensional.")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    return ranks


def _spearman_with_prior(external_scores: np.ndarray) -> float | None:
    """Within-list Spearman rho against the unchanged descending prior."""
    scores = np.asarray(external_scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) < 2:
        return None
    external_ranks = _average_ranks(scores)
    if np.ptp(external_ranks) == 0.0:
        return None
    prior_scores = np.linspace(1.0, 0.0, num=len(scores), dtype=np.float64)
    prior_ranks = _average_ranks(prior_scores)
    rho = float(np.corrcoef(external_ranks, prior_ranks)[0, 1])
    return None if not math.isfinite(rho) else rho


def _distribution_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
        }
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def _prefixed_summary(prefix: str, values: Iterable[float]) -> dict:
    return {
        f"{prefix}_{key}": value
        for key, value in _distribution_summary(values).items()
    }


def _resolve_d5_file(root: Path, relative: str, record: Mapping) -> Path:
    path = root / relative
    actual = file_fingerprint(path)
    if (
        actual["sha256"] != record.get("sha256")
        or int(actual["size_bytes"]) != int(record.get("size_bytes", -1))
    ):
        raise ValueError(f"Frozen D5 artifact checksum mismatch: {path}")
    return path


def l1(args: argparse.Namespace) -> dict:
    import importlib.metadata
    import platform
    import sys
    import time

    started = time.time()
    approval_record = _load_json(args.approval_record)
    if (
        approval_record.get("l1_protocol_id") != L1_PROTOCOL_ID
        or approval_record.get("l1_status")
        != "approved_forward_only_no_retraining_or_retuning"
        or not approval_record.get("l1_approval_date")
        or not approval_record.get("l1_approval_quote")
    ):
        raise PermissionError("L1 forward-only diagnostic lacks explicit approval.")
    approval = _approval_provenance(args.approval_record, args.analysis_plan)
    approval.update(
        {
            "l1_protocol_id": approval_record["l1_protocol_id"],
            "l1_status": approval_record["l1_status"],
            "l1_approval_date": approval_record["l1_approval_date"],
            "l1_approval_quote": approval_record["l1_approval_quote"],
        }
    )
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite L1 output: {output}")

    d5_root = Path(args.d5_root).resolve()
    test_manifest_path = d5_root / "test_results" / "manifest.json"
    freeze_path = d5_root / "freeze" / "model_selection.json"
    test_manifest = _load_json(test_manifest_path)
    freeze = _load_json(freeze_path)
    if (
        test_manifest.get("protocol_id") != D5_PROTOCOL_ID
        or freeze.get("protocol_id") != D5_PROTOCOL_ID
        or tuple(int(seed) for seed in freeze.get("seeds", ())) != D5_SEEDS
        or not test_manifest.get("test_partition_loaded_only_after_freeze")
        or freeze.get("test_partition_loaded") is not False
    ):
        raise PermissionError("L1 requires the frozen post-selection D5 artifacts.")
    _resolve_d5_file(
        d5_root,
        "freeze/model_selection.json",
        test_manifest["selection_freeze"],
    )
    fingerprint_path = _resolve_d5_file(
        d5_root,
        "test_results/test_fingerprints.npz",
        test_manifest["test_fingerprints"],
    )

    prediction_rows = {}
    for seed in D5_SEEDS:
        path = _resolve_d5_file(
            d5_root,
            f"test_results/predictions_seed_{seed}.csv",
            test_manifest["prediction_files"][str(seed)],
        )
        prediction_rows[seed] = _read_csv(path)
    reference_ids = [int(row["reaction_id"]) for row in prediction_rows[D5_SEEDS[0]]]
    candidate_counts = np.asarray(
        [int(row["candidate_count"]) for row in prediction_rows[D5_SEEDS[0]]],
        dtype=np.int64,
    )
    if len(reference_ids) != 3985 or len(set(reference_ids)) != 3985:
        raise ValueError("L1 expects 3,985 uniquely identified covered D5 reactions.")
    for seed in D5_SEEDS[1:]:
        if [int(row["reaction_id"]) for row in prediction_rows[seed]] != reference_ids:
            raise ValueError("D5 prediction reaction order differs across seeds.")
        if [int(row["candidate_count"]) for row in prediction_rows[seed]] != candidate_counts.tolist():
            raise ValueError("D5 candidate counts differ across seeds.")

    from scipy import sparse
    import torch
    from torch.utils.data import DataLoader

    from rerank.experiments.run_external_ebm import _masked_energies, resolve_device
    from rerank.external_ebm import (
        RXN_EBM_COMMIT,
        UPSTREAM_CORE_FILES,
        SparseQueryDataset,
        import_pinned_rxn_ebm,
        model_args,
        verify_pinned_repository,
    )

    repository = verify_pinned_repository(
        args.rxn_ebm_repo, RXN_EBM_COMMIT, UPSTREAM_CORE_FILES
    )
    matrix = sparse.load_npz(fingerprint_path).tocsr()
    if matrix.shape[0] != len(reference_ids):
        raise ValueError("D5 fingerprint rows do not match frozen predictions.")
    device = resolve_device(args.device)
    loader = DataLoader(
        SparseQueryDataset(matrix),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )
    _, ff_module, _ = import_pinned_rxn_ebm(args.rxn_ebm_repo)
    scores_by_seed = []
    trial_by_seed = {int(trial["seed"]): trial for trial in freeze["trials"]}
    if set(trial_by_seed) != set(D5_SEEDS):
        raise ValueError("D5 freeze does not contain the three published seeds.")
    checkpoint_inputs = {}
    for seed in D5_SEEDS:
        trial = trial_by_seed[seed]
        checkpoint_path = _resolve_d5_file(
            d5_root,
            f"fit/seed_{seed}/best_checkpoint.pt",
            trial["checkpoint"],
        )
        model = ff_module.FeedforwardEBM(model_args()).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        parts = []
        with torch.no_grad():
            for batch, mask in loader:
                batch = batch.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                energies = _masked_energies(model, batch, mask)
                parts.append((-energies).cpu().numpy())
        scores = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        if scores.shape != (len(reference_ids), 10):
            raise ValueError("L1 frozen model produced an unexpected score matrix.")
        for index, count in enumerate(candidate_counts):
            scores[index, int(count) :] = np.nan
        scores_by_seed.append(scores)
        checkpoint_inputs[str(seed)] = file_fingerprint(checkpoint_path)
        del checkpoint, model, scores

    score_cube = np.stack(scores_by_seed, axis=0)
    per_list_rows = []
    for seed_index, seed in enumerate(D5_SEEDS):
        for reaction_index, reaction_id in enumerate(reference_ids):
            count = int(candidate_counts[reaction_index])
            values = score_cube[seed_index, reaction_index, :count].astype(np.float64)
            rho = _spearman_with_prior(values)
            per_list_rows.append(
                {
                    "seed": seed,
                    "reaction_id": reaction_id,
                    "candidate_count": count,
                    "external_score_population_sd": float(np.std(values, ddof=0)),
                    "external_score_range": float(np.ptp(values)),
                    "spearman_external_vs_prior": rho,
                    "spearman_defined": rho is not None,
                }
            )

    summary_rows = []
    for seed in (*D5_SEEDS, "pooled"):
        selected = per_list_rows if seed == "pooled" else [row for row in per_list_rows if row["seed"] == seed]
        correlations = [
            float(row["spearman_external_vs_prior"])
            for row in selected
            if row["spearman_defined"]
        ]
        zero_tolerance = 1e-12
        summary_rows.append(
            {
                "scope": "pooled_seed_reaction_pairs" if seed == "pooled" else "seed",
                "seed": seed,
                "list_count": len(selected),
                "singleton_count": sum(int(row["candidate_count"]) == 1 for row in selected),
                "non_singleton_count": sum(int(row["candidate_count"]) >= 2 for row in selected),
                "spearman_defined_count": len(correlations),
                "spearman_undefined_count": len(selected) - len(correlations),
                "spearman_positive_count": sum(value > zero_tolerance for value in correlations),
                "spearman_zero_count": sum(abs(value) <= zero_tolerance for value in correlations),
                "spearman_negative_count": sum(value < -zero_tolerance for value in correlations),
                **_prefixed_summary(
                    "score_sd",
                    (row["external_score_population_sd"] for row in selected),
                ),
                **_prefixed_summary(
                    "score_range",
                    (row["external_score_range"] for row in selected),
                ),
                **_prefixed_summary("spearman", correlations),
            }
        )

    output.mkdir(parents=True, exist_ok=True)
    raw_score_path = output / "l1_frozen_external_scores.npz"
    with raw_score_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            seeds=np.asarray(D5_SEEDS, dtype=np.int64),
            reaction_ids=np.asarray(reference_ids, dtype=np.int64),
            candidate_counts=candidate_counts,
            external_scores=score_cube,
        )
    _write_csv(output / "l1_per_list_diagnostics.csv", per_list_rows)
    _write_csv(output / "l1_score_summary.csv", summary_rows)
    result = {
        "schema_version": 1,
        "record_kind": "dd_round_l1_frozen_rxn_ebm_score_diagnostic",
        "protocol_id": L1_PROTOCOL_ID,
        "comparator": "frozen cap-10 candidate-prior score order",
        "single_intended_change": "retain and descriptively analyze frozen D5 raw within-list energies",
        "inputs": {
            "analysis_code": file_fingerprint(Path(__file__).resolve()),
            "d5_test_manifest": file_fingerprint(test_manifest_path),
            "d5_model_freeze": file_fingerprint(freeze_path),
            "d5_test_fingerprints": file_fingerprint(fingerprint_path),
            "d5_checkpoints": checkpoint_inputs,
            "rxn_ebm_repository": repository,
        },
        "settings": {
            "seeds": list(D5_SEEDS),
            "device": device,
            "batch_size": args.batch_size,
            "external_higher_is_better_score": "negative frozen model energy",
            "prior_score": "1-(rank-1)/max(candidate_count-1,1)",
            "score_sd": "population SD (ddof=0) within each candidate list",
            "score_range": "maximum minus minimum within each candidate list",
            "spearman": "within-list average ranks for exact ties",
            "undefined_correlation": "singletons or constant external scores; excluded, not imputed",
            "inferential_test": None,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scipy", "torch")
            },
        },
        "runtime_seconds": time.time() - started,
        "counts": {
            "reactions": len(reference_ids),
            "seeds": len(D5_SEEDS),
            "seed_reaction_pairs": len(per_list_rows),
        },
        "outputs": {
            "raw_scores": file_fingerprint(raw_score_path),
            "per_list": file_fingerprint(output / "l1_per_list_diagnostics.csv"),
            "summary": file_fingerprint(output / "l1_score_summary.csv"),
        },
        "round_jk_approval": approval,
        "training_performed": False,
        "retuning_performed": False,
        "fingerprint_generation_performed": False,
        "candidate_generation_performed": False,
        "test_partition_loaded_from_frozen_post-selection_artifacts_only": True,
    }
    atomic_json(output / "manifest.json", result)
    return result


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approval-record", required=True)
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--output-dir", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    j12 = sub.add_parser("j1-j2")
    _common(j12)
    j12.add_argument("--anchor-baseline-csv", required=True)
    j12.add_argument("--anchor-prediction-root", required=True)
    j12.add_argument("--expanded-manifest", action="append", required=True)
    j34 = sub.add_parser("j3-j4")
    _common(j34)
    j34.add_argument("--anchor-prediction-root", required=True)
    j34.add_argument("--bootstrap-samples", type=int, default=10_000)
    j34.add_argument("--rng-seed", type=int, default=2026)
    k2_parser = sub.add_parser("k2")
    _common(k2_parser)
    k2_parser.add_argument("--anchor-prediction-root", required=True)
    k2_parser.add_argument("--regenerated-cap50-jsonl", required=True)
    k2_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    k2_parser.add_argument("--rng-seed", type=int, default=2026)
    k1_parser = sub.add_parser("k1")
    _common(k1_parser)
    k1_parser.add_argument("--result-manifest", action="append", required=True)
    k1_parser.add_argument("--source-csv", default="data/uspto_smiles.csv")
    k1_parser.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    k1_parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    k1_parser.add_argument("--rng-seed", type=int, default=2026)
    l1_parser = sub.add_parser("l1")
    _common(l1_parser)
    l1_parser.add_argument("--d5-root", required=True)
    l1_parser.add_argument("--rxn-ebm-repo", required=True)
    l1_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    l1_parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.command == "j1-j2" and len(args.expanded_manifest) != 3:
        parser.error("j1-j2 needs three expanded manifests in AiZ, LocalRetro, merged order")
    if args.command == "k1" and len(args.result_manifest) != 3:
        parser.error("k1 needs three result manifests in AiZ, LocalRetro, merged order")
    return args


def main() -> None:
    args = parse_args()
    commands = {"j1-j2": j1_j2, "j3-j4": j3_j4, "k1": k1, "k2": k2, "l1": l1}
    print(json.dumps(commands[args.command](args), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
