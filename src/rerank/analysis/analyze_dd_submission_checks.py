"""Frozen-artifact checks requested for the Digital Discovery final round.

This module does not train a model, generate candidates, or select a
configuration.  Each command writes to a new output directory and records the
input fingerprints used for the calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import platform
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rerank.study_data import canonicalize_smiles, file_fingerprint
from rerank.ws_e_streaming import atomic_json
from rerank.analysis.analyze_predictions import (
    clustered_sign_flip,
    crossed_product_seed_bootstrap,
    product_cluster_bootstrap,
)


P1_PROTOCOL_ID = "dd-p1-corrected-headroom-v1"
N5_PROTOCOL_ID = "dd-n5-crossed-transfer-inference-v1"
N1_PROTOCOL_ID = "dd-n1-matched-list-joint-sensitivity-v2"
SEEDS = tuple(range(42, 62))
POOL_ORDER = (
    "historical_cap10",
    "aizynthfinder_only",
    "localretro_only",
    "merged",
)


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _seed_metric(manifest: Mapping, arm: str, metric: str) -> float:
    per_seed = manifest["per_seed_metrics"][arm]
    expected = {str(seed) for seed in SEEDS}
    if set(per_seed) != expected:
        raise ValueError(f"Expected seeds 42--61 for {arm}; found {sorted(per_seed)}")
    return float(np.mean([float(per_seed[str(seed)][metric]) for seed in SEEDS]))


def _expanded_seed_metric(manifest: Mapping, arm: str, metric: str) -> float:
    per_seed = manifest["per_seed_metrics"][arm]
    expected = {str(seed) for seed in SEEDS}
    if set(per_seed) != expected:
        raise ValueError(f"Expected seeds 42--61 for {arm}; found {sorted(per_seed)}")
    return float(
        np.mean(
            [float(per_seed[str(seed)]["within_pool"][metric]) for seed in SEEDS]
        )
    )


def corrected_headroom_rows(
    historical: Mapping,
    expanded: Mapping[str, Mapping],
) -> list[dict]:
    """Return P1 rows using the matched prior+2D model as denominator."""

    if set(expanded) != set(POOL_ORDER[1:]):
        raise ValueError("P1 requires exactly the three final expanded pools.")

    historical_baseline = _seed_metric(historical, "baseline", "top1")
    historical_augmented = _seed_metric(historical, "augmented", "top1")
    rows = [
        {
            "pool": "historical_cap10",
            "coverage_count": 3985,
            "coverage_total": 5004,
            "coverage": 3985 / 5004,
            "prior_2d_top1_within_pool": historical_baseline,
            "augmented_top1_within_pool": historical_augmented,
        }
    ]

    for pool in POOL_ORDER[1:]:
        manifest = expanded[pool]
        if manifest.get("pool_name") != pool:
            raise ValueError(f"Expanded manifest pool mismatch for {pool}.")
        covered = int(manifest["test_reactions_covered"])
        total = int(manifest["test_reactions_total"])
        rows.append(
            {
                "pool": pool,
                "coverage_count": covered,
                "coverage_total": total,
                "coverage": covered / total,
                "prior_2d_top1_within_pool": _expanded_seed_metric(
                    manifest, "baseline", "top1"
                ),
                "augmented_top1_within_pool": _expanded_seed_metric(
                    manifest, "augmented", "top1"
                ),
            }
        )

    for row in rows:
        baseline = float(row["prior_2d_top1_within_pool"])
        effect = float(row["augmented_top1_within_pool"]) - baseline
        headroom = 1.0 - baseline
        row["observed_augmented_minus_prior_2d_top1"] = effect
        row["headroom_after_prior_2d"] = headroom
        row["corrected_capture_rate"] = None if headroom == 0.0 else effect / headroom
        row["denominator_definition"] = "1 - prior+2D within-pool Top-1"
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_p1(args: argparse.Namespace) -> dict:
    started = time.time()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite P1 output: {output}")

    historical_path = Path(args.historical_manifest)
    expanded_paths = [Path(path) for path in args.expanded_manifest]
    if len(expanded_paths) != 3:
        raise ValueError("P1 requires three expanded manifests in AiZ/LocalRetro/merged order.")
    historical = _load_json(historical_path)
    expanded = {
        pool: _load_json(path)
        for pool, path in zip(POOL_ORDER[1:], expanded_paths, strict=True)
    }
    rows = corrected_headroom_rows(historical, expanded)

    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "p1_corrected_headroom.csv"
    _write_csv(table_path, rows)
    result = {
        "schema_version": 1,
        "record_kind": "dd_final_submission_numeric_correction",
        "protocol_id": P1_PROTOCOL_ID,
        "comparator": "frozen 20-seed prior+2D and augmented own-pool metrics",
        "single_intended_change": (
            "use headroom after the matched prior+2D baseline rather than headroom after the raw prior"
        ),
        "inputs": {
            "historical_manifest": file_fingerprint(historical_path),
            "expanded_manifests": [file_fingerprint(path) for path in expanded_paths],
        },
        "outputs": {"corrected_headroom": file_fingerprint(table_path)},
        "settings": {
            "seeds": list(SEEDS),
            "denominator": "1 - prior+2D within-pool Top-1",
        },
        "training_performed": False,
        "retuning_performed": False,
        "candidate_generation_performed": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output / "manifest.json", result)
    return result


def _rank_outcome(rank: int, metric: str) -> float:
    if metric == "top1":
        return float(rank == 1)
    if metric == "mrr":
        return 1.0 / rank
    raise ValueError(f"Unsupported N5 metric: {metric}")


def _anchor_effect_matrices(
    prediction_root: Path,
    reaction_ids: Sequence[int],
) -> dict[str, np.ndarray]:
    position = {reaction_id: index for index, reaction_id in enumerate(reaction_ids)}
    matrices = {
        "top1": np.empty((len(SEEDS), len(reaction_ids)), dtype=np.float64),
        "mrr": np.empty((len(SEEDS), len(reaction_ids)), dtype=np.float64),
    }
    for seed_index, seed in enumerate(SEEDS):
        baseline = {
            int(row["reaction_id"]): row
            for row in _read_csv(prediction_root / f"baseline_seed_{seed}.csv")
        }
        augmented = {
            int(row["reaction_id"]): row
            for row in _read_csv(prediction_root / f"augmented_seed_{seed}.csv")
        }
        if not set(position).issubset(baseline) or not set(position).issubset(augmented):
            raise ValueError(f"Historical seed {seed} misses an N5 common reaction.")
        for reaction_id, column in position.items():
            baseline_rank = int(baseline[reaction_id]["reranked_rank"])
            augmented_rank = int(augmented[reaction_id]["reranked_rank"])
            for metric in matrices:
                matrices[metric][seed_index, column] = _rank_outcome(
                    augmented_rank, metric
                ) - _rank_outcome(baseline_rank, metric)
    return matrices


def _prediction_file(manifest_path: Path, manifest: Mapping) -> Path:
    recorded = Path(str(manifest["predictions"]["path"]))
    candidate = recorded if recorded.is_file() else manifest_path.parent / recorded.name
    actual = file_fingerprint(candidate)
    if actual["sha256"] != manifest["predictions"]["sha256"]:
        raise ValueError(f"Frozen K1 prediction checksum mismatch: {candidate}")
    return candidate


def _expanded_effect_matrices(
    manifest_path: Path,
    pool: str,
    reaction_ids: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict]:
    manifest = _load_json(manifest_path)
    if (
        manifest.get("pool_name") != pool
        or int(manifest.get("candidate_cap", 0)) != 10
        or manifest.get("source_pool_protocol_id")
        != "ws-e-localretro-three-pools-filtered-v2"
    ):
        raise ValueError(f"N5 received the wrong frozen K1 manifest for {pool}.")
    rows = {
        int(row["reaction_id"]): row
        for row in _read_jsonl(_prediction_file(manifest_path, manifest))
    }
    if not set(reaction_ids).issubset(rows):
        raise ValueError(f"{pool} misses an N5 common reaction.")
    matrices = {
        "top1": np.empty((len(SEEDS), len(reaction_ids)), dtype=np.float64),
        "mrr": np.empty((len(SEEDS), len(reaction_ids)), dtype=np.float64),
    }
    for seed_index, seed in enumerate(SEEDS):
        for column, reaction_id in enumerate(reaction_ids):
            row = rows[reaction_id]
            baseline_rank = int(row[f"baseline_rank_seed_{seed}"])
            augmented_rank = int(row[f"augmented_rank_seed_{seed}"])
            for metric in matrices:
                matrices[metric][seed_index, column] = _rank_outcome(
                    augmented_rank, metric
                ) - _rank_outcome(baseline_rank, metric)
    return matrices, manifest


def matched_list_cohorts(
    descriptor_rows: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[int]], dict[int, str]]:
    """Freeze the N1 1,745/2,069 matched-list partition from M2 descriptors."""

    by_reaction: dict[int, dict[str, Mapping[str, str]]] = {}
    products: dict[int, str] = {}
    for row in descriptor_rows:
        pool = row["pool"]
        if pool not in {"historical_cap10", "aizynthfinder_only"}:
            continue
        reaction_id = int(row["reaction_id"])
        product = row["canonical_product"]
        if reaction_id in products and products[reaction_id] != product:
            raise ValueError("N1 canonical product differs between descriptor rows.")
        products[reaction_id] = product
        by_reaction.setdefault(reaction_id, {})[pool] = row

    if len(by_reaction) != 3814 or any(
        set(rows) != {"historical_cap10", "aizynthfinder_only"}
        for rows in by_reaction.values()
    ):
        raise ValueError("N1 requires paired historical/AiZ descriptors for 3,814 reactions.")

    cohorts = {"matched_candidate_identity_and_order": [], "changed_candidate_list": []}
    for reaction_id in sorted(by_reaction):
        historical = by_reaction[reaction_id]["historical_cap10"]
        aizynth = by_reaction[reaction_id]["aizynthfinder_only"]
        historical_count = int(historical["candidate_count"])
        aizynth_count = int(aizynth["candidate_count"])
        kendall = aizynth["kendall_order_vs_historical"]
        matched = (
            historical_count == aizynth_count
            and int(aizynth["shared_candidate_count"]) == historical_count
            and np.isclose(float(aizynth["jaccard_vs_historical"]), 1.0)
            and (
                historical_count == 1
                or (kendall != "" and np.isclose(float(kendall), 1.0))
            )
        )
        key = (
            "matched_candidate_identity_and_order"
            if matched
            else "changed_candidate_list"
        )
        cohorts[key].append(reaction_id)

    counts = {key: len(value) for key, value in cohorts.items()}
    if counts != {
        "matched_candidate_identity_and_order": 1745,
        "changed_candidate_list": 2069,
    }:
        raise ValueError(f"N1 cohort gate failed: {counts}")
    return cohorts, products


def _n1_prior_audit(
    historical_feature_cache: Path,
    aizynth_pool_jsonl: Path,
    matched_reaction_ids: Sequence[int],
    products: Mapping[int, str],
) -> dict:
    """Reproduce the zero exact-prior-list gate on the matched-list cohort."""

    with historical_feature_cache.open("rb") as handle:
        payload = pickle.load(handle)["payload"]
    historical_priors: dict[int, list[float]] = {}
    for metadata, product_and_candidates in zip(
        payload["eval_metadata"], payload["eval_pwc"], strict=True
    ):
        reaction_id = int(metadata["reaction_id"])
        if reaction_id in products:
            _, candidates = product_and_candidates
            historical_priors[reaction_id] = [float(item["prior"]) for item in candidates]

    product_to_reaction = {
        canonicalize_smiles(products[item]): item for item in matched_reaction_ids
    }
    if len(product_to_reaction) != len(matched_reaction_ids):
        raise ValueError("N1 matched cohort requires unique canonical products.")
    aizynth_priors = {item: [] for item in matched_reaction_ids}
    canonical_product_cache: dict[str, str] = {}
    with aizynth_pool_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["candidate_rank"]) > 10:
                continue
            raw_product = row["product"]
            if raw_product not in canonical_product_cache:
                canonical_product_cache[raw_product] = canonicalize_smiles(raw_product)
            reaction_id = product_to_reaction.get(
                canonical_product_cache[raw_product]
            )
            if reaction_id is not None:
                aizynth_priors[reaction_id].append(float(row["prior"]))

    exact = 0
    for reaction_id in matched_reaction_ids:
        historical = historical_priors.get(reaction_id)
        aizynth = aizynth_priors.get(reaction_id)
        if historical is None or len(historical) != len(aizynth):
            raise ValueError(f"N1 prior audit cannot align reaction {reaction_id}.")
        exact += int(np.array_equal(np.asarray(historical), np.asarray(aizynth)))
    if exact != 0:
        raise ValueError(f"N1 expected zero exact stored-prior matches; found {exact}.")
    return {
        "audited_reactions": len(matched_reaction_ids),
        "exact_stored_prior_list_matches": exact,
        "interpretation": "stored prior calibration differs for every matched list",
    }


def n1_sensitivity_rows(
    historical: Mapping[str, np.ndarray],
    aizynth: Mapping[str, np.ndarray],
    reaction_ids: Sequence[int],
    products: Mapping[int, str],
    cohorts: Mapping[str, Sequence[int]],
    bootstrap_samples: int,
    sign_flip_samples: int,
    rng_seed: int,
) -> tuple[list[dict], list[dict]]:
    """Estimate frozen within-pipeline effects and their joint-pipeline contrast."""

    position = {reaction_id: index for index, reaction_id in enumerate(reaction_ids)}
    effect_rows: list[dict] = []
    contrast_rows: list[dict] = []
    for cohort_index, (cohort, cohort_ids) in enumerate(cohorts.items()):
        columns = np.asarray([position[item] for item in cohort_ids], dtype=np.int64)
        clusters = np.asarray([products[item] for item in cohort_ids], dtype=object)
        for metric_index, metric in enumerate(("top1", "mrr")):
            matrices = {
                "historical_cap10": historical[metric][:, columns],
                "aizynthfinder_only": aizynth[metric][:, columns],
            }
            for pipeline_index, (pipeline, matrix) in enumerate(matrices.items()):
                seed = rng_seed + cohort_index * 100 + metric_index * 10 + pipeline_index
                product_ci = product_cluster_bootstrap(
                    matrix, clusters, bootstrap_samples, seed
                )
                crossed_ci = crossed_product_seed_bootstrap(
                    matrix, clusters, bootstrap_samples, seed + 1000
                )
                effect_rows.append(
                    {
                        "cohort": cohort,
                        "pipeline": pipeline,
                        "metric": metric,
                        "n_reactions": len(cohort_ids),
                        "n_seeds": len(SEEDS),
                        "mean_augmented_minus_baseline": product_ci["effect"],
                        "product_cluster_ci95_low": product_ci["ci95_low"],
                        "product_cluster_ci95_high": product_ci["ci95_high"],
                        "crossed_product_seed_ci95_low": crossed_ci["ci95_low"],
                        "crossed_product_seed_ci95_high": crossed_ci["ci95_high"],
                    }
                )

            contrast = matrices["aizynthfinder_only"] - matrices["historical_cap10"]
            contrast_seed = rng_seed + 5000 + cohort_index * 100 + metric_index * 10
            product_ci = product_cluster_bootstrap(
                contrast, clusters, bootstrap_samples, contrast_seed
            )
            crossed_ci = crossed_product_seed_bootstrap(
                contrast, clusters, bootstrap_samples, contrast_seed + 1000
            )
            p_value = clustered_sign_flip(
                contrast, clusters, sign_flip_samples, contrast_seed + 2000
            )
            contrast_rows.append(
                {
                    "cohort": cohort,
                    "metric": metric,
                    "n_reactions": len(cohort_ids),
                    "n_seeds": len(SEEDS),
                    "aizynth_minus_historical_increment": product_ci["effect"],
                    "product_cluster_ci95_low": product_ci["ci95_low"],
                    "product_cluster_ci95_high": product_ci["ci95_high"],
                    "crossed_product_seed_ci95_low": crossed_ci["ci95_low"],
                    "crossed_product_seed_ci95_high": crossed_ci["ci95_high"],
                    "product_cluster_sign_flip_p": p_value,
                    "interpretation_scope": (
                        "joint frozen-pipeline sensitivity conditional on matching "
                        "candidate identities and order; other feature inputs "
                        "not verified identical"
                        if cohort == "matched_candidate_identity_and_order"
                        else "joint frozen-pipeline sensitivity including candidate-list changes"
                    ),
                }
            )
    return effect_rows, contrast_rows


def _assert_n1_numeric_equivalence(
    new_rows: Sequence[Mapping], reference_path: Path
) -> None:
    reference_rows = _read_csv(reference_path)
    if len(new_rows) != len(reference_rows):
        raise ValueError(f"N1 v2 row count differs from v1: {reference_path}")
    for index, (new, previous) in enumerate(
        zip(new_rows, reference_rows, strict=True)
    ):
        if set(new) != set(previous):
            raise ValueError(f"N1 v2 columns differ from v1 at row {index}.")
        for key, old_value in previous.items():
            if key != "interpretation_scope" and str(new[key]) != old_value:
                raise ValueError(
                    f"N1 v2 numeric or cohort value differs at row {index}, {key}."
                )


def crossed_transfer_intervals(
    transfer_matrices: Mapping[str, Mapping[str, np.ndarray]],
    draws: int,
    rng_seed: int,
    batch_size: int = 100,
) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Joint-product/independent-pool-seed nonparametric bootstrap."""

    pools = tuple(transfer_matrices)
    if not pools:
        raise ValueError("N5 transfer matrices are empty.")
    shapes = {
        matrix.shape
        for pool in pools
        for matrix in transfer_matrices[pool].values()
    }
    if len(shapes) != 1:
        raise ValueError(f"N5 transfer matrix shapes differ: {shapes}")
    n_seeds, n_products = next(iter(shapes))
    if n_seeds != len(SEEDS) or n_products <= 0:
        raise ValueError("N5 requires 20 seeds and a non-empty common product set.")

    rng = np.random.default_rng(rng_seed)
    boot = {
        (pool, metric): np.empty(draws, dtype=np.float64)
        for pool in pools
        for metric in ("top1", "mrr")
    }
    scale_products = 1.0 / n_products
    scale_seeds = 1.0 / n_seeds
    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        count = stop - start
        sampled_products = rng.integers(
            0, n_products, size=(count, n_products), dtype=np.int32
        )
        product_weights = np.zeros((count, n_products), dtype=np.float32)
        np.add.at(
            product_weights,
            (
                np.repeat(np.arange(count, dtype=np.int32), n_products),
                sampled_products.ravel(),
            ),
            scale_products,
        )
        for pool in pools:
            sampled_seeds = rng.integers(
                0, n_seeds, size=(count, n_seeds), dtype=np.int16
            )
            seed_weights = np.zeros((count, n_seeds), dtype=np.float32)
            np.add.at(
                seed_weights,
                (
                    np.repeat(np.arange(count, dtype=np.int32), n_seeds),
                    sampled_seeds.ravel(),
                ),
                scale_seeds,
            )
            for metric in ("top1", "mrr"):
                sampled_product_means = (
                    product_weights @ transfer_matrices[pool][metric].T
                )
                boot[(pool, metric)][start:stop] = np.sum(
                    sampled_product_means * seed_weights, axis=1
                )

    result = {}
    for key, values in boot.items():
        matrix = transfer_matrices[key[0]][key[1]]
        result[key] = (
            float(matrix.mean()),
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        )
    return result


def _approval_record(path: Path, plan_path: Path, protocol_id: str) -> dict:
    approval = _load_json(path)
    if (
        approval.get("status") != "approved"
        or protocol_id not in approval.get("approved_protocol_ids", [])
        or not approval.get("approval_quote")
    ):
        raise PermissionError(
            f"{protocol_id} final-round approval is absent or incomplete."
        )
    plan = file_fingerprint(plan_path)
    if plan["sha256"] != approval.get("analysis_plan_sha256"):
        raise PermissionError(
            f"{protocol_id} approval is bound to a different analysis plan."
        )
    return {**approval, "analysis_plan": plan}


def run_n1(args: argparse.Namespace) -> dict:
    started = time.time()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite N1 output: {output}")
    plan_path = Path(args.analysis_plan)
    approval = _approval_record(
        Path(args.approval_record), plan_path, N1_PROTOCOL_ID
    )

    descriptor_path = Path(args.m2_descriptor_csv)
    cohorts, products = matched_list_cohorts(_read_csv(descriptor_path))
    reaction_ids = sorted(products)
    historical = _anchor_effect_matrices(
        Path(args.anchor_prediction_root), reaction_ids
    )
    k1_manifest_path = Path(args.k1_aiz_manifest)
    aizynth, k1_manifest = _expanded_effect_matrices(
        k1_manifest_path, "aizynthfinder_only", reaction_ids
    )
    prior_audit = _n1_prior_audit(
        Path(args.historical_feature_cache),
        Path(args.aizynth_pool_jsonl),
        cohorts["matched_candidate_identity_and_order"],
        products,
    )
    effect_rows, contrast_rows = n1_sensitivity_rows(
        historical,
        aizynth,
        reaction_ids,
        products,
        cohorts,
        bootstrap_samples=args.bootstrap_samples,
        sign_flip_samples=args.sign_flip_samples,
        rng_seed=args.rng_seed,
    )
    reference_effect_path = Path(args.reference_effect_csv)
    reference_contrast_path = Path(args.reference_contrast_csv)
    _assert_n1_numeric_equivalence(effect_rows, reference_effect_path)
    _assert_n1_numeric_equivalence(contrast_rows, reference_contrast_path)

    output.mkdir(parents=True, exist_ok=True)
    cohort_path = output / "n1_cohorts.csv"
    cohort_rows = [
        {
            "reaction_id": reaction_id,
            "canonical_product": products[reaction_id],
            "cohort": cohort,
            "stored_prior_status": "different_calibration",
        }
        for cohort, identifiers in cohorts.items()
        for reaction_id in identifiers
    ]
    effect_path = output / "n1_within_pipeline_effects.csv"
    contrast_path = output / "n1_joint_pipeline_contrasts.csv"
    _write_csv(cohort_path, cohort_rows)
    _write_csv(effect_path, effect_rows)
    _write_csv(contrast_path, contrast_rows)

    prediction_file = _prediction_file(k1_manifest_path, k1_manifest)
    anchor_root = Path(args.anchor_prediction_root)
    anchor_files = [
        anchor_root / f"{arm}_seed_{seed}.csv"
        for seed in SEEDS
        for arm in ("baseline", "augmented")
    ]
    result = {
        "schema_version": 1,
        "record_kind": "dd_secondary_matched_list_joint_pipeline_sensitivity",
        "protocol_id": N1_PROTOCOL_ID,
        "comparator": (
            "frozen historical-cap10 and final-K1 AiZynthFinder 20-seed pipelines"
        ),
        "single_intended_change": (
            "stratify frozen pipeline contrasts by canonical candidate-list identity/order match"
        ),
        "cohort_gate": {
            "common_covered_reactions": len(reaction_ids),
            "matched_candidate_identity_and_order": len(
                cohorts["matched_candidate_identity_and_order"]
            ),
            "changed_candidate_list": len(cohorts["changed_candidate_list"]),
            "criteria": (
                "equal count; Jaccard=1; all candidates shared; Kendall=1, "
                "with singleton order accepted by definition"
            ),
        },
        "prior_audit": prior_audit,
        "settings": {
            "bootstrap_samples": args.bootstrap_samples,
            "sign_flip_samples": args.sign_flip_samples,
            "rng_seed": args.rng_seed,
            "seeds": list(SEEDS),
            "primary_holm_family_changed": False,
        },
        "inputs": {
            "analysis_plan": file_fingerprint(plan_path),
            "approval_record": file_fingerprint(args.approval_record),
            "m2_descriptor_csv": file_fingerprint(descriptor_path),
            "historical_feature_cache": file_fingerprint(
                args.historical_feature_cache
            ),
            "aizynth_pool_jsonl": file_fingerprint(args.aizynth_pool_jsonl),
            "anchor_prediction_files": [
                file_fingerprint(path) for path in anchor_files
            ],
            "k1_aiz_manifest": file_fingerprint(k1_manifest_path),
            "k1_aiz_predictions": file_fingerprint(prediction_file),
            "v1_effects_for_numeric_equivalence": file_fingerprint(
                reference_effect_path
            ),
            "v1_contrasts_for_numeric_equivalence": file_fingerprint(
                reference_contrast_path
            ),
        },
        "outputs": {
            "cohorts": file_fingerprint(cohort_path),
            "within_pipeline_effects": file_fingerprint(effect_path),
            "joint_pipeline_contrasts": file_fingerprint(contrast_path),
        },
        "approval": approval,
        "interpretation_scope": (
            "joint frozen-pipeline sensitivity; no training, normalizer, prior, "
            "candidate-list or feature-input component is isolated"
        ),
        "baseline_top1_candidate_agreement": {
            "status": "unavailable_from_frozen_rank_summary",
            "reason": (
                "final-K1 reaction summaries retain reference ranks but not "
                "top-predicted candidate identities"
            ),
            "new_inference_performed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "training_performed": False,
        "inference_performed": False,
        "retuning_performed": False,
        "candidate_generation_performed": False,
        "test_partition_loaded_from_frozen_predictions_only": True,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output / "manifest.json", result)
    return result


def run_n5(args: argparse.Namespace) -> dict:
    started = time.time()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite N5 output: {output}")
    plan_path = Path(args.analysis_plan)
    approval = _approval_record(
        Path(args.approval_record), plan_path, N5_PROTOCOL_ID
    )

    m2_path = Path(args.m2_transfer_csv)
    m2_rows = _read_csv(m2_path)
    identities: dict[int, str] = {}
    pool_counts = {pool: 0 for pool in POOL_ORDER[1:]}
    for row in m2_rows:
        pool = row["pool"]
        if pool not in pool_counts:
            raise ValueError(f"Unexpected N5 M2 pool: {pool}")
        reaction_id = int(row["reaction_id"])
        product = row["canonical_product"]
        if reaction_id in identities and identities[reaction_id] != product:
            raise ValueError("N5 canonical product differs across pools.")
        identities[reaction_id] = product
        pool_counts[pool] += 1
    if set(pool_counts.values()) != {3814} or len(identities) != 3814:
        raise ValueError(f"N5 requires 3,814 aligned reactions per pool: {pool_counts}")
    if len(set(identities.values())) != 3814:
        raise ValueError("N5 requires one unique canonical product per reaction.")
    reaction_ids = sorted(identities)

    historical = _anchor_effect_matrices(Path(args.anchor_prediction_root), reaction_ids)
    manifest_paths = [Path(path) for path in args.k1_manifest]
    if len(manifest_paths) != 3:
        raise ValueError("N5 requires AiZ/LocalRetro/merged K1 manifests in order.")
    transfer_matrices = {}
    manifests = {}
    for pool, manifest_path in zip(POOL_ORDER[1:], manifest_paths, strict=True):
        expanded, manifest = _expanded_effect_matrices(
            manifest_path, pool, reaction_ids
        )
        transfer_matrices[pool] = {
            metric: expanded[metric] - historical[metric]
            for metric in ("top1", "mrr")
        }
        manifests[pool] = manifest

    intervals = crossed_transfer_intervals(
        transfer_matrices,
        draws=args.bootstrap_samples,
        rng_seed=args.rng_seed,
        batch_size=args.batch_size,
    )
    rows = []
    for pool in POOL_ORDER[1:]:
        for metric in ("top1", "mrr"):
            effect, low, high = intervals[(pool, metric)]
            rows.append(
                {
                    "pool": pool,
                    "metric": metric,
                    "n_reactions": 3814,
                    "n_seeds": len(SEEDS),
                    "mean_transfer_loss": effect,
                    "crossed_product_seed_ci95_low": low,
                    "crossed_product_seed_ci95_high": high,
                    "interval_relation_to_zero": (
                        "negative_ci_excludes_zero"
                        if high < 0
                        else "positive_ci_excludes_zero"
                        if low > 0
                        else "ci_includes_zero"
                    ),
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    table_path = output / "n5_crossed_transfer_intervals.csv"
    _write_csv(table_path, rows)
    result = {
        "schema_version": 1,
        "record_kind": "dd_secondary_crossed_transfer_inference",
        "protocol_id": N5_PROTOCOL_ID,
        "comparator": "six frozen M2b expanded-minus-historical transfer contrasts",
        "single_intended_change": "add seed-marginal uncertainty to frozen transfer losses",
        "settings": {
            "bootstrap_samples": args.bootstrap_samples,
            "rng_seed": args.rng_seed,
            "seeds": list(SEEDS),
            "product_resampling": "joint across pools",
            "seed_resampling": "paired historical/expanded seed index; independent by pool",
            "primary_holm_family_changed": False,
        },
        "inputs": {
            "analysis_plan": file_fingerprint(plan_path),
            "approval_record": file_fingerprint(args.approval_record),
            "m2_transfer_reactions": file_fingerprint(m2_path),
            "anchor_prediction_root": str(Path(args.anchor_prediction_root).resolve()),
            "k1_manifests": [file_fingerprint(path) for path in manifest_paths],
        },
        "outputs": {"crossed_intervals": file_fingerprint(table_path)},
        "approval": approval,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "training_performed": False,
        "retuning_performed": False,
        "candidate_generation_performed": False,
        "test_partition_loaded_from_frozen_predictions_only": True,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(output / "manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    p1 = subparsers.add_parser("p1-headroom")
    p1.add_argument("--historical-manifest", required=True)
    p1.add_argument("--expanded-manifest", action="append", required=True)
    p1.add_argument("--output-dir", required=True)
    n1 = subparsers.add_parser("n1-matched-list")
    n1.add_argument("--approval-record", required=True)
    n1.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    n1.add_argument("--m2-descriptor-csv", required=True)
    n1.add_argument("--anchor-prediction-root", required=True)
    n1.add_argument("--k1-aiz-manifest", required=True)
    n1.add_argument("--historical-feature-cache", required=True)
    n1.add_argument("--aizynth-pool-jsonl", required=True)
    n1.add_argument("--reference-effect-csv", required=True)
    n1.add_argument("--reference-contrast-csv", required=True)
    n1.add_argument("--bootstrap-samples", type=int, default=10_000)
    n1.add_argument("--sign-flip-samples", type=int, default=100_000)
    n1.add_argument("--rng-seed", type=int, default=2080)
    n1.add_argument("--output-dir", required=True)
    n5 = subparsers.add_parser("n5-crossed-transfer")
    n5.add_argument("--approval-record", required=True)
    n5.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    n5.add_argument("--m2-transfer-csv", required=True)
    n5.add_argument("--anchor-prediction-root", required=True)
    n5.add_argument("--k1-manifest", action="append", required=True)
    n5.add_argument("--bootstrap-samples", type=int, default=10_000)
    n5.add_argument("--rng-seed", type=int, default=2070)
    n5.add_argument("--batch-size", type=int, default=100)
    n5.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = {
        "p1-headroom": run_p1,
        "n1-matched-list": run_n1,
        "n5-crossed-transfer": run_n5,
    }[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
