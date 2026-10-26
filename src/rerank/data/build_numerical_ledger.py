"""Build the immutable, paper-facing numerical ledger for revision results.

This module does not recompute scientific statistics.  It reads frozen B--H
artifacts, validates their protocol and leakage gates, and writes compact tables
whose rows retain an exact source locator.  Existing output directories are
never overwritten.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LEDGER_PROTOCOL_ID = "jcheminform-revision-numerical-freeze-v2"
APPROVED_B1_STRATA = ("all", "rigid_zero", "intermediate_1_4", "flexible_ge5")

SOURCE_PATHS = {
    "legacy_systems": "outputs/revision_analysis/table2_main_results.csv",
    "legacy_controls": "outputs/revision_analysis/table5_feature_controls.csv",
    "b_manifest": "outputs/jcheminform_revision/conformer_aggregate/manifest.json",
    "b1_pairwise": "outputs/jcheminform_revision/conformer_aggregate/b1_pairwise_stability.csv",
    "b1_cv": "outputs/jcheminform_revision/conformer_aggregate/b1_cv_summary.csv",
    "b2": "outputs/jcheminform_revision/conformer_aggregate/b2_analysis.json",
    "b3": "outputs/jcheminform_revision/conformer_aggregate/b3_ranking/analysis/clustered_inference.json",
    "d1": "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/test_results/manifest.json",
    "d3": "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/capacity_test_results/manifest.json",
    "d4": "outputs/jcheminform_revision/tuned_primary/conformer_seed_42/lightgbm/test_results/manifest.json",
    "d5": "outputs/jcheminform_revision/external_rerankers/rxn_ebm_ff_cap10/analysis/paired_inference.json",
    "encoder_manifest": "outputs/jcheminform_revision/encoder_attribution/manifest.json",
    "encoder_summary": "outputs/jcheminform_revision/encoder_attribution/encoder_attribution_summary.csv",
    "morgan": "outputs/jcheminform_revision/encoder_controls/morgan_atom/ranking/test_results/manifest.json",
    "grover": "outputs/jcheminform_revision/encoder_controls/grover_concatenation/ranking/test_results/manifest.json",
    "c2": "outputs/jcheminform_revision/projected_probe/c2_primary/test_results/manifest.json",
    "wse_manifest": "outputs/jcheminform_revision/ws_e_analysis_canonical_identity/manifest.json",
    "wse_stats": "outputs/jcheminform_revision/ws_e_analysis_canonical_identity/ws_e_paired_statistics.json",
    "f1_manifest": "outputs/jcheminform_revision/f1_roundtrip/results/manifest.json",
    "f1_stats": "outputs/jcheminform_revision/f1_roundtrip/results/paired_statistics.json",
    "f1_systems": "outputs/jcheminform_revision/f1_roundtrip/results/system_metrics.csv",
    "f2_manifest": "outputs/jcheminform_revision/chirality_sensitivity/analysis/manifest.json",
    "f2_stats": "outputs/jcheminform_revision/chirality_sensitivity/analysis/paired_statistics.json",
    "f3_manifest": "outputs/jcheminform_revision/salt_sensitivity/analysis/manifest.json",
    "f3_stats": "outputs/jcheminform_revision/salt_sensitivity/analysis/paired_statistics.json",
    "g_manifest": "outputs/jcheminform_revision/final_statistics_20seed/manifest.json",
    "g1": "outputs/jcheminform_revision/final_statistics_20seed/g1_seed_marginal.json",
    "g2": "outputs/jcheminform_revision/final_statistics_20seed/g2_class_tests.csv",
    "g3": "outputs/jcheminform_revision/final_statistics_20seed/g3_net_flips.csv",
    "h2": "outputs/jcheminform_revision/final_statistics_20seed/h2_logistic_models.csv",
}


SYSTEM_FIELDS = (
    "workstream",
    "analysis_id",
    "protocol_id",
    "claim_role",
    "scope",
    "system",
    "metric",
    "estimate",
    "sample_sd",
    "n_seeds",
    "n_reactions",
    "coverage",
    "source_file",
    "source_locator",
)

EFFECT_FIELDS = (
    "workstream",
    "analysis_id",
    "protocol_id",
    "claim_role",
    "comparison",
    "scope",
    "metric",
    "effect",
    "ci95_low",
    "ci95_high",
    "p_value",
    "q_value",
    "n_seeds",
    "n_reactions",
    "n_product_clusters",
    "positive_seed_count",
    "negative_seed_count",
    "zero_seed_count",
    "decision",
    "source_file",
    "source_locator",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def validate_file_record(path: Path, record: Mapping[str, Any]) -> None:
    if path.stat().st_size != int(record["size_bytes"]):
        raise RuntimeError(f"Size mismatch for frozen source: {path}")
    if sha256_file(path) != str(record["sha256"]).removeprefix("sha256:"):
        raise RuntimeError(f"SHA-256 mismatch for frozen source: {path}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def assert_close(left: float, right: float, label: str, tolerance: float = 1e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance):
        raise RuntimeError(f"Cross-artifact mismatch for {label}: {left} != {right}")


def _decision(ci_low: Any, ci_high: Any) -> str:
    if ci_low in (None, "") or ci_high in (None, ""):
        return "point_estimate_only"
    low, high = float(ci_low), float(ci_high)
    if low > 0:
        return "positive_ci_excludes_zero"
    if high < 0:
        return "negative_ci_excludes_zero"
    return "null_compatible"


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return sum(materialized) / len(materialized)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _source_locator(key: str, suffix: str) -> tuple[str, str]:
    return SOURCE_PATHS[key], suffix


def _system_row(
    *,
    workstream: str,
    analysis_id: str,
    protocol_id: str,
    claim_role: str,
    scope: str,
    system: str,
    metric: str,
    estimate: Any,
    source_key: str,
    source_locator: str,
    sample_sd: Any = "",
    n_seeds: Any = "",
    n_reactions: Any = "",
    coverage: Any = "",
) -> dict[str, Any]:
    source_file, locator = _source_locator(source_key, source_locator)
    return {
        "workstream": workstream,
        "analysis_id": analysis_id,
        "protocol_id": protocol_id,
        "claim_role": claim_role,
        "scope": scope,
        "system": system,
        "metric": metric,
        "estimate": estimate,
        "sample_sd": sample_sd,
        "n_seeds": n_seeds,
        "n_reactions": n_reactions,
        "coverage": coverage,
        "source_file": source_file,
        "source_locator": locator,
    }


def _effect_row(
    *,
    workstream: str,
    analysis_id: str,
    protocol_id: str,
    claim_role: str,
    comparison: str,
    scope: str,
    metric: str,
    effect: Any,
    source_key: str,
    source_locator: str,
    ci95_low: Any = "",
    ci95_high: Any = "",
    p_value: Any = "",
    q_value: Any = "",
    n_seeds: Any = "",
    n_reactions: Any = "",
    n_product_clusters: Any = "",
    positive_seed_count: Any = "",
    negative_seed_count: Any = "",
    zero_seed_count: Any = "",
    decision: str | None = None,
) -> dict[str, Any]:
    source_file, locator = _source_locator(source_key, source_locator)
    return {
        "workstream": workstream,
        "analysis_id": analysis_id,
        "protocol_id": protocol_id,
        "claim_role": claim_role,
        "comparison": comparison,
        "scope": scope,
        "metric": metric,
        "effect": effect,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "p_value": p_value,
        "q_value": q_value,
        "n_seeds": n_seeds,
        "n_reactions": n_reactions,
        "n_product_clusters": n_product_clusters,
        "positive_seed_count": positive_seed_count,
        "negative_seed_count": negative_seed_count,
        "zero_seed_count": zero_seed_count,
        "decision": decision or _decision(ci95_low, ci95_high),
        "source_file": source_file,
        "source_locator": locator,
    }


def _validate_gates(data: Mapping[str, Any], paths: Mapping[str, Path]) -> list[str]:
    checks: list[str] = []
    b_manifest = data["b_manifest"]
    require(b_manifest["analyses"]["B2"]["protocol_id"] == "legacy-cap10-fixed50-v1", "B2 protocol mismatch")
    require(data["b2"]["top1_robustness_gate"]["passed"] is True, "B2 robustness gate did not pass")
    checks.append("B2 crossed 5x5 robustness gate passed")

    require(data["b3"]["protocol_id"] == "cap10-tuned-b3-avg10-v1", "B3 protocol mismatch")
    validate_file_record(paths["d1"], data["b3"]["primary_manifest"])
    checks.append("B3 primary-manifest fingerprint matches D1")

    require(data["d1"]["protocol_id"] == "cap10-tuned-v1", "D1 protocol mismatch")
    require(data["d1"]["test_partition_loaded_only_after_selection_freeze"] is True, "D1 test lock missing")
    require(data["d3"]["control_id"] == "D-CAPACITY", "D3 control mismatch")
    require(data["d3"]["protocol_id"] == "cap10-tuned-v1", "D3 source protocol mismatch")
    require(
        data["d3"]["capacity_assertion"]["baseline"]["parameters"]
        == data["d3"]["capacity_assertion"]["augmented"]["parameters"]
        == 289,
        "D3 parameter-count gate failed",
    )
    require(data["d3"]["test_partition_loaded_only_after_capacity_freeze"] is True, "D3 test lock missing")
    require(data["d4"]["protocol_id"] == "cap10-lightgbm-v1", "D4 protocol mismatch")
    require(data["d4"]["test_partition_loaded_only_after_freeze"] is True, "D4 test lock missing")
    require(data["d5"]["protocol_id"] == "D-EXTERNAL-RXN-EBM-FF-CAP10-v1", "D5 protocol mismatch")
    require(data["d5"]["test_partition_used_for_training_or_selection"] is False, "D5 test leakage gate failed")
    validate_file_record(paths["d1"], data["d5"]["inputs"]["primary_manifest"])
    checks.append("D1--D5 post-freeze official-test gates passed")

    encoder = data["encoder_manifest"]
    require(encoder["protocol_id"] == "cap10-tuned-encoder-attribution-v1", "Encoder protocol mismatch")
    require(encoder["pairing_gate"]["status"] == "exact", "Encoder pairing gate failed")
    validate_file_record(paths["encoder_summary"], encoder["summary_csv"])
    require(data["c2"]["test_partition_loaded_only_after_c2_freeze"] is True, "C2 test lock missing")
    checks.append("C1 exact pairing and C2 post-freeze test gates passed")

    require(data["wse_manifest"]["protocol_id"] == "ws-e-three-pool-frozen-ranker-v1", "WS-E protocol mismatch")
    require(data["wse_stats"]["protocol_id"] == data["wse_manifest"]["protocol_id"], "WS-E manifest/statistics mismatch")
    require(data["f1_manifest"]["protocol_id"] == "f1-chemformer-forward-roundtrip-v1", "F1 protocol mismatch")
    require(data["f1_manifest"]["test_partition_used_for_training_or_selection"] is False, "F1 test leakage gate failed")
    require(data["f2_manifest"]["protocol_id"] == "f2-morgan-chirality-v1", "F2 protocol mismatch")
    require(data["f3_manifest"]["protocol_id"] == "f3-salt-removal-v1", "F3 protocol mismatch")
    checks.append("WS-E and F1--F3 protocol gates passed")

    g = data["g_manifest"]
    require(g["protocol_id"] == "cap10-final-statistics-v1", "G/H protocol mismatch")
    require(g["paired_seeds"] == list(range(42, 62)), "G1 does not contain seeds 42--61")
    for name, expected in g["generated_sha256"].items():
        generated = paths["g_manifest"].parent / name
        require(generated.is_file(), f"Missing G/H output: {generated}")
        require(sha256_file(generated) == expected, f"G/H checksum mismatch: {name}")
    require(g["h2_all_class"]["bootstrap_success"] == 10000, "H2 all-class bootstrap incomplete")
    require(g["h2_class3"]["bootstrap_success"] == 10000, "H2 class-3 bootstrap incomplete")
    checks.append("G1 seeds 42--61 and all G/H output fingerprints verified")
    return checks


def _build_b1_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in data["b1_pairwise"]:
        if row["stratum"] in APPROVED_B1_STRATA:
            grouped[(row["scalar"], row["stratum"])].append(row)
    cv_index = {
        (row["scalar"], row["stratum"]): row
        for row in data["b1_cv"]
        if row["stratum"] in APPROVED_B1_STRATA
    }
    rows: list[dict[str, Any]] = []
    for (scalar, stratum), values in sorted(grouped.items()):
        require(len(values) == 10, f"B1 requires ten conformer pairs for {scalar}/{stratum}")
        cv = cv_index[(scalar, stratum)]
        rows.append(
            {
                "scalar": scalar,
                "stratum": stratum,
                "n_conformer_pairs": 10,
                "n_unique_molecule_pairs": int(values[0]["abs_diff_n"]),
                "mean_pairwise_abs_diff": _mean(float(row["abs_diff_mean"]) for row in values),
                "mean_pearson": _mean(float(row["pearson"]) for row in values),
                "min_pearson": min(float(row["pearson"]) for row in values),
                "max_pearson": max(float(row["pearson"]) for row in values),
                "mean_spearman": _mean(float(row["spearman"]) for row in values),
                "min_spearman": min(float(row["spearman"]) for row in values),
                "max_spearman": max(float(row["spearman"]) for row in values),
                "cv_median": cv["cv_median"],
                "cv_q95": cv["cv_q95"],
                "cv_undefined_abs_mean_le_1e8": cv["n_cv_undefined_abs_mean_le_1e-8"],
                "source_pairwise": SOURCE_PATHS["b1_pairwise"],
                "source_cv": SOURCE_PATHS["b1_cv"],
            }
        )
    require(len(rows) == 12, "B1 paper-facing table must contain 3 scalars x 4 prespecified strata")
    return rows


def _build_system_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in data["legacy_systems"]:
        if raw["metric"] not in {"top1", "mrr"}:
            continue
        rows.append(
            _system_row(
                workstream="legacy",
                analysis_id="legacy-cap10-fixed50",
                protocol_id="legacy-cap10-fixed50-v1",
                claim_role="legacy_sensitivity",
                scope=raw["scope"],
                system=raw["system"],
                metric=raw["metric"],
                estimate=raw["mean"],
                sample_sd=raw["std"],
                n_seeds=raw["n_seeds"],
                n_reactions=raw["n_reactions"],
                source_key="legacy_systems",
                source_locator=f"row:scope={raw['scope']};system={raw['system']};metric={raw['metric']}",
            )
        )

    d1 = data["d1"]
    for arm, system in (("baseline", "prior_2d"), ("augmented", "prior_2d_unimol")):
        for metric in ("top1", "mrr"):
            item = d1["descriptive_summary"][arm][metric]
            rows.append(
                _system_row(
                    workstream="D",
                    analysis_id="D1",
                    protocol_id=d1["protocol_id"],
                    claim_role="revised_primary_5seed_descriptive",
                    scope="within_pool",
                    system=system,
                    metric=metric,
                    estimate=item["mean"],
                    sample_sd=item["sample_std"],
                    n_seeds=5,
                    n_reactions=3985,
                    coverage=3985 / 5004,
                    source_key="d1",
                    source_locator=f"$.descriptive_summary.{arm}.{metric}",
                )
            )

    for source_key, system in (("morgan", "prior_2d_morgan"), ("grover", "prior_2d_grover")):
        per_seed = data[source_key]["per_seed_metrics"]
        for metric in ("top1", "mrr"):
            rows.append(
                _system_row(
                    workstream="C",
                    analysis_id="C1",
                    protocol_id="cap10-tuned-encoder-attribution-v1",
                    claim_role="encoder_attribution",
                    scope="within_pool",
                    system=system,
                    metric=metric,
                    estimate=_mean(row[metric] for row in per_seed.values()),
                    n_seeds=5,
                    n_reactions=3985,
                    coverage=3985 / 5004,
                    source_key=source_key,
                    source_locator=f"$.per_seed_metrics.*.{metric} (mean)",
                )
            )

    c2 = data["c2"]
    for metric in ("top1", "mrr"):
        item = c2["descriptive_summary"][metric]
        rows.append(
            _system_row(
                workstream="C",
                analysis_id="C2",
                protocol_id=c2["protocol_id"],
                claim_role="upper_bound_probe",
                scope="within_pool",
                system="prior_2d_projected_unimol",
                metric=metric,
                estimate=item["mean"],
                sample_sd=item["sample_std"],
                n_seeds=5,
                n_reactions=3985,
                coverage=3985 / 5004,
                source_key="c2",
                source_locator=f"$.descriptive_summary.{metric}",
            )
        )

    for source_key, analysis_id, protocol_id, label in (
        ("d3", "D3", data["d3"]["protocol_id"], "capacity289"),
        ("d4", "D4", data["d4"]["protocol_id"], "lambdamart"),
    ):
        summary = data[source_key].get("descriptive_summary", data[source_key].get("per_arm_metrics"))
        for arm in ("baseline", "augmented"):
            for metric in ("top1", "mrr"):
                item = summary[arm][metric]
                estimate = item["mean"] if isinstance(item, dict) else item
                sample_sd = item.get("sample_std", "") if isinstance(item, dict) else ""
                rows.append(
                    _system_row(
                        workstream="D",
                        analysis_id=analysis_id,
                        protocol_id=protocol_id,
                        claim_role="learner_or_capacity_sensitivity",
                        scope="within_pool",
                        system=f"{label}_{arm}",
                        metric=metric,
                        estimate=estimate,
                        sample_sd=sample_sd,
                        n_seeds=5 if analysis_id == "D3" else "",
                        n_reactions=3985,
                        coverage=3985 / 5004,
                        source_key=source_key,
                        source_locator=f"$.{'descriptive_summary' if analysis_id == 'D3' else 'per_arm_metrics'}.{arm}.{metric}",
                    )
                )

    d5 = data["d5"]
    for metric in ("top1", "mrr"):
        item = d5["aggregate"][metric]
        for scope, prior_key, external_key in (
            ("within_pool", "candidate_prior_conditional", "external_conditional_mean"),
            ("end_to_end", "candidate_prior_end_to_end", "external_end_to_end_mean"),
        ):
            for system, key in (("candidate_prior", prior_key), ("rxn_ebm_ff", external_key)):
                rows.append(
                    _system_row(
                        workstream="D",
                        analysis_id="D5",
                        protocol_id=d5["protocol_id"],
                        claim_role="external_reranker_negative_result",
                        scope=scope,
                        system=system,
                        metric=metric,
                        estimate=item[key],
                        n_seeds=0 if system == "candidate_prior" else d5["n_seeds"],
                        n_reactions=d5["n_covered_reactions"] if scope == "within_pool" else d5["n_official_test_reactions"],
                        coverage=d5["coverage"],
                        source_key="d5",
                        source_locator=f"$.aggregate.{metric}.{key}",
                    )
                )

    for pool, pool_data in data["wse_stats"]["pools"].items():
        for scope, metrics in pool_data["scopes"].items():
            for metric, item in metrics.items():
                for arm in ("baseline", "augmented"):
                    rows.append(
                        _system_row(
                            workstream="E",
                            analysis_id="E-POOLS",
                            protocol_id=data["wse_stats"]["protocol_id"],
                            claim_role="candidate_pool_robustness",
                            scope=scope,
                            system=f"{pool}_{arm}",
                            metric=metric,
                            estimate=item[f"{arm}_mean"],
                            n_seeds=5,
                            n_reactions=item["n_reactions"],
                            coverage=pool_data["coverage"],
                            source_key="wse_stats",
                            source_locator=f"$.pools.{pool}.scopes.{scope}.{metric}.{arm}_mean",
                        )
                    )

    for raw in data["f1_systems"]:
        for metric, mean_key, sd_key in (
            ("exact_match_top1", "exact_match_top1_mean", "exact_match_top1_sd"),
            ("roundtrip_beam1", "roundtrip_beam1_mean", "roundtrip_beam1_sd"),
            ("roundtrip_beam5", "roundtrip_beam5_mean", "roundtrip_beam5_sd"),
        ):
            rows.append(
                _system_row(
                    workstream="F",
                    analysis_id="F1",
                    protocol_id=data["f1_manifest"]["protocol_id"],
                    claim_role="forward_roundtrip_sensitivity",
                    scope="within_pool",
                    system=raw["system"],
                    metric=metric,
                    estimate=raw[mean_key],
                    sample_sd=raw[sd_key],
                    n_seeds=raw["n_seeds"],
                    n_reactions=raw["n_reactions"],
                    coverage=3985 / 5004,
                    source_key="f1_systems",
                    source_locator=f"row:system={raw['system']};metric={metric}",
                )
            )
    return rows


def _build_effect_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in data["legacy_controls"]:
        if raw["variant"] == "prior_2d":
            continue
        for metric in ("top1", "mrr"):
            rows.append(
                _effect_row(
                    workstream="legacy",
                    analysis_id="legacy-feature-controls",
                    protocol_id="legacy-cap10-fixed50-v1",
                    claim_role="legacy_sensitivity",
                    comparison=f"{raw['variant']}_minus_prior_2d",
                    scope="within_pool",
                    metric=metric,
                    effect=raw[f"{metric}_delta_vs_2d"],
                    n_seeds=5,
                    n_reactions=3985,
                    decision="legacy_descriptive_point_estimate",
                    source_key="legacy_controls",
                    source_locator=f"row:variant={raw['variant']};column={metric}_delta_vs_2d",
                )
            )

    b2 = data["b2"]
    for metric in ("top1", "mrr"):
        item = b2[metric]["crossed_bootstrap"]
        rows.append(
            _effect_row(
                workstream="B",
                analysis_id="B2",
                protocol_id="legacy-cap10-fixed50-v1",
                claim_role="conformer_robustness",
                comparison="unimol_augmented_minus_2d",
                scope="within_pool_crossed_5x5",
                metric=metric,
                effect=item["point_estimate"],
                ci95_low=item["ci95"][0],
                ci95_high=item["ci95"][1],
                n_seeds=5,
                n_reactions=item["n_reactions"],
                n_product_clusters=item["n_product_clusters"],
                positive_seed_count=b2[metric]["positive_cells"],
                negative_seed_count=b2[metric]["negative_cells"],
                zero_seed_count=b2[metric]["zero_cells"],
                decision="robust" if metric == "top1" else _decision(*item["ci95"]),
                source_key="b2",
                source_locator=f"$.{metric}.crossed_bootstrap",
            )
        )

    for comparison, metrics in data["b3"]["comparisons"].items():
        for metric, item in metrics.items():
            rows.append(
                _effect_row(
                    workstream="B",
                    analysis_id="B3",
                    protocol_id=data["b3"]["protocol_id"],
                    claim_role="conformer_averaging_sensitivity",
                    comparison=comparison,
                    scope="within_pool",
                    metric=metric,
                    effect=item["effect"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_seeds=item["n_seeds"],
                    n_reactions=item["n_reactions"],
                    n_product_clusters=item["n_product_clusters"],
                    source_key="b3",
                    source_locator=f"$.comparisons.{comparison}.{metric}",
                )
            )

    encoder = data["encoder_manifest"]
    for raw in data["encoder_summary"]:
        pair_decision = encoder["pair_decisions"][raw["comparison"]]
        if pair_decision["superior_both_endpoints_95"]:
            decision = "pair_superior_on_both_endpoints_95"
        elif pair_decision["inferior_both_endpoints_95"]:
            decision = "pair_inferior_on_both_endpoints_95"
        elif pair_decision["equivalent_both_endpoints"]:
            decision = "pair_equivalent_on_both_endpoints"
        else:
            decision = "mixed_or_inconclusive_encoder_attribution"
        rows.append(
            _effect_row(
                workstream="C",
                analysis_id="C1",
                protocol_id=encoder["protocol_id"],
                claim_role="encoder_attribution",
                comparison=raw["comparison"],
                scope="within_pool",
                metric=raw["metric"],
                effect=raw["effect"],
                ci95_low=raw["ci95_low"],
                ci95_high=raw["ci95_high"],
                n_seeds=raw["n_seeds"],
                n_reactions=raw["n_reactions"],
                n_product_clusters=raw["n_product_clusters"],
                decision=decision,
                source_key="encoder_summary",
                source_locator=f"row:comparison={raw['comparison']};metric={raw['metric']}",
            )
        )

    c2, d1 = data["c2"], data["d1"]
    for metric in ("top1", "mrr"):
        projected = float(c2["descriptive_summary"][metric]["mean"])
        for comparison, arm in (("projected_minus_2d", "baseline"), ("projected_minus_unimol_scalar", "augmented")):
            rows.append(
                _effect_row(
                    workstream="C",
                    analysis_id="C2",
                    protocol_id=c2["protocol_id"],
                    claim_role="upper_bound_probe",
                    comparison=comparison,
                    scope="within_pool",
                    metric=metric,
                    effect=projected - float(d1["descriptive_summary"][arm][metric]["mean"]),
                    n_seeds=5,
                    n_reactions=3985,
                    decision="upper_bound_descriptive_point_estimate",
                    source_key="c2",
                    source_locator=f"derived:$.descriptive_summary.{metric}.mean minus {SOURCE_PATHS['d1']} $.descriptive_summary.{arm}.{metric}.mean",
                )
            )

    for source_key, analysis_id, protocol_id, summary_key in (
        ("d3", "D3", data["d3"]["protocol_id"], "descriptive_summary"),
        ("d4", "D4", data["d4"]["protocol_id"], "per_arm_metrics"),
    ):
        summary = data[source_key][summary_key]
        for metric in ("top1", "mrr"):
            baseline = summary["baseline"][metric]
            augmented = summary["augmented"][metric]
            baseline_value = baseline["mean"] if isinstance(baseline, dict) else baseline
            augmented_value = augmented["mean"] if isinstance(augmented, dict) else augmented
            rows.append(
                _effect_row(
                    workstream="D",
                    analysis_id=analysis_id,
                    protocol_id=protocol_id,
                    claim_role="learner_or_capacity_sensitivity",
                    comparison="augmented_minus_baseline",
                    scope="within_pool",
                    metric=metric,
                    effect=float(augmented_value) - float(baseline_value),
                    n_seeds=5 if analysis_id == "D3" else "",
                    n_reactions=3985,
                    decision="descriptive_point_estimate_no_clustered_ci",
                    source_key=source_key,
                    source_locator=f"derived:$.{summary_key}.augmented.{metric} minus $.{summary_key}.baseline.{metric}",
                )
            )

    d5 = data["d5"]
    for metric, item in d5["aggregate"].items():
        rows.append(
            _effect_row(
                workstream="D",
                analysis_id="D5",
                protocol_id=d5["protocol_id"],
                claim_role="external_reranker_negative_result",
                comparison="rxn_ebm_ff_minus_candidate_prior",
                scope="within_pool",
                metric=metric,
                effect=item["delta_external_minus_prior"],
                ci95_low=item["ci95"][0],
                ci95_high=item["ci95"][1],
                p_value=item["paired_cluster_sign_flip_p"],
                n_seeds=d5["n_seeds"],
                n_reactions=d5["n_covered_reactions"],
                n_product_clusters=d5["n_product_clusters"],
                positive_seed_count=item["positive_seed_count"],
                negative_seed_count=item["negative_seed_count"],
                zero_seed_count=item["zero_seed_count"],
                source_key="d5",
                source_locator=f"$.aggregate.{metric}",
            )
        )
        rows.append(
            _effect_row(
                workstream="D",
                analysis_id="D5",
                protocol_id=d5["protocol_id"],
                claim_role="external_reranker_negative_result",
                comparison="rxn_ebm_ff_minus_candidate_prior",
                scope="end_to_end",
                metric=metric,
                effect=item["end_to_end_delta_external_minus_prior"],
                n_seeds=d5["n_seeds"],
                n_reactions=d5["n_official_test_reactions"],
                decision="descriptive_point_estimate_no_separate_ci",
                source_key="d5",
                source_locator=f"$.aggregate.{metric}.end_to_end_delta_external_minus_prior",
            )
        )

    for pool, pool_data in data["wse_stats"]["pools"].items():
        for scope, metrics in pool_data["scopes"].items():
            for metric, item in metrics.items():
                rows.append(
                    _effect_row(
                        workstream="E",
                        analysis_id="E-POOLS",
                        protocol_id=data["wse_stats"]["protocol_id"],
                        claim_role="candidate_pool_robustness",
                        comparison=f"{pool}_augmented_minus_baseline",
                        scope=scope,
                        metric=metric,
                        effect=item["delta"],
                        ci95_low=item["ci95"][0],
                        ci95_high=item["ci95"][1],
                        p_value=item["paired_cluster_sign_flip_p"],
                        n_seeds=5,
                        n_reactions=item["n_reactions"],
                        n_product_clusters=item["n_product_clusters"],
                        decision=pool_data["top1_assessment"] if metric == "top1" else _decision(*item["ci95"]),
                        source_key="wse_stats",
                        source_locator=f"$.pools.{pool}.scopes.{scope}.{metric}",
                    )
                )

    f1 = data["f1_stats"]
    for metric, item in f1.items():
        if metric == "settings":
            continue
        rows.append(
            _effect_row(
                workstream="F",
                analysis_id="F1",
                protocol_id=data["f1_manifest"]["protocol_id"],
                claim_role="forward_roundtrip_sensitivity",
                comparison="prior_2d_unimol_minus_prior_2d",
                scope="within_pool",
                metric=metric,
                effect=item["augmented_minus_2d"],
                ci95_low=item["ci95"][0],
                ci95_high=item["ci95"][1],
                p_value=item["paired_cluster_sign_flip_p"],
                n_seeds=20,
                n_reactions=3985,
                n_product_clusters=3985,
                positive_seed_count=item["positive_seed_count"],
                negative_seed_count=item["negative_seed_count"],
                zero_seed_count=item["zero_seed_count"],
                source_key="f1_stats",
                source_locator=f"$.{metric}",
            )
        )

    for source_key, analysis_id, protocol_id, suffix in (
        ("f2_stats", "F2", data["f2_manifest"]["protocol_id"], "true_minus_false"),
        ("f3_stats", "F3", data["f3_manifest"]["protocol_id"], "salt_minus_current"),
    ):
        for arm, arm_data in data[source_key].items():
            for metric in ("top1", "mrr"):
                rows.append(
                    _effect_row(
                        workstream="F",
                        analysis_id=analysis_id,
                        protocol_id=protocol_id,
                        claim_role="chemistry_side_sensitivity",
                        comparison=f"{arm}_{suffix}",
                        scope="within_pool",
                        metric=metric,
                        effect=arm_data[f"{metric}_delta_{suffix}"],
                        ci95_low=arm_data[f"{metric}_ci95"][0],
                        ci95_high=arm_data[f"{metric}_ci95"][1],
                        p_value=arm_data[f"{metric}_paired_cluster_sign_flip_p"],
                        n_seeds=arm_data["n_seeds"],
                        n_reactions=arm_data["n_reactions"],
                        n_product_clusters=arm_data["n_product_clusters"],
                        source_key=source_key,
                        source_locator=f"$.{arm}.{metric}",
                    )
                )

    for metric, methods in data["g1"].items():
        for inference_scope, item in methods.items():
            rows.append(
                _effect_row(
                    workstream="G",
                    analysis_id="G1",
                    protocol_id=data["g_manifest"]["protocol_id"],
                    claim_role="revised_primary_headline" if inference_scope == "seed_marginal" else "companion_product_cluster_ci",
                    comparison="prior_2d_unimol_minus_prior_2d",
                    scope=inference_scope,
                    metric=metric,
                    effect=item["effect"],
                    ci95_low=item["ci95_low"],
                    ci95_high=item["ci95_high"],
                    n_seeds=20,
                    n_reactions=3985,
                    n_product_clusters=3985,
                    source_key="g1",
                    source_locator=f"$.{metric}.{inference_scope}",
                )
            )

    for raw in data["g2"]:
        class_id = raw["reaction_class"]
        for metric in ("top1", "mrr"):
            is_top1 = metric == "top1"
            significant = raw["top1_bh_significant_q05"].lower() == "true" if is_top1 else False
            rows.append(
                _effect_row(
                    workstream="G",
                    analysis_id="G2",
                    protocol_id=data["g_manifest"]["protocol_id"],
                    claim_role="prespecified_class_top1_family" if is_top1 else "secondary_class_mrr_no_p_family",
                    comparison=f"class_{class_id}_augmented_minus_2d",
                    scope="within_pool_reaction_class",
                    metric=metric,
                    effect=raw[f"{metric}_effect"],
                    ci95_low=raw[f"{metric}_ci95_low"],
                    ci95_high=raw[f"{metric}_ci95_high"],
                    p_value=raw["top1_raw_p"] if is_top1 else "",
                    q_value=raw["top1_bh_q"] if is_top1 else "",
                    n_seeds=20,
                    n_reactions=raw["n_reactions"],
                    n_product_clusters=raw["n_reactions"],
                    decision=("bh_significant_q05" if significant else "not_bh_significant_q05") if is_top1 else "descriptive_secondary_endpoint",
                    source_key="g2",
                    source_locator=f"row:reaction_class={class_id};metric={metric}",
                )
            )
    return rows


def _build_count_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, value in data["g_manifest"]["net_flip_totals"].items():
        rows.append(
            {
                "workstream": "G",
                "analysis_id": "G3",
                "protocol_id": data["g_manifest"]["protocol_id"],
                "scope": "20_seeds_x_3985_reactions",
                "count_name": name,
                "count": value,
                "source_file": SOURCE_PATHS["g_manifest"],
                "source_locator": f"$.net_flip_totals.{name}",
            }
        )
    for name, value in data["d5"]["rank_shift_totals_across_seed_reaction_pairs"].items():
        rows.append(
            {
                "workstream": "D",
                "analysis_id": "D5",
                "protocol_id": data["d5"]["protocol_id"],
                "scope": "3_seeds_x_3985_reactions",
                "count_name": name,
                "count": value,
                "source_file": SOURCE_PATHS["d5"],
                "source_locator": f"$.rank_shift_totals_across_seed_reaction_pairs.{name}",
            }
        )
    return rows


def _build_h2_rows(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = data["g_manifest"]
    rows: list[dict[str, Any]] = []
    for raw in data["h2"]:
        model = raw["model"]
        meta = manifest["h2_all_class" if model == "all_class" else "h2_class3"]
        rows.append(
            {
                **raw,
                "n_observations": meta["n_observations"],
                "n_product_clusters": meta["n_product_clusters"],
                "bootstrap_success": meta["bootstrap_success"],
                "bootstrap_failed_or_separated": meta["bootstrap_failed_or_separated"],
                "interpretation": meta["interpretation"],
                "source_file": SOURCE_PATHS["h2"],
                "source_locator": f"row:model={model};term={raw['term']}",
            }
        )
    return rows


def _cross_checks(data: Mapping[str, Any], system_rows: Sequence[Mapping[str, Any]], effect_rows: Sequence[Mapping[str, Any]]) -> list[str]:
    checks: list[str] = []
    encoder_unimol = next(
        row for row in effect_rows
        if row["analysis_id"] == "C1" and row["comparison"] == "unimol_minus_2d" and row["metric"] == "top1"
    )
    assert_close(encoder_unimol["effect"], data["d1"]["descriptive_summary"]["paired_delta"]["top1"]["mean"], "C1/D1 Top-1")
    checks.append("C1 Uni-Mol-minus-2D Top-1 equals the D1 five-seed delta")

    assert_close(data["f1_stats"]["exact_match_top1"]["augmented_minus_2d"], data["g1"]["top1"]["product_only"]["effect"], "F1/G1 Top-1")
    checks.append("F1 exact-match Top-1 equals the G1 20-seed point estimate")

    significant = [row["reaction_class"] for row in data["g2"] if row["top1_bh_significant_q05"].lower() == "true"]
    require(significant == ["3"], f"Unexpected G2 significant classes: {significant}")
    checks.append("G2 BH q=0.05 significance is restricted to reaction class 3")

    for analysis_id in ("F2", "F3"):
        relevant = [row for row in effect_rows if row["analysis_id"] == analysis_id]
        require(all(float(row["ci95_low"]) <= 0 <= float(row["ci95_high"]) for row in relevant), f"{analysis_id} sensitivity CI unexpectedly excludes zero")
    checks.append("All F2/F3 chemistry-side sensitivity intervals include zero")

    require(all(pool["top1_assessment"] == "no_clear_difference" for pool in data["wse_stats"]["pools"].values()), "WS-E assessment mismatch")
    checks.append("All three WS-E pool assessments are no-clear-difference")

    d5_rows = [row for row in effect_rows if row["analysis_id"] == "D5" and row["scope"] == "within_pool"]
    require(all(float(row["ci95_high"]) < 0 for row in d5_rows), "D5 negative-result interval mismatch")
    checks.append("Both D5 conditional intervals are strictly negative")

    headline = [row for row in effect_rows if row["claim_role"] == "revised_primary_headline"]
    require(len(headline) == 2 and {row["metric"] for row in headline} == {"top1", "mrr"}, "Headline must be exactly G1 Top-1 and MRR")
    checks.append("Exactly two paper headline effects are designated: G1 Top-1 and MRR")
    return checks


def build_numerical_ledger(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    output = output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite numerical freeze directory: {output}")

    paths = {key: root / relative for key, relative in SOURCE_PATHS.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen numerical sources:\n" + "\n".join(missing))

    json_keys = {
        "b_manifest", "b2", "b3", "d1", "d3", "d4", "d5", "encoder_manifest",
        "morgan", "grover", "c2", "wse_manifest", "wse_stats", "f1_manifest",
        "f1_stats", "f2_manifest", "f2_stats", "f3_manifest", "f3_stats", "g_manifest", "g1",
    }
    data: dict[str, Any] = {
        key: load_json(path) if key in json_keys else load_csv(path)
        for key, path in paths.items()
    }
    gate_checks = _validate_gates(data, paths)
    b1_rows = _build_b1_rows(data)
    system_rows = _build_system_rows(data)
    effect_rows = _build_effect_rows(data)
    count_rows = _build_count_rows(data)
    h2_rows = _build_h2_rows(data)
    consistency_checks = _cross_checks(data, system_rows, effect_rows)

    output.mkdir(parents=True, exist_ok=False)
    b1_path = output / "b1_conformer_stability_summary.csv"
    systems_path = output / "system_performance.csv"
    effects_path = output / "paired_effects.csv"
    counts_path = output / "promotion_degradation_counts.csv"
    h2_path = output / "h2_posthoc_models.csv"

    _write_csv(b1_path, tuple(b1_rows[0]), b1_rows)
    _write_csv(systems_path, SYSTEM_FIELDS, system_rows)
    _write_csv(effects_path, EFFECT_FIELDS, effect_rows)
    _write_csv(counts_path, tuple(count_rows[0]), count_rows)
    _write_csv(h2_path, tuple(h2_rows[0]), h2_rows)

    outputs = {path.name: file_record(path, root) for path in (b1_path, systems_path, effects_path, counts_path, h2_path)}
    manifest = {
        "schema_version": 1,
        "protocol_id": LEDGER_PROTOCOL_ID,
        "record_kind": "paper_facing_numerical_freeze",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_intended_change": "consolidate already-frozen B--H results without recomputation",
        "legacy_policy": "legacy-cap10-fixed50-v1 is retained only as explicitly labeled sensitivity evidence",
        "headline_policy": "G1 seed-marginal Top-1 and MRR are the revised-primary headline effects",
        "source_policy": "every numerical row identifies its immutable source file and field/row locator",
        "supersedes": {
            "path": "outputs/jcheminform_revision/numerical_freeze_v1",
            "reason": "v1 used an inferred D3 protocol label and a synthetic D4 seed count; values were unchanged",
        },
        "approved_b1_strata": list(APPROVED_B1_STRATA),
        "input_sources": {key: file_record(path, root) for key, path in paths.items()},
        "validation_checks": gate_checks + consistency_checks,
        "row_counts": {
            "b1_conformer_stability_summary": len(b1_rows),
            "system_performance": len(system_rows),
            "paired_effects": len(effect_rows),
            "promotion_degradation_counts": len(count_rows),
            "h2_posthoc_models": len(h2_rows),
        },
        "outputs": outputs,
        "paper_gate": {
            "numerical_sources_frozen": True,
            "latex_updated": False,
            "identity_funding_doi_placeholders_resolved": False,
        },
    }
    manifest_path = output / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/jcheminform_revision/numerical_freeze_v2"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir
    if not output.is_absolute():
        output = args.repo_root / output
    manifest = build_numerical_ledger(args.repo_root, output)
    print(json.dumps({"status": "complete", "output": str(output.resolve()), "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
