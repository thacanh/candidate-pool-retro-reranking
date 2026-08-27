"""Exploratory frozen-artifact mechanism diagnostics for Digital Discovery.

M1 formalizes heterogeneity across the already frozen J3 candidate-count
strata.  M2 describes candidate-pool composition and links it to already frozen
rank shifts.  Neither command trains a model or generates candidates.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import kendalltau, rankdata, spearmanr

from rerank.analysis.analyze_revision_predictions import (
    clustered_sign_flip,
    product_cluster_bootstrap,
)
from rerank.analysis.analyze_round_jk import (
    D5_SEEDS,
    POOL_ORDER,
    SEEDS,
    _candidate_bin,
    _expanded_records,
    _load_anchor_matrices,
    _read_csv,
    _write_csv,
)
from rerank.experiments.run_round_jk import (
    PROTOCOL_ID as K1_PROTOCOL_ID,
    _approval_provenance,
    _assert_isolated_output,
)
from rerank.study_data import (
    canonicalize_reactant_set,
    canonicalize_smiles,
    file_fingerprint,
    load_reactions,
)
from rerank.ws_e_streaming import atomic_json


M1_PROTOCOL_ID = "dd-mechanism-heterogeneity-v1"
M2_PROTOCOL_ID = "dd-candidate-pool-shift-diagnostic-v2"
M2B_PROTOCOL_ID = "dd-candidate-pool-transfer-inference-v1"
M_STATUS = "approved_exploratory_frozen_artifacts_only"
HISTORICAL_POOL_SHA256 = "9ec1cf192c49eeb7d74a320dd721287fabdef9863cc06f95d0f13baab8c3ff85"
K1_CANDIDATE_ORDERING = "first rows of the frozen pool order; no re-sorting"
K1_SOURCE_POOL_PROTOCOL_ID = "ws-e-localretro-three-pools-filtered-v2"
M1_BIN_ORDER = ("1", "2-3", "4-6", "7-9", "10")
M1_NON_SINGLETON = ("2-3", "4-6", "7-9", "10")
POOL_FILE_NAMES = {
    "aizynthfinder_only": "aizynthfinder_only.jsonl",
    "localretro_only": "localretro_only.jsonl",
    "merged": "merged_canonical_union.jsonl",
}
M2_FEATURES = (
    "candidate_count",
    "jaccard_vs_historical",
    "shared_candidate_count",
    "kendall_order_vs_historical",
    "normalized_stored_prior_mass_entropy",
    "mean_pairwise_morgan_distance",
    "max_nonreference_similarity_to_truth",
)
M2_TRANSFER_FEATURES = (
    "candidate_count_shift_vs_historical",
    "jaccard_vs_historical",
    "shared_candidate_count",
    "kendall_order_vs_historical",
    "mean_pairwise_morgan_distance_shift_vs_historical",
    "max_nonreference_similarity_shift_vs_historical",
)


def _m_approval(approval_path: str | Path, plan_path: str | Path, protocol_id: str) -> dict:
    raw = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    expected_key = "m1_protocol_id" if protocol_id == M1_PROTOCOL_ID else "m2_protocol_id"
    if (
        raw.get("m_status") != M_STATUS
        or raw.get(expected_key) != protocol_id
        or not raw.get("m_approval_date")
        or not raw.get("m_approval_quote")
    ):
        raise PermissionError(f"{protocol_id} lacks a signed M-round approval.")
    result = _approval_provenance(approval_path, plan_path)
    result.update(
        {
            "m_status": raw["m_status"],
            "m_approval_date": raw["m_approval_date"],
            "m_approval_quote": raw["m_approval_quote"],
            expected_key: raw[expected_key],
        }
    )
    return result


def _m2b_approval(approval_path: str | Path, plan_path: str | Path) -> dict:
    raw = json.loads(Path(approval_path).read_text(encoding="utf-8"))
    if (
        raw.get("final_inference_status") != "approved_frozen_artifacts_only"
        or raw.get("m2b_protocol_id") != M2B_PROTOCOL_ID
        or not raw.get("m2b_approval_date")
        or not raw.get("m2b_approval_quote")
    ):
        raise PermissionError("M2b lacks a signed frozen-artifact approval.")
    result = _approval_provenance(approval_path, plan_path)
    result.update(
        {
            "m2b_protocol_id": raw["m2b_protocol_id"],
            "m2b_approval_date": raw["m2b_approval_date"],
            "m2b_approval_quote": raw["m2b_approval_quote"],
        }
    )
    return result


def _environment() -> dict:
    packages = {}
    for name in ("numpy", "scipy", "rdkit"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_file(path: str | Path, expected: Mapping, label: str) -> dict:
    actual = file_fingerprint(path)
    if (
        actual["sha256"] != str(expected.get("sha256"))
        or int(actual["size_bytes"]) != int(expected.get("size_bytes", -1))
    ):
        raise ValueError(f"{label} differs from its frozen fingerprint: {path}")
    return actual


def _resolve_frozen_file(expected: Mapping, fallback: str | Path, label: str) -> tuple[Path, dict]:
    candidates = [Path(str(expected.get("path", ""))), Path(fallback)]
    checked = set()
    for candidate in candidates:
        if not str(candidate) or candidate in checked or not candidate.is_file():
            continue
        checked.add(candidate)
        actual = file_fingerprint(candidate)
        if (
            actual["sha256"] == str(expected.get("sha256"))
            and int(actual["size_bytes"]) == int(expected.get("size_bytes", -1))
        ):
            return candidate, actual
    raise ValueError(f"Cannot resolve the frozen {label} by SHA-256 and size.")


def _anchor_prediction_fingerprints(root: str | Path) -> dict[str, dict]:
    base = Path(root)
    return {
        f"{arm}_seed_{seed}": file_fingerprint(base / f"{arm}_seed_{seed}.csv")
        for seed in SEEDS
        for arm in ("baseline", "augmented")
    }


def _reaction_effects(prediction_root: str | Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    reference, baseline, augmented = _load_anchor_matrices(prediction_root)
    baseline_ranks = np.asarray(
        [[int(row["reranked_rank"]) for row in seed_rows] for seed_rows in baseline],
        dtype=np.int16,
    )
    augmented_ranks = np.asarray(
        [[int(row["reranked_rank"]) for row in seed_rows] for seed_rows in augmented],
        dtype=np.int16,
    )
    top1 = (augmented_ranks == 1).astype(np.float64) - (
        baseline_ranks == 1
    ).astype(np.float64)
    mrr = 1.0 / augmented_ranks.astype(np.float64) - 1.0 / baseline_ranks.astype(np.float64)
    return reference, top1, mrr


def _between_group_statistic(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    if values.ndim != 1 or labels.shape != values.shape:
        raise ValueError("Values and labels must be aligned one-dimensional arrays.")
    grand = float(values.mean())
    statistic = 0.0
    for label in np.unique(labels):
        selected = values[labels == label]
        statistic += len(selected) * (float(selected.mean()) - grand) ** 2
    return float(statistic)


def _omnibus_permutation(
    values: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    rng_seed: int,
) -> tuple[float, float]:
    observed = _between_group_statistic(values, labels)
    rng = np.random.default_rng(rng_seed)
    extreme = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        extreme += _between_group_statistic(values, permuted) >= observed - 1e-15
    return observed, float((extreme + 1) / (permutations + 1))


def _mean_contrast(values: np.ndarray, labels: np.ndarray, left: str, right: Sequence[str]) -> float:
    left_values = values[labels == left]
    right_values = values[np.isin(labels, tuple(right))]
    if not len(left_values) or not len(right_values):
        raise ValueError("M1 contrast contains an empty group.")
    return float(left_values.mean() - right_values.mean())


def _contrast_permutation(
    values: np.ndarray,
    labels: np.ndarray,
    left: str,
    right: Sequence[str],
    permutations: int,
    rng_seed: int,
) -> tuple[float, float]:
    selected = (labels == left) | np.isin(labels, tuple(right))
    sample = values[selected]
    binary = labels[selected] == left
    observed = float(sample[binary].mean() - sample[~binary].mean())
    n_left = int(binary.sum())
    rng = np.random.default_rng(rng_seed)
    extreme = 0
    for _ in range(permutations):
        permuted = rng.permutation(sample)
        statistic = float(permuted[:n_left].mean() - permuted[n_left:].mean())
        extreme += abs(statistic) >= abs(observed) - 1e-15
    return observed, float((extreme + 1) / (permutations + 1))


def _contrast_bootstrap(
    values: np.ndarray,
    labels: np.ndarray,
    left: str,
    right: Sequence[str],
    draws: int,
    rng_seed: int,
) -> tuple[float, float]:
    left_values = values[labels == left]
    right_values = values[np.isin(labels, tuple(right))]
    rng = np.random.default_rng(rng_seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        estimates[index] = float(
            rng.choice(left_values, len(left_values), replace=True).mean()
            - rng.choice(right_values, len(right_values), replace=True).mean()
        )
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    adjusted = np.empty(len(values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * float(values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def run_m1(args: argparse.Namespace) -> dict:
    started = time.time()
    approval = _m_approval(args.approval_record, args.analysis_plan, M1_PROTOCOL_ID)
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite M1 output: {output}")
    reference, top1_seed, mrr_seed = _reaction_effects(args.anchor_prediction_root)
    if len(reference) != 3985 or top1_seed.shape != (len(SEEDS), 3985):
        raise ValueError("M1 requires 3,985 reactions and paired seeds 42--61.")
    canonical_products = [canonicalize_smiles(str(row["product_smiles"])) for row in reference]
    if None in canonical_products or len(set(canonical_products)) != len(reference):
        raise ValueError("M1 requires one unique canonical product per frozen reaction.")
    labels = np.asarray([_candidate_bin(int(row["candidate_count"])) for row in reference])
    top1 = top1_seed.mean(axis=0)
    mrr = mrr_seed.mean(axis=0)
    singleton = labels == "1"
    if not np.all(top1_seed[:, singleton] == 0.0) or not np.all(mrr_seed[:, singleton] == 0.0):
        raise RuntimeError("M1 singleton structural audit failed.")

    reaction_rows = []
    for index, row in enumerate(reference):
        reaction_rows.append(
            {
                "reaction_id": int(row["reaction_id"]),
                "canonical_product": canonicalize_smiles(row["product_smiles"]),
                "candidate_count": int(row["candidate_count"]),
                "candidate_count_bin": labels[index],
                "mean_20seed_delta_top1": float(top1[index]),
                "mean_20seed_delta_mrr": float(mrr[index]),
            }
        )

    omnibus_rows = []
    primary_rows = []
    pairwise_rows = []
    for metric_index, (metric, values) in enumerate((("top1", top1), ("mrr", mrr))):
        seed = args.permutation_seed + metric_index
        for scope, included in (
            (("all_five_bins", M1_BIN_ORDER), ("non_singleton_bins", M1_NON_SINGLETON))
        ):
            mask = np.isin(labels, included)
            statistic, p_value = _omnibus_permutation(
                values[mask], labels[mask], args.permutations, seed
            )
            omnibus_rows.append(
                {
                    "metric": metric,
                    "scope": scope,
                    "n_reactions": int(mask.sum()),
                    "between_bin_statistic": statistic,
                    "omnibus_permutation_p_upper_tail": p_value,
                    "permutations": args.permutations,
                }
            )
        effect, p_value = _contrast_permutation(
            values,
            labels,
            "4-6",
            ("2-3", "7-9", "10"),
            args.permutations,
            seed,
        )
        low, high = _contrast_bootstrap(
            values,
            labels,
            "4-6",
            ("2-3", "7-9", "10"),
            args.bootstrap_samples,
            args.bootstrap_seed + metric_index,
        )
        primary_rows.append(
            {
                "metric": metric,
                "contrast": "4-6 minus pooled 2-3,7-9,10",
                "effect": effect,
                "bootstrap_ci95_low": low,
                "bootstrap_ci95_high": high,
                "permutation_p_two_sided": p_value,
                "permutations": args.permutations,
                "bootstrap_samples": args.bootstrap_samples,
            }
        )
        raw_rows = []
        for other in ("2-3", "7-9", "10"):
            pair_effect, pair_p = _contrast_permutation(
                values,
                labels,
                "4-6",
                (other,),
                args.permutations,
                seed,
            )
            raw_rows.append(
                {
                    "metric": metric,
                    "contrast": f"4-6 minus {other}",
                    "effect": pair_effect,
                    "permutation_p_two_sided": pair_p,
                    "permutations": args.permutations,
                }
            )
        adjusted = _holm_adjust([row["permutation_p_two_sided"] for row in raw_rows])
        for row, value in zip(raw_rows, adjusted, strict=True):
            row["holm_p_within_metric"] = value
            pairwise_rows.append(row)

    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "m1_reaction_effects.csv", reaction_rows)
    _write_csv(output / "m1_omnibus.csv", omnibus_rows)
    _write_csv(output / "m1_primary_contrast.csv", primary_rows)
    _write_csv(output / "m1_pairwise_contrasts.csv", pairwise_rows)
    manifest = {
        "schema_version": 1,
        "record_kind": "dd_exploratory_mechanism_heterogeneity",
        "protocol_id": M1_PROTOCOL_ID,
        "comparator": "frozen historical cap-10 baseline versus augmented predictions",
        "single_intended_change": "exploratory inference across frozen J3 candidate-count bins",
        "settings": {
            "seeds": list(SEEDS),
            "candidate_count_bins": list(M1_BIN_ORDER),
            "permutations": args.permutations,
            "permutation_rng_seeds": [args.permutation_seed, args.permutation_seed + 1],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_rng_seeds": [args.bootstrap_seed, args.bootstrap_seed + 1],
            "multiplicity": "Holm within endpoint for three secondary pairwise contrasts",
            "claim_status": "exploratory_post_hoc",
            "seed_estimand": "reaction effects averaged across the 20 frozen paired training seeds",
            "omnibus_permutation_tail": "upper",
        },
        "inputs": {
            "prediction_files": _anchor_prediction_fingerprints(args.anchor_prediction_root),
            "canonical_product_clusters": len(set(canonical_products)),
            "maximum_cluster_size": 1,
        },
        "outputs": {
            name: file_fingerprint(output / file_name)
            for name, file_name in {
                "reaction_effects": "m1_reaction_effects.csv",
                "omnibus": "m1_omnibus.csv",
                "primary_contrast": "m1_primary_contrast.csv",
                "pairwise_contrasts": "m1_pairwise_contrasts.csv",
            }.items()
        },
        "singleton_structural_audit": "pass",
        "round_jk_approval": approval,
        "environment": _environment(),
        "runtime_seconds": time.time() - started,
        "training_performed": False,
        "retuning_performed": False,
        "candidate_generation_performed": False,
        "test_partition_loaded_from_frozen_predictions_only": True,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _stream_top10_pool(
    path: str | Path,
    target_products: set[str],
    expected_protocol_id: str | None = None,
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    last_raw_product = None
    last_product = None
    closed_products: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if expected_protocol_id is not None and row.get("protocol_id") != expected_protocol_id:
                raise ValueError(f"Pool row has the wrong protocol ID: {path}")
            raw_product = str(row["product"])
            if raw_product != last_raw_product:
                if expected_protocol_id is not None and last_product is not None:
                    closed_products.add(last_product)
                last_raw_product = raw_product
                last_product = canonicalize_smiles(raw_product)
                if expected_protocol_id is not None and last_product in closed_products:
                    raise ValueError(f"Pool product rows are not contiguous: {path}")
            if last_product not in target_products or len(result[last_product]) >= 10:
                continue
            identity = canonicalize_reactant_set(str(row["reactant"]))
            if identity is None:
                raise ValueError(f"Non-canonicalizable candidate in {path}")
            if identity in {item["identity"] for item in result[last_product]}:
                raise ValueError(f"Canonical duplicate within a frozen candidate list: {path}")
            expected_rank = len(result[last_product]) + 1
            if expected_protocol_id is not None and int(row.get("candidate_rank", 0)) != expected_rank:
                raise ValueError(f"Pool candidate ranks are not contiguous from one: {path}")
            result[last_product].append(
                {"identity": identity, "prior": float(row.get("prior", 0.0))}
            )
    missing = target_products.difference(result)
    if missing:
        raise ValueError(f"Pool {path} lacks {len(missing)} target products.")
    return dict(result)


def _normalized_historical_top10_pool(
    path: str | Path, target_products: set[str]
) -> dict[str, list[dict]]:
    winners: dict[str, dict[str, dict]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            product = canonicalize_smiles(str(row["product"]))
            if product not in target_products:
                continue
            identity = canonicalize_reactant_set(str(row["reactant"]))
            prior = float(row["prior"])
            if identity is None or not math.isfinite(prior):
                continue
            existing = winners[product].get(identity)
            if existing is None:
                winners[product][identity] = {
                    "identity": identity,
                    "prior": prior,
                    "first_seen_line": line_number,
                }
            elif prior > float(existing["prior"]):
                existing["prior"] = prior
    missing = target_products.difference(winners)
    if missing:
        raise ValueError(f"Historical pool lacks {len(missing)} target products.")
    return {
        product: [
            {"identity": row["identity"], "prior": float(row["prior"])}
            for row in sorted(
                records.values(),
                key=lambda item: (-float(item["prior"]), int(item["first_seen_line"])),
            )[:10]
        ]
        for product, records in winners.items()
    }


def _historical_top10(reference: Sequence[Mapping], target_ids: set[int]) -> dict[str, list[dict]]:
    result = {}
    for row in reference:
        if int(row["reaction_id"]) not in target_ids:
            continue
        product = canonicalize_smiles(str(row["product_smiles"]))
        identities = [
            canonicalize_reactant_set(value)
            for value in json.loads(str(row["baseline_candidates_json"]))
        ]
        if product is None or any(identity is None for identity in identities):
            raise ValueError("Historical pool contains a non-canonicalizable identity.")
        count = len(identities)
        priors = np.linspace(1.0, 0.0, num=count).tolist() if count > 1 else [1.0]
        records = [
            {"identity": identity, "prior": float(prior)}
            for identity, prior in zip(identities, priors, strict=True)
        ]
        previous = result.setdefault(product, records)
        if previous != records:
            raise ValueError("Duplicate historical product has conflicting candidate lists.")
    return result


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=True
)


@lru_cache(maxsize=300_000)
def _morgan(identity: str):
    molecule = Chem.MolFromSmiles(identity)
    if molecule is None:
        raise ValueError(f"Cannot fingerprint candidate identity: {identity}")
    return _MORGAN.GetFingerprint(molecule)


def _normalized_entropy(priors: Sequence[float]) -> float | None:
    values = np.maximum(np.asarray(priors, dtype=np.float64), 0.0)
    if len(values) < 2 or float(values.sum()) <= 0.0:
        return None
    probabilities = values / values.sum()
    positive = probabilities > 0.0
    return float(-np.sum(probabilities[positive] * np.log(probabilities[positive])) / np.log(len(values)))


def _order_concordance(reference: Sequence[str], current: Sequence[str]) -> float | None:
    common = set(reference).intersection(current)
    if len(common) < 2:
        return None
    left = {value: index for index, value in enumerate(reference)}
    right = {value: index for index, value in enumerate(current)}
    ordered = sorted(common, key=left.__getitem__)
    value = kendalltau(
        [left[item] for item in ordered],
        [right[item] for item in ordered],
    ).statistic
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _list_descriptors(
    candidates: Sequence[Mapping], historical: Sequence[Mapping], ground_truth: str
) -> dict:
    identities = [str(row["identity"]) for row in candidates]
    historical_ids = [str(row["identity"]) for row in historical]
    current_set = set(identities)
    historical_set = set(historical_ids)
    shared = current_set.intersection(historical_set)
    union = current_set.union(historical_set)
    fingerprints = [_morgan(value) for value in identities]
    distances = [
        1.0 - float(DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right]))
        for left, right in combinations(range(len(fingerprints)), 2)
    ]
    truth_fp = _morgan(ground_truth)
    incorrect = [
        float(DataStructs.TanimotoSimilarity(truth_fp, fingerprint))
        for identity, fingerprint in zip(identities, fingerprints, strict=True)
        if identity != ground_truth
    ]
    return {
        "candidate_count": len(identities),
        "jaccard_vs_historical": len(shared) / len(union),
        "shared_candidate_count": len(shared),
        "kendall_order_vs_historical": _order_concordance(historical_ids, identities),
        "normalized_stored_prior_mass_entropy": _normalized_entropy(
            [float(row["prior"]) for row in candidates]
        ),
        "mean_pairwise_morgan_distance": float(np.mean(distances)) if distances else None,
        "max_nonreference_similarity_to_truth": max(incorrect) if incorrect else None,
    }


def _safe_spearman(left: Iterable[float], right: Iterable[float]) -> tuple[int, float | None]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(left, right, strict=True)
        if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 3:
        return len(pairs), None
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return len(pairs), None
    value = spearmanr(x, y).statistic
    return len(pairs), None if not math.isfinite(float(value)) else float(value)


def _quartile_rows(rows: list[dict], pool: str, feature: str, metric: str) -> list[dict]:
    selected = [
        row for row in rows
        if row[feature] is not None and math.isfinite(float(row[feature]))
    ]
    values = np.asarray([float(row[feature]) for row in selected], dtype=np.float64)
    if len(values) < 4 or len(np.unique(values)) < 2:
        return []
    quartiles = np.clip(np.ceil(4.0 * rankdata(values, method="average") / len(values)), 1, 4).astype(int)
    result = []
    for quartile in range(1, 5):
        mask = quartiles == quartile
        if not np.any(mask):
            continue
        effects = np.asarray([float(selected[index][metric]) for index in np.flatnonzero(mask)])
        result.append(
            {
                "pool": pool,
                "feature": feature,
                "metric": metric,
                "quartile": f"Q{quartile}",
                "n_reactions": int(mask.sum()),
                "feature_min": float(values[mask].min()),
                "feature_max": float(values[mask].max()),
                "mean_effect": float(effects.mean()),
            }
        )
    return result


def _expanded_effects(
    records: Mapping[int, Mapping], reaction_ids: Iterable[int] | None = None
) -> dict[int, dict[str, float]]:
    result = {}
    selected_ids = sorted(records) if reaction_ids is None else sorted(set(reaction_ids))
    for reaction_id in selected_ids:
        row = records[reaction_id]
        if not bool(row.get("covered", True)):
            raise ValueError(f"Cannot compute a rank effect for uncovered reaction {reaction_id}.")
        top1 = []
        mrr = []
        for seed in SEEDS:
            baseline = int(row[f"baseline_rank_seed_{seed}"])
            augmented = int(row[f"augmented_rank_seed_{seed}"])
            top1.append(float(augmented == 1) - float(baseline == 1))
            mrr.append(1.0 / augmented - 1.0 / baseline)
        result[reaction_id] = {
            "mean_20seed_delta_top1": float(np.mean(top1)),
            "mean_20seed_delta_mrr": float(np.mean(mrr)),
        }
    return result


def _validate_k1_chain(
    manifest_path: str | Path,
    pool: str,
    pool_jsonl: str | Path,
    source_csv: str | Path,
    metadata_csv: str | Path,
) -> tuple[dict, dict[int, dict], dict]:
    path = Path(manifest_path)
    manifest, records = _expanded_records(path)
    required = {
        "protocol_id": K1_PROTOCOL_ID,
        "pool_name": pool,
        "candidate_cap": 10,
        "candidate_ordering": K1_CANDIDATE_ORDERING,
        "source_pool_protocol_id": K1_SOURCE_POOL_PROTOCOL_ID,
        "test_reactions_total": 5004,
        "test_partition_loaded_only_after_model_freeze": True,
    }
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Invalid frozen K1 {pool} manifest field {key!r}.")
    seed_keys = {str(seed) for seed in SEEDS}
    for arm in ("baseline", "augmented"):
        if set(manifest.get("per_seed_metrics", {}).get(arm, {})) != seed_keys:
            raise ValueError(f"Frozen K1 {pool} test manifest has the wrong seed set.")
    covered_count = sum(bool(row.get("covered")) for row in records.values())
    if covered_count != int(manifest.get("test_reactions_covered", -1)):
        raise ValueError(f"Frozen K1 {pool} coverage count is inconsistent.")

    model_fallback = path.parent.parent / "models" / "freeze.json"
    model_path, model_actual = _resolve_frozen_file(
        manifest["model_freeze"], model_fallback, f"{pool} K1 model freeze"
    )
    model = _load_json(model_path)
    model_required = {
        "complete": True,
        "protocol_id": K1_PROTOCOL_ID,
        "pool_name": pool,
        "candidate_cap": 10,
        "candidate_ordering": K1_CANDIDATE_ORDERING,
        "source_pool_protocol_id": K1_SOURCE_POOL_PROTOCOL_ID,
        "test_partition_loaded": False,
    }
    for key, expected in model_required.items():
        if model.get(key) != expected:
            raise ValueError(f"Invalid frozen K1 {pool} model-freeze field {key!r}.")
    if tuple(int(seed) for seed in model.get("seeds", ())) != tuple(SEEDS):
        raise ValueError(f"Frozen K1 {pool} model freeze has the wrong seed set.")

    selection_fallback = model_path.parent.parent / "selection" / "manifest.json"
    selection_path, selection_actual = _resolve_frozen_file(
        model["selection_manifest"], selection_fallback, f"{pool} K1 selection manifest"
    )
    selection = _load_json(selection_path)
    selection_required = {
        "protocol_id": K1_PROTOCOL_ID,
        "pool_name": pool,
        "candidate_cap": 10,
        "candidate_ordering": K1_CANDIDATE_ORDERING,
        "source_pool_protocol_id": K1_SOURCE_POOL_PROTOCOL_ID,
        "test_partition_loaded": False,
        "test_ground_truth_loaded": False,
    }
    for key, expected in selection_required.items():
        if selection.get(key) != expected:
            raise ValueError(f"Invalid frozen K1 {pool} selection field {key!r}.")
    if tuple(int(seed) for seed in selection.get("seeds", ())) != tuple(SEEDS):
        raise ValueError(f"Frozen K1 {pool} selection has the wrong seed set.")

    pool_actual = _verify_file(pool_jsonl, selection["inputs"]["pool"], f"{pool} pool JSONL")
    source_actual = _verify_file(
        source_csv, selection["inputs"]["source_csv"], f"{pool} source CSV"
    )
    metadata_actual = _verify_file(
        metadata_csv, selection["inputs"]["metadata_csv"], f"{pool} metadata CSV"
    )
    _verify_file(source_csv, manifest["source_csv"], f"{pool} test source CSV")
    _verify_file(metadata_csv, manifest["metadata_csv"], f"{pool} test metadata CSV")
    feature_path, feature_actual = _resolve_frozen_file(
        manifest["feature_freeze"],
        selection["inputs"]["feature_freeze"]["path"],
        f"{pool} feature freeze",
    )
    if (
        feature_actual["sha256"] != selection["inputs"]["feature_freeze"]["sha256"]
        or int(feature_actual["size_bytes"])
        != int(selection["inputs"]["feature_freeze"]["size_bytes"])
    ):
        raise ValueError(f"Frozen K1 {pool} selection and test use different feature freezes.")
    return manifest, records, {
        "test_manifest": file_fingerprint(path),
        "predictions": manifest["predictions"],
        "model_freeze": model_actual,
        "selection_manifest": selection_actual,
        "feature_freeze": feature_actual,
        "feature_freeze_path": str(feature_path.resolve()),
        "pool": pool_actual,
        "source_csv": source_actual,
        "metadata_csv": metadata_actual,
        "covered_reactions": covered_count,
    }


def _l1_external_links(
    l1_path: str | Path, d5_root: str | Path
) -> tuple[list[dict], list[dict], dict]:
    l1_path = Path(l1_path)
    l1_manifest_path = l1_path.parent / "manifest.json"
    l1_manifest = _load_json(l1_manifest_path)
    if l1_manifest.get("protocol_id") != "dd-rxn-ebm-score-diagnostic-v1":
        raise ValueError("L1 diagnostic belongs to the wrong protocol.")
    l1_actual = _verify_file(
        l1_path, l1_manifest["outputs"]["per_list"], "L1 per-list diagnostics"
    )

    d5_manifest_path = Path(d5_root) / "test_results" / "manifest.json"
    d5_manifest = _load_json(d5_manifest_path)
    if (
        d5_manifest.get("protocol_id") != "D-EXTERNAL-RXN-EBM-FF-CAP10-v1"
        or d5_manifest.get("test_partition_loaded_only_after_freeze") is not True
        or int(d5_manifest.get("test_reactions", 0)) != 3985
    ):
        raise ValueError("D5 official-test manifest failed its frozen protocol gate.")
    _verify_file(
        d5_manifest_path,
        l1_manifest["inputs"]["d5_test_manifest"],
        "D5 manifest bound into L1",
    )

    diagnostics = _read_csv(l1_path)
    by_reaction: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in diagnostics:
        reaction_id = int(row["reaction_id"])
        by_reaction[reaction_id]["l1_score_sd"].append(float(row["external_score_population_sd"]))
        by_reaction[reaction_id]["l1_score_range"].append(float(row["external_score_range"]))
        if str(row["spearman_defined"]).lower() == "true" and row["spearman_external_vs_prior"]:
            by_reaction[reaction_id]["l1_spearman_vs_prior"].append(float(row["spearman_external_vs_prior"]))
    outcomes: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    prediction_inputs = {}
    for seed in D5_SEEDS:
        prediction_path, prediction_actual = _resolve_frozen_file(
            d5_manifest["prediction_files"][str(seed)],
            Path(d5_root) / "test_results" / f"predictions_seed_{seed}.csv",
            f"D5 predictions seed {seed}",
        )
        prediction_inputs[str(seed)] = prediction_actual
        for row in _read_csv(prediction_path):
            reaction_id = int(row["reaction_id"])
            prior = int(row["prior_true_rank"])
            external = int(row["external_true_rank"])
            outcomes[reaction_id]["mean_external_minus_prior_rank"].append(float(external - prior))
            outcomes[reaction_id]["mean_external_minus_prior_top1"].append(
                float(external == 1) - float(prior == 1)
            )
    rows = []
    for reaction_id in sorted(set(by_reaction).intersection(outcomes)):
        rows.append(
            {
                "reaction_id": reaction_id,
                **{
                    key: float(np.mean(values)) if values else None
                    for key, values in by_reaction[reaction_id].items()
                },
                **{
                    key: float(np.mean(values))
                    for key, values in outcomes[reaction_id].items()
                },
            }
        )
    links = []
    for feature in ("l1_spearman_vs_prior", "l1_score_sd", "l1_score_range"):
        for outcome in ("mean_external_minus_prior_rank", "mean_external_minus_prior_top1"):
            n, rho = _safe_spearman(
                [row.get(feature) for row in rows], [row[outcome] for row in rows]
            )
            links.append(
                {
                    "feature": feature,
                    "outcome": outcome,
                    "n_reactions": n,
                    "spearman_rho": rho,
                }
            )
    return rows, links, {
        "l1_manifest": file_fingerprint(l1_manifest_path),
        "l1_per_list_diagnostics": l1_actual,
        "d5_test_manifest": file_fingerprint(d5_manifest_path),
        "d5_predictions": prediction_inputs,
    }


def run_m2(args: argparse.Namespace) -> dict:
    started = time.time()
    approval = _m_approval(args.approval_record, args.analysis_plan, M2_PROTOCOL_ID)
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite M2 output: {output}")
    reference, historical_top1, historical_mrr = _reaction_effects(args.anchor_prediction_root)
    reference_by_id = {int(row["reaction_id"]): (index, row) for index, row in enumerate(reference)}
    expanded_records = {}
    expanded_inputs = {}
    expanded_manifests = {}
    for pool, path, pool_jsonl in zip(
        POOL_ORDER[1:], args.k1_manifest, args.pool_jsonl, strict=True
    ):
        manifest, records, audit = _validate_k1_chain(
            path, pool, pool_jsonl, args.source_csv, args.metadata_csv
        )
        expanded_manifests[pool] = manifest
        expanded_records[pool] = records
        expanded_inputs[pool] = audit
    common_ids = set(reference_by_id)
    for records in expanded_records.values():
        common_ids.intersection_update(
            reaction_id for reaction_id, row in records.items() if bool(row["covered"])
        )
    if len(common_ids) != args.expected_common_count:
        raise ValueError(
            f"Expected {args.expected_common_count} common covered reactions, found {len(common_ids)}."
        )
    reactions = {
        reaction.reaction_id: reaction
        for reaction in load_reactions(args.source_csv, args.metadata_csv)
        if reaction.reaction_id in common_ids
    }
    if reactions.keys() != common_ids:
        raise ValueError("M2 failed to resolve every common reaction in the source data.")
    target_products = {reaction.product_key for reaction in reactions.values()}
    historical = _historical_top10(reference, common_ids)
    pools = {"historical_cap10": historical}
    historical_input = file_fingerprint(args.historical_pool_jsonl)
    if historical_input["sha256"] != HISTORICAL_POOL_SHA256:
        raise ValueError("Historical candidate pool differs from the signed analysis-plan SHA-256.")
    pool_inputs = {"historical_cap10": historical_input}
    # Validate the compact historical list against the original frozen pool.
    original_historical = _normalized_historical_top10_pool(
        args.historical_pool_jsonl, target_products
    )
    for product in target_products:
        if [row["identity"] for row in original_historical[product]] != [
            row["identity"] for row in historical[product]
        ]:
            raise ValueError("Historical compact prediction list differs from frozen pool JSONL.")
        historical[product] = original_historical[product]
    for pool, path in zip(POOL_ORDER[1:], args.pool_jsonl, strict=True):
        pools[pool] = _stream_top10_pool(
            path, target_products, expected_protocol_id=K1_SOURCE_POOL_PROTOCOL_ID
        )
        pool_inputs[pool] = file_fingerprint(path)

    effects = {
        "historical_cap10": {
            int(row["reaction_id"]): {
                "mean_20seed_delta_top1": float(historical_top1[:, index].mean()),
                "mean_20seed_delta_mrr": float(historical_mrr[:, index].mean()),
            }
            for index, row in enumerate(reference)
        }
    }
    for pool in POOL_ORDER[1:]:
        effects[pool] = _expanded_effects(expanded_records[pool], common_ids)

    descriptor_rows = []
    for reaction_id in sorted(common_ids):
        reaction = reactions[reaction_id]
        historical_candidates = historical[reaction.product_key]
        if reaction.ground_truth_key not in {row["identity"] for row in historical_candidates}:
            raise ValueError("M2 common historical list does not contain its reference reactants.")
        for pool in POOL_ORDER:
            candidates = pools[pool][reaction.product_key]
            identities = [row["identity"] for row in candidates]
            if reaction.ground_truth_key not in identities:
                raise ValueError(f"M2 {pool} list does not contain its recorded reference reactants.")
            if pool == "historical_cap10":
                frozen = reference_by_id[reaction_id][1]
                frozen_count = int(frozen["candidate_count"])
                frozen_rank = int(frozen["baseline_rank"])
            else:
                frozen = expanded_records[pool][reaction_id]
                frozen_count = int(frozen["candidate_count"])
                frozen_rank = int(frozen["prior_rank"])
            if len(candidates) != frozen_count:
                raise ValueError(f"M2 {pool} list length differs from frozen candidate_count.")
            if identities.index(reaction.ground_truth_key) + 1 != frozen_rank:
                raise ValueError(f"M2 {pool} reference position differs from frozen prior_rank.")
            descriptor_rows.append(
                {
                    "reaction_id": reaction_id,
                    "canonical_product": reaction.product_key,
                    "pool": pool,
                    **_list_descriptors(candidates, historical_candidates, reaction.ground_truth_key),
                    **effects[pool][reaction_id],
                }
            )

    descriptor_by_key = {
        (int(row["reaction_id"]), str(row["pool"])): row for row in descriptor_rows
    }
    transfer_rows = []
    for reaction_id in sorted(common_ids):
        historical_row = descriptor_by_key[(reaction_id, "historical_cap10")]
        for pool in POOL_ORDER[1:]:
            row = descriptor_by_key[(reaction_id, pool)]

            def shift(field: str) -> float | None:
                left, right = row[field], historical_row[field]
                if left is None or right is None:
                    return None
                return float(left) - float(right)

            transfer_rows.append(
                {
                    "reaction_id": reaction_id,
                    "canonical_product": row["canonical_product"],
                    "pool": pool,
                    "transfer_delta_top1": (
                        float(row["mean_20seed_delta_top1"])
                        - float(historical_row["mean_20seed_delta_top1"])
                    ),
                    "transfer_delta_mrr": (
                        float(row["mean_20seed_delta_mrr"])
                        - float(historical_row["mean_20seed_delta_mrr"])
                    ),
                    "candidate_count_shift_vs_historical": shift("candidate_count"),
                    "jaccard_vs_historical": row["jaccard_vs_historical"],
                    "shared_candidate_count": row["shared_candidate_count"],
                    "kendall_order_vs_historical": row["kendall_order_vs_historical"],
                    "mean_pairwise_morgan_distance_shift_vs_historical": shift(
                        "mean_pairwise_morgan_distance"
                    ),
                    "max_nonreference_similarity_shift_vs_historical": shift(
                        "max_nonreference_similarity_to_truth"
                    ),
                }
            )

    association_rows = []
    quartile_rows = []
    for pool in POOL_ORDER:
        selected = [row for row in descriptor_rows if row["pool"] == pool]
        for feature in M2_FEATURES:
            for metric in ("mean_20seed_delta_top1", "mean_20seed_delta_mrr"):
                n, rho = _safe_spearman(
                    [row[feature] for row in selected], [row[metric] for row in selected]
                )
                association_rows.append(
                    {
                        "pool": pool,
                        "feature": feature,
                        "metric": metric,
                        "n_reactions": n,
                        "spearman_rho": rho,
                    }
                )
                quartile_rows.extend(_quartile_rows(selected, pool, feature, metric))

    transfer_association_rows = []
    transfer_quartile_rows = []
    for pool in POOL_ORDER[1:]:
        selected = [row for row in transfer_rows if row["pool"] == pool]
        for feature in M2_TRANSFER_FEATURES:
            for metric in ("transfer_delta_top1", "transfer_delta_mrr"):
                n, rho = _safe_spearman(
                    [row[feature] for row in selected], [row[metric] for row in selected]
                )
                transfer_association_rows.append(
                    {
                        "pool": pool,
                        "feature": feature,
                        "metric": metric,
                        "n_reactions": n,
                        "spearman_rho": rho,
                    }
                )
                transfer_quartile_rows.extend(
                    _quartile_rows(selected, pool, feature, metric)
                )

    transfer_summary_rows = []
    for pool in POOL_ORDER[1:]:
        selected = [row for row in transfer_rows if row["pool"] == pool]
        transfer_summary_rows.append(
            {
                "pool": pool,
                "n_reactions": len(selected),
                "mean_transfer_delta_top1": float(
                    np.mean([row["transfer_delta_top1"] for row in selected])
                ),
                "mean_transfer_delta_mrr": float(
                    np.mean([row["transfer_delta_mrr"] for row in selected])
                ),
            }
        )

    summary_rows = []
    for pool in POOL_ORDER:
        selected = [row for row in descriptor_rows if row["pool"] == pool]
        record = {"pool": pool, "n_reactions": len(selected)}
        for feature in M2_FEATURES:
            values = np.asarray(
                [float(row[feature]) for row in selected if row[feature] is not None],
                dtype=np.float64,
            )
            record[f"{feature}_defined_n"] = len(values)
            record[f"{feature}_mean"] = float(values.mean()) if len(values) else None
            record[f"{feature}_median"] = float(np.median(values)) if len(values) else None
        record["mean_delta_top1"] = float(
            np.mean([row["mean_20seed_delta_top1"] for row in selected])
        )
        record["mean_delta_mrr"] = float(
            np.mean([row["mean_20seed_delta_mrr"] for row in selected])
        )
        summary_rows.append(record)

    own_coverage = {
        "historical_cap10": len(reference_by_id),
        **{
            pool: int(expanded_manifests[pool]["test_reactions_covered"])
            for pool in POOL_ORDER[1:]
        },
    }
    coverage_rows = [
        {
            "scope": "own_top10_coverage",
            "pool": pool,
            "denominator": 5004,
            "covered_reactions": count,
            "excluded_reactions": 5004 - count,
            "coverage_fraction": count / 5004,
        }
        for pool, count in own_coverage.items()
    ]
    coverage_rows.extend(
        [
            {
                "scope": "four_pool_common_top10_coverage",
                "pool": "all_four",
                "denominator": 5004,
                "covered_reactions": len(common_ids),
                "excluded_reactions": 5004 - len(common_ids),
                "coverage_fraction": len(common_ids) / 5004,
            },
            {
                "scope": "common_within_historical_anchor",
                "pool": "all_four",
                "denominator": len(reference_by_id),
                "covered_reactions": len(common_ids),
                "excluded_reactions": len(reference_by_id) - len(common_ids),
                "coverage_fraction": len(common_ids) / len(reference_by_id),
            },
        ]
    )

    l1_rows, l1_links, l1_d5_inputs = _l1_external_links(
        args.l1_diagnostics, args.d5_root
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "m2_reaction_pool_descriptors.csv", descriptor_rows)
    _write_csv(output / "m2_pool_summary.csv", summary_rows)
    _write_csv(output / "m2_descriptor_effect_associations.csv", association_rows)
    _write_csv(output / "m2_descriptor_effect_quartiles.csv", quartile_rows)
    _write_csv(output / "m2_l1_external_reaction_links.csv", l1_rows)
    _write_csv(output / "m2_l1_external_associations.csv", l1_links)
    _write_csv(output / "m2_transfer_loss_reactions.csv", transfer_rows)
    _write_csv(output / "m2_transfer_loss_summary.csv", transfer_summary_rows)
    _write_csv(output / "m2_transfer_loss_associations.csv", transfer_association_rows)
    _write_csv(output / "m2_transfer_loss_quartiles.csv", transfer_quartile_rows)
    _write_csv(output / "m2_coverage_selection_audit.csv", coverage_rows)
    manifest = {
        "schema_version": 1,
        "record_kind": "dd_exploratory_candidate_pool_shift_diagnostic",
        "protocol_id": M2_PROTOCOL_ID,
        "comparator": "four frozen candidate pools truncated to cap 10",
        "single_intended_change": (
            "descriptive measurement of frozen candidate-list composition and its association "
            "with frozen within-pool effect and paired loss of transfer"
        ),
        "settings": {
            "seeds": list(SEEDS),
            "common_covered_reactions": len(common_ids),
            "official_test_reactions": 5004,
            "historical_anchor_covered_reactions": len(reference_by_id),
            "candidate_cap": 10,
            "candidate_identity": "canonical isomeric SMILES; fragment-order invariant",
            "morgan": "radius=2; fpSize=2048; includeChirality=True",
            "association": "Spearman; descriptive/post hoc; undefined values excluded without imputation",
            "transfer_loss": (
                "expanded-pool mean 20-seed effect minus matched historical mean 20-seed effect"
            ),
            "stored_prior_mass_entropy": (
                "within-pool dispersion only; not calibrated confidence and not used as a cross-pool shift"
            ),
            "quartiles": "average-rank quartiles; constant descriptors omitted",
            "claim_status": "exploratory_post_hoc",
        },
        "inputs": {
            "anchor_prediction_files": _anchor_prediction_fingerprints(
                args.anchor_prediction_root
            ),
            "k1_frozen_chains": expanded_inputs,
            "candidate_pools": pool_inputs,
            "source_csv": file_fingerprint(args.source_csv),
            "metadata_csv": file_fingerprint(args.metadata_csv),
            "l1_d5_frozen_chain": l1_d5_inputs,
        },
        "outputs": {
            name: file_fingerprint(output / file_name)
            for name, file_name in {
                "reaction_pool_descriptors": "m2_reaction_pool_descriptors.csv",
                "pool_summary": "m2_pool_summary.csv",
                "descriptor_effect_associations": "m2_descriptor_effect_associations.csv",
                "descriptor_effect_quartiles": "m2_descriptor_effect_quartiles.csv",
                "l1_external_reaction_links": "m2_l1_external_reaction_links.csv",
                "l1_external_associations": "m2_l1_external_associations.csv",
                "transfer_loss_reactions": "m2_transfer_loss_reactions.csv",
                "transfer_loss_summary": "m2_transfer_loss_summary.csv",
                "transfer_loss_associations": "m2_transfer_loss_associations.csv",
                "transfer_loss_quartiles": "m2_transfer_loss_quartiles.csv",
                "coverage_selection_audit": "m2_coverage_selection_audit.csv",
            }.items()
        },
        "round_jk_approval": approval,
        "environment": _environment(),
        "runtime_seconds": time.time() - started,
        "training_performed": False,
        "retuning_performed": False,
        "candidate_generation_performed": False,
        "effect_outcomes_loaded_from_frozen_predictions": True,
        "official_test_labels_loaded_post_freeze": True,
        "official_test_label_use": (
            "reference-containment audit and label-aware non-reference-candidate similarity "
            "on the frozen common-covered subset"
        ),
        "official_test_labels_used_for_training_selection_or_retuning": False,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _transfer_inference(
    values: Sequence[float],
    products: Sequence[str],
    bootstrap_samples: int,
    permutations: int,
    bootstrap_seed: int,
    permutation_seed: int,
) -> dict:
    vector = np.asarray(values, dtype=np.float64)
    clusters = np.asarray(products, dtype=object)
    if vector.ndim != 1 or len(vector) != len(clusters) or not len(vector):
        raise ValueError("M2b values and product clusters must be aligned and non-empty.")
    if not np.all(np.isfinite(vector)) or len(set(clusters.tolist())) != len(clusters):
        raise ValueError("M2b requires finite values and one unique canonical product per reaction.")
    interval = product_cluster_bootstrap(
        vector[None, :], clusters, bootstrap_samples, bootstrap_seed
    )
    p_value = clustered_sign_flip(
        vector[None, :], clusters, permutations, permutation_seed
    )
    return {
        **interval,
        "paired_sign_flip_p_two_sided": p_value,
    }


def run_m2b(args: argparse.Namespace) -> dict:
    started = time.time()
    approval = _m2b_approval(args.approval_record, args.analysis_plan)
    output = _assert_isolated_output(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite M2b output: {output}")

    m2_manifest_path = Path(args.m2_manifest)
    m2_manifest = _load_json(m2_manifest_path)
    if (
        m2_manifest.get("protocol_id") != M2_PROTOCOL_ID
        or int(m2_manifest.get("settings", {}).get("common_covered_reactions", 0)) != 3814
        or m2_manifest.get("training_performed") is not False
        or m2_manifest.get("retuning_performed") is not False
        or m2_manifest.get("candidate_generation_performed") is not False
    ):
        raise ValueError("M2b requires the frozen 3,814-reaction M2 v2 manifest.")
    transfer_path, transfer_input = _resolve_frozen_file(
        m2_manifest["outputs"]["transfer_loss_reactions"],
        m2_manifest_path.parent / "m2_transfer_loss_reactions.csv",
        "M2 transfer-loss reactions",
    )
    rows = _read_csv(transfer_path)
    by_pool = {pool: [] for pool in POOL_ORDER[1:]}
    for row in rows:
        pool = str(row["pool"])
        if pool not in by_pool:
            raise ValueError(f"Unexpected M2b pool: {pool}")
        by_pool[pool].append(row)

    expected_identity = None
    result_rows = []
    for pool_index, pool in enumerate(POOL_ORDER[1:]):
        selected = sorted(by_pool[pool], key=lambda row: int(row["reaction_id"]))
        identity = [
            (int(row["reaction_id"]), str(row["canonical_product"])) for row in selected
        ]
        if len(selected) != 3814 or len(set(identity)) != 3814:
            raise ValueError(f"M2b {pool} does not contain 3,814 unique reaction/product pairs.")
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError("M2b pool reaction/product pairing differs.")
        products = [canonicalize_smiles(product) for _, product in identity]
        if None in products or len(set(products)) != len(products):
            raise ValueError("M2b requires one unique canonical product per reaction.")
        for metric_index, metric in enumerate(("top1", "mrr")):
            offset = pool_index * 2 + metric_index
            inference = _transfer_inference(
                [float(row[f"transfer_delta_{metric}"]) for row in selected],
                products,
                args.bootstrap_samples,
                args.permutations,
                args.bootstrap_seed + offset,
                args.permutation_seed + offset,
            )
            relation = (
                "negative_ci_excludes_zero"
                if inference["ci95_high"] < 0
                else "positive_ci_excludes_zero"
                if inference["ci95_low"] > 0
                else "ci_includes_zero"
            )
            result_rows.append(
                {
                    "pool": pool,
                    "metric": metric,
                    "n_reactions": len(selected),
                    "mean_transfer_loss": inference["effect"],
                    "product_cluster_ci95_low": inference["ci95_low"],
                    "product_cluster_ci95_high": inference["ci95_high"],
                    "paired_sign_flip_p_two_sided": inference[
                        "paired_sign_flip_p_two_sided"
                    ],
                    "interval_relation_to_zero": relation,
                    "bootstrap_samples": args.bootstrap_samples,
                    "bootstrap_rng_seed": args.bootstrap_seed + offset,
                    "sign_flip_draws": args.permutations,
                    "sign_flip_rng_seed": args.permutation_seed + offset,
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "m2b_transfer_loss_inference.csv"
    _write_csv(result_path, result_rows)
    manifest = {
        "schema_version": 1,
        "record_kind": "dd_exploratory_direct_transfer_loss_inference",
        "protocol_id": M2B_PROTOCOL_ID,
        "comparator": "expanded-pool effect minus matched historical effect",
        "single_intended_change": (
            "add direct paired frozen-reaction inference to the approved M2 transfer-loss estimand"
        ),
        "settings": {
            "common_covered_reactions": 3814,
            "seeds_already_averaged_in_frozen_m2": list(SEEDS),
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_rng_seeds": list(range(args.bootstrap_seed, args.bootstrap_seed + 6)),
            "sign_flip_draws": args.permutations,
            "sign_flip_rng_seeds": list(range(args.permutation_seed, args.permutation_seed + 6)),
            "claim_status": "exploratory_post_hoc",
            "inference_unit": "unique canonical product/reaction after averaging 20 paired seeds",
        },
        "inputs": {
            "m2_manifest": file_fingerprint(m2_manifest_path),
            "m2_transfer_loss_reactions": transfer_input,
        },
        "outputs": {"transfer_loss_inference": file_fingerprint(result_path)},
        "round_jk_approval": approval,
        "environment": _environment(),
        "runtime_seconds": time.time() - started,
        "training_performed": False,
        "retuning_performed": False,
        "candidate_generation_performed": False,
        "test_partition_loaded_from_frozen_m2_only": True,
    }
    atomic_json(output / "manifest.json", manifest)
    return manifest


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--approval-record", default="docs/round_jk_approval.json")
    parser.add_argument("--analysis-plan", default="docs/analysis_plan.md")
    parser.add_argument("--output-dir", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m1 = sub.add_parser("m1")
    _common(m1)
    m1.add_argument("--anchor-prediction-root", required=True)
    m1.add_argument("--permutations", type=int, default=100_000)
    m1.add_argument("--permutation-seed", type=int, default=2030)
    m1.add_argument("--bootstrap-samples", type=int, default=10_000)
    m1.add_argument("--bootstrap-seed", type=int, default=2040)
    m2 = sub.add_parser("m2")
    _common(m2)
    m2.add_argument("--anchor-prediction-root", required=True)
    m2.add_argument("--k1-manifest", action="append", required=True)
    m2.add_argument("--pool-jsonl", action="append", required=True)
    m2.add_argument("--historical-pool-jsonl", required=True)
    m2.add_argument("--source-csv", default="data/uspto_smiles.csv")
    m2.add_argument("--metadata-csv", default="data/uspto_reaction_metadata.csv")
    m2.add_argument("--l1-diagnostics", required=True)
    m2.add_argument("--d5-root", required=True)
    # Frozen K1 cap-10 coverage intersection.  J1's 3,939 count belongs to the
    # untruncated full pools and is intentionally not reused here.
    m2.add_argument("--expected-common-count", type=int, default=3814)
    m2b = sub.add_parser("m2b")
    _common(m2b)
    m2b.add_argument("--m2-manifest", required=True)
    m2b.add_argument("--bootstrap-samples", type=int, default=10_000)
    m2b.add_argument("--bootstrap-seed", type=int, default=2050)
    m2b.add_argument("--permutations", type=int, default=100_000)
    m2b.add_argument("--permutation-seed", type=int, default=2060)
    args = parser.parse_args()
    if args.command == "m2" and (len(args.k1_manifest) != 3 or len(args.pool_jsonl) != 3):
        parser.error("m2 needs exactly three K1 manifests and three pool JSONLs in AiZ, LocalRetro, merged order")
    return args


def main() -> None:
    args = parse_args()
    result = {"m1": run_m1, "m2": run_m2, "m2b": run_m2b}[args.command](args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
