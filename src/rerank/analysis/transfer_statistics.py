"""Frozen 20-seed own-pool and common-cohort transfer effects for N3/N4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from rerank.analysis.analyze_predictions import product_cluster_bootstrap
from rerank.study_data import canonicalize_smiles


SEEDS = tuple(range(42, 62))
METRICS = ("top1", "mrr")


def common_cohort(path: str | Path) -> tuple[list[int], np.ndarray]:
    products = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["pool"] == "aizynthfinder_only":
                rid = int(row["reaction_id"])
                if rid in products:
                    raise ValueError("Duplicate common reaction ID.")
                products[rid] = canonicalize_smiles(row["canonical_product"])
    if len(products) != 3814 or len(set(products.values())) != 3814:
        raise ValueError("Transfer analysis requires exactly 3,814 distinct common products.")
    ids = sorted(products)
    return ids, np.asarray([products[rid] for rid in ids], dtype=object)


def historical_g1_ranks(path: str | Path, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    arrays = []
    for arm in ("baseline", "augmented"):
        by_seed = []
        for seed in SEEDS:
            with (Path(path) / f"{arm}_seed_{seed}.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = {
                    int(row["reaction_id"]): int(row["reranked_rank"])
                    for row in csv.DictReader(handle)
                }
            if not set(ids).issubset(rows):
                raise ValueError("Historical G1 prediction missing a common reaction.")
            by_seed.append([rows[rid] for rid in ids])
        arrays.append(np.asarray(by_seed, dtype=np.int16))
    return arrays[0], arrays[1]


def k1_ranks(path: str | Path, ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    rows = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rid = int(row["reaction_id"])
            if rid in ids:
                rows[rid] = row
    if set(rows) != set(ids) or not all(rows[rid].get("covered") for rid in ids):
        raise ValueError("Expanded pool does not cover the common cohort.")
    outputs = []
    for arm in ("baseline", "augmented"):
        outputs.append(
            np.asarray(
                [[int(rows[rid][f"{arm}_rank_seed_{seed}"]) for rid in ids] for seed in SEEDS],
                dtype=np.int16,
            )
        )
    return outputs[0], outputs[1]


def _scores(ranks: np.ndarray, metric: str) -> np.ndarray:
    values = np.asarray(ranks)
    if values.ndim != 2 or values.shape[0] != len(SEEDS) or np.any(values < 1):
        raise ValueError("Rank matrix must be 20 paired seeds by covered reactions.")
    return (values == 1).astype(np.float64) if metric == "top1" else 1.0 / values


def _independent_crossed_interval(
    old: np.ndarray, new: np.ndarray, clusters: np.ndarray, draws: int, seed: int
) -> tuple[float, float]:
    labels = np.asarray(clusters, dtype=object)
    groups = [np.flatnonzero(labels == value) for value in dict.fromkeys(labels.tolist())]
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        picked = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[item] for item in picked])
        old_seeds = rng.integers(0, old.shape[0], size=old.shape[0])
        new_seeds = rng.integers(0, new.shape[0], size=new.shape[0])
        estimates[index] = float(
            new[np.ix_(new_seeds, rows)].mean() - old[np.ix_(old_seeds, rows)].mean()
        )
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def transfer_summary(
    historical: tuple[np.ndarray, np.ndarray],
    expanded: dict[str, tuple[np.ndarray, np.ndarray]],
    clusters: np.ndarray,
    *,
    draws: int = 10_000,
    seed: int = 2180,
) -> list[dict]:
    rows = []
    for pool_index, (pool, arms) in enumerate(expanded.items()):
        for metric_index, metric in enumerate(METRICS):
            old_effect = _scores(historical[1], metric) - _scores(historical[0], metric)
            new_effect = _scores(arms[1], metric) - _scores(arms[0], metric)
            if old_effect.shape != new_effect.shape or old_effect.shape[1] != len(clusters):
                raise ValueError("Transfer arrays are not matched on the frozen common cohort.")
            loss = new_effect - old_effect
            p_ci = product_cluster_bootstrap(
                loss, clusters, draws, seed + pool_index * 10 + metric_index
            )
            crossed_low, crossed_high = _independent_crossed_interval(
                old_effect, new_effect, clusters, draws, seed + 100 + pool_index * 10 + metric_index
            )
            rows.append(
                {
                    "pool": pool,
                    "metric": metric,
                    "common_reactions": len(clusters),
                    "historical_effect": float(old_effect.mean()),
                    "expanded_effect": float(new_effect.mean()),
                    "transfer_loss": float(loss.mean()),
                    "product_ci95_low": p_ci["ci95_low"],
                    "product_ci95_high": p_ci["ci95_high"],
                    "crossed_ci95_low": crossed_low,
                    "crossed_ci95_high": crossed_high,
                }
            )
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty effect table.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
